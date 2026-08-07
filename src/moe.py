"""
Mixture-of-Experts layers.

Notebook 17 derives these; this module makes them reusable.

Provides:
- Router: top-k gating, with both balancing schemes in use today
- MoEFeedForward: a drop-in replacement for a dense feed-forward layer
- moe_aux_losses: collect the auxiliary losses from a whole model

The point of MoE is to decouple two numbers that are welded together in a dense
model: total parameters and FLOPs per token. Replace one FFN with E experts and
route each token to only top_k of them, and the layer holds E times the
parameters while doing top_k/E of the work. Mixtral-8x7B has ~47B parameters but
spends the compute of a ~13B dense model per token.

What you buy with that is capacity; what you pay is memory bandwidth and a
routing problem. Routing is the hard part -- left alone, a router collapses onto
a few favourite experts and the rest never train. Both mechanisms below exist to
prevent exactly that.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modern import SwiGLU


class Router(nn.Module):
    """
    Top-k router with load balancing.

    A single linear layer scores every expert per token; the top_k highest win.
    Two balancing schemes are implemented, and they are alternatives:

    1. Auxiliary loss (Switch Transformer, GShard). Add a differentiable
       penalty that is minimized when load is uniform. Simple and effective,
       but it fights the language-modeling loss: it pushes tokens toward
       experts that are *not* the best fit, so it costs a little quality.

    2. Loss-free bias (DeepSeek-V3). Keep a per-expert bias used *only* to
       pick the top_k, and nudge it after each step: overloaded experts get a
       lower bias, starved experts a higher one. It never enters the loss and
       never distorts a gradient. Because the returned gate weights come from
       the *unbiased* scores, the bias steers routing without corrupting the
       values the experts are combined with. This is the current default in
       frontier MoE models.

    Setting bias_update_rate > 0 enables scheme 2; aux_loss_coef > 0 enables
    scheme 1. Enabling both is legal but redundant.
    """

    def __init__(self, d_model, num_experts, top_k=2, aux_loss_coef=0.01,
                 z_loss_coef=0.001, bias_update_rate=0.0, normalize_gates=True):
        """
        Args:
            d_model: Model dimension
            num_experts: Number of experts (E)
            top_k: Experts activated per token
            aux_loss_coef: Weight for the load-balancing auxiliary loss
            z_loss_coef: Weight for the router z-loss, which keeps router
                logits small and stops the softmax from saturating
            bias_update_rate: Step size for loss-free bias balancing. 0 disables.
            normalize_gates: Renormalize the top_k gate weights to sum to 1
        """
        super().__init__()

        assert 1 <= top_k <= num_experts, "top_k must be in [1, num_experts]"

        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef
        self.bias_update_rate = bias_update_rate
        self.normalize_gates = normalize_gates

        # No bias term: a bias here would be a constant preference for an
        # expert regardless of the token, which is the opposite of routing.
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # Not a Parameter. It is updated by a rule, not by a gradient, so it
        # must not appear in the optimizer -- but it does belong in state_dict,
        # hence a persistent buffer.
        self.register_buffer("expert_bias", torch.zeros(num_experts))

    def forward(self, x_flat):
        """
        Args:
            x_flat: (num_tokens, d_model)

        Returns:
            topk_idx: (num_tokens, top_k) chosen expert indices
            gates: (num_tokens, top_k) combine weights, from unbiased scores
            info: dict with 'aux_loss', 'z_loss', 'load_fraction',
                'mean_prob', and 'max_load_ratio' (1.0 == perfectly balanced,
                num_experts == total collapse)
        """
        logits = self.gate(x_flat)                      # (N, E)
        probs = F.softmax(logits, dim=-1)

        # Selection may use the bias; the returned weights never do.
        scores = probs + self.expert_bias if self.bias_update_rate > 0 else probs
        _, topk_idx = torch.topk(scores, self.top_k, dim=-1)

        gates = probs.gather(-1, topk_idx)
        if self.normalize_gates:
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        info = self._balance_stats(probs, topk_idx, logits)

        if self.bias_update_rate > 0 and self.training:
            self._update_bias(info["load_fraction"])

        return topk_idx, gates, info

    def _balance_stats(self, probs, topk_idx, logits):
        """Compute the balancing losses and load statistics."""
        num_tokens = probs.size(0)
        E = self.num_experts

        # f_i: fraction of all assignments that went to expert i
        one_hot = F.one_hot(topk_idx, E).sum(dim=1).float()  # (N, E) counts
        load_fraction = one_hot.sum(dim=0) / max(num_tokens * self.top_k, 1)

        # P_i: mean router probability for expert i
        mean_prob = probs.mean(dim=0)

        # Switch Transformer auxiliary loss: E * sum_i f_i * P_i.
        # Uniform load gives exactly 1.0; concentration drives it toward E.
        # f_i is not differentiable (it comes from a topk), but P_i is -- so
        # the gradient flows through the probabilities, nudging the router to
        # assign less probability to experts that are already busy.
        aux_loss = E * torch.sum(load_fraction.detach() * mean_prob)

        # Router z-loss: penalize large logits so the softmax stays in a
        # well-conditioned range. Cheap insurance against router instability.
        z_loss = logits.logsumexp(dim=-1).pow(2).mean()

        return {
            "aux_loss": self.aux_loss_coef * aux_loss,
            "z_loss": self.z_loss_coef * z_loss,
            "load_fraction": load_fraction.detach(),
            "mean_prob": mean_prob.detach(),
            "max_load_ratio": (load_fraction.max() * E).detach(),
        }

    @torch.no_grad()
    def _update_bias(self, load_fraction):
        """
        Loss-free balancing update (DeepSeek-V3).

        Target load is 1/E per expert. Push the bias down where load is above
        target and up where it is below. Only the sign of the error is used, so
        the update size is constant and cannot blow up on an outlier batch.
        """
        target = 1.0 / self.num_experts
        error = load_fraction - target
        self.expert_bias -= self.bias_update_rate * torch.sign(error)


class MoEFeedForward(nn.Module):
    """
    Sparse mixture-of-experts feed-forward layer.

    Swaps in for SwiGLU or FeedForward inside a Transformer block.

    Two design details that matter in practice:

    Capacity. Experts are batched to a fixed size on real hardware, so each
    gets a token budget: capacity = capacity_factor * top_k * N / E. Tokens
    arriving past it are *dropped* -- they skip the layer and pass through on
    the residual stream. A capacity factor of 1.0 is tight and drops a lot;
    1.25 is the usual compromise.

    Shared experts (DeepSeekMoE). One always-on expert handles the general
    patterns every token needs, which frees the routed experts to specialize
    instead of each relearning the common case.

    This implementation loops over experts, which is clear but not fast. A
    production kernel sorts tokens by expert and issues one grouped GEMM, and
    across devices the dispatch is an all-to-all collective (see notebook 22).
    """

    def __init__(self, d_model, num_experts=8, top_k=2, d_ff=None,
                 num_shared_experts=0, capacity_factor=1.25, dropout=0.0,
                 aux_loss_coef=0.01, z_loss_coef=0.001, bias_update_rate=0.0):
        """
        Args:
            d_model: Model dimension
            num_experts: Number of routed experts
            top_k: Experts activated per token
            d_ff: Hidden dimension of each expert (default: SwiGLU's default)
            num_shared_experts: Always-on experts applied to every token
            capacity_factor: Token budget multiplier per expert. None disables
                dropping entirely (useful for tests and toy runs).
            dropout: Dropout rate inside experts
            aux_loss_coef: Load-balancing loss weight
            z_loss_coef: Router z-loss weight
            bias_update_rate: Loss-free balancing step size. 0 disables.
        """
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.router = Router(
            d_model, num_experts, top_k,
            aux_loss_coef=aux_loss_coef,
            z_loss_coef=z_loss_coef,
            bias_update_rate=bias_update_rate,
        )

        self.experts = nn.ModuleList([
            SwiGLU(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        self.shared_experts = nn.ModuleList([
            SwiGLU(d_model, d_ff, dropout) for _ in range(num_shared_experts)
        ])

        # Auxiliary losses cannot be returned without breaking the drop-in
        # signature, so they are stashed here and collected by moe_aux_losses().
        self.last_info = None

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, d = x.shape
        x_flat = x.reshape(-1, d)                    # (N, d)
        num_tokens = x_flat.size(0)

        topk_idx, gates, info = self.router(x_flat)

        capacity = None
        if self.capacity_factor is not None:
            capacity = int(
                self.capacity_factor * self.top_k * num_tokens / self.num_experts
            )
            capacity = max(capacity, 1)

        out = torch.zeros_like(x_flat)
        dropped = 0

        for e, expert in enumerate(self.experts):
            # Every (token, slot) pair that selected this expert
            token_idx, slot_idx = (topk_idx == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue

            if capacity is not None and token_idx.numel() > capacity:
                dropped += token_idx.numel() - capacity
                # Keep the earliest arrivals, as a real dispatch buffer would
                token_idx = token_idx[:capacity]
                slot_idx = slot_idx[:capacity]

            weight = gates[token_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)

        # Shared experts run on everything, ungated
        for expert in self.shared_experts:
            out = out + expert(x_flat)

        info["dropped_tokens"] = dropped
        info["drop_rate"] = dropped / max(num_tokens * self.top_k, 1)
        info["capacity"] = capacity
        self.last_info = info

        return out.view(B, T, d)


def moe_aux_losses(module):
    """
    Sum the auxiliary losses from every MoEFeedForward in a model.

    Add the result to the language-modeling loss before calling backward:

        logits, loss, _ = model(x, targets=y)
        loss = loss + moe_aux_losses(model)
        loss.backward()

    Forgetting this is the classic MoE bug: training looks fine, then you check
    the load statistics and find two experts doing all the work.

    Args:
        module: Any nn.Module; its submodules are searched

    Returns:
        Scalar tensor (0.0 if the model has no MoE layers)
    """
    total = 0.0
    for m in module.modules():
        if isinstance(m, MoEFeedForward) and m.last_info is not None:
            total = total + m.last_info["aux_loss"] + m.last_info["z_loss"]
    return total


def moe_load_report(module):
    """
    Per-layer load statistics, for spotting collapse.

    Returns:
        List of dicts with 'layer', 'max_load_ratio', 'drop_rate', and
        'load_fraction'. A max_load_ratio near 1.0 is balanced; near
        num_experts means the router has collapsed onto one expert.
    """
    report = []
    for i, m in enumerate(
        [m for m in module.modules() if isinstance(m, MoEFeedForward)]
    ):
        if m.last_info is None:
            continue
        report.append({
            "layer": i,
            "max_load_ratio": float(m.last_info["max_load_ratio"]),
            "drop_rate": m.last_info["drop_rate"],
            "load_fraction": m.last_info["load_fraction"].tolist(),
        })
    return report
