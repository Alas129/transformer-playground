"""
Low-Rank Adaptation (LoRA).

Notebook 12 derives this; this module makes it reusable, and notebook 26 needs
the multi-adapter variant for serving.

Provides:
- LoRALinear: a frozen nn.Linear plus a trainable low-rank update
- apply_lora: swap matching Linear layers in a model for LoRALinear
- merge_lora: fold adapters into the base weights (zero inference cost)
- lora_parameters / lora_summary: what is actually being trained
- MultiAdapterLoRALinear: many adapters, one batch (multi-tenant serving)

The idea: a full fine-tune updates W (d_out x d_in) entirely, but the *update*
a fine-tune discovers has low intrinsic rank. So do not learn dW directly --
learn B @ A with r << min(d_in, d_out) and keep W frozen:

    W' = W + (alpha / r) * B @ A

For d_model = 4096 and r = 8 that is 65k trainable parameters instead of 16.7M,
a 250x reduction. Optimizer state shrinks with it, which is usually the real
memory win.

B is initialized to zero, so at step 0 the adapter contributes exactly nothing
and the model is byte-for-byte the base model. Training starts from the
pretrained function rather than from a perturbation of it.
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    A frozen linear layer with a trainable low-rank update.

    Wraps an existing nn.Linear rather than replacing it, so the pretrained
    weights stay exactly where they were and can be restored or merged.
    """

    def __init__(self, base_layer, r=8, alpha=16, dropout=0.0):
        """
        Args:
            base_layer: The nn.Linear to adapt. Its parameters are frozen.
            r: Rank of the update. 4-16 is typical; higher mostly wastes memory.
            alpha: Scaling numerator. The update is scaled by alpha/r so that
                changing r does not require re-tuning the learning rate.
                alpha = 2r is a common default.
            dropout: Dropout applied to the LoRA branch input only
        """
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(base_layer).__name__}")

        self.base = base_layer
        for param in self.base.parameters():
            param.requires_grad = False

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # A is randomly initialized, B is zero. Both zero would leave the
        # gradient of each stuck at zero; both random would corrupt the base
        # model at step 0. This asymmetry is the point.
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):
        """
        Args:
            x: (..., in_features)
        Returns:
            (..., out_features)
        """
        out = self.base(x)

        if self.merged:
            # Already folded into base.weight; adding it again would double it.
            return out

        # x @ A.T -> (..., r), then @ B.T -> (..., out_features).
        # Going through the rank-r bottleneck is what makes this cheap: two
        # thin matmuls instead of one d_out x d_in.
        update = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + update * self.scaling

    @torch.no_grad()
    def merge(self):
        """
        Fold the adapter into the base weight.

        After merging, this layer is an ordinary Linear with no extra work at
        inference time. That is LoRA's other advantage over adapter layers that
        add depth: the deployed model has identical latency to the base.
        """
        if self.merged:
            return
        self.base.weight += self.scaling * (self.lora_B @ self.lora_A)
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        """Undo merge(), restoring the separate adapter."""
        if not self.merged:
            return
        self.base.weight -= self.scaling * (self.lora_B @ self.lora_A)
        self.merged = False

    def extra_repr(self):
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, merged={self.merged}"
        )


def apply_lora(model, target_names=("W_q", "W_v"), r=8, alpha=16, dropout=0.0):
    """
    Replace matching nn.Linear submodules with LoRALinear, in place.

    Args:
        model: The model to adapt
        target_names: Attribute names to match. The LoRA paper found adapting
            the query and value projections is the best quality-per-parameter
            trade-off, which is why that is the default. Adapting everything
            including the MLP does slightly better for more memory.
        r: Rank
        alpha: Scaling numerator
        dropout: Dropout on the LoRA branch

    Returns:
        List of the qualified names that were adapted.
    """
    adapted = []

    for module_name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if child_name in target_names and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                adapted.append(
                    f"{module_name}.{child_name}" if module_name else child_name
                )

    if not adapted:
        raise ValueError(
            f"no nn.Linear submodule matched {target_names}. "
            "Check the attribute names on your model."
        )

    # Freeze everything that is not a LoRA parameter.
    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name

    return adapted


def merge_lora(model):
    """
    Merge every LoRALinear in a model.

    Returns:
        Number of layers merged.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            count += 1
    return count


def unmerge_lora(model):
    """Unmerge every LoRALinear in a model. Returns the count."""
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
            count += 1
    return count


def lora_parameters(model):
    """Yield only the trainable LoRA parameters, for the optimizer."""
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield param


def lora_summary(model):
    """
    Trainable vs total parameter counts.

    Returns:
        dict with 'trainable', 'total', 'percent'
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "percent": 100.0 * trainable / max(total, 1),
    }


class MultiAdapterLoRALinear(nn.Module):
    """
    One frozen base weight, many adapters, mixed within a single batch.

    This is the layer that makes multi-tenant LoRA serving work (the S-LoRA
    idea, notebook 26). A provider hosting one base model for a thousand
    customers cannot merge -- merging bakes in one adapter. Nor can it batch
    per-adapter, which would starve the GPU.

    Instead: keep adapters stacked as tensors, and gather the right one per
    sequence. The base matmul stays a single big GEMM shared by the whole batch;
    only the thin rank-r branch is per-adapter. Since the base weight dominates
    the FLOPs, throughput stays close to serving a single model.

    Adapter index -1 means "base model only", so untuned requests share the
    batch too.
    """

    def __init__(self, base_layer, num_adapters, r=8, alpha=16):
        """
        Args:
            base_layer: The shared frozen nn.Linear
            num_adapters: How many adapters to hold
            r: Rank (shared by all adapters here; real servers allow per-adapter
                ranks and pad to the max)
            alpha: Scaling numerator
        """
        super().__init__()

        self.base = base_layer
        for param in self.base.parameters():
            param.requires_grad = False

        self.num_adapters = num_adapters
        self.r = r
        self.scaling = alpha / r

        # Stacked adapter banks: (num_adapters, r, in) and (num_adapters, out, r)
        self.lora_A = nn.Parameter(
            torch.zeros(num_adapters, r, base_layer.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(num_adapters, base_layer.out_features, r)
        )
        for i in range(num_adapters):
            nn.init.kaiming_uniform_(self.lora_A.data[i], a=math.sqrt(5))

    def forward(self, x, adapter_ids):
        """
        Args:
            x: (batch, seq_len, in_features)
            adapter_ids: (batch,) long tensor. Entry -1 selects base only.

        Returns:
            (batch, seq_len, out_features)
        """
        out = self.base(x)

        active = adapter_ids >= 0
        if not bool(active.any()):
            return out

        idx = adapter_ids.clamp_min(0)
        # Gather each sequence's adapter: (batch, r, in) and (batch, out, r)
        A = self.lora_A[idx]
        B = self.lora_B[idx]

        # Per-sequence batched matmuls. einsum keeps the batch dimension
        # separate so every sequence uses its own adapter in one kernel.
        h = torch.einsum("bsi,bri->bsr", x, A)
        update = torch.einsum("bsr,bor->bso", h, B) * self.scaling

        # Zero out the rows that asked for base-only
        update = update * active.view(-1, 1, 1).to(update.dtype)

        return out + update
