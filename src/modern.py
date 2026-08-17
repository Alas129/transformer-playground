"""
Modern Transformer components (LLaMA-era).

Notebook 11 derives these; this module turns them into reusable code so the
later notebooks (14-26) can build on a modern stack instead of the 2017 one.

Provides:
- RMSNorm: LayerNorm without mean-centering
- RotaryEmbedding / apply_rope: relative position by rotating Q and K
- SwiGLU: gated feed-forward network
- GroupedQueryAttention: many query heads sharing few K/V heads, with KV cache
- ModernBlock: Pre-RMSNorm + GQA + SwiGLU
- ModernGPT: the decoder-only stack used by LLaMA-style models

Contrast with embeddings.py / transformer.py / gpt.py, which implement the
original 2017 design (LayerNorm + learned absolute positions + GELU MLP + MHA).
The two stacks are deliberately kept side by side so they can be compared.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Compared with LayerNorm, two things are gone: the mean subtraction and the
    bias. Only re-scaling is left. That removes two reductions per call and
    costs essentially no quality, which is why LLaMA-era models all use it.
    """

    def __init__(self, d_model, eps=1e-6):
        """
        Args:
            d_model: Model dimension
            eps: Added inside the sqrt for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        Args:
            x: (..., d_model)
        Returns:
            (..., d_model)
        """
        # The reduction runs in float32 even under mixed precision: squaring
        # activations in fp16 overflows easily.
        dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * rms).to(dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Precomputed cos/sin tables for Rotary Position Embeddings (RoPE).

    Position is applied by *rotating* each 2-D slice of Q and K by an angle
    proportional to the position. The dot product q_m . k_n then depends only on
    the relative offset (m - n), never on m or n alone.

    Frequencies are geometrically spaced: the fastest-rotating pairs resolve
    local order, the slowest-rotating pairs carry long-range position. That
    frequency view is what all the context-extension tricks in notebook 15
    (position interpolation, NTK-aware scaling, YaRN) manipulate.
    """

    def __init__(self, head_dim, max_seq_len=4096, base=10000.0):
        """
        Args:
            head_dim: Per-head dimension (must be even)
            max_seq_len: Longest position to precompute
            base: The "theta" base; larger spreads frequencies further apart
        """
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.head_dim = head_dim
        self.base = base

        # inv_freq[i] = 1 / base^(2i/head_dim), for i in [0, head_dim/2)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float)
        angles = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim/2)

        self.register_buffer("cos_table", angles.cos(), persistent=False)
        self.register_buffer("sin_table", angles.sin(), persistent=False)

    def forward(self, seq_len, offset=0):
        """
        Args:
            seq_len: Number of positions needed
            offset: Absolute position of the first one. Non-zero during cached
                generation, where x holds only the newest token but its true
                position is len(cache).

        Returns:
            cos, sin: each (seq_len, head_dim/2)
        """
        end = offset + seq_len
        assert end <= self.cos_table.size(0), (
            f"position {end} exceeds precomputed max_seq_len "
            f"{self.cos_table.size(0)}"
        )
        return self.cos_table[offset:end], self.sin_table[offset:end]


def apply_rope(x, cos, sin):
    """
    Rotate x by the given angles.

    Args:
        x: (batch, num_heads, seq_len, head_dim)
        cos: (seq_len, head_dim/2)
        sin: (seq_len, head_dim/2)

    Returns:
        Rotated tensor, same shape as x.

    Pairing convention: dimension i is paired with dimension i + head_dim/2
    (the "rotate half" layout used by LLaMA implementations). The original
    paper pairs adjacent dimensions (0,1), (2,3), ... The two differ only by a
    fixed permutation of the head dimension, so a model trained with one is
    self-consistent; they just cannot share weights.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]

    # (seq_len, half) -> (1, 1, seq_len, half) to broadcast over batch and heads
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)

    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit feed-forward network.

    SwiGLU(x) = W_down( silu(W_gate x) * W_up x )

    Three matrices instead of two. The elementwise product is a *gate*: the
    up-projection proposes a value, the gate decides how much of it survives.

    Note the hidden width. A GELU MLP uses d_ff = 4*d and two matrices, so
    8*d^2 parameters. To keep the same budget with three matrices, d_ff shrinks
    to about (8/3)*d -- which is where LLaMA's odd-looking hidden sizes come
    from. They are then rounded up to a hardware-friendly multiple.
    """

    def __init__(self, d_model, d_ff=None, dropout=0.0, multiple_of=32):
        """
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension. Defaults to (8/3)*d_model rounded up to
                a multiple of `multiple_of`, matching a 4x GELU MLP's budget.
            dropout: Dropout rate
            multiple_of: Round d_ff up to this multiple
        """
        super().__init__()

        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)
        self.d_ff = d_ff

        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention with RoPE and an optional KV cache.

    num_heads query heads share num_kv_heads key/value heads:
        num_kv_heads == num_heads  -> standard MHA
        num_kv_heads == 1          -> MQA
        in between                 -> GQA

    The point is the KV cache, not the FLOPs. Cache size scales with
    num_kv_heads, so dropping from 32 to 8 K/V heads shrinks it 4x -- and
    during decode the model is memory-bandwidth-bound, so that shrink is
    close to a 4x speedup. Quality loss is small because query heads still
    number 32; they just look at shared keys.

    Causal masking is always applied: this is a decoder attention.
    """

    def __init__(self, d_model, num_heads, num_kv_heads=None, max_seq_len=4096,
                 dropout=0.0, rope_base=10000.0):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (default: num_heads = MHA)
            max_seq_len: Maximum sequence length (sizes the RoPE tables)
            dropout: Dropout rate on attention weights
            rope_base: RoPE theta base
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        assert num_heads % num_kv_heads == 0, (
            "num_heads must be divisible by num_kv_heads"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        # How many query heads share each K/V head
        self.num_groups = num_heads // num_kv_heads

        self.W_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(num_heads * self.d_k, d_model, bias=False)

        self.rope = RotaryEmbedding(self.d_k, max_seq_len, rope_base)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        """
        Args:
            x: (batch, seq_len, d_model). During cached generation seq_len is 1.
            past_kv: Optional (past_k, past_v), each
                (batch, num_kv_heads, past_len, d_k)
            use_cache: If True, also return the updated (k, v) for reuse

        Returns:
            output: (batch, seq_len, d_model)
            present: (k, v) if use_cache else None
        """
        B, T, _ = x.shape
        past_len = 0 if past_kv is None else past_kv[0].size(2)

        # Project and split into heads. Note K/V get fewer heads than Q.
        q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)

        # RoPE is applied at the *absolute* position, hence the offset. It goes
        # on Q and K only -- V carries content, not position.
        cos, sin = self.rope(T, offset=past_len)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Append to the cache. Crucially, the cache stores post-RoPE keys, so
        # history never needs re-rotating.
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present = (k, v) if use_cache else None

        # Expand K/V so every query head has a partner. repeat_interleave keeps
        # group members adjacent, matching how the heads were laid out.
        # A real kernel skips this copy and indexes the shared K/V directly.
        k_exp = k.repeat_interleave(self.num_groups, dim=1)
        v_exp = v.repeat_interleave(self.num_groups, dim=1)

        total = k_exp.size(2)
        scores = torch.matmul(q, k_exp.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Query i sits at absolute position past_len + i and may attend to key
        # positions 0 .. past_len + i. On a (T, total) grid that is
        # tril(diagonal=total - T), which also handles the T == 1 decode case.
        allowed = torch.ones(T, total, dtype=torch.bool, device=x.device).tril(
            diagonal=total - T
        )
        scores = scores.masked_fill(~allowed, float("-inf"))

        weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(weights, v_exp)  # (B, num_heads, T, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.d_k)
        return self.W_o(out), present

    def kv_cache_bytes(self, batch_size, seq_len, bytes_per_element=2):
        """
        Size of this layer's KV cache. Handy for the memory budgets in
        notebooks 15 and 21.
        """
        elements = 2 * batch_size * self.num_kv_heads * seq_len * self.d_k
        return elements * bytes_per_element


class ModernBlock(nn.Module):
    """
    A LLaMA-style decoder block: Pre-RMSNorm + GQA + SwiGLU, no biases.

        x = x + GQA(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    Pre-norm (normalize the branch input, leave the residual path clean) is
    what makes deep stacks trainable without a warmup-heavy schedule.
    """

    def __init__(self, d_model, num_heads, num_kv_heads=None, max_seq_len=4096,
                 d_ff=None, dropout=0.0, rope_base=10000.0):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (default: num_heads)
            max_seq_len: Maximum sequence length
            d_ff: SwiGLU hidden dimension (default: (8/3)*d_model, rounded)
            dropout: Dropout rate
            rope_base: RoPE theta base
        """
        super().__init__()

        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model, num_heads, num_kv_heads, max_seq_len, dropout, rope_base
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        """
        Args:
            x: (batch, seq_len, d_model)
            past_kv: Optional (past_k, past_v) for this layer
            use_cache: If True, also return the updated (k, v)

        Returns:
            output: (batch, seq_len, d_model)
            present: (k, v) if use_cache else None
        """
        attn_out, present = self.attn(self.attn_norm(x), past_kv, use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, present


class ModernGPT(nn.Module):
    """
    A LLaMA-style decoder-only language model.

    Differences from the GPT in gpt.py, all introduced in notebook 11:
      - RMSNorm instead of LayerNorm
      - RoPE inside attention instead of a learned absolute position embedding
        (so there is no position embedding table at all)
      - SwiGLU instead of a GELU MLP
      - Grouped-query attention instead of plain MHA
      - No biases anywhere
      - Token embeddings are not scaled by sqrt(d_model)

    Because position is relative, this model can run past its training length
    (badly, until rescaled -- that is the subject of notebook 15). The absolute
    position table in gpt.py cannot even index past max_seq_len.
    """

    def __init__(self, vocab_size, d_model=256, num_heads=8, num_kv_heads=None,
                 num_layers=6, max_seq_len=512, d_ff=None, dropout=0.0,
                 rope_base=10000.0, tie_weights=True):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Model dimension
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (default: num_heads = MHA)
            num_layers: Number of blocks
            max_seq_len: Maximum sequence length
            d_ff: SwiGLU hidden dimension
            dropout: Dropout rate
            rope_base: RoPE theta base
            tie_weights: Share the embedding matrix with the output head
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ModernBlock(
                d_model, num_heads, num_kv_heads, max_seq_len,
                d_ff, dropout, rope_base,
            )
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small normal values, as in gpt.py."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None, past_kvs=None, use_cache=False):
        """
        Args:
            input_ids: (batch, seq_len)
            targets: Optional (batch, seq_len) for the loss. Label -1 is ignored,
                which is how notebook 12 masks prompt tokens.
            past_kvs: Optional list of per-layer (k, v) from a previous call
            use_cache: If True, also return the updated per-layer caches

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: Cross-entropy loss, or None if targets is None
            presents: List of per-layer (k, v) if use_cache else None
        """
        x = self.dropout(self.token_embedding(input_ids))

        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = None if past_kvs is None else past_kvs[i]
            x, present = block(x, past_kv, use_cache)
            if use_cache:
                presents.append(present)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return logits, loss, presents

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, use_cache=True):
        """
        Generate autoregressively.

        Args:
            input_ids: (batch, seq_len) prompt
            max_new_tokens: Number of tokens to generate
            temperature: Softmax temperature. 0 means greedy.
            top_k: If set, sample only from the top k tokens
            top_p: If set, sample from the smallest set with cumulative prob >= p
            use_cache: Use the KV cache. Off recomputes the whole prefix each
                step -- much slower, and the reference the cache is checked
                against in tests/test_modern.py.

        Returns:
            (batch, seq_len + max_new_tokens)
        """
        # Sampling must run with dropout off, but this method is also called
        # *during* training to print a sample. Restore whatever mode we found,
        # or training silently continues without dropout.
        was_training = self.training
        self.eval()
        try:
            return self._generate(
                input_ids, max_new_tokens, temperature, top_k, top_p, use_cache
            )
        finally:
            self.train(was_training)

    def _generate(self, input_ids, max_new_tokens, temperature, top_k, top_p,
                  use_cache):
        """The sampling loop itself. Assumes the caller has set eval mode."""
        past_kvs = None

        for step in range(max_new_tokens):
            if use_cache and past_kvs is not None:
                # Only the newest token needs a forward pass; its keys and
                # values append to the cache and history is never recomputed.
                model_input = input_ids[:, -1:]
            else:
                model_input = input_ids[:, -self.max_seq_len:]

            logits, _, presents = self.forward(
                model_input, past_kvs=past_kvs, use_cache=use_cache
            )
            past_kvs = presents

            logits = logits[:, -1, :]

            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))

                if top_p is not None:
                    ordered, order = torch.sort(logits, descending=True, dim=-1)
                    cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
                    # Keep everything up to and including the token that crosses p
                    drop = cumulative - torch.softmax(ordered, dim=-1) >= top_p
                    ordered = ordered.masked_fill(drop, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        -1, order, ordered
                    )

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # The cache holds absolute positions, so it cannot be cropped the
            # way the recompute path crops its input. Stop before RoPE runs off
            # its precomputed table.
            if use_cache and input_ids.size(1) >= self.max_seq_len:
                break

        return input_ids

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_modern_small(vocab_size, max_seq_len=256):
    """A small LLaMA-style model, sized to match create_gpt_small in gpt.py."""
    return ModernGPT(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_kv_heads=2,
        num_layers=4,
        max_seq_len=max_seq_len,
        dropout=0.1,
    )
