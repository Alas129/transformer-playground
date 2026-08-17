"""
Shape contracts for every module in src/.

These are the cheapest possible regression tests: they catch a transposed
matmul or a forgotten reshape immediately, which is the most common way a
Transformer implementation breaks while still running.
"""

import torch

from src.attention import (
    CausalSelfAttention,
    MultiHeadAttention,
    ScaledDotProductAttention,
)
from src.embeddings import (
    LearnablePositionalEncoding,
    PositionalEncoding,
    TokenEmbedding,
    TransformerEmbedding,
)
from src.gpt import GPT, create_gpt_medium, create_gpt_small
from src.moe import MoEFeedForward
from src.modern import (
    GroupedQueryAttention,
    ModernBlock,
    ModernGPT,
    RMSNorm,
    SwiGLU,
    create_modern_small,
)
from src.transformer import (
    DecoderBlock,
    FeedForward,
    TransformerBlock,
    TransformerDecoder,
    TransformerEncoder,
)


class TestEmbeddings:
    def test_token_embedding(self, vocab_size, dims):
        emb = TokenEmbedding(vocab_size, dims["d_model"])
        ids = torch.randint(0, vocab_size, (dims["batch"], dims["seq_len"]))
        assert emb(ids).shape == (dims["batch"], dims["seq_len"], dims["d_model"])

    def test_sinusoidal_positional_encoding(self, dims):
        pe = PositionalEncoding(dims["d_model"], dims["max_seq_len"], dropout=0.0)
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        assert pe(x).shape == x.shape

    def test_learnable_positional_encoding(self, dims):
        pe = LearnablePositionalEncoding(
            dims["d_model"], dims["max_seq_len"], dropout=0.0
        )
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        assert pe(x).shape == x.shape

    def test_offset_shifts_positions(self, dims):
        """A non-zero offset must read a different slice of the position table."""
        pe = PositionalEncoding(dims["d_model"], dims["max_seq_len"], dropout=0.0)
        x = torch.zeros(1, 1, dims["d_model"])
        assert not torch.allclose(pe(x, offset=0), pe(x, offset=5))

    def test_transformer_embedding_both_modes(self, vocab_size, dims):
        ids = torch.randint(0, vocab_size, (dims["batch"], dims["seq_len"]))
        for learnable in (True, False):
            emb = TransformerEmbedding(
                vocab_size, dims["d_model"], dims["max_seq_len"],
                dropout=0.0, learnable_pos=learnable,
            )
            assert emb(ids).shape == (
                dims["batch"], dims["seq_len"], dims["d_model"]
            )


class TestAttention:
    def test_scaled_dot_product(self, dims):
        attn = ScaledDotProductAttention(dropout=0.0)
        shape = (dims["batch"], dims["num_heads"], dims["seq_len"], 8)
        q = k = v = torch.randn(*shape)
        out, weights = attn(q, k, v)
        assert out.shape == shape
        assert weights.shape == (
            dims["batch"], dims["num_heads"], dims["seq_len"], dims["seq_len"]
        )

    def test_attention_weights_are_a_distribution(self, dims):
        attn = ScaledDotProductAttention(dropout=0.0)
        shape = (dims["batch"], dims["num_heads"], dims["seq_len"], 8)
        _, weights = attn(torch.randn(*shape), torch.randn(*shape), torch.randn(*shape))
        assert torch.allclose(
            weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-5
        )

    def test_multihead_attention(self, dims):
        mha = MultiHeadAttention(dims["d_model"], dims["num_heads"], dropout=0.0)
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        out, weights = mha(x, x, x)
        assert out.shape == x.shape
        assert weights.shape == (
            dims["batch"], dims["num_heads"], dims["seq_len"], dims["seq_len"]
        )

    def test_causal_self_attention(self, dims):
        csa = CausalSelfAttention(
            dims["d_model"], dims["num_heads"], dims["max_seq_len"], dropout=0.0
        )
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        out, _ = csa(x)
        assert out.shape == x.shape

    def test_grouped_query_attention_all_regimes(self, dims):
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        # 4 -> MHA, 2 -> GQA, 1 -> MQA
        for num_kv_heads in (4, 2, 1):
            gqa = GroupedQueryAttention(
                dims["d_model"], dims["num_heads"], num_kv_heads,
                dims["max_seq_len"], dropout=0.0,
            )
            out, present = gqa(x, use_cache=True)
            assert out.shape == x.shape
            assert present[0].shape == (
                dims["batch"], num_kv_heads, dims["seq_len"],
                dims["d_model"] // dims["num_heads"],
            )

    def test_kv_cache_shrinks_with_fewer_kv_heads(self, dims):
        """The whole point of GQA: cache size scales with num_kv_heads."""
        mha = GroupedQueryAttention(dims["d_model"], 4, 4, dims["max_seq_len"])
        mqa = GroupedQueryAttention(dims["d_model"], 4, 1, dims["max_seq_len"])
        assert mha.kv_cache_bytes(1, 128) == 4 * mqa.kv_cache_bytes(1, 128)


class TestFeedForward:
    def test_feed_forward(self, dims):
        ffn = FeedForward(dims["d_model"], dropout=0.0)
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        assert ffn(x).shape == x.shape

    def test_swiglu(self, dims):
        ffn = SwiGLU(dims["d_model"], dropout=0.0)
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        assert ffn(x).shape == x.shape

    def test_swiglu_default_width_matches_gelu_budget(self):
        """
        SwiGLU uses 3 matrices, so d_ff shrinks to ~(8/3)d to keep a 4x GELU
        MLP's parameter budget. Allow 15% for the rounding to a multiple of 32.
        """
        d_model = 512
        swiglu = SwiGLU(d_model)
        swiglu_params = sum(p.numel() for p in swiglu.parameters())
        gelu_params = 2 * d_model * (4 * d_model)
        assert abs(swiglu_params - gelu_params) / gelu_params < 0.15

    def test_moe_feed_forward(self, dims):
        moe = MoEFeedForward(
            dims["d_model"], num_experts=4, top_k=2, capacity_factor=None
        )
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        assert moe(x).shape == x.shape

    def test_moe_is_drop_in_for_swiglu(self, dims):
        """Both must accept and return the same shapes."""
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        dense = SwiGLU(dims["d_model"])
        sparse = MoEFeedForward(dims["d_model"], num_experts=4, capacity_factor=None)
        assert dense(x).shape == sparse(x).shape


class TestNormalization:
    def test_rmsnorm(self, dims):
        norm = RMSNorm(dims["d_model"])
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        assert norm(x).shape == x.shape

    def test_rmsnorm_matches_definition(self):
        """x / sqrt(mean(x^2) + eps), elementwise, no mean subtraction."""
        x = torch.randn(3, 5, 16)
        norm = RMSNorm(16, eps=1e-6)
        expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        assert torch.allclose(norm(x), expected, atol=1e-5)

    def test_rmsnorm_does_not_center(self):
        """
        The one behavioural difference from LayerNorm: RMSNorm re-scales but does
        not re-center. Given a constant-shifted input, LayerNorm drives the mean
        to zero and RMSNorm does not.
        """
        x = torch.randn(4, 16) + 10.0

        rms_mean = RMSNorm(16)(x).mean().abs().item()
        layer_mean = torch.nn.LayerNorm(16)(x).mean().abs().item()

        assert layer_mean < 1e-5, "LayerNorm should center"
        assert rms_mean > 0.5, "RMSNorm should not center"

    def test_rmsnorm_is_scale_invariant(self):
        """Scaling the input must not change the output."""
        x = torch.randn(4, 16)
        norm = RMSNorm(16)
        assert torch.allclose(norm(x), norm(x * 7.0), atol=1e-4)


class TestBlocks:
    def test_transformer_block(self, dims):
        block = TransformerBlock(dims["d_model"], dims["num_heads"], dropout=0.0)
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        out, _ = block(x)
        assert out.shape == x.shape

    def test_decoder_block(self, dims):
        block = DecoderBlock(
            dims["d_model"], dims["num_heads"], dims["max_seq_len"], dropout=0.0
        )
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        out, _ = block(x)
        assert out.shape == x.shape

    def test_modern_block(self, dims):
        block = ModernBlock(
            dims["d_model"], dims["num_heads"], num_kv_heads=2,
            max_seq_len=dims["max_seq_len"], dropout=0.0,
        )
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        out, _ = block(x)
        assert out.shape == x.shape

    def test_encoder_stack_returns_per_layer_weights(self, dims):
        """Opt-in since the buffers are large; see tests/test_module_contracts.py."""
        enc = TransformerEncoder(
            dims["d_model"], dims["num_heads"], dims["num_layers"], dropout=0.0
        )
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        out, weights = enc(x, return_attention=True)
        assert out.shape == x.shape
        assert len(weights) == dims["num_layers"]

    def test_decoder_stack_returns_per_layer_weights(self, dims):
        dec = TransformerDecoder(
            dims["d_model"], dims["num_heads"], dims["num_layers"],
            dims["max_seq_len"], dropout=0.0,
        )
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        out, weights = dec(x, return_attention=True)
        assert out.shape == x.shape
        assert len(weights) == dims["num_layers"]


class TestModels:
    def test_gpt_forward(self, vocab_size, dims):
        model = GPT(
            vocab_size, dims["d_model"], dims["num_heads"], dims["num_layers"],
            dims["max_seq_len"], dropout=0.0,
        )
        ids = torch.randint(0, vocab_size, (dims["batch"], dims["seq_len"]))
        logits, loss = model(ids)
        assert logits.shape == (dims["batch"], dims["seq_len"], vocab_size)
        assert loss is None

    def test_gpt_loss(self, vocab_size, dims):
        model = GPT(
            vocab_size, dims["d_model"], dims["num_heads"], dims["num_layers"],
            dims["max_seq_len"], dropout=0.0,
        )
        ids = torch.randint(0, vocab_size, (dims["batch"], dims["seq_len"]))
        _, loss = model(ids, targets=ids)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_untrained_loss_is_in_a_sane_band(self, vocab_size, dims):
        """
        An untrained model knows nothing, so its loss should be in the
        neighbourhood of ln(vocab_size) -- the loss of uniform guessing. This
        catches a broken init or a mis-shaped lm_head, which produce a loss that
        is orders of magnitude off.

        It lands somewhat *below* ln(V) rather than at it, and that is expected
        here: this model ties lm_head to the embedding matrix and scales
        embeddings by sqrt(d_model), so even at init the logits correlate with
        the input token. Hence a band rather than a tight equality.
        """
        import math

        model = GPT(
            vocab_size, dims["d_model"], dims["num_heads"], dims["num_layers"],
            dims["max_seq_len"], dropout=0.0,
        )
        model.eval()
        ids = torch.randint(0, vocab_size, (8, dims["seq_len"]))
        _, loss = model(ids, targets=ids)

        uniform = math.log(vocab_size)
        assert 0.5 * uniform < loss.item() < 1.5 * uniform, (
            f"loss {loss.item():.3f} is nowhere near ln(V) = {uniform:.3f}"
        )

    def test_gpt_weight_tying(self, vocab_size, dims):
        model = GPT(vocab_size, dims["d_model"], max_seq_len=dims["max_seq_len"])
        assert (
            model.lm_head.weight is model.embedding.token_embedding.embedding.weight
        )

    def test_gpt_ignores_masked_labels(self, vocab_size, dims):
        """Label -1 is skipped, which is how notebook 12 masks prompt tokens."""
        model = GPT(
            vocab_size, dims["d_model"], dims["num_heads"], dims["num_layers"],
            dims["max_seq_len"], dropout=0.0,
        )
        model.eval()
        ids = torch.randint(0, vocab_size, (dims["batch"], dims["seq_len"]))
        targets = ids.clone()
        _, full = model(ids, targets=targets)
        masked = targets.clone()
        masked[:, : dims["seq_len"] // 2] = -1
        _, partial = model(ids, targets=masked)
        assert not torch.allclose(full, partial)

    def test_modern_gpt_forward(self, vocab_size, dims):
        model = ModernGPT(
            vocab_size, dims["d_model"], dims["num_heads"], num_kv_heads=2,
            num_layers=dims["num_layers"], max_seq_len=dims["max_seq_len"],
            dropout=0.0,
        )
        ids = torch.randint(0, vocab_size, (dims["batch"], dims["seq_len"]))
        logits, loss, presents = model(ids, targets=ids)
        assert logits.shape == (dims["batch"], dims["seq_len"], vocab_size)
        assert loss.ndim == 0
        assert presents is None

    def test_modern_gpt_has_no_position_embedding(self, vocab_size, dims):
        """RoPE lives inside attention, so there is no position table at all."""
        model = ModernGPT(
            vocab_size, dims["d_model"], max_seq_len=dims["max_seq_len"]
        )
        names = [n for n, _ in model.named_parameters()]
        assert not any("position" in n for n in names)

    def test_factory_models_build_and_run(self, vocab_size):
        for factory in (create_gpt_small, create_gpt_medium, create_modern_small):
            model = factory(vocab_size, max_seq_len=32)
            ids = torch.randint(0, vocab_size, (2, 6))
            out = model(ids)
            assert out[0].shape == (2, 6, vocab_size)
            assert model.count_parameters() > 0

    def test_gradients_reach_every_parameter(self, vocab_size, dims):
        """
        A parameter with no gradient is a parameter that is silently not being
        trained -- e.g. a block left out of the forward pass.
        """
        model = ModernGPT(
            vocab_size, dims["d_model"], dims["num_heads"], num_kv_heads=2,
            num_layers=dims["num_layers"], max_seq_len=dims["max_seq_len"],
            dropout=0.0,
        )
        ids = torch.randint(0, vocab_size, (dims["batch"], dims["seq_len"]))
        _, loss, _ = model(ids, targets=ids)
        loss.backward()

        missing = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and param.grad is None
        ]
        assert missing == [], f"no gradient for: {missing}"
