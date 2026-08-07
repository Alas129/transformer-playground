"""
The causal mask must not leak the future.

This is the single most important property of a decoder-only language model. If
it leaks, training loss looks *great* -- the model is reading the answer -- and
generation is garbage, because at inference time there is no future to read.
That failure mode is silent, so it gets a dedicated test.

The method: perturb a token, then check that only positions at or after it move.
"""

import torch

from src.attention import CausalSelfAttention
from src.gpt import GPT
from src.modern import GroupedQueryAttention, ModernGPT
from src.transformer import DecoderBlock, TransformerDecoder


def assert_no_future_leak(fn, x, cut):
    """
    Perturb x from index `cut` onward and assert outputs before `cut` are
    bit-identical.

    Args:
        fn: Callable taking x and returning a tensor (batch, seq_len, ...)
        x: Input tensor (batch, seq_len, d_model)
        cut: Index from which to perturb
    """
    baseline = fn(x)

    perturbed_input = x.clone()
    perturbed_input[:, cut:, :] += 10.0
    perturbed = fn(perturbed_input)

    before = (baseline[:, :cut] - perturbed[:, :cut]).abs().max().item()
    after = (baseline[:, cut:] - perturbed[:, cut:]).abs().max().item()

    assert before == 0.0, f"future leaked into the past (max diff {before})"
    # Sanity check on the test itself: the perturbation must actually do
    # something, or "no leak" would be vacuously true.
    assert after > 0.0, "perturbation had no effect; the test proves nothing"


class TestCausalSelfAttention:
    def test_no_future_leak(self, dims):
        csa = CausalSelfAttention(
            dims["d_model"], dims["num_heads"], dims["max_seq_len"], dropout=0.0
        ).eval()
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        assert_no_future_leak(lambda t: csa(t)[0], x, cut=4)

    def test_attention_weights_are_lower_triangular(self, dims):
        csa = CausalSelfAttention(
            dims["d_model"], dims["num_heads"], dims["max_seq_len"], dropout=0.0
        ).eval()
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        _, weights = csa(x)

        upper = torch.triu(
            torch.ones(dims["seq_len"], dims["seq_len"], dtype=torch.bool),
            diagonal=1,
        )
        assert weights[..., upper].abs().max().item() == 0.0

    def test_rows_still_sum_to_one_after_masking(self, dims):
        csa = CausalSelfAttention(
            dims["d_model"], dims["num_heads"], dims["max_seq_len"], dropout=0.0
        ).eval()
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        _, weights = csa(x)
        assert torch.allclose(
            weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-5
        )


class TestGroupedQueryAttention:
    def test_no_future_leak(self, dims):
        gqa = GroupedQueryAttention(
            dims["d_model"], dims["num_heads"], num_kv_heads=2,
            max_seq_len=dims["max_seq_len"], dropout=0.0,
        ).eval()
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        assert_no_future_leak(lambda t: gqa(t)[0], x, cut=3)

    def test_no_future_leak_at_every_cut(self, dims):
        gqa = GroupedQueryAttention(
            dims["d_model"], dims["num_heads"], num_kv_heads=1,
            max_seq_len=dims["max_seq_len"], dropout=0.0,
        ).eval()
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        for cut in range(1, dims["seq_len"]):
            assert_no_future_leak(lambda t: gqa(t)[0], x, cut=cut)


class TestBlocksAndModels:
    def test_decoder_block_no_leak(self, dims):
        block = DecoderBlock(
            dims["d_model"], dims["num_heads"], dims["max_seq_len"], dropout=0.0
        ).eval()
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        assert_no_future_leak(lambda t: block(t)[0], x, cut=5)

    def test_decoder_stack_no_leak(self, dims):
        """
        Stacking is where a leak becomes catastrophic: one leaky layer
        contaminates every layer above it.
        """
        dec = TransformerDecoder(
            dims["d_model"], dims["num_heads"], dims["num_layers"],
            dims["max_seq_len"], dropout=0.0,
        ).eval()
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])
        assert_no_future_leak(lambda t: dec(t)[0], x, cut=2)

    def test_gpt_logits_no_leak(self, vocab_size, dims):
        """
        End to end on token ids: changing a later token must not change the
        prediction at an earlier position.
        """
        model = GPT(
            vocab_size, dims["d_model"], dims["num_heads"], dims["num_layers"],
            dims["max_seq_len"], dropout=0.0,
        ).eval()

        ids = torch.randint(0, vocab_size, (1, dims["seq_len"]))
        cut = 5

        with torch.no_grad():
            baseline, _ = model(ids)
            changed = ids.clone()
            changed[0, cut] = (changed[0, cut] + 1) % vocab_size
            perturbed, _ = model(changed)

        assert (baseline[:, :cut] - perturbed[:, :cut]).abs().max().item() == 0.0
        assert (baseline[:, cut] - perturbed[:, cut]).abs().max().item() > 0.0

    def test_modern_gpt_logits_no_leak(self, vocab_size, dims):
        model = ModernGPT(
            vocab_size, dims["d_model"], dims["num_heads"], num_kv_heads=2,
            num_layers=dims["num_layers"], max_seq_len=dims["max_seq_len"],
            dropout=0.0,
        ).eval()

        ids = torch.randint(0, vocab_size, (1, dims["seq_len"]))
        cut = 4

        with torch.no_grad():
            baseline, _, _ = model(ids)
            changed = ids.clone()
            changed[0, cut] = (changed[0, cut] + 1) % vocab_size
            perturbed, _, _ = model(changed)

        assert (baseline[:, :cut] - perturbed[:, :cut]).abs().max().item() == 0.0
        assert (baseline[:, cut] - perturbed[:, cut]).abs().max().item() > 0.0
