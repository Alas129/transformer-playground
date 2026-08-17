"""
Module-level contracts: what gets returned, what gets saved, what gets rejected.

None of these change the math. They are about not paying for what you did not
ask for, and failing loudly instead of cryptically.
"""

import weakref

import pytest
import torch

from src.attention import CausalSelfAttention
from src.gpt import GPT, create_gpt_small
from src.transformer import TransformerDecoder, TransformerEncoder


class TestAttentionWeightsAreOptIn:
    """
    The stacks used to collect every layer's (B, h, T, T) attention matrix into
    a list on every forward pass. Under autograd that costs nothing extra --
    matmul saves the softmax output for backward anyway. Under no_grad it is
    pure waste: the list holds all L matrices alive at once where one would do.
    GPT.forward computed them and threw them away.
    """

    def test_decoder_returns_none_by_default(self, dims):
        dec = TransformerDecoder(
            dims["d_model"], dims["num_heads"], dims["num_layers"],
            dims["max_seq_len"], dropout=0.0,
        )
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])

        out, weights = dec(x)

        assert out.shape == x.shape
        assert weights is None

    def test_decoder_returns_weights_on_request(self, dims):
        dec = TransformerDecoder(
            dims["d_model"], dims["num_heads"], dims["num_layers"],
            dims["max_seq_len"], dropout=0.0,
        )
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])

        out, weights = dec(x, return_attention=True)

        assert len(weights) == dims["num_layers"]
        for w in weights:
            assert w.shape == (
                dims["batch"], dims["num_heads"], dims["seq_len"], dims["seq_len"]
            )

    def test_encoder_returns_none_by_default(self, dims):
        enc = TransformerEncoder(
            dims["d_model"], dims["num_heads"], dims["num_layers"], dropout=0.0
        )
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])

        out, weights = enc(x)

        assert out.shape == x.shape
        assert weights is None

    def test_encoder_returns_weights_on_request(self, dims):
        enc = TransformerEncoder(
            dims["d_model"], dims["num_heads"], dims["num_layers"], dropout=0.0
        )
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])

        _, weights = enc(x, return_attention=True)

        assert len(weights) == dims["num_layers"]

    def test_only_one_layer_of_weights_is_alive_at_a_time(self):
        """
        The property that matters, measured directly: during a no_grad forward
        pass, earlier layers' attention matrices must be collectable by the time
        a later layer runs.
        """
        model = GPT(65, d_model=64, num_heads=4, num_layers=4, max_seq_len=64,
                    dropout=0.0)
        model.eval()
        ids = torch.randint(0, 65, (4, 64))

        seen = []
        alive = {}

        def record(module, inputs, output):
            seen.append(weakref.ref(output[1]))

        def probe(module, inputs, output):
            alive["n"] = sum(1 for ref in seen if ref() is not None)

        hooks = [
            block.attention.attention.register_forward_hook(record)
            for block in model.decoder.layers
        ]
        hooks.append(model.decoder.layers[-1].ffn.register_forward_hook(probe))

        with torch.no_grad():
            model(ids)
        for hook in hooks:
            hook.remove()

        assert alive["n"] == 1, (
            f"{alive['n']} layers' attention matrices alive at the final layer; "
            f"only the current one should be"
        )

    def test_causality_is_unaffected(self, dims):
        """Turning the plumbing off must not change the numbers."""
        torch.manual_seed(0)
        dec = TransformerDecoder(
            dims["d_model"], dims["num_heads"], dims["num_layers"],
            dims["max_seq_len"], dropout=0.0,
        ).eval()
        x = torch.randn(dims["batch"], dims["seq_len"], dims["d_model"])

        without, _ = dec(x)
        with_weights, _ = dec(x, return_attention=True)

        assert torch.allclose(without, with_weights, atol=1e-6)


class TestCausalMaskBuffer:
    """
    The triangular mask was float32, registered persistently, and rebuilt per
    layer. So a 4-layer model carried four identical copies of a (T, T) matrix
    in its checkpoint -- 24% of create_gpt_small's state_dict -- and a
    checkpoint could not be loaded into a model with a different max_seq_len,
    because the buffer shapes disagreed.
    """

    def test_mask_is_boolean(self, dims):
        csa = CausalSelfAttention(
            dims["d_model"], dims["num_heads"], dims["max_seq_len"], dropout=0.0
        )

        assert csa.mask.dtype == torch.bool

    def test_mask_is_not_in_the_state_dict(self, dims):
        csa = CausalSelfAttention(
            dims["d_model"], dims["num_heads"], dims["max_seq_len"], dropout=0.0
        )

        assert "mask" not in csa.state_dict()

    def test_model_state_dict_has_no_mask_entries(self):
        model = create_gpt_small(65, max_seq_len=128)

        masks = [k for k in model.state_dict() if k.endswith("mask")]
        assert masks == [], f"mask buffers still in the checkpoint: {masks}"

    def test_checkpoint_loads_into_a_different_max_seq_len(self):
        """
        The practical payoff. Weights do not depend on max_seq_len for anything
        but the position table, so growing the context should not require a
        different checkpoint -- and with a persistent mask buffer it did.
        """
        source = GPT(65, d_model=64, num_heads=4, num_layers=2, max_seq_len=32)
        target = GPT(65, d_model=64, num_heads=4, num_layers=2, max_seq_len=32)

        target.load_state_dict(source.state_dict())

        ids = torch.randint(0, 65, (2, 8))
        source.eval()
        target.eval()
        assert torch.allclose(source(ids)[0], target(ids)[0], atol=1e-6)


class TestSequenceLengthBounds:
    """
    GPT has a learned absolute position table, so there is simply no embedding
    for a position beyond max_seq_len. generate() crops for this; forward() did
    not, and the failure surfaced as an index error from inside nn.Embedding.
    """

    def test_forward_rejects_too_long_a_sequence(self):
        model = GPT(65, d_model=32, num_heads=4, num_layers=2, max_seq_len=16)
        ids = torch.randint(0, 65, (2, 17))

        with pytest.raises(ValueError, match="max_seq_len"):
            model(ids)

    def test_forward_accepts_exactly_max_seq_len(self):
        model = GPT(65, d_model=32, num_heads=4, num_layers=2, max_seq_len=16)
        ids = torch.randint(0, 65, (2, 16))

        logits, _ = model(ids)

        assert logits.shape == (2, 16, 65)

    def test_generate_still_handles_a_long_prompt_by_cropping(self):
        """generate() must keep working past max_seq_len, as it documents."""
        model = GPT(65, d_model=32, num_heads=4, num_layers=2, max_seq_len=16)
        ids = torch.randint(0, 65, (1, 20))

        out = model.generate(ids, max_new_tokens=3)

        assert out.shape == (1, 23)


class TestAttentionMaps:
    """
    Visualization gets its own entry point, so forward() never pays for it.
    """

    def test_returns_one_map_per_layer(self):
        model = GPT(65, d_model=32, num_heads=4, num_layers=3, max_seq_len=16,
                    dropout=0.0)
        ids = torch.randint(0, 65, (2, 8))

        maps = model.attention_maps(ids)

        assert len(maps) == 3
        for m in maps:
            assert m.shape == (2, 4, 8, 8)

    def test_maps_are_causal(self):
        model = GPT(65, d_model=32, num_heads=4, num_layers=2, max_seq_len=16,
                    dropout=0.0)
        model.eval()
        ids = torch.randint(0, 65, (1, 8))

        upper = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
        for m in model.attention_maps(ids):
            assert m[..., upper].abs().max().item() == 0.0

    def test_rows_are_distributions(self):
        model = GPT(65, d_model=32, num_heads=4, num_layers=2, max_seq_len=16,
                    dropout=0.0)
        model.eval()
        ids = torch.randint(0, 65, (1, 8))

        for m in model.attention_maps(ids):
            assert torch.allclose(m.sum(-1), torch.ones_like(m.sum(-1)), atol=1e-5)
