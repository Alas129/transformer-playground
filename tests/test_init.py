"""
Initialization contracts.

Initialization bugs do not raise. They show up as a model that trains slowly,
or a signal that is swamped before training starts. These tests pin down the
two scale decisions that matter.
"""

import math

import torch

from src.embeddings import TokenEmbedding, TransformerEmbedding
from src.gpt import GPT
from src.modern import ModernGPT


class TestTokenEmbeddingScaling:
    """
    The sqrt(d_model) factor in "Attention Is All You Need" exists to put a
    learned embedding on the same scale as the *fixed* sinusoidal positional
    encoding it is added to, given that paper's ~N(0, 1/d) init. Applying it on
    top of a GPT-2 style N(0, 0.02) init and a *learned* position table is
    incoherent: both sides are learned, there is no fixed scale to match, and
    the position table ends up an order of magnitude weaker at init.
    """

    def test_unscaled_by_default(self):
        emb = TokenEmbedding(11, 64)
        ids = torch.tensor([[0, 1, 2]])

        assert torch.allclose(emb(ids), emb.embedding(ids))

    def test_scale_multiplies_by_sqrt_d(self):
        emb = TokenEmbedding(11, 64, scale=True)
        ids = torch.tensor([[0, 1, 2]])

        assert torch.allclose(emb(ids), emb.embedding(ids) * math.sqrt(64))

    def test_learned_positions_are_not_swamped(self):
        """
        The property that actually matters. With learned positions, the token
        and position contributions should be within a small factor of each
        other at init -- not 11x apart.
        """
        d_model = 128
        emb = TransformerEmbedding(65, d_model, 256, dropout=0.0, learnable_pos=True)
        torch.nn.init.normal_(emb.token_embedding.embedding.weight, std=0.02)
        torch.nn.init.normal_(
            emb.position_encoding.position_embedding.weight, std=0.02
        )

        ids = torch.randint(0, 65, (8, 32))
        token_part = emb.token_embedding(ids)
        position_part = emb.position_encoding.position_embedding(torch.arange(32))

        ratio = token_part.std() / position_part.std()
        assert 0.5 < ratio < 2.0, (
            f"token signal is {ratio:.1f}x the position signal; the two should "
            f"be comparable at init"
        )

    def test_sinusoidal_path_still_scales(self):
        """
        The sinusoidal table has a fixed magnitude of order 1, so here the
        scaling is doing its original job and must stay.
        """
        emb = TransformerEmbedding(65, 128, 256, dropout=0.0, learnable_pos=False)

        assert emb.token_embedding.scale is True


class TestResidualProjectionInit:
    """
    Every block adds its attention and feed-forward output into the residual
    stream. With N blocks all initialized at the same scale, the stream's
    variance grows with depth. GPT-2 scales the projections that *write* to the
    residual stream by 1/sqrt(2 * n_layer) to hold it steady.
    """

    def test_gpt_scales_residual_projections(self):
        num_layers = 6
        model = GPT(65, d_model=128, num_heads=4, num_layers=num_layers,
                    max_seq_len=64)
        expected = 0.02 / math.sqrt(2 * num_layers)

        for name, param in model.named_parameters():
            if name.endswith("attention.W_o.weight") or name.endswith("linear2.weight"):
                assert abs(param.std().item() - expected) < 0.3 * expected, (
                    f"{name} std {param.std().item():.5f}, expected ~{expected:.5f}"
                )

    def test_gpt_leaves_other_projections_alone(self):
        model = GPT(65, d_model=128, num_heads=4, num_layers=6, max_seq_len=64)

        for name, param in model.named_parameters():
            if name.endswith("W_q.weight") or name.endswith("linear1.weight"):
                assert abs(param.std().item() - 0.02) < 0.3 * 0.02, (
                    f"{name} should keep the plain 0.02 init"
                )

    def test_modern_scales_residual_projections(self):
        num_layers = 6
        model = ModernGPT(65, d_model=128, num_heads=4, num_kv_heads=2,
                          num_layers=num_layers, max_seq_len=64)
        expected = 0.02 / math.sqrt(2 * num_layers)

        for name, param in model.named_parameters():
            if name.endswith("attn.W_o.weight") or name.endswith("w_down.weight"):
                assert abs(param.std().item() - expected) < 0.3 * expected, (
                    f"{name} std {param.std().item():.5f}, expected ~{expected:.5f}"
                )

    def test_residual_stream_does_not_grow_with_depth(self):
        """
        The behavioural consequence: a deeper stack should not produce
        wildly larger activations at init.
        """
        ids = torch.randint(0, 65, (4, 32))

        stds = []
        for num_layers in (2, 8):
            torch.manual_seed(0)
            model = ModernGPT(65, d_model=128, num_heads=4, num_kv_heads=2,
                              num_layers=num_layers, max_seq_len=64, dropout=0.0)
            model.eval()
            logits, _, _ = model(ids)
            stds.append(logits.std().item())

        growth = stds[1] / stds[0]
        assert growth < 2.0, (
            f"logit scale grew {growth:.1f}x going from 2 to 8 layers"
        )
