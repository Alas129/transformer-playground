"""
RoPE must encode *relative* position.

The defining property: after rotating q at position m and k at position n, their
dot product depends only on (m - n). That is what lets a model generalize to
position pairs it never saw during training, and what the context-extension
tricks in notebook 15 rely on.
"""

import math

import torch

from src.modern import RotaryEmbedding, apply_rope


def rotated_dot(rope, q, k, m, n):
    """Dot product of q placed at position m with k placed at position n."""
    cos_m, sin_m = rope(1, offset=m)
    cos_n, sin_n = rope(1, offset=n)
    return (apply_rope(q, cos_m, sin_m) * apply_rope(k, cos_n, sin_n)).sum().item()


class TestRelativity:
    def test_dot_product_depends_only_on_offset(self):
        """The core property. Same gap, same score, anywhere in the sequence."""
        rope = RotaryEmbedding(head_dim=16, max_seq_len=256)
        q = torch.randn(1, 1, 1, 16)
        k = torch.randn(1, 1, 1, 16)

        for gap in (0, 1, 5, 17):
            scores = [
                rotated_dot(rope, q, k, m, m - gap)
                for m in (gap, gap + 10, gap + 50, gap + 200)
            ]
            spread = max(scores) - min(scores)
            assert spread < 1e-4, f"gap {gap} gave scores {scores}"

    def test_different_offsets_give_different_scores(self):
        """
        Relativity would be trivially satisfied by ignoring position entirely,
        so confirm position still matters.
        """
        rope = RotaryEmbedding(head_dim=16, max_seq_len=256)
        q = torch.randn(1, 1, 1, 16)
        k = torch.randn(1, 1, 1, 16)
        scores = [rotated_dot(rope, q, k, 100, 100 - gap) for gap in (0, 1, 2, 8)]
        assert len({round(s, 4) for s in scores}) == len(scores)

    def test_zero_offset_is_identity(self):
        """Position 0 has zero rotation angle, so nothing should change."""
        rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
        x = torch.randn(2, 3, 1, 8)
        cos, sin = rope(1, offset=0)
        assert torch.allclose(apply_rope(x, cos, sin), x, atol=1e-6)


class TestGeometry:
    def test_rotation_preserves_norm(self):
        """A rotation cannot change a vector's length."""
        rope = RotaryEmbedding(head_dim=32, max_seq_len=128)
        x = torch.randn(2, 4, 10, 32)
        cos, sin = rope(10)
        rotated = apply_rope(x, cos, sin)
        assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)

    def test_frequencies_are_geometrically_spaced(self):
        """
        Pair i rotates at 1 / base^(2i/d). The fastest pair resolves adjacent
        tokens; the slowest carries long-range position. Notebook 15's scaling
        tricks all work by stretching this spectrum.
        """
        head_dim, base = 64, 10000.0
        rope = RotaryEmbedding(head_dim, max_seq_len=4, base=base)
        # Angle at position 1 is exactly the frequency of each pair.
        cos, sin = rope(2)
        angles = torch.atan2(sin[1], cos[1])
        expected = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        assert torch.allclose(angles, expected, atol=1e-5)

    def test_larger_base_slows_rotation(self):
        """
        Raising the base is the crudest context-extension trick: every pair
        rotates more slowly, so a given angle covers more positions.
        """
        fast = RotaryEmbedding(16, max_seq_len=8, base=10000.0)
        slow = RotaryEmbedding(16, max_seq_len=8, base=1000000.0)
        _, sin_fast = fast(2)
        _, sin_slow = slow(2)
        assert sin_slow[1].abs().sum() < sin_fast[1].abs().sum()


class TestContract:
    def test_odd_head_dim_rejected(self):
        """RoPE rotates 2-D slices, so the head dimension must be even."""
        try:
            RotaryEmbedding(head_dim=15)
        except AssertionError:
            return
        raise AssertionError("expected an AssertionError for odd head_dim")

    def test_beyond_table_rejected(self):
        """
        Asking for a position past the precomputed table is a real bug (usually
        a cache growing unbounded), so it must fail loudly rather than wrap.
        """
        rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
        try:
            rope(4, offset=14)
        except AssertionError:
            return
        raise AssertionError("expected an AssertionError past max_seq_len")

    def test_offset_matches_full_sequence_slice(self):
        """
        Requesting (seq_len=1, offset=t) must equal row t of a full request.
        Cached generation depends on this.
        """
        rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
        cos_full, sin_full = rope(20)
        for t in (0, 1, 7, 19):
            cos_one, sin_one = rope(1, offset=t)
            assert torch.allclose(cos_one[0], cos_full[t], atol=1e-6)
            assert torch.allclose(sin_one[0], sin_full[t], atol=1e-6)
