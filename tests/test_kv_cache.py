"""
The KV cache must be an optimization, not an approximation.

Caching keys and values changes the computation but must not change the result.
A subtly wrong cache -- an off-by-one mask, a forgotten position offset -- still
produces fluent text, so it is easy to ship. These tests pin the equivalence
down numerically.
"""

import torch

from src.gpt import GPT, create_gpt_small
from src.modern import ModernGPT, create_modern_small

TOL = 1e-5


class TestGPTCache:
    def test_full_prefix_matches_plain_forward(self, vocab_size, dims):
        model = GPT(
            vocab_size, dims["d_model"], dims["num_heads"], dims["num_layers"],
            dims["max_seq_len"], dropout=0.0,
        ).eval()
        ids = torch.randint(0, vocab_size, (dims["batch"], dims["seq_len"]))

        with torch.no_grad():
            plain, _ = model(ids)
            cached, presents = model.forward_cached(ids)

        assert torch.allclose(plain, cached, atol=TOL)
        assert len(presents) == dims["num_layers"]

    def test_incremental_decode_matches_plain_forward(self, vocab_size, dims):
        """
        Feed one token at a time and rebuild the full logit matrix. This is the
        real test: it exercises the offset handling and the single-row mask.
        """
        model = GPT(
            vocab_size, dims["d_model"], dims["num_heads"], dims["num_layers"],
            dims["max_seq_len"], dropout=0.0,
        ).eval()
        ids = torch.randint(0, vocab_size, (dims["batch"], dims["seq_len"]))

        with torch.no_grad():
            reference, _ = model(ids)

            past = None
            steps = []
            for t in range(dims["seq_len"]):
                logits, past = model.forward_cached(ids[:, t : t + 1], past)
                steps.append(logits)

        assert torch.allclose(torch.cat(steps, dim=1), reference, atol=TOL)

    def test_cache_grows_by_one_per_step(self, vocab_size, dims):
        model = GPT(
            vocab_size, dims["d_model"], dims["num_heads"], dims["num_layers"],
            dims["max_seq_len"], dropout=0.0,
        ).eval()
        ids = torch.randint(0, vocab_size, (1, 4))

        with torch.no_grad():
            _, past = model.forward_cached(ids)
            assert past[0][0].size(2) == 4
            for expected in (5, 6, 7):
                _, past = model.forward_cached(ids[:, :1], past)
                assert past[0][0].size(2) == expected

    def test_greedy_generation_is_identical(self, vocab_size, dims):
        """
        Same seed, same prompt: cached and recomputed generation must emit the
        exact same token ids.
        """
        model = GPT(
            vocab_size, dims["d_model"], dims["num_heads"], dims["num_layers"],
            dims["max_seq_len"], dropout=0.0,
        ).eval()
        prompt = torch.randint(0, vocab_size, (1, 3))

        torch.manual_seed(123)
        with_cache = model.generate(prompt, 12, temperature=1.0, use_cache=True)
        torch.manual_seed(123)
        without_cache = model.generate(prompt, 12, temperature=1.0, use_cache=False)

        assert torch.equal(with_cache, without_cache)

    def test_sampled_generation_is_identical(self, vocab_size, dims):
        model = GPT(
            vocab_size, dims["d_model"], dims["num_heads"], dims["num_layers"],
            dims["max_seq_len"], dropout=0.0,
        ).eval()
        prompt = torch.randint(0, vocab_size, (2, 4))

        torch.manual_seed(7)
        a = model.generate(prompt, 10, temperature=0.8, top_k=5, use_cache=True)
        torch.manual_seed(7)
        b = model.generate(prompt, 10, temperature=0.8, top_k=5, use_cache=False)

        assert torch.equal(a, b)

    def test_generation_survives_crossing_max_seq_len(self, vocab_size):
        """
        This model uses learned *absolute* positions, so past max_seq_len the
        naive path crops and re-indexes -- which invalidates the cache.
        generate() must notice and fall back rather than crash or read past the
        end of the position table.
        """
        model = create_gpt_small(vocab_size, max_seq_len=16).eval()
        prompt = torch.randint(0, vocab_size, (1, 14))

        out = model.generate(prompt, 10, temperature=0.8, use_cache=True)
        assert out.shape == (1, 24)

    def test_cached_forward_rejects_overlong_input(self, vocab_size):
        model = create_gpt_small(vocab_size, max_seq_len=8).eval()
        ids = torch.randint(0, vocab_size, (1, 9))
        try:
            model.forward_cached(ids)
        except AssertionError:
            return
        raise AssertionError("expected an AssertionError past max_seq_len")


class TestModernGPTCache:
    def test_incremental_decode_matches_plain_forward(self, vocab_size, dims):
        model = ModernGPT(
            vocab_size, dims["d_model"], dims["num_heads"], num_kv_heads=2,
            num_layers=dims["num_layers"], max_seq_len=dims["max_seq_len"],
            dropout=0.0,
        ).eval()
        ids = torch.randint(0, vocab_size, (dims["batch"], dims["seq_len"]))

        with torch.no_grad():
            reference, _, _ = model(ids)

            past = None
            steps = []
            for t in range(dims["seq_len"]):
                logits, _, past = model(
                    ids[:, t : t + 1], past_kvs=past, use_cache=True
                )
                steps.append(logits)

        assert torch.allclose(torch.cat(steps, dim=1), reference, atol=TOL)

    def test_equivalence_across_all_gqa_regimes(self, vocab_size, dims):
        """MHA, GQA and MQA all share the cache path, so all three are checked."""
        ids = torch.randint(0, vocab_size, (1, dims["seq_len"]))

        for num_kv_heads in (4, 2, 1):
            model = ModernGPT(
                vocab_size, dims["d_model"], dims["num_heads"],
                num_kv_heads=num_kv_heads, num_layers=dims["num_layers"],
                max_seq_len=dims["max_seq_len"], dropout=0.0,
            ).eval()

            with torch.no_grad():
                reference, _, _ = model(ids)
                past = None
                steps = []
                for t in range(dims["seq_len"]):
                    logits, _, past = model(
                        ids[:, t : t + 1], past_kvs=past, use_cache=True
                    )
                    steps.append(logits)

            assert torch.allclose(
                torch.cat(steps, dim=1), reference, atol=TOL
            ), f"mismatch for num_kv_heads={num_kv_heads}"

    def test_greedy_generation_is_identical(self, vocab_size, dims):
        model = ModernGPT(
            vocab_size, dims["d_model"], dims["num_heads"], num_kv_heads=2,
            num_layers=dims["num_layers"], max_seq_len=dims["max_seq_len"],
            dropout=0.0,
        ).eval()
        prompt = torch.randint(0, vocab_size, (1, 3))

        with_cache = model.generate(prompt, 10, temperature=0, use_cache=True)
        without_cache = model.generate(prompt, 10, temperature=0, use_cache=False)

        assert torch.equal(with_cache, without_cache)

    def test_cache_stores_post_rope_keys(self, vocab_size, dims):
        """
        Keys go into the cache already rotated, so history is never re-rotated.
        Checked indirectly: appending token t must leave earlier cache entries
        byte-identical.
        """
        model = ModernGPT(
            vocab_size, dims["d_model"], dims["num_heads"], num_kv_heads=2,
            num_layers=dims["num_layers"], max_seq_len=dims["max_seq_len"],
            dropout=0.0,
        ).eval()
        ids = torch.randint(0, vocab_size, (1, 5))

        with torch.no_grad():
            _, _, past_a = model(ids, use_cache=True)
            first_k = past_a[0][0].clone()
            _, _, past_b = model(ids[:, :1], past_kvs=past_a, use_cache=True)

        assert torch.equal(past_b[0][0][:, :, :5, :], first_k)


class TestCachePerformance:
    def test_cache_is_faster_for_long_generation(self, vocab_size):
        """
        Not a strict benchmark -- just a guard against the cache silently
        regressing into a full recompute. The naive path is quadratic in
        sequence length, so the gap widens with more tokens.
        """
        import time

        model = create_modern_small(vocab_size, max_seq_len=192).eval()
        prompt = torch.randint(0, vocab_size, (1, 4))

        def timed(use_cache):
            start = time.perf_counter()
            model.generate(prompt, 120, temperature=0, use_cache=use_cache)
            return time.perf_counter() - start

        timed(True)  # warm up
        cached = timed(True)
        naive = timed(False)

        assert cached < naive, f"cached {cached:.3f}s vs naive {naive:.3f}s"
