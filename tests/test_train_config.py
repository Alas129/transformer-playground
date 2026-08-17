"""
Training configuration: dataset windowing, train/val split, optimizer groups
and the learning-rate schedule.

Notebook 08 teaches all four. src/train.py did none of them, which is the gap
these tests close.
"""

import math

import torch

from src.gpt import GPT
from src.train import (
    CharTokenizer,
    TextDataset,
    configure_optimizer,
    lr_lambda_for,
    split_text,
)

CORPUS = "".join(chr(ord("a") + i % 26) for i in range(1000))


class TestDatasetWindowing:
    """
    The dataset slid its window one token at a time, so consecutive samples
    shared all but one token and an 11k-character corpus produced 11k samples
    per epoch -- 127x the corpus in tokens, for almost no new information.
    """

    def test_default_stride_is_seq_len(self):
        tokenizer = CharTokenizer(CORPUS)
        ds = TextDataset(CORPUS, tokenizer, seq_len=10)

        assert ds.stride == 10

    def test_default_windows_do_not_overlap(self):
        tokenizer = CharTokenizer(CORPUS)
        ds = TextDataset(CORPUS, tokenizer, seq_len=10)

        first, _ = ds[0]
        second, _ = ds[1]
        assert not torch.equal(first[1:], second[:-1]), "windows still overlap"

    def test_windows_tile_the_corpus(self):
        tokenizer = CharTokenizer(CORPUS)
        ds = TextDataset(CORPUS, tokenizer, seq_len=10)

        # ~1000 tokens in windows of 10, not ~1000 windows.
        assert 95 <= len(ds) <= 100

    def test_stride_one_reproduces_the_old_behaviour(self):
        tokenizer = CharTokenizer(CORPUS)
        ds = TextDataset(CORPUS, tokenizer, seq_len=10, stride=1)

        first, _ = ds[0]
        second, _ = ds[1]
        assert torch.equal(first[1:], second[:-1])
        assert len(ds) == 990

    def test_every_index_is_in_range(self):
        """A too-generous __len__ yields short tensors at the tail."""
        tokenizer = CharTokenizer(CORPUS)
        for seq_len in (7, 10, 33):
            for stride in (1, 3, seq_len):
                ds = TextDataset(CORPUS, tokenizer, seq_len=seq_len, stride=stride)
                for idx in (0, len(ds) - 1):
                    x, y = ds[idx]
                    assert x.shape == (seq_len,)
                    assert y.shape == (seq_len,)

    def test_targets_are_inputs_shifted_by_one(self):
        tokenizer = CharTokenizer(CORPUS)
        ds = TextDataset(CORPUS, tokenizer, seq_len=10)

        x, y = ds[3]
        assert torch.equal(x[1:], y[:-1])

    def test_corpus_shorter_than_window_yields_nothing(self):
        tokenizer = CharTokenizer("abc")
        ds = TextDataset("abc", tokenizer, seq_len=10)

        assert len(ds) == 0


class TestSplitText:
    """
    Held-out data is the only way to see overfitting. For text the split has to
    be contiguous -- a random split puts neighbouring, nearly identical windows
    on both sides and the validation loss then tracks the training loss.
    """

    def test_split_is_contiguous_and_lossless(self):
        train, val = split_text(CORPUS, val_fraction=0.1)

        assert train + val == CORPUS

    def test_fraction_is_respected(self):
        train, val = split_text(CORPUS, val_fraction=0.2)

        assert abs(len(val) / len(CORPUS) - 0.2) < 0.01

    def test_zero_fraction_gives_no_validation_set(self):
        train, val = split_text(CORPUS, val_fraction=0.0)

        assert train == CORPUS
        assert val == ""


class TestOptimizerGroups:
    """
    Weight decay is a prior that a weight should be small. That is meaningful
    for a matrix, and meaningless for a normalization gain or a bias -- decaying
    those just fights the layer. AdamW(model.parameters()) decays everything.
    """

    def test_two_groups(self):
        model = GPT(65, d_model=32, num_heads=4, num_layers=2, max_seq_len=24)
        optimizer = configure_optimizer(model, lr=3e-4, weight_decay=0.1)

        assert len(optimizer.param_groups) == 2

    def test_matrices_decay_and_vectors_do_not(self):
        model = GPT(65, d_model=32, num_heads=4, num_layers=2, max_seq_len=24)
        optimizer = configure_optimizer(model, lr=3e-4, weight_decay=0.1)

        decayed = {id(p) for g in optimizer.param_groups if g["weight_decay"] > 0
                   for p in g["params"]}
        undecayed = {id(p) for g in optimizer.param_groups if g["weight_decay"] == 0
                     for p in g["params"]}

        for name, param in model.named_parameters():
            if param.dim() >= 2:
                assert id(param) in decayed, f"{name} should decay"
            else:
                assert id(param) in undecayed, f"{name} should not decay"

    def test_every_parameter_appears_exactly_once(self):
        model = GPT(65, d_model=32, num_heads=4, num_layers=2, max_seq_len=24)
        optimizer = configure_optimizer(model, lr=3e-4, weight_decay=0.1)

        grouped = [id(p) for g in optimizer.param_groups for p in g["params"]]
        expected = {id(p) for p in model.parameters() if p.requires_grad}

        assert len(grouped) == len(set(grouped)), "a parameter is in two groups"
        assert set(grouped) == expected


class TestLRSchedule:
    """
    Warmup then cosine decay. The old code's comment said "optional warmup" but
    there was none -- training started at full learning rate.
    """

    def test_starts_well_below_peak(self):
        f = lr_lambda_for(warmup_steps=100, total_steps=1000)

        assert 0 < f(0) < 0.05

    def test_reaches_peak_at_end_of_warmup(self):
        f = lr_lambda_for(warmup_steps=100, total_steps=1000)

        assert abs(f(100) - 1.0) < 1e-6

    def test_warmup_is_monotonic(self):
        f = lr_lambda_for(warmup_steps=100, total_steps=1000)
        values = [f(s) for s in range(101)]

        assert all(b > a for a, b in zip(values, values[1:]))

    def test_decays_after_warmup(self):
        f = lr_lambda_for(warmup_steps=100, total_steps=1000)
        values = [f(s) for s in range(100, 1001, 50)]

        assert all(b < a for a, b in zip(values, values[1:]))

    def test_ends_at_min_ratio(self):
        f = lr_lambda_for(warmup_steps=100, total_steps=1000, min_ratio=0.1)

        assert abs(f(1000) - 0.1) < 1e-6

    def test_never_negative_past_the_end(self):
        f = lr_lambda_for(warmup_steps=100, total_steps=1000, min_ratio=0.0)

        assert f(5000) >= 0.0

    def test_zero_warmup_starts_at_peak(self):
        f = lr_lambda_for(warmup_steps=0, total_steps=1000)

        assert abs(f(0) - 1.0) < 1e-6
