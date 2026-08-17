"""
End-to-end contract for train_gpt.

The function had no test at all, which is how it kept a scheduler comment
promising warmup that was never implemented. These runs are deliberately tiny
-- a handful of steps on a few hundred characters -- because the point is the
plumbing, not the loss.
"""

import math

import pytest
import torch

from src.train import CharTokenizer, evaluate, train_gpt, TextDataset
from src.gpt import create_gpt_small
from torch.utils.data import DataLoader

CORPUS = ("the quick brown fox jumps over the lazy dog. " * 40)


@pytest.fixture
def corpus_file(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text(CORPUS)
    return path


class TestTrainGpt:
    def test_returns_model_tokenizer_and_history(self, corpus_file):
        model, tokenizer, history = train_gpt(
            corpus_file, epochs=2, batch_size=4, seq_len=16, print_every=100
        )

        assert model.vocab_size == tokenizer.vocab_size
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

    def test_loss_decreases(self, corpus_file):
        _, _, history = train_gpt(
            corpus_file, epochs=4, batch_size=4, seq_len=16, lr=3e-3,
            print_every=100
        )

        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_is_reproducible_for_a_fixed_seed(self, corpus_file):
        _, _, first = train_gpt(
            corpus_file, epochs=2, batch_size=4, seq_len=16, print_every=100,
            seed=7
        )
        _, _, second = train_gpt(
            corpus_file, epochs=2, batch_size=4, seq_len=16, print_every=100,
            seed=7
        )

        assert first["train_loss"] == second["train_loss"]

    def test_different_seeds_give_different_runs(self, corpus_file):
        _, _, first = train_gpt(
            corpus_file, epochs=1, batch_size=4, seq_len=16, print_every=100,
            seed=1
        )
        _, _, second = train_gpt(
            corpus_file, epochs=1, batch_size=4, seq_len=16, print_every=100,
            seed=2
        )

        assert first["train_loss"] != second["train_loss"]

    def test_no_validation_when_fraction_is_zero(self, corpus_file):
        _, _, history = train_gpt(
            corpus_file, epochs=1, batch_size=4, seq_len=16, print_every=100,
            val_fraction=0.0
        )

        assert history["val_loss"] == []

    def test_dropout_is_still_on_after_a_sample_is_generated(self, corpus_file):
        """
        print_every=1 samples after every epoch. If generate() leaked eval mode,
        training would silently continue without dropout from epoch 1 onward.
        """
        model, _, _ = train_gpt(
            corpus_file, epochs=2, batch_size=4, seq_len=16, print_every=1
        )

        assert model.training

    def test_saves_a_loadable_checkpoint(self, corpus_file, tmp_path):
        from src.train import load_gpt

        save_path = str(tmp_path / "run.pt")
        model, _, _ = train_gpt(
            corpus_file, epochs=1, batch_size=4, seq_len=16, print_every=100,
            save_path=save_path
        )

        restored, _ = load_gpt(save_path)

        assert restored.config == model.config


class TestEvaluate:
    def test_returns_mean_loss(self):
        tokenizer = CharTokenizer(CORPUS)
        model = create_gpt_small(tokenizer.vocab_size, max_seq_len=16)
        loader = DataLoader(TextDataset(CORPUS, tokenizer, 16), batch_size=4)

        loss = evaluate(model, loader, "cpu")

        assert loss > 0
        assert math.isfinite(loss)

    def test_restores_training_mode(self):
        tokenizer = CharTokenizer(CORPUS)
        model = create_gpt_small(tokenizer.vocab_size, max_seq_len=16)
        loader = DataLoader(TextDataset(CORPUS, tokenizer, 16), batch_size=4)
        model.train()

        evaluate(model, loader, "cpu")

        assert model.training

    def test_empty_loader_is_nan_not_a_crash(self):
        tokenizer = CharTokenizer("abc")
        model = create_gpt_small(tokenizer.vocab_size, max_seq_len=16)
        loader = DataLoader(TextDataset("abc", tokenizer, 16), batch_size=4)

        assert math.isnan(evaluate(model, loader, "cpu"))

    def test_does_not_leave_gradients_behind(self):
        """evaluate is under no_grad, so nothing it touches should need a step."""
        tokenizer = CharTokenizer(CORPUS)
        model = create_gpt_small(tokenizer.vocab_size, max_seq_len=16)
        loader = DataLoader(TextDataset(CORPUS, tokenizer, 16), batch_size=4)

        evaluate(model, loader, "cpu")

        assert all(p.grad is None for p in model.parameters())
