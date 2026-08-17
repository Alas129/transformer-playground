"""
Training-loop and tokenizer contracts.

These guard the things that go wrong *silently*: a model left in eval mode so
dropout quietly stops, a sidecar file written over the checkpoint it belongs to,
a prompt that loses characters on the way in. None of these raise, so only a
test notices them.
"""

import torch

import pytest

from src.gpt import GPT, create_gpt_small
from src.modern import ModernGPT, create_modern_small
from src.train import (
    CharTokenizer,
    generate_text,
    tokenizer_path_for,
)


class TestGenerateRestoresTrainingMode:
    """
    generate() has to switch to eval so dropout is off while sampling, but it
    must put the model back the way it found it. train_gpt() samples from the
    model *inside* the epoch loop; a generate() that leaks eval mode disables
    dropout for every remaining epoch of training and nothing reports it.
    """

    def test_gpt_generate_leaves_training_mode_untouched(self, vocab_size):
        model = create_gpt_small(vocab_size, max_seq_len=16)
        model.train()

        model.generate(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=2)

        assert model.training, "generate() left the model in eval mode"

    def test_gpt_generate_keeps_eval_when_called_in_eval(self, vocab_size):
        model = create_gpt_small(vocab_size, max_seq_len=16)
        model.eval()

        model.generate(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=2)

        assert not model.training, "generate() should not turn training back on"

    def test_modern_gpt_generate_leaves_training_mode_untouched(self, vocab_size):
        model = create_modern_small(vocab_size, max_seq_len=16)
        model.train()

        model.generate(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=2)

        assert model.training, "ModernGPT.generate() left the model in eval mode"

    def test_generate_text_leaves_training_mode_untouched(self, vocab_size):
        tokenizer = CharTokenizer("abcdefg ")
        model = create_gpt_small(tokenizer.vocab_size, max_seq_len=16)
        model.train()

        generate_text(model, tokenizer, "ab", max_tokens=2)

        assert model.training, "generate_text() left the model in eval mode"

    def test_dropout_still_active_after_generating(self, vocab_size):
        """
        The consequence, stated directly: a model that is still in training mode
        gives different outputs on two forward passes, because dropout is live.
        If generate() leaked eval mode, these two passes would be identical.
        """
        model = GPT(vocab_size, d_model=32, num_heads=4, num_layers=2,
                    max_seq_len=16, dropout=0.5)
        model.train()
        model.generate(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=1)

        ids = torch.randint(0, vocab_size, (2, 6))
        first, _ = model(ids)
        second, _ = model(ids)

        assert not torch.allclose(first, second), (
            "identical outputs mean dropout is off -- eval mode leaked"
        )


class TestTokenizerSidecarPath:
    """
    train_gpt() writes the tokenizer next to the checkpoint. The old derivation
    was save_path.replace('.pt', '_tokenizer.json'), which does nothing at all
    when the path does not contain '.pt' -- and then writes the tokenizer JSON
    straight over the checkpoint that was just saved.
    """

    def test_pt_suffix(self):
        assert tokenizer_path_for("gpt_model.pt") == "gpt_model_tokenizer.json"

    def test_other_suffix_does_not_collide(self):
        assert tokenizer_path_for("model.bin") == "model_tokenizer.json"

    def test_no_suffix_does_not_collide(self):
        assert tokenizer_path_for("checkpoint") == "checkpoint_tokenizer.json"

    def test_never_returns_the_checkpoint_path(self):
        for path in ("model.pt", "model.bin", "model.pth", "model", "a.pt.bak"):
            assert tokenizer_path_for(path) != path, (
                f"tokenizer path collides with the checkpoint for {path!r}"
            )

    def test_directory_is_preserved(self):
        assert tokenizer_path_for("out/run1/model.pt") == (
            "out/run1/model_tokenizer.json"
        )


class TestTokenizerHandlesUnknownCharacters:
    """
    A character tokenizer built from one corpus will meet characters it has
    never seen. Dropping them silently makes a prompt shorter than the user
    wrote, and an all-unknown prompt encodes to nothing at all -- which reaches
    the model as a (1, 0) tensor.
    """

    def test_unknown_characters_warn(self):
        tokenizer = CharTokenizer("abc")

        with pytest.warns(UserWarning, match="not in the vocabulary"):
            tokenizer.encode("abcZ")

    def test_known_characters_do_not_warn(self, recwarn):
        tokenizer = CharTokenizer("abc")

        tokenizer.encode("abcabc")

        assert len(recwarn) == 0

    def test_unknown_characters_are_still_dropped(self):
        """Behaviour is unchanged -- it is now merely visible."""
        tokenizer = CharTokenizer("abc")

        with pytest.warns(UserWarning):
            assert tokenizer.encode("aZb") == [0, 1]

    def test_generate_text_rejects_a_prompt_that_encodes_to_nothing(self):
        tokenizer = CharTokenizer("abc")
        model = create_gpt_small(tokenizer.vocab_size, max_seq_len=16)

        with pytest.raises(ValueError, match="encoded to zero tokens"):
            with pytest.warns(UserWarning):
                generate_text(model, tokenizer, "ZZZ", max_tokens=2)
