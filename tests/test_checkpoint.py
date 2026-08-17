"""
Checkpoints must be enough to rebuild the model.

The old checkpoint stored vocab_size and seq_len only. Everything that defines
the architecture -- width, depth, head count -- was missing, so loading one
required knowing out of band which factory built it. Get it wrong and
load_state_dict raises a wall of shape mismatches.
"""

import torch

from src.gpt import GPT, create_gpt_medium, create_gpt_small
from src.train import (
    CharTokenizer,
    checkpoint_payload,
    load_gpt,
    tokenizer_path_for,
)


class TestModelConfig:
    def test_config_names_every_constructor_argument(self):
        model = GPT(65, d_model=64, num_heads=4, num_layers=3, max_seq_len=32,
                    d_ff=128, dropout=0.2)

        assert model.config == {
            "vocab_size": 65,
            "d_model": 64,
            "num_heads": 4,
            "num_layers": 3,
            "max_seq_len": 32,
            "d_ff": 128,
            "dropout": 0.2,
        }

    def test_from_config_rebuilds_the_same_shapes(self):
        original = create_gpt_medium(65, max_seq_len=32)

        rebuilt = GPT.from_config(original.config)

        assert rebuilt.config == original.config
        originals = dict(original.named_parameters())
        for name, param in rebuilt.named_parameters():
            assert name in originals
            assert param.shape == originals[name].shape


class TestCheckpointRoundTrip:
    def test_payload_carries_the_config(self):
        model = create_gpt_small(65, max_seq_len=32)
        tokenizer = CharTokenizer("abcdef")

        payload = checkpoint_payload(model, tokenizer, seq_len=16)

        assert payload["config"] == model.config

    def test_load_reproduces_identical_outputs(self, tmp_path):
        tokenizer = CharTokenizer("abcdefghij ")
        model = create_gpt_small(tokenizer.vocab_size, max_seq_len=32)
        model.eval()

        path = tmp_path / "model.pt"
        torch.save(checkpoint_payload(model, tokenizer, seq_len=16), path)
        tokenizer.save(tokenizer_path_for(str(path)))

        restored, restored_tokenizer = load_gpt(path)
        restored.eval()

        ids = torch.randint(0, tokenizer.vocab_size, (2, 8))
        before, _ = model(ids)
        after, _ = restored(ids)

        assert torch.allclose(before, after, atol=1e-6)
        assert restored_tokenizer.char_to_idx == tokenizer.char_to_idx

    def test_load_works_without_knowing_the_factory(self, tmp_path):
        """The point of the fix: a medium model loads with no extra hints."""
        tokenizer = CharTokenizer("abcdefghij ")
        model = create_gpt_medium(tokenizer.vocab_size, max_seq_len=32)

        path = tmp_path / "medium.pt"
        torch.save(checkpoint_payload(model, tokenizer, seq_len=16), path)
        tokenizer.save(tokenizer_path_for(str(path)))

        restored, _ = load_gpt(path)

        assert restored.config == model.config
