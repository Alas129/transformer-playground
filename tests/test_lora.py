"""
LoRA invariants.

Three properties make LoRA work, and all three are easy to break:
  1. At initialization the adapter is an exact no-op (B = 0).
  2. Only the adapter trains; the base model is frozen.
  3. Merging is exact -- the merged weight computes the same function.

If (3) is subtly wrong, quality silently drops between the model you evaluated
and the model you deployed.
"""

import torch
import torch.nn as nn

from src.gpt import create_gpt_small
from src.lora import (
    LoRALinear,
    MultiAdapterLoRALinear,
    apply_lora,
    lora_parameters,
    lora_summary,
    merge_lora,
    unmerge_lora,
)


class TestLoRALinear:
    def test_starts_as_exact_no_op(self):
        """
        B is initialized to zero, so the adapted layer must be bit-identical to
        the base at step 0. Training therefore starts from the pretrained
        function, not a perturbation of it.
        """
        base = nn.Linear(32, 16)
        x = torch.randn(4, 32)
        expected = base(x).clone()
        lora = LoRALinear(base, r=4, alpha=8, dropout=0.0)
        assert torch.equal(lora(x), expected)

    def test_becomes_a_no_op_only_because_b_is_zero(self):
        """A is nonzero -- both zero would leave both gradients stuck at zero."""
        lora = LoRALinear(nn.Linear(32, 16), r=4)
        assert lora.lora_A.abs().sum() > 0
        assert lora.lora_B.abs().sum() == 0

    def test_changes_output_once_b_is_nonzero(self):
        base = nn.Linear(32, 16)
        x = torch.randn(4, 32)
        lora = LoRALinear(base, r=4, dropout=0.0)
        before = lora(x).clone()
        with torch.no_grad():
            lora.lora_B.normal_(0, 0.1)
        assert not torch.allclose(lora(x), before)

    def test_base_weights_are_frozen(self):
        lora = LoRALinear(nn.Linear(32, 16), r=4)
        assert not lora.base.weight.requires_grad
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_gradient_reaches_adapter_only(self):
        lora = LoRALinear(nn.Linear(32, 16), r=4, dropout=0.0)
        lora(torch.randn(4, 32)).pow(2).mean().backward()
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        assert lora.base.weight.grad is None

    def test_scaling_uses_alpha_over_r(self):
        lora = LoRALinear(nn.Linear(8, 8), r=4, alpha=16)
        assert lora.scaling == 4.0

    def test_rejects_non_linear_module(self):
        try:
            LoRALinear(nn.Conv1d(4, 4, 1), r=2)
        except TypeError:
            return
        raise AssertionError("expected a TypeError for a non-Linear base")


class TestMerge:
    def test_merge_is_numerically_exact(self):
        """The core guarantee. Merged output must equal unmerged output."""
        base = nn.Linear(48, 24)
        lora = LoRALinear(base, r=8, alpha=16, dropout=0.0).eval()
        with torch.no_grad():
            lora.lora_B.normal_(0, 0.05)

        x = torch.randn(6, 48)
        unmerged = lora(x).clone()
        lora.merge()
        merged = lora(x)

        assert torch.allclose(unmerged, merged, atol=1e-6)

    def test_merge_actually_changes_the_base_weight(self):
        base = nn.Linear(16, 16)
        original = base.weight.clone()
        lora = LoRALinear(base, r=4, dropout=0.0)
        with torch.no_grad():
            lora.lora_B.normal_(0, 0.1)
        lora.merge()
        assert not torch.allclose(lora.base.weight, original)

    def test_merge_is_idempotent(self):
        """Calling merge() twice must not double-apply the update."""
        lora = LoRALinear(nn.Linear(16, 16), r=4, dropout=0.0).eval()
        with torch.no_grad():
            lora.lora_B.normal_(0, 0.1)
        x = torch.randn(3, 16)
        lora.merge()
        once = lora(x).clone()
        lora.merge()
        assert torch.allclose(lora(x), once, atol=1e-6)

    def test_unmerge_restores_the_original_weight(self):
        base = nn.Linear(16, 16)
        original = base.weight.clone()
        lora = LoRALinear(base, r=4, dropout=0.0)
        with torch.no_grad():
            lora.lora_B.normal_(0, 0.1)
        lora.merge()
        lora.unmerge()
        assert torch.allclose(lora.base.weight, original, atol=1e-6)

    def test_merged_layer_has_no_extra_compute(self):
        """
        After merging, the forward pass is a single Linear. That is LoRA's
        deployment advantage over adapters that add depth.
        """
        lora = LoRALinear(nn.Linear(16, 16), r=4, dropout=0.0).eval()
        lora.merge()
        x = torch.randn(2, 16)
        assert torch.equal(lora(x), lora.base(x))


class TestApplyLoRA:
    def test_adapts_named_targets(self, vocab_size):
        model = create_gpt_small(vocab_size, max_seq_len=16)
        adapted = apply_lora(model, target_names=("W_q", "W_v"), r=4)
        assert len(adapted) > 0
        assert all(name.endswith(("W_q", "W_v")) for name in adapted)

    def test_only_adapters_are_trainable(self, vocab_size):
        model = create_gpt_small(vocab_size, max_seq_len=16)
        apply_lora(model, r=4)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert "lora_A" in name or "lora_B" in name

    def test_trains_a_small_fraction_of_parameters(self, vocab_size):
        model = create_gpt_small(vocab_size, max_seq_len=16)
        apply_lora(model, target_names=("W_q", "W_v"), r=4)
        summary = lora_summary(model)
        assert 0 < summary["percent"] < 5.0
        assert summary["trainable"] < summary["total"]

    def test_model_still_runs_and_output_is_unchanged(self, vocab_size):
        """Applying LoRA must not change the model's function at step 0."""
        model = create_gpt_small(vocab_size, max_seq_len=16).eval()
        ids = torch.randint(0, vocab_size, (2, 6))
        with torch.no_grad():
            before, _ = model(ids)
        apply_lora(model, r=4, dropout=0.0)
        model.eval()
        with torch.no_grad():
            after, _ = model(ids)
        assert torch.allclose(before, after, atol=1e-6)

    def test_end_to_end_merge_equivalence(self, vocab_size):
        """
        Full model: train the adapter briefly, then confirm merging preserves the
        logits exactly. This is the check that matters before deployment.
        """
        model = create_gpt_small(vocab_size, max_seq_len=16)
        apply_lora(model, target_names=("W_q", "W_v"), r=4, dropout=0.0)

        optimizer = torch.optim.AdamW(lora_parameters(model), lr=1e-2)
        ids = torch.randint(0, vocab_size, (2, 8))
        for _ in range(3):
            _, loss = model(ids, targets=ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            unmerged, _ = model(ids)
            count = merge_lora(model)
            merged, _ = model(ids)

        assert count > 0
        assert torch.allclose(unmerged, merged, atol=1e-5)

    def test_unmerge_round_trip_on_full_model(self, vocab_size):
        model = create_gpt_small(vocab_size, max_seq_len=16).eval()
        apply_lora(model, r=4, dropout=0.0)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    module.lora_B.normal_(0, 0.02)

        ids = torch.randint(0, vocab_size, (1, 6))
        with torch.no_grad():
            before, _ = model(ids)
            merge_lora(model)
            unmerge_lora(model)
            after, _ = model(ids)

        assert torch.allclose(before, after, atol=1e-5)

    def test_raises_when_no_target_matches(self, vocab_size):
        model = create_gpt_small(vocab_size, max_seq_len=16)
        try:
            apply_lora(model, target_names=("does_not_exist",))
        except ValueError:
            return
        raise AssertionError("expected a ValueError when nothing matched")

    def test_adapter_state_is_small(self, vocab_size):
        """
        Shipping a fine-tune should mean shipping megabytes, not gigabytes.
        Only the adapter tensors need saving.
        """
        model = create_gpt_small(vocab_size, max_seq_len=16)
        apply_lora(model, target_names=("W_q", "W_v"), r=4)
        adapter = {
            k: v for k, v in model.state_dict().items() if "lora_" in k
        }
        adapter_size = sum(v.numel() for v in adapter.values())
        full_size = sum(v.numel() for v in model.state_dict().values())
        assert adapter_size < full_size / 20


class TestMultiAdapterServing:
    def test_shapes(self):
        layer = MultiAdapterLoRALinear(nn.Linear(32, 16), num_adapters=4, r=4)
        x = torch.randn(3, 5, 32)
        ids = torch.tensor([0, 2, 1])
        assert layer(x, ids).shape == (3, 5, 16)

    def test_starts_as_no_op(self):
        base = nn.Linear(32, 16)
        layer = MultiAdapterLoRALinear(base, num_adapters=4, r=4)
        x = torch.randn(2, 5, 32)
        ids = torch.tensor([0, 1])
        assert torch.allclose(layer(x, ids), base(x), atol=1e-6)

    def test_negative_id_selects_base_only(self):
        """
        Untuned requests must be able to share the batch, otherwise a server
        needs a separate batch for them.
        """
        base = nn.Linear(32, 16)
        layer = MultiAdapterLoRALinear(base, num_adapters=4, r=4)
        with torch.no_grad():
            layer.lora_B.normal_(0, 0.1)

        x = torch.randn(2, 4, 32)
        out = layer(x, torch.tensor([-1, 0]))

        assert torch.allclose(out[0], base(x[0]), atol=1e-6)
        assert not torch.allclose(out[1], base(x[1]), atol=1e-6)

    def test_each_sequence_uses_its_own_adapter(self):
        """
        The correctness requirement for batched multi-tenant serving: mixing
        adapters in one batch must give the same answer as running each
        sequence alone.
        """
        base = nn.Linear(24, 12)
        layer = MultiAdapterLoRALinear(base, num_adapters=3, r=4)
        with torch.no_grad():
            layer.lora_B.normal_(0, 0.1)

        x = torch.randn(3, 6, 24)
        ids = torch.tensor([2, 0, 1])

        batched = layer(x, ids)
        for i in range(3):
            alone = layer(x[i : i + 1], ids[i : i + 1])
            assert torch.allclose(batched[i], alone[0], atol=1e-6)

    def test_matches_single_adapter_layer(self):
        """
        A stacked adapter bank must compute exactly what the plain LoRALinear
        computes for the same weights.
        """
        base = nn.Linear(16, 8)
        multi = MultiAdapterLoRALinear(base, num_adapters=2, r=4, alpha=8)
        with torch.no_grad():
            multi.lora_B.normal_(0, 0.1)

        single = LoRALinear(nn.Linear(16, 8), r=4, alpha=8, dropout=0.0)
        with torch.no_grad():
            single.base.weight.copy_(base.weight)
            single.base.bias.copy_(base.bias)
            single.lora_A.copy_(multi.lora_A[1])
            single.lora_B.copy_(multi.lora_B[1])

        x = torch.randn(1, 5, 16)
        assert torch.allclose(
            multi(x, torch.tensor([1])), single(x), atol=1e-6
        )

    def test_base_weight_is_shared_not_duplicated(self):
        """
        One base, many tenants. If the base were copied per adapter the memory
        argument for this design would collapse.
        """
        base = nn.Linear(64, 64)
        layer = MultiAdapterLoRALinear(base, num_adapters=100, r=4)
        base_params = base.weight.numel()
        adapter_params = layer.lora_A.numel() + layer.lora_B.numel()
        assert layer.base.weight is base.weight
        assert adapter_params < 15 * base_params
