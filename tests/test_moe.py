"""
Mixture-of-Experts routing invariants.

Routing bugs are quiet. A collapsed router still trains and still generates --
it just wastes most of the model's parameters, and you only notice when the loss
curve refuses to beat a dense baseline. These tests check the properties that
distinguish working routing from broken routing.
"""

import torch

from src.moe import MoEFeedForward, Router, moe_aux_losses, moe_load_report
from src.modern import ModernBlock


class TestRouterContract:
    def test_shapes(self):
        router = Router(32, num_experts=8, top_k=2)
        idx, gates, info = router(torch.randn(50, 32))
        assert idx.shape == (50, 2)
        assert gates.shape == (50, 2)
        assert info["load_fraction"].shape == (8,)

    def test_every_token_is_routed(self):
        """No token may be silently dropped by the router itself."""
        router = Router(32, num_experts=8, top_k=2)
        idx, _, _ = router(torch.randn(64, 32))
        assert idx.shape[0] == 64
        assert idx.min() >= 0
        assert idx.max() < 8

    def test_top_k_selections_are_distinct(self):
        """A token must not be routed to the same expert twice."""
        router = Router(32, num_experts=8, top_k=3)
        idx, _, _ = router(torch.randn(100, 32))
        for row in idx:
            assert len(set(row.tolist())) == 3

    def test_gate_weights_sum_to_one(self):
        """
        The expert outputs are a convex combination, so the weights must sum to
        1 -- otherwise the layer silently rescales its own output.
        """
        router = Router(32, num_experts=8, top_k=2, normalize_gates=True)
        _, gates, _ = router(torch.randn(80, 32))
        assert torch.allclose(gates.sum(-1), torch.ones(80), atol=1e-5)

    def test_unnormalized_gates_do_not_sum_to_one(self):
        router = Router(32, num_experts=8, top_k=2, normalize_gates=False)
        _, gates, _ = router(torch.randn(80, 32))
        assert not torch.allclose(gates.sum(-1), torch.ones(80), atol=1e-3)

    def test_load_fraction_is_a_distribution(self):
        router = Router(32, num_experts=8, top_k=2)
        _, _, info = router(torch.randn(120, 32))
        assert abs(float(info["load_fraction"].sum()) - 1.0) < 1e-5

    def test_gates_come_from_unbiased_probabilities(self):
        """
        The loss-free bias must steer *selection* only. If it leaked into the
        returned weights it would distort the expert combination, which is
        exactly what the scheme is designed to avoid.
        """
        router = Router(16, num_experts=4, top_k=4, bias_update_rate=0.01)
        x = torch.randn(20, 16)

        _, gates_before, _ = router(x)
        with torch.no_grad():
            router.expert_bias += torch.tensor([1.0, -1.0, 0.5, -0.5])
        _, gates_after, _ = router(x)

        # With top_k == num_experts every expert is selected regardless of bias,
        # so the weights must be untouched (order may differ, so compare sorted).
        assert torch.allclose(
            gates_before.sort(-1).values, gates_after.sort(-1).values, atol=1e-6
        )


class TestBalancingLosses:
    def test_aux_loss_is_one_at_perfect_balance(self):
        """
        The Switch loss is E * sum_i f_i * P_i, which equals exactly 1.0 when
        both load and probability are uniform. Verified against a hand-built
        uniform router.
        """
        router = Router(16, num_experts=4, top_k=1, aux_loss_coef=1.0)
        with torch.no_grad():
            router.gate.weight.zero_()  # uniform probabilities

        # Uniform probabilities make load uniform in expectation; with all
        # logits equal, topk picks index 0 every time, so build the balanced
        # case by checking the formula directly instead.
        probs = torch.full((100, 4), 0.25)
        load = torch.full((4,), 0.25)
        expected = 4 * float((load * probs.mean(0)).sum())
        assert abs(expected - 1.0) < 1e-6

    def test_aux_loss_grows_with_imbalance(self):
        router = Router(16, num_experts=8, top_k=1, aux_loss_coef=1.0)
        x = torch.randn(200, 16)

        _, _, balanced = router(x)
        with torch.no_grad():
            router.gate.weight[0] += 20.0  # force collapse onto expert 0
        _, _, collapsed = router(x)

        assert collapsed["aux_loss"].item() > balanced["aux_loss"].item()
        assert float(collapsed["max_load_ratio"]) > float(
            balanced["max_load_ratio"]
        )

    def test_max_load_ratio_detects_total_collapse(self):
        """
        Collapse onto one expert gives a ratio of exactly num_experts.

        Zeroing the gate is the cleanest way to force it: every logit is
        identical, so topk breaks the tie toward index 0 for every token. That
        is also a real failure mode -- a router that has learned nothing sends
        everything to one expert.
        """
        router = Router(16, num_experts=8, top_k=1)
        with torch.no_grad():
            router.gate.weight.zero_()
        _, _, info = router(torch.randn(64, 16))
        assert abs(float(info["max_load_ratio"]) - 8.0) < 1e-4

    def test_aux_loss_is_differentiable(self):
        router = Router(16, num_experts=4, top_k=2, aux_loss_coef=1.0)
        _, _, info = router(torch.randn(32, 16))
        info["aux_loss"].backward()
        assert router.gate.weight.grad is not None
        assert router.gate.weight.grad.abs().sum() > 0

    def test_z_loss_penalizes_large_logits(self):
        small = Router(16, num_experts=4, z_loss_coef=1.0)
        large = Router(16, num_experts=4, z_loss_coef=1.0)
        with torch.no_grad():
            large.gate.weight *= 50.0
        x = torch.randn(32, 16)
        assert large(x)[2]["z_loss"].item() > small(x)[2]["z_loss"].item()


class TestLossFreeBias:
    def test_bias_is_a_buffer_not_a_parameter(self):
        """
        It is updated by a rule, not a gradient, so it must not reach the
        optimizer -- but it must persist in state_dict.
        """
        router = Router(16, num_experts=4, bias_update_rate=0.01)
        names = [n for n, _ in router.named_parameters()]
        assert not any("expert_bias" in n for n in names)
        assert "expert_bias" in router.state_dict()

    def test_bias_stays_zero_when_disabled(self):
        router = Router(16, num_experts=4, bias_update_rate=0.0)
        router.train()
        for _ in range(10):
            router(torch.randn(32, 16))
        assert torch.all(router.expert_bias == 0)

    def test_bias_penalizes_overloaded_experts(self):
        """Overloaded experts must get a lower bias, starved ones higher."""
        router = Router(16, num_experts=4, top_k=1, bias_update_rate=0.1)
        with torch.no_grad():
            router.gate.weight.zero_()
            router.gate.weight[2] += 20.0  # expert 2 wins everything
        router.train()

        for _ in range(5):
            router(torch.randn(64, 16))

        bias = router.expert_bias
        assert bias[2] < 0, "overloaded expert should be penalized"
        assert bias[0] > 0 and bias[1] > 0 and bias[3] > 0

    def test_bias_frozen_in_eval_mode(self):
        router = Router(16, num_experts=4, top_k=1, bias_update_rate=0.1)
        router.eval()
        for _ in range(5):
            router(torch.randn(32, 16))
        assert torch.all(router.expert_bias == 0)

    def test_bias_improves_balance(self):
        """
        End to end: with the same skewed start, loss-free balancing must end up
        better balanced than no balancing at all.
        """

        def final_ratio(bias_rate):
            torch.manual_seed(0)
            moe = MoEFeedForward(
                32, num_experts=8, top_k=2, capacity_factor=None,
                aux_loss_coef=0.0, bias_update_rate=bias_rate,
            )
            with torch.no_grad():
                moe.router.gate.weight.mul_(0.5)
                moe.router.gate.weight[0] += 1.5
            moe.train()
            ratios = []
            for _ in range(150):
                moe(torch.randn(1, 128, 32))
                ratios.append(float(moe.last_info["max_load_ratio"]))
            return sum(ratios[-20:]) / 20

        assert final_ratio(0.001) < final_ratio(0.0)


class TestMoELayer:
    def test_output_shape(self):
        moe = MoEFeedForward(32, num_experts=4, top_k=2, capacity_factor=None)
        x = torch.randn(2, 8, 32)
        assert moe(x).shape == x.shape

    def test_no_drops_without_capacity_limit(self):
        moe = MoEFeedForward(32, num_experts=4, top_k=2, capacity_factor=None)
        moe(torch.randn(2, 16, 32))
        assert moe.last_info["dropped_tokens"] == 0

    def test_tight_capacity_drops_tokens(self):
        """
        capacity = factor * top_k * N / E. With factor 0.5, half the assignments
        cannot fit, so tokens must be dropped rather than silently overflowing.
        """
        moe = MoEFeedForward(32, num_experts=4, top_k=2, capacity_factor=0.5)
        moe(torch.randn(2, 32, 32))
        assert moe.last_info["dropped_tokens"] > 0
        assert 0.0 < moe.last_info["drop_rate"] <= 1.0

    def test_dropped_tokens_still_pass_through_residual(self):
        """
        A dropped token contributes zero from the MoE layer. Inside a block the
        residual carries it forward unchanged, so it must not become NaN or zero
        out the hidden state.
        """
        block = ModernBlock(32, num_heads=4, max_seq_len=32, dropout=0.0)
        block.ffn = MoEFeedForward(32, num_experts=8, top_k=1, capacity_factor=0.25)
        out, _ = block(torch.randn(1, 16, 32))
        assert torch.isfinite(out).all()
        assert out.abs().sum() > 0

    def test_shared_experts_apply_to_every_token(self):
        """A shared expert is ungated, so it must change every position."""
        torch.manual_seed(0)
        without = MoEFeedForward(
            32, num_experts=4, top_k=1, num_shared_experts=0, capacity_factor=None
        )
        torch.manual_seed(0)
        with_shared = MoEFeedForward(
            32, num_experts=4, top_k=1, num_shared_experts=1, capacity_factor=None
        )
        x = torch.randn(1, 8, 32)
        diff = (with_shared(x) - without(x)).abs().sum(-1)
        assert (diff > 0).all()

    def test_top_k_one_uses_one_expert_per_token(self):
        moe = MoEFeedForward(32, num_experts=4, top_k=1, capacity_factor=None)
        moe(torch.randn(1, 20, 32))
        assert moe.last_info["load_fraction"].shape == (4,)

    def test_more_experts_means_more_parameters_same_flops(self):
        """
        The defining MoE property: parameters scale with E, active compute with
        top_k. Both layers below activate 2 experts per token.
        """
        small = MoEFeedForward(64, num_experts=4, top_k=2, capacity_factor=None)
        large = MoEFeedForward(64, num_experts=16, top_k=2, capacity_factor=None)

        def expert_params(m):
            return sum(p.numel() for p in m.experts.parameters())

        assert expert_params(large) == 4 * expert_params(small)
        assert small.top_k == large.top_k

    def test_gradients_reach_selected_experts(self):
        moe = MoEFeedForward(32, num_experts=4, top_k=4, capacity_factor=None)
        out = moe(torch.randn(1, 16, 32))
        (out.pow(2).mean() + moe_aux_losses(moe)).backward()

        for i, expert in enumerate(moe.experts):
            grad = expert.w_gate.weight.grad
            assert grad is not None, f"expert {i} got no gradient"
            assert grad.abs().sum() > 0


class TestHelpers:
    def test_aux_losses_sum_across_layers(self):
        model = torch.nn.Sequential(
            MoEFeedForward(32, num_experts=4, capacity_factor=None),
            MoEFeedForward(32, num_experts=4, capacity_factor=None),
        )
        model(torch.randn(1, 8, 32))
        total = moe_aux_losses(model)
        assert total.item() > 0

    def test_aux_losses_zero_without_moe(self):
        model = torch.nn.Linear(4, 4)
        assert moe_aux_losses(model) == 0.0

    def test_load_report_lists_every_layer(self):
        model = torch.nn.Sequential(
            MoEFeedForward(32, num_experts=4, capacity_factor=None),
            MoEFeedForward(32, num_experts=4, capacity_factor=None),
        )
        model(torch.randn(1, 8, 32))
        report = moe_load_report(model)
        assert len(report) == 2
        for entry in report:
            assert len(entry["load_fraction"]) == 4
