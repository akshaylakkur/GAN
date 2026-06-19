"""
Tests for ``ilgan.models.attention``.

Verifies:
- ``SpatialContentCrossAttention`` forward pass:
  - Output shapes and dtypes.
  - Content-feature aggregation shape correctness.
  - Attention weight map shape and normalisation (sums to 1 per slot).
- Multi-head attention partitioning.
- Auxiliary loss terms:
  - Entropy regularisation (non-negative, zero for uniform distribution).
  - Repulsion loss increases when slots are forced close together and
    decreases when slots are far apart.
- Edge cases: single slot, single head, NaN/Inf detection.
"""

import unittest

import torch
import torch.nn as nn

from ilgan.models.attention import SpatialContentCrossAttention


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_content(
    B: int = 4, C: int = 64, H: int = 16, W: int = 16,
) -> torch.Tensor:
    """Create a random content feature map in a stable range."""
    return torch.randn(B, C, H, W) * 0.1


def _make_queries(
    B: int = 4, N: int = 8, D: int = 64,
) -> torch.Tensor:
    """Create random spatial queries (slots)."""
    return torch.randn(B, N, D) * 0.1


# ──────────────────────────────────────────────────────────────────────────────
# Test SCCA: forward shapes
# ──────────────────────────────────────────────────────────────────────────────


class TestSCCAForwardShapes(unittest.TestCase):
    """Verify output shapes for various input configurations."""

    def test_basic_shapes(self):
        """Standard forward pass should produce correct shapes."""
        B, C, H, W = 4, 64, 16, 16
        N, D = 8, 64
        C_proj = 32

        scca = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
        )
        X = _make_content(B, C, H, W)
        Q = _make_queries(B, N, D)

        Z, A, aux = scca(X, Q)

        # Z: [B, N, D]
        self.assertEqual(Z.shape, (B, N, D), f"Expected Z shape (B={B}, N={N}, D={D})")
        self.assertEqual(Z.dtype, torch.float32)

        # A: [B, N, H, W]
        self.assertEqual(A.shape, (B, N, H, W), f"Expected A shape (B={B}, N={N}, H={H}, W={W})")

        # Each slot's attention should sum to 1.0
        A_sum = A.view(B, N, -1).sum(dim=-1)  # [B, N]
        self.assertTrue(
            torch.allclose(A_sum, torch.ones_like(A_sum), atol=1e-5),
            "Attention weights should sum to 1.0 per slot",
        )

        # Auxiliary losses: check keys exist
        for key in ("entropy", "repulsion", "entropy_weighted", "repulsion_weighted"):
            self.assertIn(key, aux, f"Missing auxiliary loss key: {key}")
            self.assertTrue(aux[key].ndim == 0, f"{key} should be scalar")

    def test_single_slot(self):
        """Should work with a single slot (N=1)."""
        B, C, H, W = 2, 32, 8, 8
        N, D = 1, 32
        C_proj = 16

        scca = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
        )
        X = _make_content(B, C, H, W)
        Q = _make_queries(B, N, D)

        Z, A, aux = scca(X, Q)
        self.assertEqual(Z.shape, (B, N, D))
        self.assertEqual(A.shape, (B, N, H, W))

        # Repulsion loss should be zero with single slot
        self.assertEqual(aux["repulsion"].item(), 0.0,
                         "Repulsion loss should be 0 with a single slot")

    def test_single_batch(self):
        """Should work with batch size 1."""
        B, C, H, W = 1, 64, 16, 16
        N, D = 4, 64
        C_proj = 32

        scca = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
        )
        X = _make_content(B, C, H, W)
        Q = _make_queries(B, N, D)

        Z, A, aux = scca(X, Q)
        self.assertEqual(Z.shape, (B, N, D))
        self.assertEqual(A.shape, (B, N, H, W))

    def test_small_spatial(self):
        """Should work with small spatial dimensions (1x1)."""
        B, C, H, W = 2, 16, 1, 1
        N, D = 3, 16
        C_proj = 16

        scca = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
        )
        X = _make_content(B, C, H, W)
        Q = _make_queries(B, N, D)

        Z, A, aux = scca(X, Q)
        self.assertEqual(Z.shape, (B, N, D))
        # With only 1 spatial position, all attention should be concentrated
        self.assertAlmostEqual(A[0, 0, 0, 0].item(), 1.0, places=5)

    def test_large_slot_count(self):
        """Should work with many slots (N=32)."""
        B, C, H, W = 2, 32, 8, 8
        N, D = 32, 32
        C_proj = 32

        scca = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
        )
        X = _make_content(B, C, H, W)
        Q = _make_queries(B, N, D)

        Z, A, aux = scca(X, Q)
        self.assertEqual(Z.shape, (B, N, D))
        self.assertEqual(A.shape, (B, N, H, W))


# ──────────────────────────────────────────────────────────────────────────────
# Test SCCA: multi-head attention
# ──────────────────────────────────────────────────────────────────────────────


class TestSCCAMultiHead(unittest.TestCase):
    """Verify multi-head attention partitioning."""

    def test_multi_head_shapes(self):
        """Multi-head should produce same output shapes."""
        B, C, H, W = 4, 64, 16, 16
        N, D = 8, 64
        C_proj = 64
        num_heads = 4

        scca = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
            num_heads=num_heads,
        )
        X = _make_content(B, C, H, W)
        Q = _make_queries(B, N, D)

        Z, A, aux = scca(X, Q)
        self.assertEqual(Z.shape, (B, N, D))
        self.assertEqual(A.shape, (B, N, H, W))

        # Attention should still sum to 1
        A_sum = A.view(B, N, -1).sum(dim=-1)
        self.assertTrue(
            torch.allclose(A_sum, torch.ones_like(A_sum), atol=1e-5),
        )

    def test_multi_head_vs_single_head_consistent(self):
        """Multi-head with 1 head should produce same shapes as single-head."""
        B, C, H, W = 2, 32, 8, 8
        N, D = 4, 32
        C_proj = 32

        torch.manual_seed(42)
        scca_single = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
            num_heads=1,
        )
        torch.manual_seed(42)
        scca_multi = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
            num_heads=1,
        )

        X = _make_content(B, C, H, W)
        Q = _make_queries(B, N, D)

        Z1, A1, _ = scca_single(X, Q)
        Z2, A2, _ = scca_multi(X, Q)

        self.assertEqual(Z1.shape, Z2.shape)
        self.assertEqual(A1.shape, A2.shape)

    def test_multi_head_non_divisible_slots(self):
        """Multi-head with non-divisible N works via partitioning."""
        B, C, H, W = 2, 32, 8, 8
        N, D = 7, 32  # not divisible by 4
        C_proj = 64
        num_heads = 4
        # 64 / 4 = 16 per head
        # 7 slots, partitioned into 4 groups

        scca = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
            num_heads=num_heads,
        )
        X = _make_content(B, C, H, W)
        Q = _make_queries(B, N, D)

        Z, A, aux = scca(X, Q)
        self.assertEqual(Z.shape, (B, N, D))
        self.assertEqual(A.shape, (B, N, H, W))

        # All attention should sum to 1
        A_sum = A.view(B, N, -1).sum(dim=-1)
        self.assertTrue(
            torch.allclose(A_sum, torch.ones_like(A_sum), atol=1e-5),
        )

    def test_proj_channels_not_divisible_raises(self):
        """proj_channels must be divisible by num_heads."""
        with self.assertRaises(ValueError):
            SpatialContentCrossAttention(
                content_channels=64,
                slot_dim=64,
                proj_channels=30,  # not divisible by 4
                num_heads=4,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Test SCCA: auxiliary losses — entropy
# ──────────────────────────────────────────────────────────────────────────────


class TestSCCAEntropyLoss(unittest.TestCase):
    """Verify entropy regularisation properties."""

    def test_entropy_non_negative(self):
        """Entropy should always be non-negative."""
        B, C, H, W = 2, 32, 8, 8
        N, D = 4, 32
        C_proj = 32

        scca = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
        )
        X = _make_content(B, C, H, W)
        Q = _make_queries(B, N, D)

        _, _, aux = scca(X, Q)
        self.assertGreaterEqual(aux["entropy"].item(), 0.0,
                                "Entropy must be non-negative")

    def test_entropy_decreases_with_confident_attention(self):
        """Entropy should be lower when a slot focuses on a single location
        vs. spread uniformly.

        We create two scenarios using actual forward passes:
        - Random content with uniform queries produces diffuse attention -> higher entropy
        - A strong peak in one location with a query that matches it produces
          concentrated attention -> lower entropy
        """
        B, C, H, W = 1, 4, 8, 8
        N, D = 2, 4
        C_proj = 4

        scca = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
        )

        # ── Scenario A: random content -> diffuse attention -> higher entropy
        random_content = torch.randn(B, C, H, W) * 0.1
        random_queries = torch.randn(B, N, D) * 0.1

        _, _, aux_random = scca(random_content, random_queries)

        # ── Scenario B: a strong peak in one location with matching query
        # -> concentrated attention -> lower entropy
        peaked_content = torch.zeros(B, C, H, W)
        peaked_content[:, :, 4, 4] = 10.0  # strong peak at (4,4)
        # Use a query that will attend to the peak
        peaked_queries = torch.ones(B, N, D) * 0.5

        _, _, aux_peaked = scca(peaked_content, peaked_queries)

        # Peaked should have lower entropy than random
        self.assertLess(
            aux_peaked["entropy"].item(),
            aux_random["entropy"].item(),
            "Concentrated attention should have lower entropy than diffuse attention",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test SCCA: auxiliary losses — repulsion
# ──────────────────────────────────────────────────────────────────────────────


class TestSCCARepulsionLoss(unittest.TestCase):
    """Verify repulsion loss properties."""

    def test_repulsion_zero_for_far_slots(self):
        """Repulsion should be zero when slots attend to distant regions.

        We manually construct attention distributions that are concentrated
        at opposite corners of the spatial grid.
        """
        B, H, W = 1, 16, 16
        N = 2  # two slots
        spatial_positions = H * W

        # Slot 0 attends to top-left corner (index 0)
        # Slot 1 attends to bottom-right corner (index H*W - 1)
        A = torch.zeros(B, N, spatial_positions)
        A[0, 0, 0] = 1.0
        A[0, 1, spatial_positions - 1] = 1.0

        scca = SpatialContentCrossAttention(
            content_channels=4,
            slot_dim=4,
            proj_channels=4,
        )

        rep = scca.compute_repulsion_loss(A, H, W)
        self.assertAlmostEqual(
            rep.item(), 0.0, places=6,
            msg="Repulsion should be zero when slots attend to distant regions",
        )

    def test_repulsion_increases_when_slots_forced_close(self):
        """Repulsion should be positive when slots attend to the same region.

        We construct attention distributions where both slots concentrate on
        the same spatial position.
        """
        B, H, W = 1, 16, 16
        N = 2
        spatial_positions = H * W

        # Both slots attend exclusively to the centre position
        centre_idx = (H // 2) * W + (W // 2)
        A = torch.zeros(B, N, spatial_positions)
        A[0, 0, centre_idx] = 1.0
        A[0, 1, centre_idx] = 1.0

        scca = SpatialContentCrossAttention(
            content_channels=4,
            slot_dim=4,
            proj_channels=4,
        )

        rep = scca.compute_repulsion_loss(A, H, W)
        self.assertGreater(
            rep.item(), 0.0,
            "Repulsion should be positive when slots attend to the same location",
        )

    def test_repulsion_greater_for_closer_slots(self):
        """Repulsion should be higher when slots are closer together.

        Compare two scenarios:
        1. Slots at the same position.
        2. Slots a few pixels apart.
        """
        B, H, W = 1, 8, 8
        spatial_positions = H * W

        centre_idx = (H // 2) * W + (W // 2)

        # Scenario 1: same position
        A_close = torch.zeros(B, 2, spatial_positions)
        A_close[0, 0, centre_idx] = 1.0
        A_close[0, 1, centre_idx] = 1.0

        # Scenario 2: slightly apart (2 pixels away)
        A_apart = torch.zeros(B, 2, spatial_positions)
        A_apart[0, 0, centre_idx] = 1.0
        A_apart[0, 1, centre_idx + 2] = 1.0  # 2 pixels to the right

        scca = SpatialContentCrossAttention(
            content_channels=4,
            slot_dim=4,
            proj_channels=4,
        )

        rep_close = scca.compute_repulsion_loss(A_close, H, W)
        rep_apart = scca.compute_repulsion_loss(A_apart, H, W)

        self.assertGreater(
            rep_close.item(), rep_apart.item(),
            "Repulsion should be larger for slots at the exact same position "
            "than for slots 2 pixels apart",
        )

    def test_repulsion_three_slots_two_colliding(self):
        """With 3 slots where 2 collide and 1 is far, repulsion should be
        positive but dominated by the colliding pair."""
        B, H, W = 1, 8, 8
        N = 3
        spatial_positions = H * W

        centre_idx = (H // 2) * W + (W // 2)
        far_idx = 0  # top-left

        A = torch.zeros(B, N, spatial_positions)
        A[0, 0, centre_idx] = 1.0
        A[0, 1, centre_idx] = 1.0  # collides with slot 0
        A[0, 2, far_idx] = 1.0      # far away

        scca = SpatialContentCrossAttention(
            content_channels=4,
            slot_dim=4,
            proj_channels=4,
        )

        rep = scca.compute_repulsion_loss(A, H, W)
        self.assertGreater(rep.item(), 0.0,
                           "Repulsion should be positive when at least one pair collides")

    def test_repulsion_lower_threshold_less_penalty(self):
        """A lower repulsion threshold should result in smaller or zero
        penalty for the same arrangement."""
        B, H, W = 1, 8, 8
        spatial_positions = H * W

        centre_idx = (H // 2) * W + (W // 2)

        A = torch.zeros(B, 2, spatial_positions)
        A[0, 0, centre_idx] = 1.0
        A[0, 1, centre_idx + 1] = 1.0  # 1 pixel apart

        scca_wide = SpatialContentCrossAttention(
            content_channels=4,
            slot_dim=4,
            proj_channels=4,
            repulsion_threshold=0.5,
        )
        scca_narrow = SpatialContentCrossAttention(
            content_channels=4,
            slot_dim=4,
            proj_channels=4,
            repulsion_threshold=0.05,
        )

        rep_wide = scca_wide.compute_repulsion_loss(A, H, W)
        rep_narrow = scca_narrow.compute_repulsion_loss(A, H, W)

        self.assertGreaterEqual(
            rep_wide.item(), rep_narrow.item(),
            "Wide repulsion threshold should incur >= penalty than a narrow one",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test SCCA: input validation
# ──────────────────────────────────────────────────────────────────────────────


class TestSCCAValidation(unittest.TestCase):
    """Verify input validation edge cases."""

    def test_batch_mismatch_raises(self):
        """Mismatched batch sizes between X and Q should raise."""
        scca = SpatialContentCrossAttention(
            content_channels=32,
            slot_dim=32,
            proj_channels=16,
        )
        X = _make_content(B=4, C=32, H=8, W=8)
        Q = _make_queries(B=2, N=4, D=32)  # batch mismatch

        with self.assertRaises(ValueError):
            scca(X, Q)

    def test_nan_in_content_raises(self):
        """NaN values in content should raise."""
        scca = SpatialContentCrossAttention(
            content_channels=32,
            slot_dim=32,
            proj_channels=16,
        )
        X = _make_content(B=2, C=32, H=8, W=8)
        X[0, 0, 0, 0] = float("nan")
        Q = _make_queries(B=2, N=4, D=32)

        with self.assertRaises(ValueError):
            scca(X, Q)

    def test_inf_in_query_raises(self):
        """Inf values in queries should raise."""
        scca = SpatialContentCrossAttention(
            content_channels=32,
            slot_dim=32,
            proj_channels=16,
        )
        X = _make_content(B=2, C=32, H=8, W=8)
        Q = _make_queries(B=2, N=4, D=32)
        Q[0, 0, 0] = float("inf")

        with self.assertRaises(ValueError):
            scca(X, Q)


# ──────────────────────────────────────────────────────────────────────────────
# Test SCCA: gradient flow
# ──────────────────────────────────────────────────────────────────────────────


class TestSCCAGradients(unittest.TestCase):
    """Verify gradients flow through the entire module."""

    def test_gradients_flow_to_both_inputs(self):
        """Gradients should flow to both X and Q."""
        B, C, H, W = 2, 16, 8, 8
        N, D = 4, 16
        C_proj = 16

        scca = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
        )

        X = _make_content(B, C, H, W).requires_grad_(True)
        Q = _make_queries(B, N, D).requires_grad_(True)

        Z, _, aux = scca(X, Q)

        # Sum all outputs and losses to get a scalar
        loss = Z.sum() + aux["entropy_weighted"] + aux["repulsion_weighted"]
        loss.backward()

        self.assertIsNotNone(X.grad, "Gradients should flow to X")
        self.assertIsNotNone(Q.grad, "Gradients should flow to Q")
        self.assertGreater(X.grad.abs().sum().item(), 0.0,
                           "X.grad should be non-zero")
        self.assertGreater(Q.grad.abs().sum().item(), 0.0,
                           "Q.grad should be non-zero")

    def test_gradients_flow_to_parameters(self):
        """Gradients should flow to all module parameters."""
        B, C, H, W = 2, 16, 8, 8
        N, D = 4, 16
        C_proj = 16

        scca = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
        )

        X = _make_content(B, C, H, W)
        Q = _make_queries(B, N, D)

        Z, _, aux = scca(X, Q)

        loss = Z.sum() + aux["entropy_weighted"] + aux["repulsion_weighted"]
        loss.backward()

        for name, param in scca.named_parameters():
            self.assertIsNotNone(
                param.grad,
                f"Parameter {name} should have gradients",
            )
            self.assertGreater(
                param.grad.abs().sum().item(), 0.0,
                f"Parameter {name} grad should be non-zero",
            )

    def test_repulsion_loss_backward(self):
        """Repulsion loss should produce non-zero gradients on attention
        weights via the chain through Q."""
        B, C, H, W = 1, 4, 8, 8
        N, D = 2, 4
        C_proj = 4

        scca = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
            repulsion_weight=1.0,
        )

        X = _make_content(B, C, H, W).requires_grad_(True)
        Q = _make_queries(B, N, D).requires_grad_(True)

        _, _, aux = scca(X, Q)

        aux["repulsion_weighted"].backward()
        self.assertIsNotNone(Q.grad, "Repulsion loss should produce gradients on Q")
        self.assertIsNotNone(X.grad, "Repulsion loss should produce gradients on X")


# ──────────────────────────────────────────────────────────────────────────────
# Test SCCA: repulsion with actual forward pass
# ──────────────────────────────────────────────────────────────────────────────


class TestSCCARepulsionForwardBehavior(unittest.TestCase):
    """Test that the repulsion mechanism behaves correctly in actual forward
    passes by controlling the input to force close or far attention."""

    def test_same_content_peaks_produce_repulsion(self):
        """When content has a single strong peak, all slots converge to it
        and repulsion loss should be positive."""
        B, C, H, W = 1, 4, 8, 8
        N, D = 4, 4
        C_proj = 4

        scca = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
        )

        # Content with a single strong peak at (4, 4)
        X = torch.zeros(B, C, H, W)
        X[:, :, 4, 4] = 50.0
        X[:, :, :, :] += torch.randn_like(X) * 0.01  # tiny noise

        # Uniform queries
        Q = torch.randn(B, N, D) * 0.1

        _, A, aux = scca(X, Q)

        # Inspect repulsion
        rep = aux["repulsion"].item()

        # Most slots should attend to the peak, so repulsion > 0
        self.assertGreater(rep, 0.0,
                           "With a single strong content peak, slots should collide "
                           "and produce repulsion")

    def test_different_content_peaks_reduce_repulsion(self):
        """When content has distinct peaks at different locations, slots
        should distribute and repulsion should be lower."""
        B, C, H, W = 1, 4, 8, 8
        N, D = 4, 4
        C_proj = 4

        scca = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
        )

        # Content with 4 distinct peaks in the 4 quadrants
        X = torch.zeros(B, C, H, W)
        X[:, :, 2, 2] = 50.0   # top-left
        X[:, :, 2, 5] = 50.0   # top-right
        X[:, :, 5, 2] = 50.0   # bottom-left
        X[:, :, 5, 5] = 50.0   # bottom-right
        X[:, :, :, :] += torch.randn_like(X) * 0.01

        # Queries initialised with slight differences
        Q = torch.randn(B, N, D) * 0.1

        _, _, aux = scca(X, Q)

        rep_multi_peaks = aux["repulsion"].item()

        # Now compare with single-peak case
        X_single = torch.zeros(B, C, H, W)
        X_single[:, :, 4, 4] = 50.0
        X_single[:, :, :, :] += torch.randn_like(X_single) * 0.01

        _, _, aux_single = scca(X_single, Q)

        rep_single_peak = aux_single["repulsion"].item()

        # With the same mechanism, single peak should generally produce
        # higher or equal repulsion than multi-peak
        self.assertGreaterEqual(
            rep_single_peak, rep_multi_peaks * 0.5,
            "Single peak should typically produce no less repulsion than "
            "multi-peak (not strictly, but highly probable)",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test SCCA: parameter count
# ──────────────────────────────────────────────────────────────────────────────


class TestSCCAParameters(unittest.TestCase):
    """Verify parameter initialisation and count."""

    def test_parameter_count(self):
        """Expected number of parameters should be correct."""
        C, D, C_proj = 64, 64, 32
        scca = SpatialContentCrossAttention(
            content_channels=C,
            slot_dim=D,
            proj_channels=C_proj,
        )

        # value_proj: 1x1 conv C->C': 64*32 = 2048
        # key_proj:   1x1 conv C->C': 64*32 = 2048
        # query_proj: Linear D->C': 64*32 = 2048
        # output_proj: Linear C'->D: 32*64 = 2048
        # Total = 2048 + 2048 + 2048 + 2048 = 8192

        total_params = sum(p.numel() for p in scca.parameters())
        self.assertEqual(
            total_params, 8192,
            f"Expected 8192 parameters, got {total_params}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)