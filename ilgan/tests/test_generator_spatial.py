"""
Tests for ``ilgan.models.generator.SpatialHead``.

Verifies:

- **Forward shapes**: ``SpatialHead`` with ``max_boxes=20`` produces output
  tensors of the correct shape: ``boxes [B, 20, 4]``, ``class_logits [B, 20, C]``,
  ``confidences [B, 20, 1]``.
- **Value ranges**: box coordinates in ``[0, 1]``, confidences in ``[0, 1]``.
- **Auxiliary losses**: dictionary keys exist and losses are scalar.
- **Coarse-to-fine refinement**: running on multiple resolution levels
  produces different attention patterns than a single level.
- **Repulsion loss behaviour**: repulsion decreases when training on a
  synthetic dataset with well-separated objects (gradient optimisation).
- **Edge cases**: single box, single batch, mismatched skip features.
- **Gradient flow**: gradients flow to both the trainable spatial queries
  and the SCCA parameters.
- **Skip feature count mismatch**: raises ``ValueError``.

Note: all tests use conservative model sizes and batch sizes to fit
within limited CPU memory (~1 GB).
"""

import unittest
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ilgan.models.generator import (
    ContentDecoder,
    FeatureProjector,
    SlotMLP,
    SpatialHead,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_skip_features(
    batch_size: int,
    skip_channels: List[int],
    resolutions: List[int],
) -> List[torch.Tensor]:
    """Create a list of mock skip feature maps with small memory footprint."""
    features: List[torch.Tensor] = []
    for ch, res in zip(skip_channels, resolutions):
        features.append(torch.randn(batch_size, ch, res, res) * 0.1)
    return features


def _count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _make_spatial_head(
    max_boxes: int = 8,
    slot_dim: int = 32,
    num_classes: int = 10,
    skip_channels: List[int] | None = None,
    **kwargs,
) -> SpatialHead:
    """Create a compact SpatialHead for testing (small memory footprint)."""
    if skip_channels is None:
        skip_channels = [64, 32, 16]  # 3 levels, small channels
    return SpatialHead(
        max_boxes=max_boxes,
        skip_channels=skip_channels,
        slot_dim=slot_dim,
        num_classes=num_classes,
        **kwargs,
    )


def _make_synthetic_objects_single_level(
    batch_size: int,
    num_objects: int,
    feature_size: int,
    channels: int = 16,
) -> torch.Tensor:
    """Create a single-level synthetic feature map with well-separated blobs.

    Places ``num_objects`` Gaussian blobs at evenly-spaced positions along
    a circle, producing a feature map where distinct objects are present at
    distinct spatial locations.
    """
    angles = torch.linspace(0, 2 * torch.pi, num_objects + 1)[:num_objects]
    radius = feature_size * 0.30
    cx = (feature_size // 2 + (radius * torch.cos(angles)).long()).clamp(2, feature_size - 2)
    cy = (feature_size // 2 + (radius * torch.sin(angles)).long()).clamp(2, feature_size - 2)

    feat = torch.zeros(batch_size, channels, feature_size, feature_size)
    for obj_idx in range(num_objects):
        cxi = int(cx[obj_idx].item())
        cyi = int(cy[obj_idx].item())
        y, x = torch.meshgrid(
            torch.arange(feature_size, dtype=torch.float32),
            torch.arange(feature_size, dtype=torch.float32),
            indexing="ij",
        )
        gaussian = torch.exp(-((x - cxi) ** 2 + (y - cyi) ** 2) / (2 * 3.0 ** 2))
        feat[:, :, :, :] += gaussian[None, None, :, :] * 10.0
    feat += torch.randn_like(feat) * 0.1
    return feat


# ──────────────────────────────────────────────────────────────────────────────
# Test SpatialHead: forward shapes (small model)
# ──────────────────────────────────────────────────────────────────────────────


class TestSpatialHeadForwardShapes(unittest.TestCase):
    """Verify output shapes for the standard forward pass."""

    def setUp(self):
        self.max_boxes = 8
        self.slot_dim = 32
        self.num_classes = 10
        self.batch_size = 2
        # 3 resolution levels with small channels
        self.skip_channels = [64, 32, 16]
        self.resolutions = [8, 16, 32]

        self.head = SpatialHead(
            max_boxes=self.max_boxes,
            skip_channels=self.skip_channels,
            slot_dim=self.slot_dim,
            num_classes=self.num_classes,
        )

    def test_forward_output_shapes(self):
        """Output tensors should have correct shapes."""
        skip_features = _make_skip_features(
            self.batch_size, self.skip_channels, self.resolutions,
        )
        boxes, class_logits, confidences, aux_losses = self.head(skip_features)

        self.assertEqual(
            boxes.shape, (self.batch_size, self.max_boxes, 4),
        )
        self.assertEqual(
            class_logits.shape,
            (self.batch_size, self.max_boxes, self.num_classes),
        )
        self.assertEqual(
            confidences.shape, (self.batch_size, self.max_boxes, 1),
        )

    def test_value_ranges(self):
        """Box coordinates should be in [0, 1] and confidences in [0, 1]."""
        skip_features = _make_skip_features(
            self.batch_size, self.skip_channels, self.resolutions,
        )
        boxes, _, confidences, _ = self.head(skip_features)

        self.assertTrue(torch.all(boxes >= 0.0).item())
        self.assertTrue(torch.all(boxes <= 1.0).item())
        self.assertTrue(torch.all(confidences >= 0.0).item())
        self.assertTrue(torch.all(confidences <= 1.0).item())

    def test_auxiliary_losses_keys(self):
        """Auxiliary losses dict should contain the expected keys."""
        skip_features = _make_skip_features(
            self.batch_size, self.skip_channels, self.resolutions,
        )
        _, _, _, aux_losses = self.head(skip_features)

        for key in ("entropy", "repulsion", "entropy_weighted", "repulsion_weighted"):
            self.assertIn(key, aux_losses)

    def test_auxiliary_losses_scalar(self):
        """Auxiliary losses should be scalar tensors."""
        skip_features = _make_skip_features(
            self.batch_size, self.skip_channels, self.resolutions,
        )
        _, _, _, aux_losses = self.head(skip_features)

        for key, value in aux_losses.items():
            self.assertEqual(value.ndim, 0, f"'{key}' should be scalar")

    def test_repulsion_loss_positive_with_random_data(self):
        """Repulsion loss should typically be positive with random data."""
        skip_features = _make_skip_features(
            self.batch_size, self.skip_channels, self.resolutions,
        )
        _, _, _, aux_losses = self.head(skip_features)
        self.assertGreater(aux_losses["repulsion"].item(), 0.0)

    def test_batch_size_one(self):
        """Should work with B=1."""
        head = _make_spatial_head()
        skip_features = _make_skip_features(1, [64, 32, 16], [8, 16, 32])
        boxes, class_logits, confidences, _ = head(skip_features)
        self.assertEqual(boxes.shape, (1, 8, 4))
        self.assertEqual(class_logits.shape, (1, 8, 10))
        self.assertEqual(confidences.shape, (1, 8, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Test SpatialHead: configurations
# ──────────────────────────────────────────────────────────────────────────────


class TestSpatialHeadConfigs(unittest.TestCase):
    """Verify SpatialHead works with various hyperparameter configurations."""

    def test_single_box(self):
        """Should work with max_boxes=1."""
        head = SpatialHead(
            max_boxes=1, skip_channels=[32], slot_dim=16, num_classes=5,
        )
        skip_features = [torch.randn(2, 32, 8, 8)]
        boxes, class_logits, confidences, _ = head(skip_features)
        self.assertEqual(boxes.shape, (2, 1, 4))
        self.assertEqual(class_logits.shape, (2, 1, 5))
        self.assertEqual(confidences.shape, (2, 1, 1))

    def test_single_resolution_level(self):
        """Should work with a single resolution level."""
        head = SpatialHead(
            max_boxes=5, skip_channels=[32], slot_dim=16, num_classes=5,
        )
        skip_features = [torch.randn(2, 32, 16, 16)]
        boxes, class_logits, confidences, _ = head(skip_features)
        self.assertEqual(boxes.shape, (2, 5, 4))
        self.assertEqual(class_logits.shape, (2, 5, 5))
        self.assertEqual(confidences.shape, (2, 5, 1))

    def test_many_boxes(self):
        """Should work with a large number of boxes (max_boxes=50)."""
        head = SpatialHead(
            max_boxes=50, skip_channels=[16], slot_dim=16, num_classes=5,
        )
        skip_features = [torch.randn(2, 16, 8, 8)]
        boxes, class_logits, confidences, _ = head(skip_features)
        self.assertEqual(boxes.shape, (2, 50, 4))
        self.assertEqual(class_logits.shape, (2, 50, 5))
        self.assertEqual(confidences.shape, (2, 50, 1))

    def test_custom_proj_channels(self):
        """Should work with custom proj_channels != slot_dim."""
        head = SpatialHead(
            max_boxes=5, skip_channels=[32, 16], slot_dim=32,
            num_classes=5, proj_channels=16, num_heads=2,
        )
        skip_features = [
            torch.randn(2, ch, res, res)
            for ch, res in zip([32, 16], [8, 16])
        ]
        boxes, _, _, _ = head(skip_features)
        self.assertEqual(boxes.shape, (2, 5, 4))

    def test_empty_skip_channels_raises(self):
        """Empty skip_channels should raise ValueError."""
        with self.assertRaises(ValueError):
            SpatialHead(max_boxes=5, skip_channels=[], slot_dim=32, num_classes=10)

    def test_mismatched_skip_features_raises(self):
        """Mismatched number of skip features should raise ValueError."""
        head = SpatialHead(
            max_boxes=5, skip_channels=[32, 16], slot_dim=32, num_classes=10,
        )
        skip_features = [
            torch.randn(2, 32, 8, 8),
            torch.randn(2, 16, 16, 16),
            torch.randn(2, 8, 32, 32),
        ]
        with self.assertRaises(ValueError):
            head(skip_features)


# ──────────────────────────────────────────────────────────────────────────────
# Test SpatialHead: gradient flow
# ──────────────────────────────────────────────────────────────────────────────


class TestSpatialHeadGradients(unittest.TestCase):
    """Verify gradients flow through the entire SpatialHead."""

    def setUp(self):
        self.head = _make_spatial_head(
            max_boxes=5, slot_dim=32, num_classes=5,
            skip_channels=[64, 32, 16],
        )
        self.skip_features = _make_skip_features(2, [64, 32, 16], [8, 16, 32])

    def test_gradients_flow_to_spatial_queries(self):
        """Gradients should flow to the learned spatial query parameters."""
        boxes, class_logits, confidences, aux_losses = self.head(self.skip_features)
        loss = (
            boxes.sum() + class_logits.sum() + confidences.sum()
            + aux_losses["repulsion_weighted"] + aux_losses["entropy_weighted"]
        )
        loss.backward()

        self.assertIsNotNone(self.head.spatial_queries.grad)
        self.assertGreater(self.head.spatial_queries.grad.abs().sum().item(), 0.0)

    def test_gradients_flow_to_all_parameters(self):
        """Every trainable parameter should receive a non-zero gradient."""
        boxes, class_logits, confidences, aux_losses = self.head(self.skip_features)
        loss = (
            boxes.sum() + class_logits.sum() + confidences.sum()
            + aux_losses["repulsion_weighted"] + aux_losses["entropy_weighted"]
        )
        loss.backward()

        for name, param in self.head.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"'{name}' should have gradients")
                self.assertGreater(param.grad.abs().sum().item(), 0.0,
                                   f"'{name}' grad should be non-zero")

    def test_gradients_flow_through_all_scca_modules(self):
        """All SCCA modules should receive gradients."""
        boxes, class_logits, confidences, aux_losses = self.head(self.skip_features)
        loss = (
            boxes.sum() + class_logits.sum() + confidences.sum()
            + aux_losses["repulsion_weighted"]
        )
        loss.backward()

        for idx, scca in enumerate(self.head.scca_modules):
            for name, param in scca.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(
                        param.grad,
                        f"SCCA[{idx}] '{name}' should have gradients",
                    )


# ──────────────────────────────────────────────────────────────────────────────
# Test SpatialHead: repulsion loss training
# ──────────────────────────────────────────────────────────────────────────────


class TestSpatialHeadRepulsionTraining(unittest.TestCase):
    """Verify that the repulsion loss decreases during optimisation."""

    def test_repulsion_decreases_with_optimisation(self):
        """Repulsion loss should decrease after optimising on well-separated
        objects."""
        torch.manual_seed(42)

        head = SpatialHead(
            max_boxes=3,
            skip_channels=[16],
            slot_dim=16,
            num_classes=3,
            proj_channels=16,
            num_heads=2,
            repulsion_threshold=0.3,
            repulsion_weight=2.0,
            entropy_weight=0.05,
        )

        # Single resolution level with 3 well-separated Gaussian blobs
        feat = _make_synthetic_objects_single_level(
            batch_size=2, num_objects=3, feature_size=16, channels=16,
        )
        skip_features = [feat]

        optimiser = torch.optim.Adam(head.parameters(), lr=0.02)

        with torch.no_grad():
            _, _, _, aux_before = head(skip_features)
            rep_before = aux_before["repulsion"].item()

        for _ in range(40):
            optimiser.zero_grad()
            _, _, _, aux = head(skip_features)
            loss = aux["repulsion_weighted"]
            loss.backward()
            optimiser.step()

        with torch.no_grad():
            _, _, _, aux_after = head(skip_features)
            rep_after = aux_after["repulsion"].item()

        self.assertLess(
            rep_after, rep_before * 0.8,
            f"Repulsion should decrease: before={rep_before:.6f}, after={rep_after:.6f}",
        )

    def test_boxes_become_diverse_with_training(self):
        """Box coordinates should become more diverse after training."""
        torch.manual_seed(42)

        head = SpatialHead(
            max_boxes=3,
            skip_channels=[16],
            slot_dim=16,
            num_classes=3,
            proj_channels=16,
            num_heads=2,
            repulsion_threshold=0.3,
            repulsion_weight=2.0,
            entropy_weight=0.05,
        )

        feat = _make_synthetic_objects_single_level(
            batch_size=2, num_objects=3, feature_size=16, channels=16,
        )
        skip_features = [feat]

        optimiser = torch.optim.Adam(head.parameters(), lr=0.02)

        def _mean_pairwise_dist(boxes: torch.Tensor) -> float:
            cx = boxes[:, :, 0]
            cy = boxes[:, :, 1]
            dx = cx[:, :, None] - cx[:, None, :]
            dy = cy[:, :, None] - cy[:, None, :]
            dists = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
            triu = torch.triu_indices(3, 3, offset=1)
            return dists[:, triu[0], triu[1]].mean().item()

        with torch.no_grad():
            boxes_before, _, _, _ = head(skip_features)
            dist_before = _mean_pairwise_dist(boxes_before)

        for _ in range(40):
            optimiser.zero_grad()
            boxes, _, _, aux = head(skip_features)
            loss = aux["repulsion_weighted"]
            loss.backward()
            optimiser.step()

        with torch.no_grad():
            boxes_after, _, _, _ = head(skip_features)
            dist_after = _mean_pairwise_dist(boxes_after)

        self.assertGreater(
            dist_after, dist_before * 1.1,
            f"Box centres should spread: before={dist_before:.4f}, after={dist_after:.4f}",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test SpatialHead: coarse-to-fine
# ──────────────────────────────────────────────────────────────────────────────


class TestSpatialHeadCoarseToFine(unittest.TestCase):
    """Verify that multi-resolution processing alters the outputs."""

    def test_multi_level_vs_single_level_differ(self):
        """Multi-level should produce different box outputs than single-level."""
        skip_channels = [64, 32, 16]
        resolutions = [8, 16, 32]

        head_multi = SpatialHead(
            max_boxes=5, skip_channels=skip_channels,
            slot_dim=32, num_classes=5,
        )

        head_single = SpatialHead(
            max_boxes=5, skip_channels=[16],
            slot_dim=32, num_classes=5,
        )

        # Copy output heads so they start identically
        head_single.box_head.load_state_dict(head_multi.box_head.state_dict())
        head_single.class_head.load_state_dict(head_multi.class_head.state_dict())
        head_single.confidence_head.load_state_dict(
            head_multi.confidence_head.state_dict()
        )

        skip_features = _make_skip_features(2, skip_channels, resolutions)

        with torch.no_grad():
            boxes_multi, _, _, _ = head_multi(skip_features)
            boxes_single, _, _, _ = head_single([skip_features[-1]])

        diff = (boxes_multi - boxes_single).abs().mean().item()
        self.assertGreater(diff, 1e-6,
                           "Multi-level and single-level outputs should differ")

    def test_skip_level_changes_boxes(self):
        """Adding a resolution level should change the output boxes."""
        head_full = SpatialHead(
            max_boxes=4, skip_channels=[64, 32, 16],
            slot_dim=32, num_classes=5,
        )
        head_partial = SpatialHead(
            max_boxes=4, skip_channels=[64, 32],
            slot_dim=32, num_classes=5,
        )

        # Copy matching parameters
        head_partial.feature_projectors[0].load_state_dict(
            head_full.feature_projectors[0].state_dict()
        )
        head_partial.feature_projectors[1].load_state_dict(
            head_full.feature_projectors[1].state_dict()
        )
        head_partial.slot_mlp.load_state_dict(head_full.slot_mlp.state_dict())
        head_partial.box_head.load_state_dict(head_full.box_head.state_dict())
        head_partial.class_head.load_state_dict(head_full.class_head.state_dict())
        head_partial.confidence_head.load_state_dict(
            head_full.confidence_head.state_dict()
        )

        skip_features = _make_skip_features(2, [64, 32, 16], [8, 16, 32])

        with torch.no_grad():
            boxes_full, _, _, _ = head_full(skip_features)
            boxes_partial, _, _, _ = head_partial(skip_features[:2])

        diff = (boxes_full - boxes_partial).abs().mean().item()
        self.assertGreater(
            diff, 1e-6,
            "Adding the final resolution level should change box outputs",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test FeatureProjector
# ──────────────────────────────────────────────────────────────────────────────


class TestFeatureProjector(unittest.TestCase):
    """Verify the FeatureProjector module individually."""

    def test_forward_shape(self):
        """Output should have the target number of channels."""
        projector = FeatureProjector(in_channels=64, out_channels=32)
        x = torch.randn(2, 64, 16, 16)
        y = projector(x)
        self.assertEqual(y.shape, (2, 32, 16, 16))

    def test_preserves_spatial_size(self):
        """1×1 conv should not change spatial dimensions."""
        projector = FeatureProjector(in_channels=32, out_channels=16)
        x = torch.randn(2, 32, 8, 8)
        y = projector(x)
        self.assertEqual(y.shape[2], 8)
        self.assertEqual(y.shape[3], 8)

    def test_gradient_flow(self):
        """Gradients should flow through the projector."""
        projector = FeatureProjector(in_channels=16, out_channels=16)
        x = torch.randn(2, 16, 8, 8, requires_grad=True)
        y = projector(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Test SlotMLP
# ──────────────────────────────────────────────────────────────────────────────


class TestSlotMLP(unittest.TestCase):
    """Verify the SlotMLP module individually."""

    def test_forward_shape(self):
        """Output should have same dimensions as input."""
        mlp = SlotMLP(d_model=64)
        x = torch.randn(4, 20, 64)
        y = mlp(x)
        self.assertEqual(y.shape, (4, 20, 64))

    def test_gradient_flow(self):
        """Gradients should flow through the MLP."""
        mlp = SlotMLP(d_model=32)
        x = torch.randn(2, 10, 32, requires_grad=True)
        y = mlp(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Test SpatialHead: parameter count
# ──────────────────────────────────────────────────────────────────────────────


class TestSpatialHeadParameters(unittest.TestCase):
    """Verify parameter count is reasonable."""

    def test_parameter_count_non_zero(self):
        """SpatialHead should have a non-zero number of parameters."""
        head = _make_spatial_head()
        self.assertGreater(_count_parameters(head), 0)

    def test_parameter_count_range(self):
        """SpatialHead should have a reasonable parameter count."""
        head = SpatialHead(
            max_boxes=20, skip_channels=[512, 256, 128, 64, 64],
            slot_dim=128, num_classes=80,
        )
        num_params = _count_parameters(head)
        self.assertGreater(num_params, 500_000)
        self.assertLess(num_params, 50_000_000)


# ──────────────────────────────────────────────────────────────────────────────
# Test SpatialHead: integration with ContentDecoder
# ──────────────────────────────────────────────────────────────────────────────


class TestSpatialHeadIntegration(unittest.TestCase):
    """Verify SpatialHead works with real ContentDecoder skip features."""

    def test_integration_with_content_decoder(self):
        """SpatialHead should consume ContentDecoder's skip features."""
        decoder = ContentDecoder(
            latent_dim=64, gen_base_channels=16, image_size=32,
        )
        head = SpatialHead(
            max_boxes=5,
            skip_channels=decoder._skip_channels,
            slot_dim=32,
            num_classes=10,
        )

        z = torch.randn(2, 64)
        with torch.no_grad():
            img, skip_features = decoder(z)

        boxes, class_logits, confidences, aux_losses = head(skip_features)

        self.assertEqual(boxes.shape, (2, 5, 4))
        self.assertEqual(class_logits.shape, (2, 5, 10))
        self.assertEqual(confidences.shape, (2, 5, 1))
        self.assertEqual(img.shape, (2, 3, 32, 32))
        self.assertGreaterEqual(aux_losses["repulsion"].item(), 0.0)

    def test_integration_gradient_flow(self):
        """Gradients should flow through all head parameters."""
        decoder = ContentDecoder(
            latent_dim=64, gen_base_channels=16, image_size=32,
        )
        head = SpatialHead(
            max_boxes=5,
            skip_channels=decoder._skip_channels,
            slot_dim=32,
            num_classes=5,
        )

        z = torch.randn(2, 64).requires_grad_(True)
        _, skip_features = decoder(z)

        boxes, class_logits, confidences, aux = head(skip_features)
        # Include all outputs so every parameter receives gradients
        loss = boxes.sum() + class_logits.sum() + confidences.sum() + aux["repulsion_weighted"]
        loss.backward()

        # Gradients should flow to head parameters
        for name, param in head.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"Head parameter '{name}' should have gradients",
                )


# ──────────────────────────────────────────────────────────────────────────────
# Test SpatialHead: batch independence
# ──────────────────────────────────────────────────────────────────────────────


class TestSpatialHeadBatchIndependence(unittest.TestCase):
    """Verify different batch elements produce different outputs."""

    def test_different_batch_elements_different_boxes(self):
        """Two different skip features should produce different boxes."""
        head = _make_spatial_head()
        skip_features = _make_skip_features(2, [64, 32, 16], [8, 16, 32])

        with torch.no_grad():
            boxes, _, _, _ = head(skip_features)

        diff = (boxes[0] - boxes[1]).abs().sum().item()
        self.assertGreater(diff, 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)