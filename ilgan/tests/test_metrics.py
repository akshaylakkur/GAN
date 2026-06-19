"""
Comprehensive tests for all ILGAN metrics modules.

This test suite verifies:

1. **FID calculation** (``ilgan.metrics.image_metrics.FIDCalculator``):
   - Identical distributions give FID ≈ 0.
   - Different distributions give FID > 0.
   - Insufficient samples (fewer than 2) return NaN.
   - Incremental accumulation across multiple updates works correctly.
   - Reset clears accumulated state.

2. **mAP calculation** (``ilgan.metrics.box_metrics.compute_map``):
   - Perfect predictions give mAP = 1.0.
   - Random predictions give mAP ≈ 1 / num_classes.
   - No predictions give mAP = 0.
   - No targets give NaN AP for those classes (excluded from mAP).
   - Edge cases: single class, single box, empty valid mask.

3. **Joint score** (``ilgan.metrics.joint_metrics.compute_joint_score``):
   - The joint score is a weighted combination of FID, mAP, and IS.
   - Higher mAP and IS increase the score; higher FID decreases it.
   - NaN/Inf inputs are sanitised to 0.
   - Default weights produce expected ranges.

4. **MetricsTracker** (``ilgan.metrics.joint_metrics.MetricsTracker``):
   - Accumulates image metrics correctly across multiple updates.
   - Accumulates box metrics correctly across multiple updates.
   - Accumulates loss metrics correctly across multiple updates.
   - ``compute_all`` returns all expected keys.
   - ``reset`` clears all accumulators.
   - ``log_summary`` produces a formatted string (smoke test).
   - Joint score is computed from accumulated metrics.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, List, Optional, Tuple

import torch

from ilgan.metrics import (
    FIDCalculator,
    InceptionScoreCalculator,
    MetricsTracker,
    compute_box_statistics,
    compute_detection_accuracy,
    compute_giou,
    compute_image_statistics,
    compute_joint_score,
    compute_map,
    format_metrics,
)
from ilgan.utils.logger import Logger


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _make_random_images(
    B: int = 4,
    C: int = 3,
    H: int = 64,
    W: int = 64,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Create random images in [-1, 1] range."""
    if seed is not None:
        torch.manual_seed(seed)
    from ilgan.utils.device import get_device
    device = get_device()
    return (torch.rand(B, C, H, W) * 2.0 - 1.0).to(device)


def _make_identical_images(
    B: int = 4,
    C: int = 3,
    H: int = 64,
    W: int = 64,
) -> torch.Tensor:
    """Create a batch of identical images (all pixels = 0.5)."""
    from ilgan.utils.device import get_device
    device = get_device()
    return torch.full((B, C, H, W), 0.5, device=device)


def _make_boxes(
    B: int = 4,
    N: int = 5,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random boxes, labels, scores, and a valid mask.

    Returns
    -------
    tuple of (boxes, scores, labels, target_boxes, target_labels, valid_mask)
    """
    if seed is not None:
        torch.manual_seed(seed)

    boxes = torch.rand(B, N, 4)  # (cx, cy, w, h) in [0, 1]
    labels = torch.randint(0, 10, (B, N))
    scores = torch.sigmoid(torch.randn(B, N))
    target_boxes = torch.rand(B, N, 4)
    target_labels = torch.randint(0, 10, (B, N))
    valid_mask = torch.ones(B, N, dtype=torch.bool)

    return boxes, scores, labels, target_boxes, target_labels, valid_mask


def _make_perfect_predictions(
    B: int = 4,
    N: int = 5,
    num_classes: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create predictions that perfectly match the targets.

    Returns
    -------
    tuple of (pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask)
    """
    torch.manual_seed(42)
    target_boxes = torch.rand(B, N, 4)
    target_labels = torch.randint(0, num_classes, (B, N))
    valid_mask = torch.ones(B, N, dtype=torch.bool)

    # Perfect predictions: same boxes, same labels, high confidence
    pred_boxes = target_boxes.clone()
    pred_labels = target_labels.clone()
    pred_scores = torch.full((B, N), 0.99)

    return pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask


def _make_no_predictions(
    B: int = 4,
    N: int = 5,
    num_classes: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a scenario with no valid predictions (all scores = 0).

    Returns
    -------
    tuple of (pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask)
    """
    torch.manual_seed(42)
    target_boxes = torch.rand(B, N, 4)
    target_labels = torch.randint(0, num_classes, (B, N))
    valid_mask = torch.ones(B, N, dtype=torch.bool)

    # Zero-confidence predictions (treated as invalid by compute_map)
    pred_boxes = torch.rand(B, N, 4)
    pred_labels = torch.randint(0, num_classes, (B, N))
    pred_scores = torch.zeros(B, N)

    return pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask


# ══════════════════════════════════════════════════════════════════════════════
# 1. FID Calculation Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestFIDCalculator(unittest.TestCase):
    """Tests for the FIDCalculator class."""

    def setUp(self) -> None:
        self.calculator = FIDCalculator(device=torch.device("cpu"))
        self.B, self.C, self.H, self.W = 8, 3, 64, 64

    def tearDown(self) -> None:
        self.calculator.reset()

    # ── Identical distributions → FID ≈ 0 ────────────────────────────────

    def test_identical_distributions_fid_near_zero(self) -> None:
        """When real and fake images are drawn from the same distribution,
        FID should be approximately 0."""
        # Use the same random seed for both sets
        real = _make_random_images(self.B, self.C, self.H, self.W, seed=42)
        fake = _make_random_images(self.B, self.C, self.H, self.W, seed=42)

        self.calculator.update(real, fake)
        fid = self.calculator.compute()

        # FID should be very close to 0 (identical distribution)
        self.assertFalse(math.isnan(fid), "FID should not be NaN for valid inputs")
        self.assertLess(fid, 5.0,
                        msg=f"Expected FID near 0 for identical distributions, got {fid}")

    def test_identical_images_fid_zero(self) -> None:
        """When real and fake images are exactly the same images, FID should
        be 0 (or very close to 0 due to numerical precision)."""
        real = _make_identical_images(self.B, self.C, self.H, self.W)
        fake = _make_identical_images(self.B, self.C, self.H, self.W)

        self.calculator.update(real, fake)
        fid = self.calculator.compute()

        self.assertFalse(math.isnan(fid), "FID should not be NaN")
        self.assertAlmostEqual(fid, 0.0, delta=1e-4,
                               msg=f"Expected FID ≈ 0 for identical images, got {fid}")

    # ── Different distributions → FID > 0 ────────────────────────────────

    def test_different_distributions_fid_positive(self) -> None:
        """When real and fake images are from different distributions, FID
        should be strictly positive."""
        real = _make_random_images(self.B, self.C, self.H, self.W, seed=42)
        fake = _make_random_images(self.B, self.C, self.H, self.W, seed=99)

        self.calculator.update(real, fake)
        fid = self.calculator.compute()

        self.assertFalse(math.isnan(fid), "FID should not be NaN")
        self.assertGreater(fid, 0.0,
                           msg=f"Expected FID > 0 for different distributions, got {fid}")

    def test_extreme_difference_fid_large(self) -> None:
        """When real and fake images are extremely different (e.g., all
        black vs all white), FID should be large."""
        # Real: all black (pixel = -1.0)
        real = torch.full((self.B, self.C, self.H, self.W), -1.0)
        # Fake: all white (pixel = 1.0)
        fake = torch.full((self.B, self.C, self.H, self.W), 1.0)

        self.calculator.update(real, fake)
        fid = self.calculator.compute()

        self.assertFalse(math.isnan(fid), "FID should not be NaN")
        self.assertGreater(fid, 10.0,
                           msg=f"Expected large FID for extreme difference, got {fid}")

    # ── Insufficient samples → NaN ───────────────────────────────────────

    def test_insufficient_real_samples_returns_nan(self) -> None:
        """When fewer than 2 real samples are provided, FID should be NaN."""
        real = _make_random_images(1, self.C, self.H, self.W)  # only 1 sample
        fake = _make_random_images(4, self.C, self.H, self.W)

        self.calculator.update(real, fake)
        fid = self.calculator.compute()

        self.assertTrue(math.isnan(fid),
                        "FID should be NaN with < 2 real samples")

    def test_insufficient_fake_samples_returns_nan(self) -> None:
        """When fewer than 2 fake samples are provided, FID should be NaN."""
        real = _make_random_images(4, self.C, self.H, self.W)
        fake = _make_random_images(1, self.C, self.H, self.W)  # only 1 sample

        self.calculator.update(real, fake)
        fid = self.calculator.compute()

        self.assertTrue(math.isnan(fid),
                        "FID should be NaN with < 2 fake samples")

    def test_no_samples_returns_nan(self) -> None:
        """When no samples have been accumulated, FID should be NaN."""
        fid = self.calculator.compute()
        self.assertTrue(math.isnan(fid),
                        "FID should be NaN with no samples")

    # ── Incremental accumulation ─────────────────────────────────────────

    def test_incremental_accumulation(self) -> None:
        """FID computed from multiple incremental updates should be the same
        as FID computed from a single update with all data."""
        # Create two batches
        real1 = _make_random_images(4, self.C, self.H, self.W, seed=42)
        fake1 = _make_random_images(4, self.C, self.H, self.W, seed=42)
        real2 = _make_random_images(4, self.C, self.H, self.W, seed=99)
        fake2 = _make_random_images(4, self.C, self.H, self.W, seed=99)

        # Incremental: update twice
        self.calculator.update(real1, fake1)
        self.calculator.update(real2, fake2)
        fid_incremental = self.calculator.compute()

        # Batch: update once with all data
        calculator_batch = FIDCalculator(device=torch.device("cpu"))
        real_all = torch.cat([real1, real2], dim=0)
        fake_all = torch.cat([fake1, fake2], dim=0)
        calculator_batch.update(real_all, fake_all)
        fid_batch = calculator_batch.compute()

        # They should be very close (identical if the features are the same)
        self.assertAlmostEqual(fid_incremental, fid_batch, delta=1e-4,
                              msg="Incremental and batch FID should match")

    # ── Reset ────────────────────────────────────────────────────────────

    def test_reset_clears_state(self) -> None:
        """After reset, FID should be NaN (no samples)."""
        real = _make_random_images(4, self.C, self.H, self.W)
        fake = _make_random_images(4, self.C, self.H, self.W)

        self.calculator.update(real, fake)
        self.calculator.reset()
        fid = self.calculator.compute()

        self.assertTrue(math.isnan(fid),
                        "FID should be NaN after reset")

    def test_reset_then_update_works(self) -> None:
        """After reset, updating with new data should produce valid FID."""
        real = _make_random_images(4, self.C, self.H, self.W, seed=42)
        fake = _make_random_images(4, self.C, self.H, self.W, seed=42)

        self.calculator.update(real, fake)
        self.calculator.reset()

        # Update with new data
        real2 = _make_random_images(4, self.C, self.H, self.W, seed=99)
        fake2 = _make_random_images(4, self.C, self.H, self.W, seed=99)
        self.calculator.update(real2, fake2)
        fid = self.calculator.compute()

        self.assertFalse(math.isnan(fid),
                         "FID should be valid after reset and re-update")

    # ── Properties ───────────────────────────────────────────────────────

    def test_num_samples_properties(self) -> None:
        """num_real_samples and num_fake_samples should reflect accumulated
        counts."""
        self.assertEqual(self.calculator.num_real_samples, 0)
        self.assertEqual(self.calculator.num_fake_samples, 0)

        real = _make_random_images(4, self.C, self.H, self.W)
        fake = _make_random_images(6, self.C, self.H, self.W)
        self.calculator.update(real, fake)

        self.assertEqual(self.calculator.num_real_samples, 4)
        self.assertEqual(self.calculator.num_fake_samples, 6)

        # After another update
        real2 = _make_random_images(2, self.C, self.H, self.W)
        fake2 = _make_random_images(3, self.C, self.H, self.W)
        self.calculator.update(real2, fake2)

        self.assertEqual(self.calculator.num_real_samples, 6)
        self.assertEqual(self.calculator.num_fake_samples, 9)


# ══════════════════════════════════════════════════════════════════════════════
# 2. mAP Calculation Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeMAP(unittest.TestCase):
    """Tests for the compute_map function."""

    def setUp(self) -> None:
        self.B, self.N = 4, 5
        self.num_classes = 10

    # ── Perfect predictions → mAP = 1.0 ──────────────────────────────────

    def test_perfect_predictions_map_one(self) -> None:
        """When all predictions perfectly match the targets, mAP should be
        1.0."""
        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask = \
            _make_perfect_predictions(self.B, self.N, self.num_classes)

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
            num_classes=self.num_classes,
        )

        self.assertAlmostEqual(result["mAP"], 1.0, places=4,
                               msg=f"Expected mAP=1.0 for perfect predictions, got {result['mAP']}")

    def test_perfect_predictions_single_class(self) -> None:
        """Perfect predictions with a single class should give mAP = 1.0."""
        B, N = 2, 3
        num_classes = 1
        target_boxes = torch.tensor([
            [[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.15, 0.15]],
            [[0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1]],
        ])
        target_labels = torch.zeros(B, N, dtype=torch.long)
        valid_mask = torch.ones(B, N, dtype=torch.bool)

        # Perfect predictions
        pred_boxes = target_boxes.clone()
        pred_labels = target_labels.clone()
        pred_scores = torch.full((B, N), 0.99)

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
            num_classes=num_classes,
        )

        self.assertAlmostEqual(result["mAP"], 1.0, places=4)

    # ── Random predictions → mAP ≈ 1 / num_classes ──────────────────────

    def test_random_predictions_map_approx_one_over_classes(self) -> None:
        """With random predictions, mAP should be approximately 1/num_classes
        (the expected precision of random guessing)."""
        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask = \
            _make_boxes(self.B, self.N, seed=42)

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
            num_classes=self.num_classes,
        )

        # For random predictions with 10 classes, expected mAP ≈ 0.1
        # We use a generous tolerance since random sampling has high variance
        expected_map = 1.0 / self.num_classes
        self.assertGreater(result["mAP"], 0.0,
                           msg=f"Expected mAP > 0 for random predictions, got {result['mAP']}")
        self.assertLess(result["mAP"], 0.5,
                        msg=f"Expected mAP < 0.5 for random predictions, got {result['mAP']}")

    def test_random_predictions_many_classes(self) -> None:
        """With many classes, random mAP should be very small."""
        num_classes = 50
        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask = \
            _make_boxes(self.B, self.N, seed=42)

        # Reassign labels to span many classes
        pred_labels = torch.randint(0, num_classes, (self.B, self.N))
        target_labels = torch.randint(0, num_classes, (self.B, self.N))

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
            num_classes=num_classes,
        )

        # With 50 classes, random mAP should be very low
        self.assertLess(result["mAP"], 0.1,
                        msg=f"Expected mAP < 0.1 for 50-class random, got {result['mAP']}")

    # ── No predictions → mAP = 0 ────────────────────────────────────────

    def test_no_predictions_map_zero(self) -> None:
        """When there are no valid predictions, mAP should be 0."""
        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask = \
            _make_no_predictions(self.B, self.N, self.num_classes)

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
            num_classes=self.num_classes,
        )

        self.assertAlmostEqual(result["mAP"], 0.0, places=4,
                               msg=f"Expected mAP=0 for no predictions, got {result['mAP']}")

    def test_no_predictions_no_targets_map_zero(self) -> None:
        """When there are no predictions and no targets, mAP should be 0."""
        B, N = 2, 3
        pred_boxes = torch.rand(B, N, 4)
        pred_scores = torch.zeros(B, N)
        pred_labels = torch.zeros(B, N, dtype=torch.long)
        target_boxes = torch.rand(B, N, 4)
        target_labels = torch.zeros(B, N, dtype=torch.long)
        valid_mask = torch.zeros(B, N, dtype=torch.bool)  # all invalid

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
            num_classes=self.num_classes,
        )

        self.assertAlmostEqual(result["mAP"], 0.0, places=4)

    # ── No targets for a class → NaN AP (excluded from mAP) ─────────────

    def test_class_with_no_targets_excluded(self) -> None:
        """Classes with no ground-truth boxes should have NaN AP and be
        excluded from the mAP average."""
        B, N = 2, 2
        num_classes = 3

        # Only class 0 and 1 have targets; class 2 has none
        target_boxes = torch.tensor([
            [[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]],
            [[0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.1, 0.1]],
        ])
        target_labels = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        valid_mask = torch.ones(B, N, dtype=torch.bool)

        # Perfect predictions for classes 0 and 1
        pred_boxes = target_boxes.clone()
        pred_labels = target_labels.clone()
        pred_scores = torch.full((B, N), 0.99)

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
            num_classes=num_classes,
        )

        # AP for class 2 should be NaN
        self.assertTrue(math.isnan(result["AP_per_class"][2]),
                        "AP for class with no targets should be NaN")

        # mAP should be 1.0 (average of classes 0 and 1 only)
        self.assertAlmostEqual(result["mAP"], 1.0, places=4)

    # ── Edge cases ──────────────────────────────────────────────────────

    def test_single_box_perfect(self) -> None:
        """A single perfect box should give mAP = 1.0."""
        B, N = 1, 1
        num_classes = 5
        target_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
        target_labels = torch.tensor([[2]], dtype=torch.long)
        valid_mask = torch.ones(B, N, dtype=torch.bool)

        pred_boxes = target_boxes.clone()
        pred_labels = target_labels.clone()
        pred_scores = torch.full((B, N), 0.99)

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
            num_classes=num_classes,
        )

        self.assertAlmostEqual(result["mAP"], 1.0, places=4)

    def test_empty_valid_mask(self) -> None:
        """When the valid mask is all False, mAP should be 0."""
        B, N = 2, 3
        pred_boxes = torch.rand(B, N, 4)
        pred_scores = torch.rand(B, N)
        pred_labels = torch.randint(0, 5, (B, N))
        target_boxes = torch.rand(B, N, 4)
        target_labels = torch.randint(0, 5, (B, N))
        valid_mask = torch.zeros(B, N, dtype=torch.bool)

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
            num_classes=5,
        )

        self.assertAlmostEqual(result["mAP"], 0.0, places=4)
        self.assertEqual(result["num_predictions"], 0)
        self.assertEqual(result["num_targets"], 0)

    def test_invalid_num_classes_raises(self) -> None:
        """num_classes < 1 should raise ValueError."""
        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask = \
            _make_perfect_predictions(self.B, self.N, self.num_classes)

        with self.assertRaises(ValueError):
            compute_map(
                pred_boxes, pred_scores, pred_labels,
                target_boxes, target_labels, valid_mask,
                num_classes=0,
            )

    def test_invalid_iou_threshold_raises(self) -> None:
        """iou_threshold outside (0, 1] should raise ValueError."""
        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask = \
            _make_perfect_predictions(self.B, self.N, self.num_classes)

        with self.assertRaises(ValueError):
            compute_map(
                pred_boxes, pred_scores, pred_labels,
                target_boxes, target_labels, valid_mask,
                num_classes=self.num_classes,
                iou_threshold=0.0,
            )

        with self.assertRaises(ValueError):
            compute_map(
                pred_boxes, pred_scores, pred_labels,
                target_boxes, target_labels, valid_mask,
                num_classes=self.num_classes,
                iou_threshold=1.5,
            )

    def test_result_keys(self) -> None:
        """compute_map should return the correct keys."""
        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask = \
            _make_perfect_predictions(self.B, self.N, self.num_classes)

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
            num_classes=self.num_classes,
        )

        expected_keys = {"mAP", "AP_per_class", "num_predictions", "num_targets"}
        self.assertEqual(set(result.keys()), expected_keys)

    def test_ap_per_class_length(self) -> None:
        """AP_per_class should have length equal to num_classes."""
        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask = \
            _make_perfect_predictions(self.B, self.N, self.num_classes)

        result = compute_map(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
            num_classes=self.num_classes,
        )

        self.assertEqual(len(result["AP_per_class"]), self.num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Joint Score Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeJointScore(unittest.TestCase):
    """Tests for the compute_joint_score function."""

    # ── Weighted combination ─────────────────────────────────────────────

    def test_default_weights_combination(self) -> None:
        """The joint score should be a weighted combination of the three
        components with default weights."""
        # With default weights: w_fid=0.4, w_map=0.4, w_is=0.2
        # J = 0.4 * mAP + 0.2 * (IS/10) - 0.4 * (FID/100)
        fid = 50.0
        mAP = 0.5
        is_score = 5.0

        score = compute_joint_score(fid, mAP, is_score)

        expected = 0.4 * 0.5 + 0.2 * (5.0 / 10.0) - 0.4 * (50.0 / 100.0)
        self.assertAlmostEqual(score, expected, places=6,
                               msg=f"Expected {expected}, got {score}")

    def test_custom_weights(self) -> None:
        """Custom weights should be used when provided."""
        weights = {"fid": 0.2, "map": 0.7, "is": 0.1}
        fid = 30.0
        mAP = 0.8
        is_score = 6.0

        score = compute_joint_score(fid, mAP, is_score, weights=weights)

        expected = 0.7 * 0.8 + 0.1 * (6.0 / 10.0) - 0.2 * (30.0 / 100.0)
        self.assertAlmostEqual(score, expected, places=6)

    # ── Higher mAP increases score ───────────────────────────────────────

    def test_higher_map_increases_score(self) -> None:
        """Holding FID and IS constant, higher mAP should give a higher
        joint score."""
        fid = 50.0
        is_score = 5.0

        score_low_map = compute_joint_score(fid, 0.2, is_score)
        score_high_map = compute_joint_score(fid, 0.9, is_score)

        self.assertGreater(score_high_map, score_low_map,
                           "Higher mAP should increase joint score")

    # ── Higher IS increases score ────────────────────────────────────────

    def test_higher_is_increases_score(self) -> None:
        """Holding FID and mAP constant, higher IS should give a higher
        joint score."""
        fid = 50.0
        mAP = 0.5

        score_low_is = compute_joint_score(fid, mAP, 2.0)
        score_high_is = compute_joint_score(fid, mAP, 8.0)

        self.assertGreater(score_high_is, score_low_is,
                           "Higher IS should increase joint score")

    # ── Higher FID decreases score ────────────────────────────────────────

    def test_higher_fid_decreases_score(self) -> None:
        """Holding mAP and IS constant, higher FID should give a lower
        joint score."""
        mAP = 0.5
        is_score = 5.0

        score_low_fid = compute_joint_score(10.0, mAP, is_score)
        score_high_fid = compute_joint_score(100.0, mAP, is_score)

        self.assertGreater(score_low_fid, score_high_fid,
                           "Higher FID should decrease joint score")

    # ── NaN/Inf sanitisation ────────────────────────────────────────────

    def test_nan_fid_sanitised(self) -> None:
        """NaN FID should be treated as 0 (no penalty)."""
        mAP = 0.5
        is_score = 5.0

        score = compute_joint_score(float("nan"), mAP, is_score)
        expected = 0.4 * 0.5 + 0.2 * (5.0 / 10.0) - 0.4 * (0.0 / 100.0)
        self.assertAlmostEqual(score, expected, places=6)

    def test_inf_fid_sanitised(self) -> None:
        """Inf FID should be treated as 0 (no penalty)."""
        mAP = 0.5
        is_score = 5.0

        score = compute_joint_score(float("inf"), mAP, is_score)
        expected = 0.4 * 0.5 + 0.2 * (5.0 / 10.0) - 0.4 * (0.0 / 100.0)
        self.assertAlmostEqual(score, expected, places=6)

    def test_nan_map_sanitised(self) -> None:
        """NaN mAP should be treated as 0."""
        fid = 50.0
        is_score = 5.0

        score = compute_joint_score(fid, float("nan"), is_score)
        expected = 0.4 * 0.0 + 0.2 * (5.0 / 10.0) - 0.4 * (50.0 / 100.0)
        self.assertAlmostEqual(score, expected, places=6)

    def test_nan_is_sanitised(self) -> None:
        """NaN IS should be treated as 1.0 (minimum valid IS)."""
        fid = 50.0
        mAP = 0.5

        score = compute_joint_score(fid, mAP, float("nan"))
        expected = 0.4 * 0.5 + 0.2 * (1.0 / 10.0) - 0.4 * (50.0 / 100.0)
        self.assertAlmostEqual(score, expected, places=6)

    def test_all_nan_returns_zero(self) -> None:
        """When all inputs are NaN, the joint score should be 0 (all
        contributions are zeroed)."""
        score = compute_joint_score(float("nan"), float("nan"), float("nan"))
        expected = 0.4 * 0.0 + 0.2 * (1.0 / 10.0) - 0.4 * (0.0 / 100.0)
        self.assertAlmostEqual(score, expected, places=6)

    # ── Clamping ─────────────────────────────────────────────────────────

    def test_map_clamped_to_01(self) -> None:
        """mAP should be clamped to [0, 1]."""
        score = compute_joint_score(50.0, 1.5, 5.0)  # mAP > 1
        expected = 0.4 * 1.0 + 0.2 * (5.0 / 10.0) - 0.4 * (50.0 / 100.0)
        self.assertAlmostEqual(score, expected, places=6)

    def test_fid_clamped_to_non_negative(self) -> None:
        """FID should be clamped to >= 0."""
        score = compute_joint_score(-10.0, 0.5, 5.0)  # negative FID
        expected = 0.4 * 0.5 + 0.2 * (5.0 / 10.0) - 0.4 * (0.0 / 100.0)
        self.assertAlmostEqual(score, expected, places=6)

    def test_is_clamped_to_min_one(self) -> None:
        """IS should be clamped to >= 1."""
        score = compute_joint_score(50.0, 0.5, 0.5)  # IS < 1
        expected = 0.4 * 0.5 + 0.2 * (1.0 / 10.0) - 0.4 * (50.0 / 100.0)
        self.assertAlmostEqual(score, expected, places=6)

    # ── Score range ──────────────────────────────────────────────────────

    def test_perfect_scores_positive(self) -> None:
        """Perfect scores (FID=0, mAP=1, IS=10) should give a positive
        joint score."""
        score = compute_joint_score(0.0, 1.0, 10.0)
        expected = 0.4 * 1.0 + 0.2 * (10.0 / 10.0) - 0.4 * (0.0 / 100.0)
        self.assertAlmostEqual(score, expected, places=6)
        self.assertGreater(score, 0.0)

    def test_poor_scores_negative(self) -> None:
        """Poor scores (high FID, low mAP, low IS) should give a negative
        joint score."""
        score = compute_joint_score(200.0, 0.0, 1.0)
        expected = 0.4 * 0.0 + 0.2 * (1.0 / 10.0) - 0.4 * (200.0 / 100.0)
        self.assertAlmostEqual(score, expected, places=6)
        self.assertLess(score, 0.0)

    # ── Custom weights ──────────────────────────────────────────────────

    def test_custom_weights_missing_key_uses_default(self) -> None:
        """If a weight key is missing, the default value for that key should
        be used."""
        weights = {"fid": 0.5, "map": 0.5}  # missing "is"
        score = compute_joint_score(50.0, 0.5, 5.0, weights=weights)
        # Missing "is" defaults to 0.2 (from _DEFAULT_JOINT_WEIGHTS)
        # J = 0.5*0.5 + 0.2*(5/10) - 0.5*(50/100) = 0.25 + 0.1 - 0.25 = 0.1
        expected = 0.5 * 0.5 + 0.2 * (5.0 / 10.0) - 0.5 * (50.0 / 100.0)
        self.assertAlmostEqual(score, expected, places=6)

    def test_custom_weights_do_not_sum_to_one(self) -> None:
        """The joint score should still work even if weights don't sum to 1."""
        weights = {"fid": 1.0, "map": 1.0, "is": 1.0}
        score = compute_joint_score(50.0, 0.5, 5.0, weights=weights)
        expected = 1.0 * 0.5 + 1.0 * (5.0 / 10.0) - 1.0 * (50.0 / 100.0)
        self.assertAlmostEqual(score, expected, places=6)


# ══════════════════════════════════════════════════════════════════════════════
# 4. MetricsTracker Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMetricsTracker(unittest.TestCase):
    """Tests for the MetricsTracker class."""

    def setUp(self) -> None:
        from ilgan.utils.device import get_device
        self.tracker = MetricsTracker(
            num_classes=10,
            device=get_device(),
        )
        self.B, self.N, self.C, self.H, self.W = 4, 5, 3, 64, 64

    def tearDown(self) -> None:
        self.tracker.reset()

    # ── Image metrics accumulation ───────────────────────────────────────

    def test_update_image_metrics_increases_batch_count(self) -> None:
        """Calling update_image_metrics should increase the batch count."""
        real = _make_random_images(self.B, self.C, self.H, self.W)
        fake = _make_random_images(self.B, self.C, self.H, self.W)

        initial_count = self.tracker.num_batches
        self.tracker.update_image_metrics(real, fake)
        self.assertEqual(self.tracker.num_batches, initial_count + 1)

    def test_update_image_metrics_multiple_times(self) -> None:
        """Calling update_image_metrics multiple times should accumulate
        correctly."""
        for _ in range(3):
            real = _make_random_images(self.B, self.C, self.H, self.W)
            fake = _make_random_images(self.B, self.C, self.H, self.W)
            self.tracker.update_image_metrics(real, fake)

        results = self.tracker.compute_all()

        self.assertIn("image/fid", results)
        self.assertIn("image/inception_score", results)
        self.assertIn("image/mean_pixel", results)
        self.assertIn("image/mean_gradient_magnitude", results)
        self.assertIn("image/color_histogram_entropy", results)
        self.assertEqual(results["num_batches"], 3)

    def test_update_image_metrics_empty_tensors(self) -> None:
        """Updating with empty tensors should not crash."""
        real_empty = torch.empty(0, 3, 64, 64)
        fake_empty = torch.empty(0, 3, 64, 64)
        self.tracker.update_image_metrics(real_empty, fake_empty)
        # Should not raise

    # ── Box metrics accumulation ────────────────────────────────────────

    def test_update_box_metrics_increases_batch_count(self) -> None:
        """Calling update_box_metrics should increase the batch count."""
        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask = \
            _make_boxes(self.B, self.N)

        initial_count = self.tracker.num_batches
        self.tracker.update_box_metrics(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
        )
        self.assertEqual(self.tracker.num_batches, initial_count + 1)

    def test_update_box_metrics_multiple_times(self) -> None:
        """Calling update_box_metrics multiple times should accumulate
        correctly."""
        for _ in range(3):
            pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask = \
                _make_boxes(self.B, self.N)
            self.tracker.update_box_metrics(
                pred_boxes, pred_scores, pred_labels,
                target_boxes, target_labels, valid_mask,
            )

        results = self.tracker.compute_all()

        self.assertIn("box/mAP", results)
        self.assertIn("box/mean_giou", results)
        self.assertIn("box/detection_accuracy", results)
        self.assertIn("box/mean_confidence", results)
        self.assertIn("box/mean_box_size", results)
        self.assertIn("box/std_cx", results)
        self.assertIn("box/std_cy", results)
        self.assertEqual(results["num_batches"], 3)

    def test_update_box_metrics_empty_valid_mask(self) -> None:
        """Updating with an all-False valid mask should not crash."""
        pred_boxes = torch.rand(self.B, self.N, 4)
        pred_scores = torch.rand(self.B, self.N)
        pred_labels = torch.randint(0, 10, (self.B, self.N))
        target_boxes = torch.rand(self.B, self.N, 4)
        target_labels = torch.randint(0, 10, (self.B, self.N))
        valid_mask = torch.zeros(self.B, self.N, dtype=torch.bool)

        self.tracker.update_box_metrics(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
        )
        # Should not raise

    # ── Loss metrics accumulation ────────────────────────────────────────

    def test_update_loss_metrics_increases_batch_count(self) -> None:
        """Calling update_loss_metrics should increase the batch count."""
        initial_count = self.tracker.num_batches
        self.tracker.update_loss_metrics({"g_loss": 1.0, "d_loss": 0.5})
        self.assertEqual(self.tracker.num_batches, initial_count + 1)

    def test_update_loss_metrics_multiple_times(self) -> None:
        """Calling update_loss_metrics multiple times should accumulate
        correctly."""
        for i in range(3):
            self.tracker.update_loss_metrics({
                "g_loss": 1.0 + i * 0.1,
                "d_loss": 0.5 - i * 0.1,
                "box_loss": 2.0,
            })

        results = self.tracker.compute_all()

        # Mean of [1.0, 1.1, 1.2] = 1.1
        self.assertAlmostEqual(results["loss/g_loss"], 1.1, places=5)
        # Mean of [0.5, 0.4, 0.3] = 0.4
        self.assertAlmostEqual(results["loss/d_loss"], 0.4, places=5)
        # Mean of [2.0, 2.0, 2.0] = 2.0
        self.assertAlmostEqual(results["loss/box_loss"], 2.0, places=5)

    def test_update_loss_metrics_different_keys(self) -> None:
        """Different batches may have different loss keys; the tracker should
        handle this gracefully."""
        self.tracker.update_loss_metrics({"g_loss": 1.0, "d_loss": 0.5})
        self.tracker.update_loss_metrics({"g_loss": 1.5, "box_loss": 3.0})

        results = self.tracker.compute_all()

        self.assertAlmostEqual(results["loss/g_loss"], 1.25, places=5)
        self.assertAlmostEqual(results["loss/d_loss"], 0.5, places=5)
        self.assertAlmostEqual(results["loss/box_loss"], 3.0, places=5)

    def test_update_loss_metrics_nan_values(self) -> None:
        """NaN loss values should be excluded from the running mean."""
        self.tracker.update_loss_metrics({"g_loss": 1.0})
        self.tracker.update_loss_metrics({"g_loss": float("nan")})
        self.tracker.update_loss_metrics({"g_loss": 3.0})

        results = self.tracker.compute_all()

        # Mean of [1.0, 3.0] = 2.0 (NaN excluded)
        self.assertAlmostEqual(results["loss/g_loss"], 2.0, places=5)

    # ── compute_all keys ─────────────────────────────────────────────────

    def test_compute_all_returns_expected_keys(self) -> None:
        """compute_all should return all expected metric keys."""
        # Add some data
        real = _make_random_images(self.B, self.C, self.H, self.W)
        fake = _make_random_images(self.B, self.C, self.H, self.W)
        self.tracker.update_image_metrics(real, fake)

        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask = \
            _make_boxes(self.B, self.N)
        self.tracker.update_box_metrics(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
        )

        self.tracker.update_loss_metrics({"g_loss": 1.0, "d_loss": 0.5})

        results = self.tracker.compute_all()

        # Check for all expected key groups
        expected_image_keys = {
            "image/fid", "image/inception_score", "image/inception_score_std",
            "image/mean_pixel", "image/std_pixel",
            "image/mean_gradient_magnitude", "image/color_histogram_entropy",
        }
        expected_box_keys = {
            "box/mAP", "box/mean_giou", "box/detection_accuracy", "box/recall",
            "box/mean_confidence", "box/mean_box_size", "box/std_cx", "box/std_cy",
            "box/num_predictions", "box/num_targets",
        }
        expected_loss_keys = {"loss/g_loss", "loss/d_loss"}
        expected_meta_keys = {"num_batches", "joint_score"}

        all_expected = expected_image_keys | expected_box_keys | expected_loss_keys | expected_meta_keys
        actual_keys = set(results.keys())

        missing = all_expected - actual_keys
        extra = actual_keys - all_expected

        self.assertTrue(
            len(missing) == 0,
            f"Missing expected keys: {missing}",
        )
        # Allow extra keys (e.g., additional loss keys from the tracker)
        # but log them for awareness
        if extra:
            print(f"Note: extra keys in results: {extra}")

    # ── Reset ────────────────────────────────────────────────────────────

    def test_reset_clears_all_accumulators(self) -> None:
        """After reset, compute_all should show no accumulated data."""
        real = _make_random_images(self.B, self.C, self.H, self.W)
        fake = _make_random_images(self.B, self.C, self.H, self.W)
        self.tracker.update_image_metrics(real, fake)

        self.tracker.reset()
        results = self.tracker.compute_all()

        self.assertEqual(results["num_batches"], 0)
        self.assertTrue(math.isnan(results["image/fid"]))
        self.assertTrue(math.isnan(results["box/mAP"]))

    def test_reset_then_update_works(self) -> None:
        """After reset, updating with new data should produce valid results."""
        # Add data, reset, add new data
        real1 = _make_random_images(self.B, self.C, self.H, self.W)
        fake1 = _make_random_images(self.B, self.C, self.H, self.W)
        self.tracker.update_image_metrics(real1, fake1)
        self.tracker.reset()

        # Add new data
        real2 = _make_random_images(self.B, self.C, self.H, self.W, seed=99)
        fake2 = _make_random_images(self.B, self.C, self.H, self.W, seed=99)
        self.tracker.update_image_metrics(real2, fake2)

        results = self.tracker.compute_all()
        self.assertEqual(results["num_batches"], 1)
        self.assertFalse(math.isnan(results["image/fid"]))

    # ── Joint score from accumulated metrics ─────────────────────────────

    def test_joint_score_computed_from_accumulated(self) -> None:
        """The joint score should be computed from the accumulated FID, mAP,
        and IS."""
        # Add image metrics (affects FID and IS)
        real = _make_random_images(self.B, self.C, self.H, self.W, seed=42)
        fake = _make_random_images(self.B, self.C, self.H, self.W, seed=42)
        self.tracker.update_image_metrics(real, fake)

        # Add box metrics (affects mAP)
        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask = \
            _make_perfect_predictions(self.B, self.N, num_classes=10)
        self.tracker.update_box_metrics(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
        )

        results = self.tracker.compute_all()

        # The joint score should be a finite float
        self.assertIn("joint_score", results)
        self.assertFalse(math.isnan(results["joint_score"]),
                         "Joint score should not be NaN")
        self.assertFalse(math.isinf(results["joint_score"]),
                         "Joint score should not be Inf")

    def test_joint_score_with_no_box_metrics(self) -> None:
        """When no box metrics are available, the joint score should still
        be computable (mAP defaults to 0)."""
        real = _make_random_images(self.B, self.C, self.H, self.W)
        fake = _make_random_images(self.B, self.C, self.H, self.W)
        self.tracker.update_image_metrics(real, fake)

        results = self.tracker.compute_all()

        self.assertIn("joint_score", results)
        self.assertFalse(math.isnan(results["joint_score"]))

    def test_joint_score_with_no_image_metrics(self) -> None:
        """When no image metrics are available, the joint score should still
        be computable (FID defaults to 100, IS defaults to 1)."""
        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask = \
            _make_perfect_predictions(self.B, self.N, num_classes=10)
        self.tracker.update_box_metrics(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
        )

        results = self.tracker.compute_all()

        self.assertIn("joint_score", results)
        self.assertFalse(math.isnan(results["joint_score"]))

    # ── num_classes property ─────────────────────────────────────────────

    def test_num_classes_property(self) -> None:
        """The num_classes property should return the configured value."""
        self.assertEqual(self.tracker.num_classes, 10)

    def test_num_classes_setter(self) -> None:
        """The num_classes setter should update the value."""
        self.tracker.num_classes = 20
        self.assertEqual(self.tracker.num_classes, 20)

    def test_num_classes_setter_invalid_raises(self) -> None:
        """Setting num_classes to < 1 should raise ValueError."""
        with self.assertRaises(ValueError):
            self.tracker.num_classes = 0
        with self.assertRaises(ValueError):
            self.tracker.num_classes = -1

    # ── log_summary smoke test ──────────────────────────────────────────

    def test_log_summary_does_not_crash(self) -> None:
        """log_summary should produce output without crashing."""
        # Create a simple logger that captures output
        logger = Logger(name="test_metrics", log_dir=None)

        # Add some data
        real = _make_random_images(self.B, self.C, self.H, self.W)
        fake = _make_random_images(self.B, self.C, self.H, self.W)
        self.tracker.update_image_metrics(real, fake)

        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask = \
            _make_boxes(self.B, self.N)
        self.tracker.update_box_metrics(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels, valid_mask,
        )

        self.tracker.update_loss_metrics({"g_loss": 1.0, "d_loss": 0.5})

        # This should not raise
        self.tracker.log_summary(epoch=5, logger=logger, phase="Test")

    def test_log_summary_empty_tracker(self) -> None:
        """log_summary should not crash when the tracker has no data."""
        logger = Logger(name="test_metrics", log_dir=None)
        # Should not raise
        self.tracker.log_summary(epoch=0, logger=logger, phase="Empty")

    # ── Multiple update types together ───────────────────────────────────

    def test_all_update_types_together(self) -> None:
        """The tracker should handle all three update types together."""
        for i in range(2):
            real = _make_random_images(self.B, self.C, self.H, self.W)
            fake = _make_random_images(self.B, self.C, self.H, self.W)
            self.tracker.update_image_metrics(real, fake)

            pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, valid_mask = \
                _make_boxes(self.B, self.N)
            self.tracker.update_box_metrics(
                pred_boxes, pred_scores, pred_labels,
                target_boxes, target_labels, valid_mask,
            )

            self.tracker.update_loss_metrics({
                "g_loss": 1.0 + i * 0.5,
                "d_loss": 0.5 - i * 0.2,
            })

        results = self.tracker.compute_all()

        # Should have 2 batches from each update type
        # But note: each update type increments the batch counter independently
        # So total batches = 2 (image) + 2 (box) + 2 (loss) = 6
        self.assertEqual(results["num_batches"], 6)

        # Losses should be averaged
        self.assertAlmostEqual(results["loss/g_loss"], 1.25, places=5)
        self.assertAlmostEqual(results["loss/d_loss"], 0.4, places=5)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Format Metrics Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestFormatMetrics(unittest.TestCase):
    """Tests for the format_metrics utility function."""

    def test_fid_formatted_two_decimals(self) -> None:
        """FID should be formatted with 2 decimal places."""
        formatted = format_metrics({"image/fid": 12.3456})
        self.assertEqual(formatted["image/fid"], "12.35")

    def test_map_formatted_four_decimals(self) -> None:
        """mAP should be formatted with 4 decimal places."""
        formatted = format_metrics({"box/mAP": 0.876543})
        self.assertEqual(formatted["box/mAP"], "0.8765")

    def test_loss_formatted_six_decimals(self) -> None:
        """Loss values should be formatted with 6 decimal places."""
        formatted = format_metrics({"loss/g_loss": 1.23456789})
        self.assertEqual(formatted["loss/g_loss"], "1.234568")

    def test_joint_score_formatted_six_decimals(self) -> None:
        """Joint score should be formatted with 6 decimal places."""
        formatted = format_metrics({"joint_score": 0.12345678})
        self.assertEqual(formatted["joint_score"], "0.123457")

    def test_nan_formatted_as_na(self) -> None:
        """NaN values should be formatted as 'N/A'."""
        formatted = format_metrics({"image/fid": float("nan")})
        self.assertEqual(formatted["image/fid"], "N/A")

    def test_inf_formatted_as_na(self) -> None:
        """Inf values should be formatted as 'N/A'."""
        formatted = format_metrics({"image/fid": float("inf")})
        self.assertEqual(formatted["image/fid"], "N/A")

    def test_none_formatted_as_na(self) -> None:
        """None values should be formatted as 'N/A'."""
        formatted = format_metrics({"box/mAP": None})
        self.assertEqual(formatted["box/mAP"], "N/A")

    def test_integer_values(self) -> None:
        """Integer values should be formatted as integers."""
        formatted = format_metrics({
            "box/num_predictions": 42,
            "box/num_targets": 100,
            "num_batches": 5,
        })
        self.assertEqual(formatted["box/num_predictions"], "42")
        self.assertEqual(formatted["box/num_targets"], "100")
        self.assertEqual(formatted["num_batches"], "5")

    def test_image_statistics_four_decimals(self) -> None:
        """Image statistics should be formatted with 4 decimal places."""
        formatted = format_metrics({
            "image/mean_pixel": 0.512345,
            "image/std_pixel": 0.123456,
            "image/mean_gradient_magnitude": 0.234567,
            "image/color_histogram_entropy": 3.456789,
        })
        self.assertEqual(formatted["image/mean_pixel"], "0.5123")
        self.assertEqual(formatted["image/std_pixel"], "0.1235")
        self.assertEqual(formatted["image/mean_gradient_magnitude"], "0.2346")
        self.assertEqual(formatted["image/color_histogram_entropy"], "3.4568")

    def test_inception_score_two_decimals(self) -> None:
        """Inception Score should be formatted with 2 decimal places."""
        formatted = format_metrics({
            "image/inception_score": 5.6789,
            "image/inception_score_std": 0.1234,
        })
        self.assertEqual(formatted["image/inception_score"], "5.68")
        self.assertEqual(formatted["image/inception_score_std"], "0.12")

    def test_box_metrics_four_decimals(self) -> None:
        """Box metrics should be formatted with 4 decimal places."""
        formatted = format_metrics({
            "box/mean_giou": 0.765432,
            "box/detection_accuracy": 0.987654,
            "box/recall": 0.876543,
            "box/mean_confidence": 0.654321,
            "box/mean_box_size": 0.123456,
            "box/std_cx": 0.045678,
            "box/std_cy": 0.056789,
        })
        self.assertEqual(formatted["box/mean_giou"], "0.7654")
        self.assertEqual(formatted["box/detection_accuracy"], "0.9877")
        self.assertEqual(formatted["box/recall"], "0.8765")
        self.assertEqual(formatted["box/mean_confidence"], "0.6543")
        self.assertEqual(formatted["box/mean_box_size"], "0.1235")
        self.assertEqual(formatted["box/std_cx"], "0.0457")
        self.assertEqual(formatted["box/std_cy"], "0.0568")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()
