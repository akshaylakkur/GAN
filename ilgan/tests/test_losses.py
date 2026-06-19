"""Tests for ILGAN loss functions.

This module tests all loss functions defined in ``ilgan.losses``, including
adversarial losses, box regression losses, collapse prevention losses,
consistency losses, and the central :class:`LossAggregator`.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict

import torch

from ilgan.losses import (
    LossAggregator,
    attention_entropy_loss,
    class_loss,
    compute_adversarial_losses,
    compute_box_losses,
    compute_collapse_losses,
    compute_consistency_loss,
    confidence_loss,
    consistency_loss,
    cosine_similarity,
    feature_diversity_loss,
    giou_loss,
    gradient_penalty,
    l1_box_loss,
    latent_diversity_loss,
    repulsion_loss,
    wgan_discriminator_loss,
    wgan_generator_loss,
)
from ilgan.losses.consistency import BoxFeatureEncoder, ImageFeatureEncoder
from ilgan.utils.config import Config


def _make_minimal_config() -> Config:
    """Create a minimal Config with all required keys for testing."""
    return Config.from_dict({
        "data": {
            "image_size": 32,
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {
            "latent_dim": 32,
            "gen_base_channels": 8,
            "disc_base_channels": 8,
            "num_attention_heads": 2,
            "max_boxes": 5,
            "num_classes": 2,
        },
        "loss": {
            "adv_weight": 1.0,
            "box_weight": 5.0,
            "class_weight": 1.0,
            "confidence_weight": 1.0,
            "diversity_weight": 0.1,
            "consistency_weight": 0.5,
            "entropy_weight": 0.1,
            "repulsion_weight": 1.0,
            "feature_diversity_weight": 0.1,
            "latent_diversity_weight": 0.01,
            "spectral_reg_weight": 0.001,
            "gp_weight": 10.0,
            "w_global": 0.5,
            "repulsion_threshold": 0.2,
            "noise_schedule_initial": 0.1,
            "noise_schedule_final": 0.001,
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.0002,
            "beta1": 0.0,
            "beta2": 0.9,
            "n_critic": 1,
            "gradient_accumulation_steps": 1,
            "use_mixed_precision": False,
            "grad_checkpoint": False,
            "clip_grad_norm": 1.0,
            "adaptive_lr": False,
            "gradient_balance": False,
            "representation_anchor_frequency": 500,
            "representation_anchor_weight": 0.01,
        },
        "logging": {
            "log_interval": 10,
            "save_interval": 50,
            "eval_interval": 100,
            "use_wandb": False,
        },
        "metrics": {
            "fid_sample_size": 100,
            "map_iou_threshold": 0.5,
            "eval_sample_size": 100,
            "compute_fid": False,
            "compute_is": False,
            "compute_map": False,
            "num_fid_splits": 10,
        },
        "augmentation": {
            "use_mosaic": False,
            "use_mixup": False,
            "mosaic_prob": 0.0,
            "mixup_prob": 0.0,
            "mixup_alpha": 0.5,
            "random_erasing_prob": 0.0,
            "random_erasing_fill": "random",
            "random_erasing_max": 1,
            "hflip_prob": 0.0,
            "color_jitter_prob": 0.0,
            "affine_prob": 0.0,
            "cutout_prob": 0.0,
            "cutout_max_holes": 1,
            "cutout_max_size": 0.1,
        },
        "profiling": {
            "profile_memory": False,
            "log_gradients": False,
            "detect_anomalies": False,
            "log_spectral_norms": False,
            "log_parameter_stats": False,
            "profile_frequency": 100,
            "track_ema_ratio": False,
        },
        "paths": {
            "data_root": "./data",
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./logs",
        },
    })


class TestAdversarialLosses(unittest.TestCase):
    """Tests for WGAN-GP adversarial losses."""

    def setUp(self) -> None:
        self.B = 4
        self.H = 4
        self.W = 4
        self.real_local = torch.randn(self.B, 1, self.H, self.W)
        self.real_global = torch.randn(self.B, 1)
        self.fake_local = torch.randn(self.B, 1, self.H, self.W)
        self.fake_global = torch.randn(self.B, 1)

    def test_wgan_discriminator_loss_shape(self) -> None:
        loss = wgan_discriminator_loss(
            self.real_local, self.real_global,
            self.fake_local, self.fake_global,
        )
        self.assertEqual(loss.ndim, 0)

    def test_wgan_generator_loss_shape(self) -> None:
        loss = wgan_generator_loss(self.fake_local, self.fake_global)
        self.assertEqual(loss.ndim, 0)

    def test_wgan_negative_w_global_raises(self) -> None:
        with self.assertRaises(ValueError):
            wgan_discriminator_loss(
                self.real_local, self.real_global,
                self.fake_local, self.fake_global,
                w_global=-0.1,
            )
        with self.assertRaises(ValueError):
            wgan_generator_loss(
                self.fake_local, self.fake_global,
                w_global=-0.1,
            )

    def test_compute_adversarial_losses_keys(self) -> None:
        # Create a minimal discriminator for gradient penalty
        from ilgan.models.discriminator import ImageDiscriminator
        disc = ImageDiscriminator(
            disc_base_channels=8,
            image_size=32,
        )
        real_imgs = torch.randn(self.B, 3, 32, 32)
        fake_imgs = torch.randn(self.B, 3, 32, 32)

        losses = compute_adversarial_losses(
            discriminator=disc,
            real_images=real_imgs,
            fake_images=fake_imgs,
            real_scores_local=self.real_local,
            real_scores_global=self.real_global,
            fake_scores_local=self.fake_local,
            fake_scores_global=self.fake_global,
        )
        expected_keys = {"d_loss", "g_loss", "gp_loss", "gp_value"}
        self.assertEqual(set(losses.keys()), expected_keys)
        for v in losses.values():
            self.assertEqual(v.ndim, 0)


class TestBoxRegressionLosses(unittest.TestCase):
    """Tests for GIoU, L1, class, and confidence losses."""

    def setUp(self) -> None:
        self.B, self.N = 4, 10
        self.pred = torch.rand(self.B, self.N, 4)
        self.target = torch.rand(self.B, self.N, 4)
        self.mask = torch.rand(self.B, self.N) > 0.5

    def test_giou_loss_shape(self) -> None:
        loss = giou_loss(self.pred, self.target, self.mask)
        self.assertEqual(loss.ndim, 0)

    def test_giou_loss_empty_mask(self) -> None:
        empty_mask = torch.zeros(self.B, self.N, dtype=torch.bool)
        loss = giou_loss(self.pred, self.target, empty_mask)
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_giou_loss_perfect_overlap(self) -> None:
        boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
        mask = torch.tensor([[True]])
        loss = giou_loss(boxes, boxes, mask)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_l1_loss_shape(self) -> None:
        loss = l1_box_loss(self.pred, self.target, self.mask)
        self.assertEqual(loss.ndim, 0)

    def test_l1_loss_empty_mask(self) -> None:
        empty_mask = torch.zeros(self.B, self.N, dtype=torch.bool)
        loss = l1_box_loss(self.pred, self.target, empty_mask)
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_compute_box_losses_keys(self) -> None:
        losses = compute_box_losses(self.pred, self.target, self.mask)
        expected_keys = {"giou_loss", "l1_loss", "box_loss"}
        self.assertEqual(set(losses.keys()), expected_keys)

    def test_class_loss_shape(self) -> None:
        logits = torch.randn(self.B, self.N, 5)
        labels = torch.randint(0, 5, (self.B, self.N))
        result = class_loss(logits, labels, self.mask)
        self.assertIn("class_loss", result)
        self.assertEqual(result["class_loss"].ndim, 0)

    def test_confidence_loss_shape(self) -> None:
        conf = torch.sigmoid(torch.randn(self.B, self.N))
        target = self.mask.to(torch.float)
        result = confidence_loss(conf, target, self.mask)
        self.assertIn("confidence_loss", result)
        self.assertEqual(result["confidence_loss"].ndim, 0)


class TestCollapsePreventionLosses(unittest.TestCase):
    """Tests for attention entropy, repulsion, feature diversity, and latent diversity losses."""

    def setUp(self) -> None:
        self.B, self.N, self.H, self.W = 2, 4, 8, 8
        # Create valid attention distributions (sum to 1 over HW)
        self.attn = torch.rand(self.B, self.N, self.H, self.W)
        self.attn = self.attn / self.attn.view(self.B, self.N, -1).sum(dim=-1, keepdim=True).view(
            self.B, self.N, 1, 1
        )
        self.skip = [torch.randn(self.B, 8, 8, 8), torch.randn(self.B, 4, 16, 16)]
        self.z = torch.randn(self.B, 32)

    def test_attention_entropy_loss_shape(self) -> None:
        loss = attention_entropy_loss(self.attn)
        self.assertEqual(loss.ndim, 0)

    def test_attention_entropy_uniform(self) -> None:
        uniform = torch.ones(self.B, self.N, self.H, self.W) / (self.H * self.W)
        loss = attention_entropy_loss(uniform)
        # Entropy of uniform distribution over 64 positions = ln(64) ≈ 4.1589
        self.assertAlmostEqual(loss.item(), 4.1589, places=3)

    def test_attention_entropy_concentrated(self) -> None:
        concentrated = torch.zeros(self.B, self.N, self.H, self.W)
        concentrated[:, :, 4, 4] = 1.0
        loss = attention_entropy_loss(concentrated)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_repulsion_loss_shape(self) -> None:
        loss = repulsion_loss(self.attn)
        self.assertEqual(loss.ndim, 0)

    def test_repulsion_loss_single_slot(self) -> None:
        single = self.attn[:, :1, :, :]
        loss = repulsion_loss(single)
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_repulsion_loss_far_slots(self) -> None:
        """Two slots at opposite corners should have zero repulsion."""
        A = torch.zeros(1, 2, self.H, self.W)
        A[0, 0, 0, 0] = 1.0
        A[0, 1, self.H - 1, self.W - 1] = 1.0
        loss = repulsion_loss(A, repulsion_threshold=0.2)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_repulsion_loss_collapsed_slots(self) -> None:
        """Two slots at the same position should have high repulsion."""
        A = torch.zeros(1, 2, self.H, self.W)
        A[0, 0, 4, 4] = 1.0
        A[0, 1, 4, 4] = 1.0
        loss = repulsion_loss(A, repulsion_threshold=0.2)
        self.assertGreater(loss.item(), 0.0)

    def test_feature_diversity_loss_shape(self) -> None:
        loss = feature_diversity_loss(self.skip)
        self.assertEqual(loss.ndim, 0)

    def test_feature_diversity_empty_list(self) -> None:
        loss = feature_diversity_loss([])
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_latent_diversity_loss_shape(self) -> None:
        loss = latent_diversity_loss(self.z)
        self.assertEqual(loss.ndim, 0)

    def test_latent_diversity_single_sample(self) -> None:
        z_single = torch.randn(1, 32)
        loss = latent_diversity_loss(z_single)
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_compute_collapse_losses_keys(self) -> None:
        losses = compute_collapse_losses(self.attn, self.skip, self.z)
        expected_keys = {
            "entropy", "repulsion", "feature_diversity",
            "latent_diversity", "collapse_loss",
        }
        self.assertEqual(set(losses.keys()), expected_keys)
        for v in losses.values():
            self.assertEqual(v.ndim, 0)


class TestConsistencyLosses(unittest.TestCase):
    """Tests for cross-modal consistency losses and encoders."""

    def setUp(self) -> None:
        self.B = 4
        self.proj_dim = 128
        self.img_enc = ImageFeatureEncoder(proj_dim=self.proj_dim)
        self.box_enc = BoxFeatureEncoder(proj_dim=self.proj_dim)
        self.images = torch.randn(self.B, 3, 64, 64)
        self.boxes = torch.rand(self.B, 10, 4)
        self.confs = torch.sigmoid(torch.randn(self.B, 10, 1))
        self.mask = torch.rand(self.B, 10) > 0.5

    def test_image_encoder_shape(self) -> None:
        features = self.img_enc(self.images)
        self.assertEqual(features.shape, (self.B, self.proj_dim))

    def test_box_encoder_shape(self) -> None:
        features = self.box_enc(self.boxes, self.confs, self.mask)
        self.assertEqual(features.shape, (self.B, self.proj_dim))

    def test_consistency_loss_shape(self) -> None:
        img_feat = self.img_enc(self.images)
        box_feat = self.box_enc(self.boxes, self.confs, self.mask)
        loss = consistency_loss(img_feat, box_feat)
        self.assertEqual(loss.ndim, 0)

    def test_consistency_loss_perfect_alignment(self) -> None:
        feat = torch.randn(self.B, self.proj_dim)
        loss = consistency_loss(feat, feat)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_consistency_loss_range(self) -> None:
        img_feat = torch.randn(self.B, self.proj_dim)
        box_feat = torch.randn(self.B, self.proj_dim)
        loss = consistency_loss(img_feat, box_feat)
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertLessEqual(loss.item(), 2.0)

    def test_cosine_similarity_shape(self) -> None:
        img_feat = torch.randn(self.B, self.proj_dim)
        box_feat = torch.randn(self.B, self.proj_dim)
        sim = cosine_similarity(img_feat, box_feat)
        self.assertEqual(sim.ndim, 0)
        self.assertGreaterEqual(sim.item(), -1.0)
        self.assertLessEqual(sim.item(), 1.0)

    def test_compute_consistency_loss_keys(self) -> None:
        result = compute_consistency_loss(
            generated_images=self.images,
            predicted_boxes=self.boxes,
            confidences=self.confs,
            valid_mask=self.mask,
            image_encoder=self.img_enc,
            box_encoder=self.box_enc,
            consistency_weight=0.5,
        )
        expected_keys = {"consistency_loss", "cosine_similarity"}
        self.assertEqual(set(result.keys()), expected_keys)


class TestLossAggregator(unittest.TestCase):
    """Tests for the central LossAggregator."""

    def setUp(self) -> None:
        self.cfg = _make_minimal_config()
        self.aggregator = LossAggregator(self.cfg)

    def test_initialisation(self) -> None:
        self.assertIsInstance(self.aggregator, LossAggregator)
        self.assertEqual(self.aggregator.adv_weight, 1.0)
        self.assertEqual(self.aggregator.box_weight, 5.0)
        self.assertEqual(self.aggregator.gp_weight, 10.0)

    def test_negative_weights_raises(self) -> None:
        """Negative weights in config should raise ValueError.

        Note: Config validates all known weights before LossAggregator
        sees them.  We test that a negative weight is caught either by
        Config validation or by LossAggregator's own validation.
        """
        bad_cfg_dict = _make_minimal_config().to_dict()
        bad_cfg_dict["loss"]["entropy_weight"] = -0.1
        with self.assertRaises((ValueError, TypeError)):
            # Config may raise ValueError (constraint check) or
            # LossAggregator may raise ValueError (own validation)
            bad_cfg = Config.from_dict(bad_cfg_dict)
            LossAggregator(bad_cfg)

    def test_repr(self) -> None:
        r = repr(self.aggregator)
        self.assertIn("LossAggregator", r)
        self.assertIn("adv_weight=1.0", r)


if __name__ == "__main__":
    unittest.main()
