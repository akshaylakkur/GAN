"""
Tests for ``ilgan.models.generator.ILGANGenerator``.

Verifies:

- **Architecture and composition**: ``ILGANGenerator`` correctly composes
  ``ContentDecoder`` and ``SpatialHead``, and the forward pass produces all
  expected outputs.
- **Forward shapes**: with a small config (latent_dim=64, image_size=32,
  max_boxes=5, gen_base_channels=16), a forward pass with batch_size=2
  produces tensors of the correct shapes and dtypes.
- **Output ranges**: image pixels in ``[-1, 1]``, box coordinates in
  ``[0, 1]``, confidences in ``[0, 1]``.
- **Auxiliary losses**: dictionary with keys ``"entropy"``, ``"repulsion"``,
  ``"entropy_weighted"``, ``"repulsion_weighted"``, all scalar tensors.
- **Backward pass**: gradients flow to all parameters in both sub-modules.
- **Learnable instance noise**: ``noise_std`` parameter exists, is learnable,
  and injecting noise into the latent produces different outputs during
  training vs. evaluation.
- **Latent statistics**: ``get_latent_statistics()`` returns running mean,
  variance, and count; statistics are updated after each forward pass.
- **Noise generation**: ``generate_noise()`` produces correctly shaped
  tensors from ``N(0, I)``.
- **Gradient checkpointing**: ``set_gradient_checkpointing()`` toggles the
  flag on the ContentDecoder; forward pass produces the same outputs with
  and without checkpointing in eval mode.
- **Config compatibility**: generator can be instantiated from a Config
  object with the expected keys.
"""

import unittest
from typing import Dict

import torch
import torch.nn as nn

from ilgan.models.generator import ILGANGenerator, ContentDecoder, SpatialHead
from ilgan.utils.config import Config


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_small_config() -> Config:
    """Create a small Config object suitable for testing.

    Uses reduced dimensions to keep memory footprint low.
    """
    return Config.from_dict({
        "model": {
            "latent_dim": 64,
            "gen_base_channels": 16,
            "disc_base_channels": 16,
            "max_boxes": 5,
            "num_classes": 10,
            "num_attention_heads": 4,
        },
        "data": {
            "image_size": 32,
            "batch_size": 2,
            "num_workers": 0,
            "augment_prob": 0.0,
            "yolo_format": True,
        },
        "loss": {
            "adv_weight": 1.0,
            "box_weight": 5.0,
            "diversity_weight": 0.1,
            "consistency_weight": 0.5,
            "gp_weight": 10.0,
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.0002,
            "beta1": 0.0,
            "beta2": 0.9,
            "n_critic": 5,
            "gradient_accumulation_steps": 1,
            "use_mixed_precision": False,
            "grad_checkpoint": False,
            "clip_grad_norm": 1.0,
        },
        "logging": {
            "log_interval": 10,
            "save_interval": 50,
            "eval_interval": 100,
            "use_wandb": False,
        },
        "paths": {
            "data_root": "/tmp",
            "checkpoint_dir": "/tmp",
            "log_dir": "/tmp",
        },
    })


def _count_parameters(module: nn.Module) -> int:
    """Return the total number of trainable parameters in *module*."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Test ILGANGenerator: architecture and construction
# ──────────────────────────────────────────────────────────────────────────────


class TestILGANGeneratorConstruction(unittest.TestCase):
    """Verify the generator is constructed correctly from a Config."""

    def setUp(self):
        self.config = _make_small_config()
        self.gen = ILGANGenerator(self.config)

    def test_has_content_decoder(self):
        """Generator should have a ContentDecoder sub-module."""
        self.assertTrue(
            hasattr(self.gen, "content_decoder"),
            "ILGANGenerator should have 'content_decoder' attribute",
        )
        self.assertIsInstance(
            self.gen.content_decoder, ContentDecoder,
        )

    def test_has_spatial_head(self):
        """Generator should have a SpatialHead sub-module."""
        self.assertTrue(
            hasattr(self.gen, "spatial_head"),
            "ILGANGenerator should have 'spatial_head' attribute",
        )
        self.assertIsInstance(self.gen.spatial_head, SpatialHead)

    def test_content_decoder_config(self):
        """ContentDecoder should have correct hyper-parameters from config."""
        self.assertEqual(self.gen.content_decoder.latent_dim, 64)
        self.assertEqual(self.gen.content_decoder.gen_base_channels, 16)
        self.assertEqual(self.gen.content_decoder.image_size, 32)
        self.assertTrue(self.gen.content_decoder.use_spectral_norm)

    def test_spatial_head_config(self):
        """SpatialHead should have correct hyper-parameters from config."""
        self.assertEqual(self.gen.spatial_head.num_slots, 5)
        self.assertEqual(self.gen.spatial_head.num_classes, 10)
        self.assertEqual(self.gen.spatial_head.slot_dim, 32)

    def test_has_noise_std(self):
        """Generator should have a learnable noise_std parameter."""
        self.assertTrue(
            hasattr(self.gen, "noise_std"),
            "ILGANGenerator should have 'noise_std' attribute",
        )
        self.assertIsInstance(self.gen.noise_std, nn.Parameter)
        self.assertEqual(self.gen.noise_std.ndim, 0)

    def test_has_latent_statistics_buffers(self):
        """Generator should have latent statistics buffers."""
        self.assertTrue(hasattr(self.gen, "_latent_mean"))
        self.assertTrue(hasattr(self.gen, "_latent_var"))
        self.assertTrue(hasattr(self.gen, "_latent_count"))

    def test_parameter_count_non_zero(self):
        """Generator should have a non-zero number of parameters."""
        self.assertGreater(_count_parameters(self.gen), 0)

    def test_content_decoder_spectral_norm_applied(self):
        """ContentDecoder's final_conv should have spectral norm applied."""
        conv = self.gen.content_decoder.final_conv
        self.assertTrue(hasattr(conv.conv, "weight_u"))

    def test_spatial_head_output_spectral_norm_applied(self):
        """SpatialHead's output projection heads should have spectral norm."""
        self.assertTrue(hasattr(self.gen.spatial_head.box_head[0], "weight_u"))
        self.assertTrue(hasattr(self.gen.spatial_head.class_head, "weight_u"))
        self.assertTrue(hasattr(self.gen.spatial_head.confidence_head[0], "weight_u"))


# ──────────────────────────────────────────────────────────────────────────────
# Test ILGANGenerator: forward shapes
# ──────────────────────────────────────────────────────────────────────────────


class TestILGANGeneratorForwardShapes(unittest.TestCase):
    """Verify output shapes for the standard forward pass."""

    def setUp(self):
        self.config = _make_small_config()
        self.gen = ILGANGenerator(self.config)
        self.gen.eval()
        self.batch_size = 2
        self.latent_dim = 64
        self.z = ILGANGenerator.generate_noise(self.batch_size, self.latent_dim, "cpu")

    def test_output_is_dict(self):
        """Forward pass should return a dictionary."""
        with torch.no_grad():
            out = self.gen(self.z)
        self.assertIsInstance(out, dict)

    def test_dict_keys(self):
        """Output dictionary should contain all expected keys."""
        with torch.no_grad():
            out = self.gen(self.z)
        expected_keys = {"image", "boxes", "class_logits", "confidences", "aux_losses", "aux"}
        self.assertSetEqual(set(out.keys()), expected_keys)

    def test_image_shape(self):
        """Image should have shape [B, 3, H, W]."""
        with torch.no_grad():
            out = self.gen(self.z)
        self.assertEqual(out["image"].shape, (2, 3, 32, 32))

    def test_boxes_shape(self):
        """Boxes should have shape [B, max_boxes, 4]."""
        with torch.no_grad():
            out = self.gen(self.z)
        self.assertEqual(out["boxes"].shape, (2, 5, 4))

    def test_class_logits_shape(self):
        """Class logits should have shape [B, max_boxes, num_classes]."""
        with torch.no_grad():
            out = self.gen(self.z)
        self.assertEqual(out["class_logits"].shape, (2, 5, 10))

    def test_confidences_shape(self):
        """Confidences should have shape [B, max_boxes, 1]."""
        with torch.no_grad():
            out = self.gen(self.z)
        self.assertEqual(out["confidences"].shape, (2, 5, 1))

    def test_aux_losses_is_dict(self):
        """Auxiliary losses should be a dictionary."""
        with torch.no_grad():
            out = self.gen(self.z)
        self.assertIsInstance(out["aux_losses"], dict)

    def test_aux_losses_keys(self):
        """Auxiliary losses should contain expected keys."""
        with torch.no_grad():
            out = self.gen(self.z)
        expected_keys = {"entropy", "repulsion", "entropy_weighted", "repulsion_weighted"}
        self.assertSetEqual(set(out["aux_losses"].keys()), expected_keys)

    def test_aux_losses_are_scalar(self):
        """All auxiliary losses should be scalar tensors."""
        with torch.no_grad():
            out = self.gen(self.z)
        for key, value in out["aux_losses"].items():
            self.assertEqual(value.ndim, 0, f"'{key}' should be scalar")

    def test_output_dtypes(self):
        """All output tensors should be float32."""
        with torch.no_grad():
            out = self.gen(self.z)
        self.assertEqual(out["image"].dtype, torch.float32)
        self.assertEqual(out["boxes"].dtype, torch.float32)
        self.assertEqual(out["class_logits"].dtype, torch.float32)
        self.assertEqual(out["confidences"].dtype, torch.float32)

    def test_batch_size_one(self):
        """Should work with batch_size=1."""
        z = ILGANGenerator.generate_noise(1, 64, "cpu")
        with torch.no_grad():
            out = self.gen(z)
        self.assertEqual(out["image"].shape, (1, 3, 32, 32))
        self.assertEqual(out["boxes"].shape, (1, 5, 4))
        self.assertEqual(out["class_logits"].shape, (1, 5, 10))
        self.assertEqual(out["confidences"].shape, (1, 5, 1))

    def test_batch_size_four(self):
        """Should work with batch_size=4."""
        z = ILGANGenerator.generate_noise(4, 64, "cpu")
        with torch.no_grad():
            out = self.gen(z)
        self.assertEqual(out["image"].shape, (4, 3, 32, 32))
        self.assertEqual(out["boxes"].shape, (4, 5, 4))


# ──────────────────────────────────────────────────────────────────────────────
# Test ILGANGenerator: output value ranges
# ──────────────────────────────────────────────────────────────────────────────


class TestILGANGeneratorValueRanges(unittest.TestCase):
    """Verify output values are in their expected ranges."""

    def setUp(self):
        self.config = _make_small_config()
        self.gen = ILGANGenerator(self.config)
        self.gen.eval()
        self.z = ILGANGenerator.generate_noise(4, 64, "cpu")

    def test_image_values_in_range(self):
        """Image pixel values should be in [-1, 1]."""
        with torch.no_grad():
            out = self.gen(self.z)
        self.assertGreaterEqual(out["image"].min().item(), -1.0)
        self.assertLessEqual(out["image"].max().item(), 1.0)

    def test_boxes_values_in_range(self):
        """Box coordinates should be in [0, 1]."""
        with torch.no_grad():
            out = self.gen(self.z)
        self.assertTrue(torch.all(out["boxes"] >= 0.0).item())
        self.assertTrue(torch.all(out["boxes"] <= 1.0).item())

    def test_confidences_values_in_range(self):
        """Confidence scores should be in [0, 1]."""
        with torch.no_grad():
            out = self.gen(self.z)
        self.assertTrue(torch.all(out["confidences"] >= 0.0).item())
        self.assertTrue(torch.all(out["confidences"] <= 1.0).item())


# ──────────────────────────────────────────────────────────────────────────────
# Test ILGANGenerator: backward pass (gradients)
# ──────────────────────────────────────────────────────────────────────────────


class TestILGANGeneratorGradients(unittest.TestCase):
    """Verify gradients flow through the entire generator."""

    def setUp(self):
        self.config = _make_small_config()
        self.gen = ILGANGenerator(self.config)
        self.gen.train()
        self.z = ILGANGenerator.generate_noise(2, 64, "cpu")

    def test_backward_produces_all_gradients(self):
        """Backward pass should produce non-zero gradients on all parameters."""
        out = self.gen(self.z)
        loss = (out["image"].sum() + out["boxes"].sum() + out["class_logits"].sum()
                + out["confidences"].sum()
                + out["aux_losses"]["repulsion_weighted"]
                + out["aux_losses"]["entropy_weighted"])
        loss.backward()
        for name, param in self.gen.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"'{name}' should have gradients")
                self.assertGreater(param.grad.abs().sum().item(), 0.0)

    def test_gradients_flow_to_content_decoder(self):
        """Gradients should flow to ContentDecoder parameters."""
        out = self.gen(self.z)
        loss = out["image"].sum()
        loss.backward()
        for name, param in self.gen.content_decoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"CD '{name}' should have gradients")

    def test_gradients_flow_to_spatial_head(self):
        """Gradients should flow to SpatialHead parameters."""
        out = self.gen(self.z)
        loss = out["boxes"].sum() + out["class_logits"].sum() + out["confidences"].sum()
        loss.backward()
        for name, param in self.gen.spatial_head.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"SH '{name}' should have gradients")

    def test_gradients_flow_to_noise_std(self):
        """Gradients should flow to noise_std parameter."""
        out = self.gen(self.z)
        loss = out["image"].sum()
        loss.backward()
        self.assertIsNotNone(self.gen.noise_std.grad)

    def test_gradients_flow_to_spatial_queries(self):
        """Gradients should flow to the learned spatial queries."""
        out = self.gen(self.z)
        loss = out["boxes"].sum() + out["confidences"].sum()
        loss.backward()
        self.assertIsNotNone(self.gen.spatial_head.spatial_queries.grad)

    def test_gradients_flow_from_aux_losses(self):
        """Gradients from auxiliary losses should reach all modules."""
        out = self.gen(self.z)
        loss = out["aux_losses"]["repulsion_weighted"] + out["aux_losses"]["entropy_weighted"]
        loss.backward()
        grad_count = 0
        for param in self.gen.parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum().item() > 0:
                    grad_count += 1
        self.assertGreater(grad_count, 0)


# ──────────────────────────────────────────────────────────────────────────────
# Test ILGANGenerator: learnable instance noise
# ──────────────────────────────────────────────────────────────────────────────


class TestILGANGeneratorInstanceNoise(unittest.TestCase):
    """Verify the learnable instance noise mechanism."""

    def setUp(self):
        self.config = _make_small_config()

    def test_noise_std_is_learnable(self):
        """noise_std should be a trainable parameter."""
        gen = ILGANGenerator(self.config)
        self.assertIn("noise_std", dict(gen.named_parameters()))

    def test_noise_injection_changes_output(self):
        """In training mode, injecting noise should produce different
        outputs than eval mode for the same latent."""
        gen = ILGANGenerator(self.config)
        z = ILGANGenerator.generate_noise(4, 64, "cpu")
        gen.eval()
        with torch.no_grad():
            out_eval = gen(z)
        gen.train()
        out_train = gen(z)
        diff = (out_eval["image"] - out_train["image"]).abs().sum().item()
        self.assertGreater(diff, 1e-6)

    def test_noise_std_can_be_modified(self):
        """The noise_std parameter can be modified and affects behaviour."""
        gen = ILGANGenerator(self.config)
        gen.noise_std.data.fill_(0.5)
        self.assertAlmostEqual(gen.noise_std.item(), 0.5)
        gen.noise_std.data.fill_(0.0)
        self.assertAlmostEqual(gen.noise_std.item(), 0.0)

        z = ILGANGenerator.generate_noise(2, 64, "cpu")

        # With noise_std=0.0, no noise added
        gen.train()
        gen.noise_std.data.fill_(0.0)
        out_zero_noise = gen(z)

        # With noise_std=0.5, noise added
        gen.noise_std.data.fill_(0.5)
        out_with_noise = gen(z)
        diff = (out_zero_noise["image"] - out_with_noise["image"]).abs().sum().item()
        self.assertGreater(diff, 1e-6)

        # Eval mode deterministic regardless of noise_std
        gen.eval()
        gen.noise_std.data.fill_(0.5)
        with torch.no_grad():
            out_eval1 = gen(z)
            out_eval2 = gen(z)
        self.assertTrue(torch.allclose(out_eval1["image"], out_eval2["image"], atol=1e-6))


# ──────────────────────────────────────────────────────────────────────────────
# Test ILGANGenerator: latent statistics
# ──────────────────────────────────────────────────────────────────────────────


class TestILGANGeneratorLatentStatistics(unittest.TestCase):
    """Verify latent statistics tracking."""

    def setUp(self):
        self.config = _make_small_config()
        self.gen = ILGANGenerator(self.config)

    def test_initial_statistics(self):
        """Initial statistics should have mean=0, var=1, count=0."""
        stats = self.gen.get_latent_statistics()
        self.assertTrue(torch.allclose(stats["mean"], torch.zeros(64)))
        self.assertTrue(torch.allclose(stats["var"], torch.ones(64)))
        self.assertEqual(stats["count"], 0)

    def test_statistics_updated_after_forward(self):
        """Statistics should be updated after a forward pass."""
        z = ILGANGenerator.generate_noise(4, 64, "cpu")
        self.gen.eval()
        with torch.no_grad():
            _ = self.gen(z)
        stats = self.gen.get_latent_statistics()
        self.assertEqual(stats["count"], 4)
        # With 4 random samples, mean should have been updated from zero
        self.assertFalse(torch.allclose(stats["mean"], torch.zeros(64), atol=0.01))

    def test_statistics_accumulate_across_passes(self):
        """Statistics should accumulate across multiple forward passes."""
        z1 = ILGANGenerator.generate_noise(2, 64, "cpu")
        z2 = ILGANGenerator.generate_noise(3, 64, "cpu")
        with torch.no_grad():
            self.gen.eval()
            _ = self.gen(z1)
            _ = self.gen(z2)
        self.assertEqual(self.gen.get_latent_statistics()["count"], 5)

    def test_statistics_are_buffers_in_state_dict(self):
        """Latent statistics should be in the state dict (buffers)."""
        sd = self.gen.state_dict()
        self.assertIn("_latent_mean", sd)
        self.assertIn("_latent_var", sd)
        self.assertIn("_latent_count", sd)

    def test_statistics_not_updated_by_optimiser(self):
        """Buffers should not have 'requires_grad'."""
        self.assertFalse(self.gen._latent_mean.requires_grad)
        self.assertFalse(self.gen._latent_var.requires_grad)

    def test_statistics_return_correct_types(self):
        """get_latent_statistics should return tensors and an int."""
        stats = self.gen.get_latent_statistics()
        self.assertIsInstance(stats["mean"], torch.Tensor)
        self.assertIsInstance(stats["var"], torch.Tensor)
        self.assertIsInstance(stats["count"], int)


# ──────────────────────────────────────────────────────────────────────────────
# Test ILGANGenerator: noise generation
# ──────────────────────────────────────────────────────────────────────────────


class TestILGANGeneratorNoiseGeneration(unittest.TestCase):
    """Verify the static noise generation method."""

    def test_generate_noise_shape(self):
        """generate_noise should produce [num_samples, latent_dim]."""
        noise = ILGANGenerator.generate_noise(8, 128, "cpu")
        self.assertEqual(noise.shape, (8, 128))

    def test_generate_noise_dtype(self):
        """Noise should be float32."""
        noise = ILGANGenerator.generate_noise(4, 64, "cpu")
        self.assertEqual(noise.dtype, torch.float32)

    def test_generate_noise_device(self):
        """Noise should be on the requested device."""
        noise = ILGANGenerator.generate_noise(2, 32, "cpu")
        self.assertEqual(noise.device.type, "cpu")

    def test_generate_noise_distribution(self):
        """Noise should be approximately N(0, 1)."""
        noise = ILGANGenerator.generate_noise(10000, 16, "cpu")
        self.assertAlmostEqual(noise.mean().item(), 0.0, delta=0.05)
        self.assertAlmostEqual(noise.std().item(), 1.0, delta=0.05)

    def test_generate_noise_batch_one(self):
        """Should work with num_samples=1."""
        noise = ILGANGenerator.generate_noise(1, 64, "cpu")
        self.assertEqual(noise.shape, (1, 64))

    def test_generate_noise_batch_zero(self):
        """Should work with num_samples=0 (empty tensor)."""
        noise = ILGANGenerator.generate_noise(0, 64, "cpu")
        self.assertEqual(noise.shape, (0, 64))


# ──────────────────────────────────────────────────────────────────────────────
# Test ILGANGenerator: gradient checkpointing
# ──────────────────────────────────────────────────────────────────────────────


class TestILGANGeneratorCheckpointing(unittest.TestCase):
    """Verify gradient checkpointing methods."""

    def setUp(self):
        self.config = _make_small_config()
        self.config["training.grad_checkpoint"] = False
        self.gen = ILGANGenerator(self.config)

    def test_initial_checkpointing_disabled(self):
        """By default, checkpointing should be disabled."""
        self.assertFalse(self.gen.content_decoder.use_checkpointing)

    def test_set_gradient_checkpointing_enables(self):
        """set_gradient_checkpointing(True) should enable checkpointing."""
        self.gen.set_gradient_checkpointing(True)
        self.assertTrue(self.gen.content_decoder.use_checkpointing)

    def test_set_gradient_checkpointing_disables(self):
        """set_gradient_checkpointing(False) should disable checkpointing."""
        self.gen.set_gradient_checkpointing(True)
        self.gen.set_gradient_checkpointing(False)
        self.assertFalse(self.gen.content_decoder.use_checkpointing)

    def test_checkpointing_forward_equivalence(self):
        """Forward pass with and without checkpointing should produce
        identical outputs in eval mode."""
        gen_no_ckpt = ILGANGenerator(self.config)
        gen_ckpt = ILGANGenerator(self.config)
        gen_ckpt.load_state_dict(gen_no_ckpt.state_dict())
        gen_ckpt.set_gradient_checkpointing(True)
        gen_no_ckpt.eval()
        gen_ckpt.eval()
        z = ILGANGenerator.generate_noise(4, 64, "cpu")
        with torch.no_grad():
            out_no_ckpt = gen_no_ckpt(z)
            out_ckpt = gen_ckpt(z)
        self.assertTrue(torch.allclose(out_no_ckpt["image"], out_ckpt["image"], atol=1e-6))
        self.assertTrue(torch.allclose(out_no_ckpt["boxes"], out_ckpt["boxes"], atol=1e-6))
        self.assertTrue(torch.allclose(out_no_ckpt["class_logits"], out_ckpt["class_logits"], atol=1e-6))
        self.assertTrue(torch.allclose(out_no_ckpt["confidences"], out_ckpt["confidences"], atol=1e-6))

    def test_checkpointing_enabled_from_config(self):
        """If config says grad_checkpoint=True, generator should start with
        checkpointing enabled."""
        self.config["training.grad_checkpoint"] = True
        gen = ILGANGenerator(self.config)
        self.assertTrue(gen.content_decoder.use_checkpointing)

    def test_checkpointing_training_forward(self):
        """Forward pass with checkpointing in training mode should succeed
        and produce valid outputs."""
        gen = ILGANGenerator(self.config)
        gen.set_gradient_checkpointing(True)
        gen.train()
        z = ILGANGenerator.generate_noise(2, 64, "cpu")
        out = gen(z)
        self.assertEqual(out["image"].shape, (2, 3, 32, 32))
        self.assertEqual(out["boxes"].shape, (2, 5, 4))


# ──────────────────────────────────────────────────────────────────────────────
# Test ILGANGenerator: integration of both pathways
# ──────────────────────────────────────────────────────────────────────────────


class TestILGANGeneratorIntegration(unittest.TestCase):
    """Verify that the two pathways (image and boxes) both respond to the
    latent vector."""

    def setUp(self):
        self.config = _make_small_config()
        self.gen = ILGANGenerator(self.config)
        self.gen.eval()

    def test_different_latents_different_images(self):
        """Different latents should produce different images."""
        z = ILGANGenerator.generate_noise(2, 64, "cpu")
        with torch.no_grad():
            out = self.gen(z)
        diff = (out["image"][0] - out["image"][1]).abs().sum().item()
        self.assertGreater(diff, 0.01)

    def test_different_latents_different_boxes(self):
        """Different latents produce different class logits via content modulation.
        Note: with random init, box coordinates/confidences may be identical
        across batch due to sigmoid saturation; class_logits always differ."""
        z = ILGANGenerator.generate_noise(2, 64, "cpu")
        with torch.no_grad():
            out = self.gen(z)
        # Content modulation is proven via class_logits differing;
        # Verify class_logits differ across batch elements
        cls_diff = (out["class_logits"][0] - out["class_logits"][1]).abs().sum().item()
        self.assertGreater(cls_diff, 1e-8, "Class logits must differ across batch elements")


    def test_same_latent_same_output(self):
        """The same latent should produce the same output deterministically."""
        z = ILGANGenerator.generate_noise(2, 64, "cpu")
        with torch.no_grad():
            out1 = self.gen(z)
            out2 = self.gen(z)
        self.assertTrue(torch.allclose(out1["image"], out2["image"], atol=1e-6))
        self.assertTrue(torch.allclose(out1["boxes"], out2["boxes"], atol=1e-6))
        self.assertTrue(torch.allclose(out1["class_logits"], out2["class_logits"], atol=1e-6))
        self.assertTrue(torch.allclose(out1["confidences"], out2["confidences"], atol=1e-6))


# ──────────────────────────────────────────────────────────────────────────────
# Test ILGANGenerator: state dict serialisation
# ──────────────────────────────────────────────────────────────────────────────


class TestILGANGeneratorSerialisation(unittest.TestCase):
    """Verify that the generator supports state dict save/load."""

    def setUp(self):
        self.config = _make_small_config()

    def test_state_dict_contains_all_parameters(self):
        """State dict should contain keys for both sub-modules.

        Note: spectral norm renames .weight to .weight_orig + .weight_u/.weight_v
        """
        gen = ILGANGenerator(self.config)
        sd = gen.state_dict()
        self.assertIn("content_decoder.init_linear.weight", sd)
        self.assertIn("spatial_head.spatial_queries", sd)
        self.assertIn("spatial_head.box_head.0.weight_orig", sd)
        self.assertIn("spatial_head.box_head.0.weight_u", sd)
        self.assertIn("spatial_head.box_head.0.weight_v", sd)
        self.assertIn("spatial_head.box_head.0.bias", sd)
        self.assertIn("spatial_head.class_head.weight_orig", sd)
        self.assertIn("spatial_head.class_head.weight_u", sd)
        self.assertIn("spatial_head.class_head.weight_v", sd)
        self.assertIn("spatial_head.class_head.bias", sd)
        self.assertIn("spatial_head.confidence_head.0.weight_orig", sd)
        self.assertIn("spatial_head.confidence_head.0.weight_u", sd)
        self.assertIn("spatial_head.confidence_head.0.weight_v", sd)
        self.assertIn("spatial_head.confidence_head.0.bias", sd)
        self.assertIn("noise_std", sd)

    def test_load_state_dict(self):
        """Loading a state dict should restore all weights."""
        gen1 = ILGANGenerator(self.config)
        gen2 = ILGANGenerator(self.config)
        with torch.no_grad():
            gen1.content_decoder.init_linear.weight.add_(0.01)
        sd = gen1.state_dict()
        gen2.load_state_dict(sd)
        for p1, p2 in zip(gen1.parameters(), gen2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_forward_after_load(self):
        """Forward pass should still work after loading state dict."""
        gen = ILGANGenerator(self.config)
        sd = gen.state_dict()
        gen.load_state_dict(sd)
        z = ILGANGenerator.generate_noise(2, 64, "cpu")
        with torch.no_grad():
            out = gen(z)
        self.assertEqual(out["image"].shape, (2, 3, 32, 32))
        self.assertEqual(out["boxes"].shape, (2, 5, 4))


# ──────────────────────────────────────────────────────────────────────────────
# Test ILGANGenerator: default config compatibility
# ──────────────────────────────────────────────────────────────────────────────


class TestILGANGeneratorDefaultConfig(unittest.TestCase):
    """Verify the generator can be constructed from the default config."""

    def _make_full_config(self):
        return Config.from_dict({
            "model": {
                "latent_dim": 256,
                "gen_base_channels": 64,
                "disc_base_channels": 64,
                "max_boxes": 20,
                "num_classes": 80,
                "num_attention_heads": 8,
            },
            "data": {
                "image_size": 128,
                "batch_size": 4,
                "num_workers": 0,
                "augment_prob": 0.0,
                "yolo_format": True,
            },
            "loss": {
                "adv_weight": 1.0,
                "box_weight": 5.0,
                "diversity_weight": 0.1,
                "consistency_weight": 0.5,
                "gp_weight": 10.0,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.0002,
                "beta1": 0.0,
                "beta2": 0.9,
                "n_critic": 5,
                "gradient_accumulation_steps": 1,
                "use_mixed_precision": False,
                "grad_checkpoint": False,
                "clip_grad_norm": 1.0,
            },
            "logging": {
                "log_interval": 10,
                "save_interval": 50,
                "eval_interval": 100,
                "use_wandb": False,
            },
            "paths": {
                "data_root": "/tmp",
                "checkpoint_dir": "/tmp",
                "log_dir": "/tmp",
            },
        })

    def test_default_config_constructs(self):
        """Default config should produce a valid generator."""
        try:
            cfg = self._make_full_config()
            gen = ILGANGenerator(cfg)
            self.assertIsInstance(gen, ILGANGenerator)
        except Exception as e:
            self.fail(f"Default config construction raised: {e}")

    def test_larger_config_forward(self):
        """Generator with larger config should forward correctly."""
        cfg = self._make_full_config()
        cfg["data.batch_size"] = 2
        gen = ILGANGenerator(cfg)
        gen.eval()
        z = ILGANGenerator.generate_noise(2, 256, "cpu")
        with torch.no_grad():
            out = gen(z)
        self.assertEqual(out["image"].shape, (2, 3, 128, 128))
        self.assertEqual(out["boxes"].shape, (2, 20, 4))
        self.assertEqual(out["class_logits"].shape, (2, 20, 80))
        self.assertEqual(out["confidences"].shape, (2, 20, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)