"""
Tests for ``ilgan.models.discriminator.ImageDiscriminator``.

Verifies:

- **Architecture and construction**: ``ImageDiscriminator`` is constructed
  correctly with various configurations (different base channels, image sizes,
  norm types, minibatch stddev toggle).
- **Forward shapes**: with ``disc_base_channels=32`` and ``image_size=64``,
  a forward pass with batch_size=4 produces:
  - ``local_scores`` of shape ``[B, 1, 4, 4]`` (since 64 / 2^4 = 4).
  - ``global_score`` of shape ``[B, 1]`` (scalar per sample).
- **Score ranges**: both local and global scores are real-valued (no
  activation constraint beyond the model's design).
- **Differentiable**: gradients flow to all parameters.
- **Minibatch discrimination**: when enabled, the final feature map before
  the score heads has an extra channel; when disabled, it does not.
- **FrozenBatchNorm2d**: statistics are never updated during training.
- **DownBlock**: correct output shapes and channel progression.
- **Norm types**: all supported norm types produce valid forward passes.
- **Edge cases**: single batch, multiple batches, different image sizes.
"""

import unittest

import torch
import torch.nn as nn

from ilgan.models.discriminator import (
    FrozenBatchNorm2d,
    DownBlock,
    minibatch_stddev,
    ImageDiscriminator,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_random_image(
    B: int = 4,
    C: int = 3,
    H: int = 64,
    W: int = 64,
) -> torch.Tensor:
    """Create a random image tensor in ``[-1, 1]``."""
    return torch.rand(B, C, H, W) * 2.0 - 1.0


def _count_parameters(module: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Test FrozenBatchNorm2d
# ──────────────────────────────────────────────────────────────────────────────


class TestFrozenBatchNorm2d(unittest.TestCase):
    """Verify FrozenBatchNorm2d behaviour."""

    def setUp(self):
        self.num_features = 8
        self.bn = FrozenBatchNorm2d(self.num_features)

    def test_initial_statistics(self):
        """Initial running_mean should be zero, running_var should be one."""
        self.assertTrue(
            torch.allclose(self.bn.running_mean, torch.zeros(self.num_features)),
        )
        self.assertTrue(
            torch.allclose(self.bn.running_var, torch.ones(self.num_features)),
        )

    def test_forward_shape(self):
        """Forward pass should preserve shape."""
        x = torch.randn(4, self.num_features, 16, 16)
        y = self.bn(x)
        self.assertEqual(y.shape, x.shape)

    def test_statistics_not_updated_during_training(self):
        """Running statistics should NOT be updated during training."""
        self.bn.train()
        x_before = self.bn.running_mean.clone()
        y_before = self.bn.running_var.clone()

        for _ in range(5):
            x = torch.randn(4, self.num_features, 8, 8)
            _ = self.bn(x)

        self.assertTrue(
            torch.allclose(self.bn.running_mean, x_before),
            "running_mean should not change during training",
        )
        self.assertTrue(
            torch.allclose(self.bn.running_var, y_before),
            "running_var should not change during training",
        )

    def test_statistics_not_updated_during_eval(self):
        """Running statistics should not change during eval either."""
        self.bn.eval()
        x_before = self.bn.running_mean.clone()
        for _ in range(3):
            x = torch.randn(2, self.num_features, 8, 8)
            with torch.no_grad():
                _ = self.bn(x)
        self.assertTrue(
            torch.allclose(self.bn.running_mean, x_before),
        )

    def test_calibrate_sets_statistics(self):
        """``calibrate()`` should set running statistics to match the data."""
        x = torch.randn(8, self.num_features, 4, 4)
        x_mean = x.mean(dim=(0, 2, 3))
        x_var = x.var(dim=(0, 2, 3), unbiased=False)

        self.bn.calibrate(x)
        self.assertTrue(
            torch.allclose(self.bn.running_mean, x_mean, atol=1e-6),
        )
        self.assertTrue(
            torch.allclose(self.bn.running_var, x_var, atol=1e-6),
        )

    def test_affine_parameters_are_trainable(self):
        """Weight and bias should be trainable parameters."""
        self.assertTrue(self.bn.weight.requires_grad)
        self.assertTrue(self.bn.bias.requires_grad)

    def test_affine_parameters_produce_expected_transform(self):
        """Setting weight=2, bias=-1 should transform normalised output."""
        x = torch.randn(2, self.num_features, 4, 4)
        self.bn.weight.data.fill_(2.0)
        self.bn.bias.data.fill_(-1.0)
        y = self.bn(x)

        # Normalise manually with frozen stats
        x_norm = (x - self.bn.running_mean.view(1, -1, 1, 1)) \
                 / torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn.eps)
        y_expected = 2.0 * x_norm - 1.0

        self.assertTrue(torch.allclose(y, y_expected, atol=1e-6))

    def test_gradients_flow(self):
        """Gradients should flow to weight and bias."""
        x = torch.randn(2, self.num_features, 4, 4)
        y = self.bn(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(self.bn.weight.grad)
        self.assertIsNotNone(self.bn.bias.grad)
        self.assertGreater(self.bn.weight.grad.abs().sum().item(), 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Test DownBlock
# ──────────────────────────────────────────────────────────────────────────────


class TestDownBlock(unittest.TestCase):
    """Verify the DownBlock module."""

    def test_forward_shape(self):
        """DownBlock should halve spatial size and double channels."""
        B, C_in, H, W = 2, 16, 64, 64
        C_out = 32
        block = DownBlock(C_in, C_out)
        x = torch.randn(B, C_in, H, W)
        y = block(x)
        expected_shape = (B, C_out, H // 2, W // 2)
        self.assertEqual(y.shape, expected_shape)

    def test_spectral_norm_by_default(self):
        """DownBlock should have spectral norm enabled by default."""
        block = DownBlock(16, 32)
        self.assertTrue(hasattr(block.conv.conv, "weight_u"))

    def test_spectral_norm_disabled(self):
        """DownBlock should not have spectral norm when disabled."""
        block = DownBlock(16, 32, use_spectral_norm=False)
        self.assertFalse(hasattr(block.conv.conv, "weight_u"))

    def test_different_norm_types(self):
        """DownBlock should work with all supported norm types."""
        for norm_type in ("frozen_bn", "instance", "layer", "group"):
            with self.subTest(norm_type=norm_type):
                block = DownBlock(16, 32, norm_type=norm_type)
                x = torch.randn(2, 16, 32, 32)
                y = block(x)
                self.assertEqual(y.shape, (2, 32, 16, 16))

    def test_activation_type(self):
        """DownBlock should use LeakyReLU with negative slope 0.2."""
        block = DownBlock(16, 32)
        self.assertIsInstance(block.activation, nn.LeakyReLU)
        self.assertEqual(block.activation.negative_slope, 0.2)

    def test_gradients_flow(self):
        """Gradients should flow through the block."""
        block = DownBlock(16, 32)
        x = torch.randn(2, 16, 32, 32)
        y = block(x)
        loss = y.sum()
        loss.backward()
        for name, param in block.named_parameters():
            self.assertIsNotNone(
                param.grad,
                f"{name} should have gradients",
            )

    def test_batch_size_one(self):
        """Should work with batch size 1."""
        block = DownBlock(16, 32)
        x = torch.randn(1, 16, 8, 8)
        y = block(x)
        self.assertEqual(y.shape, (1, 32, 4, 4))


# ──────────────────────────────────────────────────────────────────────────────
# Test minibatch_stddev
# ──────────────────────────────────────────────────────────────────────────────


class TestMinibatchStddev(unittest.TestCase):
    """Verify the minibatch standard deviation helper."""

    def test_output_shape(self):
        """Output should have one extra channel for B >= 2."""
        x = torch.randn(4, 16, 8, 8)
        y = minibatch_stddev(x)
        self.assertEqual(y.shape, (4, 17, 8, 8))

    def test_extra_channel_is_constant_across_batch(self):
        """The extra std-dev channel should be identical for all elements
        in the batch."""
        x = torch.randn(8, 8, 4, 4)
        y = minibatch_stddev(x)
        extra_ch = y[:, -1, :, :]  # [B, H, W]
        # All batch elements should have the same std-dev value
        for i in range(1, extra_ch.shape[0]):
            self.assertTrue(
                torch.allclose(extra_ch[0], extra_ch[i], atol=1e-6),
                f"Batch element 0 and {i} differ in std-dev channel",
            )

    def test_identical_inputs_produce_zero_std(self):
        """If all batch elements are identical, std should be ~zero."""
        x = torch.ones(4, 8, 4, 4)
        y = minibatch_stddev(x)
        # std with _EPS added results in sqrt(0 + 1e-8) approx 0.0001
        extra_ch = y[:, -1, :, :]
        self.assertAlmostEqual(extra_ch[0, 0, 0].item(), 0.0, places=3)

    def test_different_inputs_produce_positive_std(self):
        """If batch elements differ, std should be positive."""
        x = torch.randn(4, 8, 4, 4) * 10.0
        y = minibatch_stddev(x)
        extra_ch = y[:, -1, :, :]
        self.assertGreater(extra_ch[0, 0, 0].item(), 0.0)

    def test_preserves_input_channels(self):
        """The original channels should be unchanged."""
        x = torch.randn(4, 16, 8, 8)
        x_copy = x.clone()
        y = minibatch_stddev(x)
        self.assertTrue(
            torch.allclose(y[:, :-1, :, :], x_copy, atol=1e-6),
            "Input channels should be unchanged",
        )

    def test_batch_size_one_adds_channel(self):
        """With a single batch element, std is zero but the extra channel
        is still added to maintain consistent shape."""
        x = torch.randn(1, 8, 4, 4)
        y = minibatch_stddev(x)
        self.assertEqual(y.shape, (1, 9, 4, 4),
                         "With B=1, minibatch_stddev should still add a channel")
        # The extra channel should be zero
        extra_ch = y[:, -1, :, :]
        self.assertAlmostEqual(extra_ch[0, 0, 0].item(), 0.0, places=6)

    def test_differentiable(self):
        """The operation should be differentiable."""
        x = torch.randn(4, 8, 4, 4, requires_grad=True)
        y = minibatch_stddev(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


# ──────────────────────────────────────────────────────────────────────────────
# Test ImageDiscriminator: construction
# ──────────────────────────────────────────────────────────────────────────────


class TestImageDiscriminatorConstruction(unittest.TestCase):
    """Verify ImageDiscriminator can be constructed correctly."""

    def test_default_configuration(self):
        """Default configuration should construct without errors."""
        disc = ImageDiscriminator(
            disc_base_channels=32,
            image_size=64,
        )
        self.assertIsInstance(disc, ImageDiscriminator)

    def test_has_down_blocks(self):
        """Discriminator should have down_blocks as ModuleList."""
        disc = ImageDiscriminator(32, 64)
        self.assertTrue(hasattr(disc, "down_blocks"))
        self.assertIsInstance(disc.down_blocks, nn.ModuleList)

    def test_num_blocks_correct(self):
        """Number of down blocks should be log2(image_size) - 2."""
        for img_size, expected_blocks in [(8, 1), (16, 2), (32, 3), (64, 4), (128, 5)]:
            with self.subTest(image_size=img_size):
                disc = ImageDiscriminator(16, img_size)
                self.assertEqual(
                    disc.num_blocks, expected_blocks,
                    f"Expected {expected_blocks} blocks for image_size={img_size}",
                )

    def test_channel_progression(self):
        """Channels should double each block, capped at max_channels."""
        disc = ImageDiscriminator(
            disc_base_channels=16,
            image_size=128,  # 5 blocks: 16, 32, 64, 128, 256
            max_channels=128,
        )
        expected_channels = [3, 16, 32, 64, 128, 128]  # input + 5 blocks
        for idx, block in enumerate(disc.down_blocks):
            self.assertEqual(
                block.in_channels, expected_channels[idx],
                f"Block {idx} input channels mismatch",
            )
            self.assertEqual(
                block.out_channels, expected_channels[idx + 1],
                f"Block {idx} output channels mismatch",
            )

    def test_channel_capping(self):
        """Max channels should cap the doubling progression."""
        disc = ImageDiscriminator(
            disc_base_channels=32,
            image_size=256,  # 6 blocks: 32, 64, 128, 256, 512, 1024
            max_channels=512,
        )
        last_block = disc.down_blocks[-1]
        self.assertEqual(last_block.out_channels, 512)

    def test_spectral_norm_on_score_conv(self):
        """Score conv layers should have spectral norm by default."""
        disc = ImageDiscriminator(32, 64)
        self.assertTrue(hasattr(disc.score_conv1.conv, "weight_u"))
        self.assertTrue(hasattr(disc.score_conv2.conv, "weight_u"))

    def test_spectral_norm_on_global_head(self):
        """Global head should have spectral norm by default."""
        disc = ImageDiscriminator(32, 64)
        self.assertTrue(hasattr(disc.global_head, "weight_u"))

    def test_spectral_norm_disabled(self):
        """When disabled, no spectral norm should appear."""
        disc = ImageDiscriminator(32, 64, use_spectral_norm=False)
        self.assertFalse(hasattr(disc.score_conv1.conv, "weight_u"))
        self.assertFalse(hasattr(disc.global_head, "weight_u"))

    def test_minibatch_stddev_enabled_by_default(self):
        """Minibatch stddev should be enabled by default."""
        disc = ImageDiscriminator(32, 64)
        self.assertTrue(disc.use_minibatch_stddev)

    def test_minibatch_stddev_disabled(self):
        """Should be able to disable minibatch stddev."""
        disc = ImageDiscriminator(32, 64, use_minibatch_stddev=False)
        self.assertFalse(disc.use_minibatch_stddev)

    def test_grid_size_property(self):
        """grid_size should be image_size / 2^{num_blocks}."""
        for img_size in [8, 16, 32, 64, 128]:
            disc = ImageDiscriminator(16, img_size)
            expected = img_size // (2 ** disc.num_blocks)
            self.assertEqual(disc.grid_size, expected)

    def test_parameter_count_non_zero(self):
        """Discriminator should have a non-zero number of parameters."""
        disc = ImageDiscriminator(32, 64)
        self.assertGreater(_count_parameters(disc), 0)

    def test_invalid_image_size_raises(self):
        """Non-power-of-two or too-small image size should raise."""
        with self.assertRaises(ValueError):
            ImageDiscriminator(16, 7)
        with self.assertRaises(ValueError):
            ImageDiscriminator(16, 9)
        with self.assertRaises(ValueError):
            ImageDiscriminator(16, 10)
        with self.assertRaises(ValueError):
            ImageDiscriminator(16, 1280)

    def test_norm_types_construct(self):
        """All norm types should construct without errors."""
        for norm_type in ("frozen_bn", "instance", "layer", "group"):
            with self.subTest(norm_type=norm_type):
                disc = ImageDiscriminator(32, 64, norm_type=norm_type)
                self.assertIsInstance(disc, ImageDiscriminator)

    def test_unknown_norm_type_raises(self):
        """An unknown norm type should raise ValueError."""
        with self.assertRaises(ValueError):
            ImageDiscriminator(32, 64, norm_type="unknown")


# ──────────────────────────────────────────────────────────────────────────────
# Test ImageDiscriminator: forward shapes
# ──────────────────────────────────────────────────────────────────────────────


class TestImageDiscriminatorForwardShapes(unittest.TestCase):
    """Verify output shapes for the standard forward pass."""

    def setUp(self):
        self.disc = ImageDiscriminator(
            disc_base_channels=32,
            image_size=64,
        )
        self.disc.eval()
        self.B = 4
        self.H = 64
        self.W = 64

    def test_local_scores_shape(self):
        """local_scores should have shape [B, 1, grid_h, grid_w]."""
        x = _make_random_image(self.B, 3, self.H, self.W)
        with torch.no_grad():
            local, global_ = self.disc(x)
        expected_grid = self.disc.grid_size  # 64 / 16 = 4
        self.assertEqual(local.shape, (self.B, 1, expected_grid, expected_grid))

    def test_global_score_shape(self):
        """global_score should have shape [B, 1]."""
        x = _make_random_image(self.B, 3, self.H, self.W)
        with torch.no_grad():
            local, global_ = self.disc(x)
        self.assertEqual(global_.shape, (self.B, 1))

    def test_return_types(self):
        """local_scores and global_score should be torch.Tensor."""
        x = _make_random_image(self.B, 3, self.H, self.W)
        with torch.no_grad():
            local, global_ = self.disc(x)
        self.assertIsInstance(local, torch.Tensor)
        self.assertIsInstance(global_, torch.Tensor)

    def test_output_dtypes(self):
        """All output tensors should be float32."""
        x = _make_random_image(self.B, 3, self.H, self.W)
        with torch.no_grad():
            local, global_ = self.disc(x)
        self.assertEqual(local.dtype, torch.float32)
        self.assertEqual(global_.dtype, torch.float32)

    def test_batch_size_one(self):
        """Should work with batch_size=1."""
        disc = ImageDiscriminator(32, 64)
        disc.eval()
        x = _make_random_image(1, 3, 64, 64)
        with torch.no_grad():
            local, global_ = disc(x)
        expected_grid = disc.grid_size
        self.assertEqual(local.shape, (1, 1, expected_grid, expected_grid))
        self.assertEqual(global_.shape, (1, 1))

    def test_batch_size_eight(self):
        """Should work with batch_size=8."""
        disc = ImageDiscriminator(16, 32)
        disc.eval()
        x = _make_random_image(8, 3, 32, 32)
        with torch.no_grad():
            local, global_ = disc(x)
        expected_grid = disc.grid_size  # 32 / 8 = 4
        self.assertEqual(local.shape, (8, 1, expected_grid, expected_grid))
        self.assertEqual(global_.shape, (8, 1))

    def test_different_image_sizes(self):
        """Should work with various image sizes."""
        for img_size in [8, 16, 32, 64, 128]:
            with self.subTest(image_size=img_size):
                disc = ImageDiscriminator(16, img_size)
                disc.eval()
                x = _make_random_image(2, 3, img_size, img_size)
                with torch.no_grad():
                    local, global_ = disc(x)
                expected_grid = disc.grid_size
                self.assertEqual(
                    local.shape,
                    (2, 1, expected_grid, expected_grid),
                    f"Image size {img_size}",
                )
                self.assertEqual(global_.shape, (2, 1))

    def test_different_base_channels(self):
        """Should work with different disc_base_channels values."""
        for base_ch in [8, 16, 32, 64]:
            with self.subTest(base_channels=base_ch):
                disc = ImageDiscriminator(base_ch, 32)
                disc.eval()
                x = _make_random_image(2, 3, 32, 32)
                with torch.no_grad():
                    local, global_ = disc(x)
                self.assertEqual(local.shape, (2, 1, disc.grid_size, disc.grid_size))
                self.assertEqual(global_.shape, (2, 1))

    def test_batch_size_two_with_minibatch_stddev(self):
        """Minibatch stddev works with batch_size=2."""
        disc = ImageDiscriminator(16, 32, use_minibatch_stddev=True)
        disc.eval()
        x = _make_random_image(2, 3, 32, 32)
        with torch.no_grad():
            local, global_ = disc(x)
        self.assertEqual(local.shape, (2, 1, disc.grid_size, disc.grid_size))
        self.assertEqual(global_.shape, (2, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Test ImageDiscriminator: minibatch stddev effect on feature dimensions
# ──────────────────────────────────────────────────────────────────────────────


class TestImageDiscriminatorMinibatchStddev(unittest.TestCase):
    """Verify the effect of minibatch_stddev on internal features."""

    def test_minibatch_stddev_enabled_adds_channel(self):
        """With minibatch_stddev enabled, the score_conv1 input should have
        one extra channel."""
        disc = ImageDiscriminator(
            32, 64, use_minibatch_stddev=True,
        )
        expected = disc._final_channels + 1
        self.assertEqual(disc.score_conv1.conv.in_channels, expected)

    def test_minibatch_stddev_disabled_no_extra_channel(self):
        """With minibatch_stddev disabled, no extra channel should be added."""
        disc = ImageDiscriminator(
            32, 64, use_minibatch_stddev=False,
        )
        expected = disc._final_channels
        self.assertEqual(disc.score_conv1.conv.in_channels, expected)

    def test_minibatch_stddev_disabled_forward(self):
        """Forward pass should still work with minibatch_stddev disabled."""
        disc = ImageDiscriminator(32, 64, use_minibatch_stddev=False)
        disc.eval()
        x = _make_random_image(4, 3, 64, 64)
        with torch.no_grad():
            local, global_ = disc(x)
        self.assertEqual(local.shape, (4, 1, disc.grid_size, disc.grid_size))
        self.assertEqual(global_.shape, (4, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Test ImageDiscriminator: gradient flow
# ──────────────────────────────────────────────────────────────────────────────


class TestImageDiscriminatorGradients(unittest.TestCase):
    """Verify gradients flow through the discriminator."""

    def setUp(self):
        self.disc = ImageDiscriminator(
            disc_base_channels=32,
            image_size=64,
        )
        self.disc.train()
        self.B = 2
        self.x = _make_random_image(self.B, 3, 64, 64)

    def test_gradients_flow_to_all_parameters(self):
        """Backward pass should produce non-zero gradients on all trainable
        parameters."""
        local, global_ = self.disc(self.x)
        loss = local.sum() + global_.sum()
        loss.backward()
        for name, param in self.disc.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"'{name}' should have gradients",
                )
                self.assertGreater(
                    param.grad.abs().sum().item(), 0.0,
                    f"'{name}' grad should be non-zero",
                )

    def test_gradients_from_local_scores(self):
        """Gradients from local_scores should flow to most parameters."""
        local, _ = self.disc(self.x)
        loss = local.sum()
        loss.backward()
        grad_count = sum(
            1 for p in self.disc.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0
        )
        total_trainable = sum(1 for p in self.disc.parameters() if p.requires_grad)
        self.assertGreater(grad_count, total_trainable // 2,
                           "Most parameters should receive gradients from local_scores")

    def test_gradients_from_global_score(self):
        """Gradients from global_score should flow to most parameters."""
        _, global_ = self.disc(self.x)
        loss = global_.sum()
        loss.backward()
        grad_count = sum(
            1 for p in self.disc.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0
        )
        total_trainable = sum(1 for p in self.disc.parameters() if p.requires_grad)
        self.assertGreater(grad_count, total_trainable // 2,
                           "Most parameters should receive gradients from global_score")

    def test_gradients_flow_to_down_blocks(self):
        """Gradients should flow to all DownBlock parameters."""
        local, global_ = self.disc(self.x)
        loss = local.sum() + global_.sum()
        loss.backward()
        for idx, block in enumerate(self.disc.down_blocks):
            for name, param in block.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(
                        param.grad,
                        f"Block {idx} param '{name}' should have gradients",
                    )

    def test_gradients_flow_to_score_conv(self):
        """Gradients should flow to score convolution parameters."""
        local, global_ = self.disc(self.x)
        loss = local.sum() + global_.sum()
        loss.backward()
        for name, param in self.disc.score_conv1.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"score_conv1 {name} grad")
        for name, param in self.disc.score_conv2.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"score_conv2 {name} grad")

    def test_gradients_flow_to_global_head(self):
        """Gradients should flow to global head parameters."""
        local, global_ = self.disc(self.x)
        loss = local.sum() + global_.sum()
        loss.backward()
        for name, param in self.disc.global_head.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"global_head {name} grad")

    def test_gradients_with_minibatch_stddev_disabled(self):
        """Gradients should still flow when minibatch stddev is disabled."""
        disc = ImageDiscriminator(32, 64, use_minibatch_stddev=False)
        disc.train()
        x = _make_random_image(2, 3, 64, 64)
        local, global_ = disc(x)
        loss = local.sum() + global_.sum()
        loss.backward()
        for name, param in disc.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"{name} should have gradients")


# ──────────────────────────────────────────────────────────────────────────────
# Test ImageDiscriminator: output value ranges
# ──────────────────────────────────────────────────────────────────────────────


class TestImageDiscriminatorOutputRanges(unittest.TestCase):
    """Verify that scores are finite and have reasonable ranges."""

    def setUp(self):
        self.disc = ImageDiscriminator(32, 64)
        self.disc.eval()

    def test_local_scores_are_finite(self):
        """All local score values should be finite."""
        x = _make_random_image(4, 3, 64, 64)
        with torch.no_grad():
            local, _ = self.disc(x)
        self.assertTrue(torch.isfinite(local).all())

    def test_global_scores_are_finite(self):
        """All global score values should be finite."""
        x = _make_random_image(4, 3, 64, 64)
        with torch.no_grad():
            _, global_ = self.disc(x)
        self.assertTrue(torch.isfinite(global_).all())

    def test_scores_are_not_constant(self):
        """Scores should vary across batch elements (not collapsed)."""
        x = _make_random_image(4, 3, 64, 64)
        with torch.no_grad():
            local, global_ = self.disc(x)
        local_std = local.view(4, -1).std(dim=0).mean().item()
        global_std = global_.std().item()
        self.assertGreater(local_std, 0.0, "local_scores should vary across batch")
        self.assertGreater(global_std, 0.0, "global_scores should vary across batch")


# ──────────────────────────────────────────────────────────────────────────────
# Test ImageDiscriminator: different norm types
# ──────────────────────────────────────────────────────────────────────────────


class TestImageDiscriminatorNormTypes(unittest.TestCase):
    """Verify that different norm types produce valid forward passes."""

    def test_all_norm_types_forward(self):
        """All norm types should produce correct shapes on forward pass."""
        for norm_type in ("frozen_bn", "instance", "layer", "group"):
            with self.subTest(norm_type=norm_type):
                disc = ImageDiscriminator(
                    16, 32, norm_type=norm_type,
                )
                disc.eval()
                x = _make_random_image(2, 3, 32, 32)
                with torch.no_grad():
                    local, global_ = disc(x)
                expected_grid = disc.grid_size
                self.assertEqual(local.shape, (2, 1, expected_grid, expected_grid))
                self.assertEqual(global_.shape, (2, 1))

    def test_all_norm_types_trainable(self):
        """All norm types should allow gradient flow during training."""
        for norm_type in ("frozen_bn", "instance", "layer", "group"):
            with self.subTest(norm_type=norm_type):
                disc = ImageDiscriminator(
                    16, 32, norm_type=norm_type,
                )
                disc.train()
                x = _make_random_image(2, 3, 32, 32)
                local, global_ = disc(x)
                loss = local.sum() + global_.sum()
                loss.backward()
                has_grad = any(
                    p.grad is not None and p.grad.abs().sum().item() > 0
                    for p in disc.parameters() if p.requires_grad
                )
                self.assertTrue(has_grad, f"Norm type '{norm_type}' should have gradients")


# ──────────────────────────────────────────────────────────────────────────────
# Test ImageDiscriminator: state dict serialisation
# ──────────────────────────────────────────────────────────────────────────────


class TestImageDiscriminatorSerialisation(unittest.TestCase):
    """Verify discriminator supports save/load."""

    def setUp(self):
        self.disc = ImageDiscriminator(32, 64)

    def test_state_dict_contains_all_keys(self):
        """State dict should contain keys for all sub-modules."""
        sd = self.disc.state_dict()
        self.assertIn("down_blocks.0.conv.conv.weight_orig", sd)
        self.assertIn("down_blocks.0.norm.running_mean", sd)
        self.assertIn("score_conv1.conv.weight_orig", sd)
        self.assertIn("score_conv2.conv.weight_orig", sd)
        self.assertIn("global_head.weight_orig", sd)
        self.assertIn("global_head.bias", sd)

    def test_load_state_dict(self):
        """Loading a state dict should restore all weights."""
        disc1 = ImageDiscriminator(32, 64)
        disc2 = ImageDiscriminator(32, 64)
        with torch.no_grad():
            for p in disc1.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        sd = disc1.state_dict()
        disc2.load_state_dict(sd)
        for p1, p2 in zip(disc1.parameters(), disc2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_forward_after_load(self):
        """Forward pass should still work after loading state dict."""
        disc = ImageDiscriminator(32, 64)
        sd = disc.state_dict()
        disc.load_state_dict(sd)
        x = _make_random_image(2, 3, 64, 64)
        disc.eval()
        with torch.no_grad():
            local, global_ = disc(x)
        expected_grid = disc.grid_size
        self.assertEqual(local.shape, (2, 1, expected_grid, expected_grid))
        self.assertEqual(global_.shape, (2, 1))

    def test_state_dict_with_minibatch_stddev_disabled(self):
        """State dict should work with minibatch_stddev disabled."""
        disc = ImageDiscriminator(32, 64, use_minibatch_stddev=False)
        sd = disc.state_dict()
        self.assertIn("score_conv1.conv.weight_orig", sd)


# ──────────────────────────────────────────────────────────────────────────────
# Test ImageDiscriminator: edge cases
# ──────────────────────────────────────────────────────────────────────────────


class TestImageDiscriminatorEdgeCases(unittest.TestCase):
    """Verify discriminator handles edge cases correctly."""

    def test_batch_size_zero_raises(self):
        """Batch size 0 should raise."""
        disc = ImageDiscriminator(32, 64)
        disc.eval()
        x = torch.randn(0, 3, 64, 64)
        with self.assertRaises(RuntimeError):
            with torch.no_grad():
                disc(x)

    def test_batch_size_two_with_minibatch_stddev(self):
        """Batch size 2 should work with minibatch stddev."""
        disc = ImageDiscriminator(16, 32, use_minibatch_stddev=True)
        disc.eval()
        x = _make_random_image(2, 3, 32, 32)
        with torch.no_grad():
            local, global_ = disc(x)
        self.assertEqual(local.shape, (2, 1, disc.grid_size, disc.grid_size))
        self.assertEqual(global_.shape, (2, 1))

    def test_batch_size_two_without_minibatch_stddev(self):
        """Batch size 2 should work without minibatch stddev."""
        disc = ImageDiscriminator(16, 32, use_minibatch_stddev=False)
        disc.eval()
        x = _make_random_image(2, 3, 32, 32)
        with torch.no_grad():
            local, global_ = disc(x)
        self.assertEqual(local.shape, (2, 1, disc.grid_size, disc.grid_size))
        self.assertEqual(global_.shape, (2, 1))

    def test_all_same_image(self):
        """If all images in the batch are identical, scores should still be
        valid (though minibatch std will be zero)."""
        disc = ImageDiscriminator(16, 32)
        disc.eval()
        x = torch.ones(4, 3, 32, 32) * 0.5
        with torch.no_grad():
            local, global_ = disc(x)
        self.assertTrue(torch.isfinite(local).all())
        self.assertTrue(torch.isfinite(global_).all())

    def test_input_range_extremes(self):
        """Should handle input tensors with extreme values."""
        disc = ImageDiscriminator(16, 32)
        disc.eval()
        x_large = torch.randn(2, 3, 32, 32) * 1000.0
        with torch.no_grad():
            local, global_ = disc(x_large)
        self.assertTrue(torch.isfinite(local).all(),
                        "Should handle large input values")
        self.assertTrue(torch.isfinite(global_).all(),
                        "Should handle large input values")


# ──────────────────────────────────────────────────────────────────────────────
# Test ImageDiscriminator: calibration of FrozenBatchNorm2d
# ──────────────────────────────────────────────────────────────────────────────


class TestImageDiscriminatorCalibration(unittest.TestCase):
    """Verify that discriminator norms can be calibrated."""

    def test_calibrate_frozen_bn(self):
        """Calibrating each FrozenBatchNorm2d should set its statistics."""
        disc = ImageDiscriminator(16, 32, norm_type="frozen_bn")
        disc.train()

        calib_x = torch.randn(8, 3, 32, 32)

        h = calib_x
        first_block = disc.down_blocks[0]
        with torch.no_grad():
            h_conv = first_block.conv(h)
        first_block.norm.calibrate(h_conv)

        self.assertFalse(
            torch.allclose(first_block.norm.running_mean,
                           torch.zeros(first_block.norm.num_features),
                           atol=1e-2),
            "Running mean should have been calibrated away from zero",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)