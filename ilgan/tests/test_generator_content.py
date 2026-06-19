"""
Tests for ``ilgan.models.generator.ContentDecoder``.

Verifies:

- **Forward shapes**: ``ContentDecoder`` with ``latent_dim=256`` and
  ``image_size=128`` produces an output image of shape ``[4, 3, 128, 128]``,
  pixel values in ``[-1, 1]``, and ``skip_features`` of the expected length.
- **Skip feature resolutions**: each skip feature has the correct spatial
  size for its position in the progression.
- **Value range**: all output pixel values are in ``[-1, 1]``.
- **Gradient checkpointing**: enabling ``use_checkpointing=True`` does not
  change the output values (forward pass equivalence).
- **Batch independence**: different latent vectors produce different images.
- **Different image sizes**: ``ContentDecoder`` works for 64, 128, and 256.
- **Spectral norm**: module can be instantiated with ``use_spectral_norm``
  without errors.
- **Group norm**: module can be instantiated with ``use_group_norm``.
- **Edge cases**: ``image_size`` validation, minimal size, gradient flow.
"""

import unittest

import torch
import torch.nn as nn

from ilgan.models.generator import ContentDecoder, UpBlock, SpectralNormConv2d


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_latent(B: int = 4, latent_dim: int = 256) -> torch.Tensor:
    """Create a batch of random latent vectors from ``𝒩(0, I)``."""
    return torch.randn(B, latent_dim)


def _count_parameters(module: nn.Module) -> int:
    """Return the total number of trainable parameters in *module*."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Test ContentDecoder: forward shapes
# ──────────────────────────────────────────────────────────────────────────────


class TestContentDecoderForwardShapes(unittest.TestCase):
    """Verify output shapes for the standard forward pass."""

    def setUp(self):
        self.latent_dim = 256
        self.image_size = 128
        self.batch_size = 4
        self.decoder = ContentDecoder(
            latent_dim=self.latent_dim,
            gen_base_channels=64,
            image_size=self.image_size,
        )

    def test_output_image_shape(self):
        """Output image should be [B, 3, image_size, image_size]."""
        z = _make_latent(self.batch_size, self.latent_dim)
        img, _ = self.decoder(z)

        expected_shape = (self.batch_size, 3, self.image_size, self.image_size)
        self.assertEqual(
            img.shape,
            expected_shape,
            f"Expected image shape {expected_shape}, got {img.shape}",
        )

    def test_output_dtype(self):
        """Output image should be float32."""
        z = _make_latent(self.batch_size, self.latent_dim)
        img, _ = self.decoder(z)
        self.assertEqual(img.dtype, torch.float32)

    def test_output_values_in_range(self):
        """All pixel values should be in [-1, 1]."""
        z = _make_latent(self.batch_size, self.latent_dim)
        img, _ = self.decoder(z)

        self.assertGreaterEqual(img.min().item(), -1.0,
                                "Pixel values below -1.0")
        self.assertLessEqual(img.max().item(), 1.0,
                             "Pixel values above 1.0")

    def test_skip_features_length(self):
        """Number of skip features should equal number of up-blocks.

        For image_size=128: log2(128) - log2(4) = 7 - 2 = 5 blocks.
        """
        z = _make_latent(self.batch_size, self.latent_dim)
        _, skips = self.decoder(z)

        expected_blocks = int(torch.log2(torch.tensor(self.image_size))) - 2
        self.assertEqual(
            len(skips),
            expected_blocks,
            f"Expected {expected_blocks} skip features, got {len(skips)}",
        )

    def test_skip_feature_resolutions(self):
        """Skip features should have increasing spatial resolutions.

        Starting from 4x4, each up-block doubles the spatial size.
        For image_size=128: resolutions [8, 16, 32, 64, 128].
        """
        z = _make_latent(self.batch_size, self.latent_dim)
        _, skips = self.decoder(z)

        expected_resolutions = [8, 16, 32, 64, 128]
        self.assertEqual(
            len(skips), len(expected_resolutions),
            f"Expected {len(expected_resolutions)} skip features, got {len(skips)}",
        )

        for i, (skip, res) in enumerate(zip(skips, expected_resolutions)):
            self.assertEqual(
                skip.shape[2],
                res,
                f"Skip feature {i} expected height={res}, got {skip.shape[2]}",
            )
            self.assertEqual(
                skip.shape[3],
                res,
                f"Skip feature {i} expected width={res}, got {skip.shape[3]}",
            )

    def test_skip_feature_channel_counts(self):
        """Skip feature channel counts should halve each time.

        Starting from gen_base_channels*16/2 = 64*8 = 512 at 8x8,
        then 256, 128, 64, 64 (stays at gen_base_channels).
        """
        z = _make_latent(self.batch_size, self.latent_dim)
        _, skips = self.decoder(z)

        # Channel progression for gen_base_channels=64, image_size=128:
        # Block 0: 64*16 // 2 = 512  (at 8x8)
        # Block 1: 512 // 2 = 256     (at 16x16)
        # Block 2: 256 // 2 = 128     (at 32x32)
        # Block 3: 128 // 2 = 64      (at 64x64)
        # Block 4: 64 // 2 = 64       (at 128x128) — clamped to base
        expected_channels = [512, 256, 128, 64, 64]

        for i, (skip, ch) in enumerate(zip(skips, expected_channels)):
            self.assertEqual(
                skip.shape[1],
                ch,
                f"Skip feature {i} expected {ch} channels, got {skip.shape[1]}",
            )


# ──────────────────────────────────────────────────────────────────────────────
# Test ContentDecoder: gradient checkpointing
# ──────────────────────────────────────────────────────────────────────────────


class TestContentDecoderCheckpointing(unittest.TestCase):
    """Verify gradient checkpointing preserves forward pass outputs."""

    def test_checkpointing_forward_equivalence(self):
        """Outputs with and without checkpointing should be identical
        (within numerical tolerance)."""
        torch.manual_seed(42)

        latent_dim = 256
        image_size = 128
        batch_size = 4

        decoder_no_ckpt = ContentDecoder(
            latent_dim=latent_dim,
            gen_base_channels=64,
            image_size=image_size,
            use_checkpointing=False,
        )
        decoder_ckpt = ContentDecoder(
            latent_dim=latent_dim,
            gen_base_channels=64,
            image_size=image_size,
            use_checkpointing=True,
        )

        # Copy weights from no-checkpoint model to checkpoint model
        decoder_ckpt.load_state_dict(decoder_no_ckpt.state_dict())

        z = _make_latent(batch_size, latent_dim)

        # Evaluate mode (checkpointing only affects training)
        decoder_ckpt.eval()
        decoder_no_ckpt.eval()

        with torch.no_grad():
            img_no_ckpt, skips_no_ckpt = decoder_no_ckpt(z)
            img_ckpt, skips_ckpt = decoder_ckpt(z)

        # Images should be identical
        self.assertTrue(
            torch.allclose(img_no_ckpt, img_ckpt, atol=1e-6),
            "Image outputs differ between checkpointed and non-checkpointed forward pass",
        )

        # Skip features should be identical
        for i, (s_no, s_ckpt) in enumerate(zip(skips_no_ckpt, skips_ckpt)):
            self.assertTrue(
                torch.allclose(s_no, s_ckpt, atol=1e-6),
                f"Skip feature {i} differs between checkpointed and non-checkpointed",
            )

    def test_checkpointing_training_forward_works(self):
        """Checkpointing forward pass in training mode should complete
        without errors and produce valid outputs."""
        decoder = ContentDecoder(
            latent_dim=128,
            gen_base_channels=32,
            image_size=64,
            use_checkpointing=True,
        )
        decoder.train()

        z = _make_latent(4, 128)
        img, skips = decoder(z)

        self.assertEqual(img.shape, (4, 3, 64, 64))
        self.assertEqual(len(skips), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Test ContentDecoder: different image sizes
# ──────────────────────────────────────────────────────────────────────────────


class TestContentDecoderImageSizes(unittest.TestCase):
    """Verify ContentDecoder works for common image sizes."""

    def test_image_size_64(self):
        """Decoder for 64x64 should produce correct shapes.

        Number of blocks: log2(64) - log2(4) = 6 - 2 = 4.
        Skip features at [8, 16, 32, 64].
        """
        decoder = ContentDecoder(
            latent_dim=128,
            gen_base_channels=32,
            image_size=64,
        )
        z = _make_latent(2, 128)
        img, skips = decoder(z)

        self.assertEqual(img.shape, (2, 3, 64, 64))
        self.assertEqual(len(skips), 4)
        self.assertEqual(skips[0].shape[2], 8)
        self.assertEqual(skips[-1].shape[2], 64)

    def test_image_size_128(self):
        """Decoder for 128x128 should produce correct shapes with
        small base channels (memory-friendly)."""
        decoder = ContentDecoder(
            latent_dim=128,
            gen_base_channels=16,
            image_size=128,
        )
        z = _make_latent(2, 128)
        img, skips = decoder(z)

        self.assertEqual(img.shape, (2, 3, 128, 128))
        self.assertEqual(len(skips), 5)
        self.assertEqual(skips[0].shape[2], 8)
        self.assertEqual(skips[-1].shape[2], 128)

    def test_image_size_256(self):
        """Decoder for 256x256 should produce correct shapes.

        Number of blocks: log2(256) - log2(4) = 8 - 2 = 6.
        Skip features at [8, 16, 32, 64, 128, 256].
        """
        decoder = ContentDecoder(
            latent_dim=128,
            gen_base_channels=16,
            image_size=256,
        )
        z = _make_latent(2, 128)
        img, skips = decoder(z)

        self.assertEqual(img.shape, (2, 3, 256, 256))
        self.assertEqual(len(skips), 6)
        self.assertEqual(skips[0].shape[2], 8)
        self.assertEqual(skips[-1].shape[2], 256)


# ──────────────────────────────────────────────────────────────────────────────
# Test ContentDecoder: spectral norm
# ──────────────────────────────────────────────────────────────────────────────


class TestContentDecoderSpectralNorm(unittest.TestCase):
    """Verify the decoder can be instantiated with spectral normalisation."""

    def test_spectral_norm_instantiation(self):
        """Should instantiate without errors and produce valid output shapes."""
        decoder = ContentDecoder(
            latent_dim=128,
            gen_base_channels=32,
            image_size=64,
            use_spectral_norm=True,
        )
        z = _make_latent(4, 128)
        img, skips = decoder(z)

        self.assertEqual(img.shape, (4, 3, 64, 64))
        self.assertEqual(len(skips), 4)

    def test_spectral_norm_checkpointing_combined(self):
        """Both spectral norm and checkpointing should work together."""
        decoder = ContentDecoder(
            latent_dim=128,
            gen_base_channels=32,
            image_size=64,
            use_spectral_norm=True,
            use_checkpointing=True,
        )
        z = _make_latent(4, 128)
        img, skips = decoder(z)

        self.assertEqual(img.shape, (4, 3, 64, 64))
        self.assertEqual(len(skips), 4)
        self.assertTrue(torch.all((img >= -1.0) & (img <= 1.0)))


# ──────────────────────────────────────────────────────────────────────────────
# Test ContentDecoder: group norm
# ──────────────────────────────────────────────────────────────────────────────


class TestContentDecoderGroupNorm(unittest.TestCase):
    """Verify the decoder can use GroupNorm instead of BatchNorm."""

    def test_group_norm_instantiation(self):
        """Should instantiate and forward without errors."""
        decoder = ContentDecoder(
            latent_dim=128,
            gen_base_channels=32,
            image_size=64,
            use_group_norm=True,
        )
        z = _make_latent(2, 128)
        img, skips = decoder(z)

        self.assertEqual(img.shape, (2, 3, 64, 64))

    def test_group_norm_small_batch(self):
        """GroupNorm should work with a batch size of 1."""
        decoder = ContentDecoder(
            latent_dim=128,
            gen_base_channels=32,
            image_size=64,
            use_group_norm=True,
        )
        z = _make_latent(1, 128)
        img, skips = decoder(z)

        self.assertEqual(img.shape, (1, 3, 64, 64))


# ──────────────────────────────────────────────────────────────────────────────
# Test ContentDecoder: edge cases
# ──────────────────────────────────────────────────────────────────────────────


class TestContentDecoderEdgeCases(unittest.TestCase):
    """Verify the decoder handles edge cases correctly."""

    def test_invalid_image_size_not_power_of_two(self):
        """Non-power-of-two image_size should raise ValueError."""
        with self.assertRaises(ValueError):
            ContentDecoder(
                latent_dim=256,
                gen_base_channels=64,
                image_size=100,  # not a power of two
            )

    def test_invalid_image_size_too_small(self):
        """image_size < 8 should raise ValueError."""
        with self.assertRaises(ValueError):
            ContentDecoder(
                latent_dim=256,
                gen_base_channels=64,
                image_size=4,  # too small
            )

    def test_minimal_image_size(self):
        """Minimum valid image_size (8) should work.

        Number of blocks: log2(8) - log2(4) = 3 - 2 = 1.
        Skip features at [8].
        """
        decoder = ContentDecoder(
            latent_dim=64,
            gen_base_channels=32,
            image_size=8,
        )
        z = _make_latent(2, 64)
        img, skips = decoder(z)

        self.assertEqual(img.shape, (2, 3, 8, 8))
        self.assertEqual(len(skips), 1)
        self.assertEqual(skips[0].shape[2], 8)

    def test_batch_size_one(self):
        """Should work with a single sample (B=1)."""
        decoder = ContentDecoder(
            latent_dim=128,
            gen_base_channels=32,
            image_size=64,
        )
        z = _make_latent(1, 128)
        img, skips = decoder(z)

        self.assertEqual(img.shape, (1, 3, 64, 64))
        self.assertEqual(len(skips), 4)

    def test_modest_batch(self):
        """Should work with a modest batch (B=8) using small model."""
        decoder = ContentDecoder(
            latent_dim=128,
            gen_base_channels=32,
            image_size=64,
        )
        z = _make_latent(8, 128)
        img, skips = decoder(z)

        self.assertEqual(img.shape, (8, 3, 64, 64))
        self.assertEqual(len(skips), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Test ContentDecoder: gradient flow
# ──────────────────────────────────────────────────────────────────────────────


class TestContentDecoderGradients(unittest.TestCase):
    """Verify gradients flow through the entire decoder."""

    def test_gradients_flow_to_input(self):
        """Gradients should flow back to the latent input ``z``."""
        decoder = ContentDecoder(
            latent_dim=128,
            gen_base_channels=32,
            image_size=64,
        )
        z = _make_latent(4, 128).requires_grad_(True)
        img, _ = decoder(z)

        loss = img.sum()
        loss.backward()

        self.assertIsNotNone(z.grad, "Gradients should flow to z")
        self.assertGreater(z.grad.abs().sum().item(), 0.0,
                           "z.grad should be non-zero")

    def test_gradients_flow_to_all_parameters(self):
        """Every trainable parameter should receive a non-zero gradient."""
        decoder = ContentDecoder(
            latent_dim=128,
            gen_base_channels=32,
            image_size=64,
        )
        z = _make_latent(4, 128)
        img, _ = decoder(z)

        loss = img.sum()
        loss.backward()

        for name, param in decoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"Parameter '{name}' should have gradients",
                )
                self.assertGreater(
                    param.grad.abs().sum().item(), 0.0,
                    f"Parameter '{name}' grad should be non-zero",
                )

    def test_gradient_checkpointing_backward(self):
        """Backward pass with checkpointing enabled should succeed and
        produce gradients on all parameters."""
        decoder = ContentDecoder(
            latent_dim=128,
            gen_base_channels=32,
            image_size=64,
            use_checkpointing=True,
        )
        decoder.train()

        z = _make_latent(4, 128)
        img, _ = decoder(z)

        loss = img.sum()
        loss.backward()

        for name, param in decoder.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"Parameter '{name}' should have gradients with checkpointing",
                )


# ──────────────────────────────────────────────────────────────────────────────
# Test ContentDecoder: batch independence
# ──────────────────────────────────────────────────────────────────────────────


class TestContentDecoderBatchIndependence(unittest.TestCase):
    """Verify different latent vectors produce different outputs."""

    def test_different_latents_different_images(self):
        """Two different latent vectors should produce different images."""
        decoder = ContentDecoder(
            latent_dim=128,
            gen_base_channels=32,
            image_size=64,
        )
        z1 = torch.randn(1, 128)
        z2 = torch.randn(1, 128)

        with torch.no_grad():
            img1, _ = decoder(z1)
            img2, _ = decoder(z2)

        # Images should differ
        diff = (img1 - img2).abs().sum().item()
        self.assertGreater(diff, 0.01,
                           "Different latent vectors should produce different images")

    def test_same_latent_same_image(self):
        """The same latent vector should produce the same image
        (deterministic forward pass)."""
        decoder = ContentDecoder(
            latent_dim=128,
            gen_base_channels=32,
            image_size=64,
        )
        z = torch.randn(1, 128)

        with torch.no_grad():
            img1, _ = decoder(z)
            img2, _ = decoder(z)

        self.assertTrue(
            torch.allclose(img1, img2, atol=1e-6),
            "Same latent vector should produce the same image",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test ContentDecoder: parameter count
# ──────────────────────────────────────────────────────────────────────────────


class TestContentDecoderParameters(unittest.TestCase):
    """Verify parameter count is reasonable (sanity check)."""

    def test_parameter_count_non_zero(self):
        """Decoder should have a non-zero number of parameters."""
        decoder = ContentDecoder(
            latent_dim=256,
            gen_base_channels=64,
            image_size=128,
        )
        num_params = _count_parameters(decoder)
        self.assertGreater(num_params, 0, "ContentDecoder must have parameters")

    def test_parameter_count_range(self):
        """Decoder should have a reasonable number of parameters
        (not trivially small, not absurdly large for this config)."""
        decoder = ContentDecoder(
            latent_dim=256,
            gen_base_channels=64,
            image_size=128,
        )
        num_params = _count_parameters(decoder)
        # For this config: should be between ~5M and ~50M
        self.assertGreater(num_params, 1_000_000,
                           "ContentDecoder should have at least 1M parameters")
        self.assertLess(num_params, 100_000_000,
                        "ContentDecoder should have less than 100M parameters")


# ──────────────────────────────────────────────────────────────────────────────
# Test UpBlock separately
# ──────────────────────────────────────────────────────────────────────────────


class TestUpBlock(unittest.TestCase):
    """Verify the UpBlock module individually."""

    def test_upblock_doubles_spatial_size(self):
        """UpBlock should double the spatial dimensions."""
        block = UpBlock(in_channels=128, out_channels=64)
        x = torch.randn(2, 128, 8, 8)
        out, skip = block(x)

        self.assertEqual(out.shape, (2, 64, 16, 16),
                         "UpBlock should double spatial size")
        self.assertEqual(skip.shape, (2, 64, 16, 16),
                         "Skip feature should match output spatial size")

    def test_upblock_with_spectral_norm(self):
        """UpBlock with spectral norm should work."""
        block = UpBlock(in_channels=128, out_channels=64, use_spectral_norm=True)
        x = torch.randn(2, 128, 8, 8)
        out, skip = block(x)

        self.assertEqual(out.shape, (2, 64, 16, 16))

    def test_upblock_with_group_norm(self):
        """UpBlock with GroupNorm should work."""
        block = UpBlock(in_channels=128, out_channels=64, use_group_norm=True)
        x = torch.randn(2, 128, 8, 8)
        out, skip = block(x)

        self.assertEqual(out.shape, (2, 64, 16, 16))

    def test_upblock_gradient_flow(self):
        """Gradients should flow through UpBlock."""
        block = UpBlock(in_channels=128, out_channels=64)
        x = torch.randn(2, 128, 8, 8, requires_grad=True)

        out, skip = block(x)
        loss = out.sum() + skip.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Test SpectralNormConv2d
# ──────────────────────────────────────────────────────────────────────────────


class TestSpectralNormConv2d(unittest.TestCase):
    """Verify SpectralNormConv2d utility."""

    def test_forward_without_spectral_norm(self):
        """Conv2d without spectral norm should produce same shape as Conv2d."""
        conv = SpectralNormConv2d(64, 128, kernel_size=3, padding=1)
        x = torch.randn(2, 64, 16, 16)
        y = conv(x)
        self.assertEqual(y.shape, (2, 128, 16, 16))

    def test_forward_with_spectral_norm(self):
        """Conv2d with spectral norm should produce correct shape."""
        conv = SpectralNormConv2d(64, 128, kernel_size=3, padding=1,
                                  use_spectral_norm=True)
        x = torch.randn(2, 64, 16, 16)
        y = conv(x)
        self.assertEqual(y.shape, (2, 128, 16, 16))

    def test_spectral_norm_has_u_and_v(self):
        """Spectral norm should add ``weight_u`` and ``weight_v`` buffers."""
        conv = SpectralNormConv2d(64, 128, kernel_size=3, padding=1,
                                  use_spectral_norm=True)
        self.assertTrue(
            hasattr(conv.conv, "weight_u"),
            "Spectral norm should add weight_u buffer",
        )
        self.assertTrue(
            hasattr(conv.conv, "weight_v"),
            "Spectral norm should add weight_v buffer",
        )

    def test_spectral_norm_no_extra_state_without_flag(self):
        """Without spectral norm, no extra buffers should be added."""
        conv = SpectralNormConv2d(64, 128, kernel_size=3, padding=1,
                                  use_spectral_norm=False)
        self.assertFalse(
            hasattr(conv.conv, "weight_u"),
            "Without spectral norm, no weight_u should exist",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)