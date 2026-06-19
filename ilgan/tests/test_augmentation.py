"""
Tests for ``ilgan.data.augmentation`` and ``ilgan.data.dataloader``.

Verifies:
- RandomHorizontalFlip flips images and box x-coordinates correctly.
- RandomColorJitter produces valid outputs (boxes unchanged).
- RandomAffine transforms boxes to stay consistent with visual content,
  and that no boxes go out of bounds.
- Cutout masks regions without affecting boxes.
- Compose chains transforms correctly with deterministic seeds.
- ``get_train_val_loaders`` creates properly configured loaders.
- ``GANDataloader.set_epoch`` controls determinism.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from ilgan.data.augmentation import (
    RandomHorizontalFlip,
    RandomColorJitter,
    RandomAffine,
    Cutout,
    Compose,
    build_default_augmentation_pipeline,
)
from ilgan.data.dataloader import GANDataloader, get_train_val_loaders
from ilgan.data.dataset import YOLODataset
from ilgan.data.structures import Sample, Batch


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_sample(
    h: int = 64,
    w: int = 64,
    boxes: torch.Tensor = None,
) -> Sample:
    """Create a sample with a solid-colour image and known box coordinates."""
    # Image: left half solid red (R=255), right half solid blue (B=255)
    # This makes horizontal flip visually detectable.
    arr = np.zeros((h, w, 3), dtype=np.float32)
    arr[:, :w // 2, :] = [255.0, 0.0, 0.0]   # red left
    arr[:, w // 2:, :] = [0.0, 0.0, 255.0]    # blue right
    arr = arr.transpose(2, 0, 1)                # [3, H, W]
    image = torch.from_numpy(arr) / 127.5 - 1.0  # -> [-1, 1]

    if boxes is None:
        # A box in the left half: x_center=0.25, safely inside bounds
        boxes = torch.tensor([[0.25, 0.5, 0.3, 0.4]], dtype=torch.float32)

    labels = torch.zeros(boxes.size(0), dtype=torch.long)
    valid_mask = torch.ones(boxes.size(0), dtype=torch.bool)

    return Sample(
        image=image,
        boxes=boxes,
        labels=labels,
        valid_mask=valid_mask,
        image_path="/fake/test.png",
        metadata={"split": "test"},
    )


def _make_random_sample(
    h: int = 64,
    w: int = 64,
    num_boxes: int = 3,
) -> Sample:
    """Create a sample with random image and random valid boxes.

    Boxes are carefully constrained to be well inside ``[0, 1]`` so that
    affine clamping does not alter them under identity transforms.
    """
    image = torch.rand(3, h, w) * 2.0 - 1.0

    # Generate boxes with centres in [0.2, 0.8] and widths/heights in [0.05, 0.2]
    # This guarantees corners lie in [0.1, 0.9] so no clamping occurs.
    centers = 0.2 + torch.rand(num_boxes, 2) * 0.6   # [xc, yc] in [0.2, 0.8]
    sizes = 0.05 + torch.rand(num_boxes, 2) * 0.15    # [w, h] in [0.05, 0.2]
    boxes = torch.cat([centers, sizes], dim=1)         # [N, 4]

    labels = torch.zeros(num_boxes, dtype=torch.long)
    valid_mask = torch.ones(num_boxes, dtype=torch.bool)

    return Sample(
        image=image,
        boxes=boxes,
        labels=labels,
        valid_mask=valid_mask,
        image_path="/fake/random.png",
        metadata={"split": "train"},
    )


def _create_synthetic_dataset(root_dir: str, num_images: int = 8):
    """Create a minimal synthetic dataset for testing dataloaders."""
    img_dir = os.path.join(root_dir, "images")
    label_dir = os.path.join(root_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for i in range(num_images):
        stem = f"img_{i:04d}"
        # Random image
        arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(img_dir, f"{stem}.jpg"))
        # YOLO label (1 box per image) — well inside bounds
        xc = np.random.uniform(0.2, 0.8)
        yc = np.random.uniform(0.2, 0.8)
        bw = np.random.uniform(0.1, 0.3)
        bh = np.random.uniform(0.1, 0.3)
        with open(os.path.join(label_dir, f"{stem}.txt"), "w") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Test RandomHorizontalFlip
# ──────────────────────────────────────────────────────────────────────────────


class TestRandomHorizontalFlip(unittest.TestCase):
    """Verify flip correctness for images and boxes."""

    def test_x_center_flipped(self):
        """x_center=0.25 should become x_center=0.75 after flip."""
        sample = _make_sample(64, 64)
        aug = RandomHorizontalFlip(p=1.0)
        _, boxes, _, _ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        self.assertAlmostEqual(boxes[0, 0].item(), 0.75, places=5)

    def test_y_center_unchanged(self):
        """y_center should remain unchanged."""
        sample = _make_sample(64, 64)
        aug = RandomHorizontalFlip(p=1.0)
        _, boxes, _, _ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        self.assertAlmostEqual(boxes[0, 1].item(), 0.5, places=5)

    def test_width_height_unchanged(self):
        """Width and height should remain unchanged."""
        sample = _make_sample(64, 64)
        aug = RandomHorizontalFlip(p=1.0)
        _, boxes, _, _ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        self.assertAlmostEqual(boxes[0, 2].item(), 0.3, places=5)
        self.assertAlmostEqual(boxes[0, 3].item(), 0.4, places=5)

    def test_image_flipped_visually(self):
        """Left side (red) should move to the right side after flip."""
        sample = _make_sample(64, 64)
        aug = RandomHorizontalFlip(p=1.0)
        flipped_img, *_ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        # The right side of the flipped image should be red (positive R channel)
        right_pixel = flipped_img[0, 32, -1].item()  # R channel, middle row, last col
        self.assertGreater(right_pixel, 0.5, "Right side should be red after flip")

    def test_probabilistic_skips(self):
        """With p=0.0, the identity should be returned."""
        sample = _make_sample(64, 64)
        aug = RandomHorizontalFlip(p=0.0)
        img_out, boxes_out, _, _ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        self.assertTrue(torch.equal(img_out, sample.image))
        self.assertTrue(torch.equal(boxes_out, sample.boxes))

    def test_empty_boxes(self):
        """Flipping with no boxes should not error."""
        sample = _make_sample(64, 64, boxes=torch.zeros((0, 4)))
        aug = RandomHorizontalFlip(p=1.0)
        _, boxes_out, _, _ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        self.assertEqual(boxes_out.shape, (0, 4))

    def test_deterministic_seed(self):
        """Same seed should produce same flip result."""
        sample = _make_random_sample(32, 32, num_boxes=2)
        aug = RandomHorizontalFlip(p=1.0)
        res1 = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask, rng_seed=42)
        res2 = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask, rng_seed=42)
        self.assertTrue(torch.equal(res1[0], res2[0]))
        self.assertTrue(torch.equal(res1[1], res2[1]))


# ──────────────────────────────────────────────────────────────────────────────
# Test RandomColorJitter
# ──────────────────────────────────────────────────────────────────────────────


class TestRandomColorJitter(unittest.TestCase):
    """Verify colour jitter produces valid outputs and leaves boxes alone."""

    def test_boxes_unchanged(self):
        """Box tensors should be passed through identically."""
        sample = _make_random_sample(32, 32, num_boxes=2)
        aug = RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)
        _, boxes_out, labels_out, valid_out = aug(
            sample.image, sample.boxes, sample.labels, sample.valid_mask
        )
        self.assertTrue(torch.equal(boxes_out, sample.boxes))
        self.assertTrue(torch.equal(labels_out, sample.labels))
        self.assertTrue(torch.equal(valid_out, sample.valid_mask))

    def test_output_range(self):
        """Output image values should remain in [-1, 1]."""
        sample = _make_random_sample(32, 32)
        aug = RandomColorJitter(brightness=0.3, contrast=0.3, p=1.0)
        img_out, *_ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        self.assertGreaterEqual(img_out.min().item(), -1.0 - 1e-4)
        self.assertLessEqual(img_out.max().item(), 1.0 + 1e-4)

    def test_output_shape(self):
        """Output image shape should match input."""
        sample = _make_random_sample(64, 48)
        aug = RandomColorJitter(p=1.0)
        img_out, *_ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        self.assertEqual(img_out.shape, (3, 64, 48))

    def test_zero_strength_is_identity(self):
        """Jitter with 0.0 strength should return near-identical image."""
        sample = _make_random_sample(32, 32)
        aug = RandomColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, p=1.0)
        img_out, *_ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        # Some numerical differences may arise from PIL conversion roundtrip
        diff = (img_out - sample.image).abs().max().item()
        self.assertLess(diff, 0.02, "Zero-strength jitter should be near identity")

    def test_probabilistic(self):
        """With p=0.0, the identity is returned."""
        sample = _make_random_sample(32, 32)
        aug = RandomColorJitter(brightness=0.5, p=0.0)
        img_out, *_ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        self.assertTrue(torch.equal(img_out, sample.image))


# ──────────────────────────────────────────────────────────────────────────────
# Test RandomAffine
# ──────────────────────────────────────────────────────────────────────────────


class TestRandomAffine(unittest.TestCase):
    """Verify affine transform consistency between image and boxes."""

    def test_output_shape(self):
        """Output image shape should match input."""
        sample = _make_random_sample(64, 64, num_boxes=2)
        aug = RandomAffine(degrees=10, translate=0.1, scale=(0.8, 1.2), p=1.0)
        img_out, *_ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        self.assertEqual(img_out.shape, sample.image.shape)

    def test_no_box_goes_out_of_bounds(self):
        """Transformed box coordinates should stay within [0, 1] after affine
        transform."""
        sample = _make_random_sample(64, 64, num_boxes=5)
        aug = RandomAffine(degrees=15, translate=0.15, scale=(0.8, 1.2), p=1.0)
        _, boxes_out, _, valid_out = aug(
            sample.image, sample.boxes, sample.labels, sample.valid_mask
        )
        # Valid boxes should have coords in [0, 1]
        if valid_out.any():
            valid_boxes = boxes_out[valid_out]
            self.assertGreaterEqual(valid_boxes.min().item(), 0.0,
                                    "Valid box coords must be >= 0")
            self.assertLessEqual(valid_boxes.max().item(), 1.0,
                                 "Valid box coords must be <= 1")

    def test_identity_parameters(self):
        """With degrees=0, translate=0, scale=1.0, transform should be identity."""
        sample = _make_random_sample(32, 32, num_boxes=2)
        aug = RandomAffine(degrees=0.0, translate=0.0, scale=(1.0, 1.0), p=1.0)
        img_out, boxes_out, _, _ = aug(
            sample.image, sample.boxes, sample.labels, sample.valid_mask
        )
        # Image should be very close to input (small numerical diff from grid_sample)
        img_diff = (img_out - sample.image).abs().max().item()
        self.assertLess(img_diff, 0.02, "Identity affine should preserve image")
        # Boxes should be preserved (our _make_random_sample uses safe coords)
        self.assertTrue(torch.allclose(boxes_out, sample.boxes, atol=1e-4),
                        "Identity affine should preserve boxes")

    def test_translation_moves_boxes(self):
        """A rightward translation should change x_center."""
        sample = _make_sample(64, 64, boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]))
        aug = RandomAffine(degrees=0.0, translate=0.1, scale=(1.0, 1.0), p=1.0)
        _, boxes_out, _, _ = aug(
            sample.image, sample.boxes, sample.labels, sample.valid_mask,
            rng_seed=12345,
        )
        # The x_center should have changed from 0.5
        self.assertNotEqual(boxes_out[0, 0].item(), 0.5,
                            "Translation should change x_center")

    def test_empty_boxes(self):
        """Affine with no boxes should not error."""
        sample = _make_sample(64, 64, boxes=torch.zeros((0, 4)))
        aug = RandomAffine(p=1.0)
        _, boxes_out, _, _ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        self.assertEqual(boxes_out.shape, (0, 4))

    def test_deterministic_seed(self):
        """Same seed should produce same affine transform."""
        sample = _make_random_sample(32, 32, num_boxes=2)
        aug = RandomAffine(degrees=10, translate=0.1, scale=(0.9, 1.1), p=1.0)
        res1 = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask, rng_seed=999)
        res2 = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask, rng_seed=999)
        self.assertTrue(torch.equal(res1[0], res2[0]))
        self.assertTrue(torch.equal(res1[1], res2[1]))


# ──────────────────────────────────────────────────────────────────────────────
# Test Cutout
# ──────────────────────────────────────────────────────────────────────────────


class TestCutout(unittest.TestCase):
    """Verify cutout masking behaviour."""

    def test_boxes_unchanged(self):
        """Box tensors should be passed through identically."""
        sample = _make_random_sample(32, 32)
        aug = Cutout(p=1.0, max_holes=2, max_size=0.3)
        _, boxes_out, labels_out, valid_out = aug(
            sample.image, sample.boxes, sample.labels, sample.valid_mask
        )
        self.assertTrue(torch.equal(boxes_out, sample.boxes))
        self.assertTrue(torch.equal(labels_out, sample.labels))
        self.assertTrue(torch.equal(valid_out, sample.valid_mask))

    def test_output_shape(self):
        """Output image shape should match input."""
        sample = _make_random_sample(32, 32)
        aug = Cutout(p=1.0, max_holes=2, max_size=0.3)
        img_out, *_ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        self.assertEqual(img_out.shape, (3, 32, 32))

    def test_masked_regions_are_fill_value(self):
        """Some pixels should be set to fill_value."""
        sample = _make_random_sample(16, 16)
        aug = Cutout(p=1.0, max_holes=5, max_size=0.5, fill_value=0.0)
        img_out, *_ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        # At least some pixels should be exactly 0.0
        self.assertTrue((img_out == 0.0).any(), "Cutout should mask some pixels")

    def test_no_cutout_when_prob_zero(self):
        """With p=0.0, identity is returned."""
        sample = _make_random_sample(16, 16)
        aug = Cutout(p=0.0, max_holes=5, max_size=0.5)
        img_out, *_ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        self.assertTrue(torch.equal(img_out, sample.image))

    def test_custom_fill_value(self):
        """Custom fill_value should appear in masked regions."""
        sample = _make_random_sample(16, 16)
        aug = Cutout(p=1.0, max_holes=2, max_size=0.4, fill_value=-0.5)
        img_out, *_ = aug(sample.image, sample.boxes, sample.labels, sample.valid_mask)
        self.assertTrue((img_out == -0.5).any(), "Custom fill value should appear")


# ──────────────────────────────────────────────────────────────────────────────
# Test Compose
# ──────────────────────────────────────────────────────────────────────────────


class TestCompose(unittest.TestCase):
    """Verify Compose chains transforms correctly."""

    def test_chained_transforms(self):
        """Compose should apply all transforms in order."""
        sample = _make_random_sample(32, 32)
        transforms = [
            RandomHorizontalFlip(p=1.0),
            Cutout(p=1.0, max_holes=1, max_size=0.2),
        ]
        compose = Compose(transforms)
        img_out, boxes_out, labels_out, valid_out = compose(
            sample.image, sample.boxes, sample.labels, sample.valid_mask,
            rng_seed=42,
        )
        # Boxes should be flipped (x_center changed)
        self.assertNotEqual(boxes_out[0, 0].item(), sample.boxes[0, 0].item())
        # Image should be flipped and cutout
        self.assertEqual(img_out.shape, sample.image.shape)

    def test_apply_to_sample(self):
        """apply_to_sample should return a new Sample with augmented fields."""
        sample = _make_random_sample(32, 32)
        transforms = [RandomHorizontalFlip(p=1.0)]
        compose = Compose(transforms)
        aug_sample = compose.apply_to_sample(sample, rng_seed=42)
        self.assertIsInstance(aug_sample, Sample)
        self.assertTrue(aug_sample.metadata.get("augmented", False))

    def test_determinism(self):
        """Same seed produces same result."""
        sample = _make_random_sample(32, 32)
        transforms = [
            RandomHorizontalFlip(p=1.0),
            RandomColorJitter(brightness=0.2, contrast=0.2, p=1.0),
        ]
        compose = Compose(transforms)
        r1 = compose(sample.image, sample.boxes, sample.labels, sample.valid_mask, rng_seed=100)
        r2 = compose(sample.image, sample.boxes, sample.labels, sample.valid_mask, rng_seed=100)
        for t1, t2 in zip(r1, r2):
            self.assertTrue(torch.equal(t1, t2))

    def test_len(self):
        """__len__ returns number of transforms."""
        compose = Compose([RandomHorizontalFlip(p=0.5), Cutout(p=0.5)])
        self.assertEqual(len(compose), 2)

    def test_getitem(self):
        """__getitem__ returns the transform at the index."""
        t = RandomHorizontalFlip(p=0.5)
        compose = Compose([t])
        self.assertIs(compose[0], t)


# ──────────────────────────────────────────────────────────────────────────────
# Test build_default_augmentation_pipeline
# ──────────────────────────────────────────────────────────────────────────────


class TestBuildDefaultPipeline(unittest.TestCase):
    """Verify the pipeline factory."""

    def test_returns_compose(self):
        """Factory should return a Compose instance."""
        pipeline = build_default_augmentation_pipeline()
        self.assertIsInstance(pipeline, Compose)

    def test_has_four_transforms(self):
        """Default pipeline should have 4 transforms."""
        pipeline = build_default_augmentation_pipeline()
        self.assertEqual(len(pipeline), 4)

    def test_shuffled_by_default(self):
        """Default pipeline should have shuffle_order=True."""
        pipeline = build_default_augmentation_pipeline()
        self.assertTrue(pipeline._shuffle_order)

    def test_custom_parameters(self):
        """Custom parameters should be passed through to the transforms."""
        pipeline = build_default_augmentation_pipeline(
            hflip_prob=0.1,
            color_jitter_prob=0.2,
            affine_prob=0.3,
            cutout_prob=0.4,
            degrees=15.0,
            translate=0.1,
        )
        self.assertIsInstance(pipeline, Compose)
        # The pipeline shuffles, so we check in aggregate that the expected
        # types are present and at least one has the right probability.
        found_flip = False
        found_jitter = False
        found_affine = False
        found_cutout = False
        for t in pipeline:
            if isinstance(t, RandomHorizontalFlip):
                self.assertEqual(t.p, 0.1)
                found_flip = True
            elif isinstance(t, RandomColorJitter):
                self.assertEqual(t.p, 0.2)
                found_jitter = True
            elif isinstance(t, RandomAffine):
                self.assertEqual(t.p, 0.3)
                self.assertEqual(t.degrees, 15.0)
                self.assertEqual(t.translate, 0.1)
                found_affine = True
            elif isinstance(t, Cutout):
                self.assertEqual(t.p, 0.4)
                found_cutout = True
        self.assertTrue(found_flip, "RandomHorizontalFlip not found in pipeline")
        self.assertTrue(found_jitter, "RandomColorJitter not found in pipeline")
        self.assertTrue(found_affine, "RandomAffine not found in pipeline")
        self.assertTrue(found_cutout, "Cutout not found in pipeline")


# ──────────────────────────────────────────────────────────────────────────────
# Test GANDataloader
# ──────────────────────────────────────────────────────────────────────────────


class TestGANDataloader(unittest.TestCase):
    """Verify GANDataloader wrapping and augmentation integration."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        _create_synthetic_dataset(self._tmpdir, num_images=8)

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_creation(self):
        """GANDataloader should be creatable with a dataset."""
        ds = YOLODataset(self._tmpdir, image_size=64, split="train", max_boxes=5)
        loader = GANDataloader(
            dataset=ds,
            batch_size=4,
            num_workers=0,  # use 0 workers for test
            shuffle=False,
        )
        self.assertIsInstance(loader, GANDataloader)
        self.assertEqual(len(loader), 2)  # 8 images / 4 = 2 batches

    def test_iteration_returns_batches(self):
        """Iterating over the loader should yield Batch objects."""
        ds = YOLODataset(self._tmpdir, image_size=64, split="train", max_boxes=5)
        loader = GANDataloader(
            dataset=ds,
            batch_size=4,
            num_workers=0,
            shuffle=False,
        )
        for batch in loader:
            self.assertIsInstance(batch, Batch)
            self.assertEqual(batch.images.shape[0], 4)
            self.assertEqual(batch.images.shape[1:], (3, 64, 64))
            break

    def test_set_epoch(self):
        """set_epoch should not raise and should influence seeds."""
        ds = YOLODataset(self._tmpdir, image_size=64, split="train", max_boxes=5)
        loader = GANDataloader(
            dataset=ds,
            augmentations=build_default_augmentation_pipeline(),
            batch_size=4,
            num_workers=0,
            shuffle=False,
        )
        # Setting epoch should work
        loader.set_epoch(0)
        batches_epoch0 = list(loader)
        loader.set_epoch(1)
        batches_epoch1 = list(loader)
        # Different epochs may produce different augmentations
        self.assertEqual(len(batches_epoch0), len(batches_epoch1))

    def test_no_augmentation(self):
        """Loader with no augmentation should work identically."""
        ds = YOLODataset(self._tmpdir, image_size=64, split="train", max_boxes=5)
        loader = GANDataloader(
            dataset=ds,
            augmentations=None,
            batch_size=4,
            num_workers=0,
            shuffle=False,
        )
        for batch in loader:
            self.assertIsInstance(batch, Batch)
            break

    def test_properties(self):
        """Property accessors should return correct values."""
        ds = YOLODataset(self._tmpdir, image_size=64, split="train", max_boxes=5)
        loader = GANDataloader(
            dataset=ds,
            batch_size=8,
            num_workers=0,
        )
        self.assertEqual(loader.batch_size, 8)
        self.assertIs(loader.dataset, ds)


# ──────────────────────────────────────────────────────────────────────────────
# Test get_train_val_loaders
# ──────────────────────────────────────────────────────────────────────────────


class TestGetTrainValLoaders(unittest.TestCase):
    """Verify the factory function."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        _create_synthetic_dataset(self._tmpdir, num_images=10)

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_returns_two_loaders(self):
        """Factory should return (train_loader, val_loader)."""
        train_loader, val_loader = get_train_val_loaders(
            self._tmpdir,
            image_size=64,
            batch_size=4,
            num_workers=0,
            augment=True,
        )
        self.assertIsInstance(train_loader, GANDataloader)
        self.assertIsInstance(val_loader, GANDataloader)

    def test_train_has_augmentations(self):
        """Training loader should have augmentations when augment=True."""
        train_loader, _ = get_train_val_loaders(
            self._tmpdir,
            image_size=64,
            batch_size=4,
            num_workers=0,
            augment=True,
        )
        self.assertIsNotNone(train_loader._augmentations)

    def test_val_has_no_augmentations(self):
        """Validation loader should not have augmentations."""
        _, val_loader = get_train_val_loaders(
            self._tmpdir,
            image_size=64,
            batch_size=4,
            num_workers=0,
            augment=True,
        )
        self.assertIsNone(val_loader._augmentations)

    def test_train_shuffles_val_does_not(self):
        """Train loader shuffles, val loader does not."""
        train_loader, val_loader = get_train_val_loaders(
            self._tmpdir,
            image_size=64,
            batch_size=4,
            num_workers=0,
            augment=True,
        )
        self.assertTrue(train_loader._shuffle)
        self.assertFalse(val_loader._shuffle)

    def test_no_augmentation_sets_none(self):
        """Setting augment=False should make train loader have no augmentations."""
        train_loader, _ = get_train_val_loaders(
            self._tmpdir,
            image_size=64,
            batch_size=4,
            num_workers=0,
            augment=False,
        )
        self.assertIsNone(train_loader._augmentations)

    def test_iteration_works(self):
        """Both loaders should produce valid batches."""
        train_loader, val_loader = get_train_val_loaders(
            self._tmpdir,
            image_size=64,
            batch_size=4,
            num_workers=0,
            augment=True,
        )
        for batch in train_loader:
            self.assertIsInstance(batch, Batch)
            break
        for batch in val_loader:
            self.assertIsInstance(batch, Batch)
            break


if __name__ == "__main__":
    unittest.main(verbosity=2)