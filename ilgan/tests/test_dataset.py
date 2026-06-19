"""
Tests for ``ilgan.data.dataset``.

Verifies:
- ``resize_with_pad`` utility produces correctly shaped tensors, masks,
  scale factors, and pad amounts.
- ``YOLODataset`` initialises correctly with synthetic data.
- ``__len__`` and ``__getitem__`` return properly shaped ``Sample`` objects.
- Image values are in ``[-1, 1]``.
- Box coordinates are in ``[0, 1]``.
- Padding mask is correctly formed.
- Max-box truncation and padding work.
- Split file vs automatic split logic.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ilgan.data.dataset import YOLODataset, resize_with_pad
from ilgan.data.structures import Sample


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _create_synthetic_dataset(
    root_dir: str,
    num_images: int = 5,
    img_size: tuple = (100, 80),  # (width, height) — non-square
    num_classes: int = 3,
    boxes_per_image: int = 3,
) -> None:
    """Create a tiny synthetic dataset with random colour noise images and
    corresponding YOLO label files.

    Directory structure::

        root_dir/
        ├── images/       *.jpg
        └── labels/       *.txt
    """
    img_dir = Path(root_dir) / "images"
    label_dir = Path(root_dir) / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    w, h = img_size

    for i in range(num_images):
        stem = f"img_{i:04d}"

        # ── random image ────────────────────────────────────────────────
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        pil_img = Image.fromarray(arr)
        pil_img.save(str(img_dir / f"{stem}.jpg"), quality=85)

        # ── random YOLO labels ──────────────────────────────────────────
        # Each line: class_id xc yc bw bh  (all normalised [0,1])
        lines = []
        n_boxes = np.random.randint(1, boxes_per_image + 1)
        for _ in range(n_boxes):
            cls = np.random.randint(0, num_classes)
            xc = np.random.uniform(0.1, 0.9)
            yc = np.random.uniform(0.1, 0.9)
            bw = np.random.uniform(0.05, 0.4)
            bh = np.random.uniform(0.05, 0.4)
            # Ensure box stays within image bounds
            xc = min(xc, 1.0 - bw / 2)
            yc = min(yc, 1.0 - bh / 2)
            xc = max(xc, bw / 2)
            yc = max(yc, bh / 2)
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        with open(str(label_dir / f"{stem}.txt"), "w") as f:
            for line in lines:
                f.write(line + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Test resize_with_pad
# ──────────────────────────────────────────────────────────────────────────────


class TestResizeWithPad(unittest.TestCase):
    """Unit tests for the ``resize_with_pad`` utility."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def _image(self, w: int = 200, h: int = 100) -> Image.Image:
        """Create a solid-color test image."""
        arr = np.full((h, w, 3), 128, dtype=np.uint8)
        return Image.fromarray(arr)

    def test_output_shape(self):
        """Output tensor should be [3, target_size, target_size]."""
        tensor, mask, scale, pad = resize_with_pad(self._image(), 64)
        self.assertEqual(tensor.shape, (3, 64, 64))
        self.assertEqual(mask.shape, (1, 64, 64))
        self.assertIsInstance(scale, float)
        self.assertIsInstance(pad, tuple)
        self.assertEqual(len(pad), 2)

    def test_value_range(self):
        """Output tensor values should be in [-1, 1]."""
        tensor, *_ = resize_with_pad(self._image(), 64)
        self.assertGreaterEqual(tensor.min().item(), -1.0)
        self.assertLessEqual(tensor.max().item(), 1.0)

    def test_mask_correctness_horizontal_padding(self):
        """Landscape image: padding is on top/bottom.

        image is 200x100 (w x h), longer side = 200 → target_size = 64.
        new_w = 64, new_h = int(round(100 * 64/200)) = 32.
        pad_top = (64 - 32) // 2 = 16, pad_left = 0.
        """
        tensor, mask, scale, (pl, pt) = resize_with_pad(self._image(200, 100), 64)
        self.assertEqual(pl, 0)       # no horizontal padding
        self.assertEqual(pt, 16)      # vertical padding
        # Image region
        self.assertTrue(mask[0, 16:48, :].all())
        # Padded regions
        self.assertFalse(mask[0, :16, :].any())
        self.assertFalse(mask[0, 48:, :].any())

    def test_mask_correctness_vertical_padding(self):
        """Portrait image: padding is on left/right.

        image is 100x200 (w x h), longer side = 200 → target_size = 64.
        new_w = int(round(100 * 64/200)) = 32, new_h = 64.
        pad_left = (64 - 32) // 2 = 16, pad_top = 0.
        """
        tensor, mask, scale, (pl, pt) = resize_with_pad(self._image(100, 200), 64)
        self.assertEqual(pl, 16)      # horizontal padding
        self.assertEqual(pt, 0)       # no vertical padding
        # Image region
        self.assertTrue(mask[0, :, 16:48].all())
        # Padded regions
        self.assertFalse(mask[0, :, :16].any())
        self.assertFalse(mask[0, :, 48:].any())

    def test_square_image(self):
        """Square image: no padding needed."""
        tensor, mask, scale, (pl, pt) = resize_with_pad(self._image(128, 128), 64)
        self.assertEqual(pl, 0)
        self.assertEqual(pt, 0)
        self.assertTrue(mask.all())

    def test_scale_factor(self):
        """Scale factor should be target_size / longer_side."""
        _, _, scale, _ = resize_with_pad(self._image(200, 100), 64)
        self.assertAlmostEqual(scale, 64.0 / 200.0)
        _, _, scale, _ = resize_with_pad(self._image(100, 200), 128)
        self.assertAlmostEqual(scale, 128.0 / 200.0)

    def test_greyscale_input_is_converted(self):
        """Greyscale images should be converted to RGB."""
        grey = Image.fromarray(np.full((50, 100), 128, dtype=np.uint8), mode="L")
        tensor, mask, scale, pad = resize_with_pad(grey, 64)
        self.assertEqual(tensor.shape[0], 3)  # RGB channels

    def test_padding_mask_dtype(self):
        """Padding mask should be bool."""
        _, mask, *_ = resize_with_pad(self._image(200, 100), 64)
        self.assertEqual(mask.dtype, torch.bool)


# ──────────────────────────────────────────────────────────────────────────────
# Test YOLODataset
# ──────────────────────────────────────────────────────────────────────────────


class TestYOLODatasetInit(unittest.TestCase):
    """Verify dataset construction."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        _create_synthetic_dataset(self._tmpdir, num_images=5)

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_default_train_split(self):
        """Default split='train' should create a non-empty dataset."""
        ds = YOLODataset(self._tmpdir, image_size=64, split="train", max_boxes=10)
        self.assertGreater(len(ds), 0)

    def test_val_split(self):
        """split='val' should create a non-empty dataset."""
        ds = YOLODataset(self._tmpdir, image_size=64, split="val", max_boxes=10)
        self.assertGreater(len(ds), 0)

    def test_total_samples_matches_split(self):
        """Train + val should equal total images (without split files)."""
        ds_train = YOLODataset(self._tmpdir, image_size=64, split="train", max_boxes=10)
        ds_val = YOLODataset(self._tmpdir, image_size=64, split="val", max_boxes=10)
        self.assertEqual(len(ds_train) + len(ds_val), 5)

    def test_split_file_overrides_auto_split(self):
        """If train.txt exists, it should be used instead of auto-split."""
        # Create train.txt with 3 specific images
        train_stems = ["img_0000", "img_0001", "img_0002"]
        with open(os.path.join(self._tmpdir, "train.txt"), "w") as f:
            for s in train_stems:
                f.write(s + ".jpg\n")
        ds = YOLODataset(self._tmpdir, image_size=64, split="train", max_boxes=10)
        self.assertEqual(len(ds), 3)

    def test_val_split_file(self):
        """If val.txt exists, it should be used."""
        val_stems = ["img_0003", "img_0004"]
        with open(os.path.join(self._tmpdir, "val.txt"), "w") as f:
            for s in val_stems:
                f.write(s + "\n")
        ds = YOLODataset(self._tmpdir, image_size=64, split="val", max_boxes=10)
        self.assertEqual(len(ds), 2)

    def test_invalid_split_raises(self):
        """An unknown split name should raise ValueError."""
        with self.assertRaises(ValueError):
            YOLODataset(self._tmpdir, image_size=64, split="invalid", max_boxes=10)

    def test_nonexistent_root_raises(self):
        """A non-existent root_dir should raise NotADirectoryError."""
        with self.assertRaises(NotADirectoryError):
            YOLODataset("/nonexistent/path", image_size=64, split="train", max_boxes=10)

    def test_properties(self):
        """Check property accessors."""
        ds = YOLODataset(self._tmpdir, image_size=128, split="val", max_boxes=20)
        self.assertEqual(ds.image_size, 128)
        self.assertEqual(ds.split, "val")
        self.assertEqual(ds.max_boxes, 20)
        self.assertEqual(ds.root_dir, self._tmpdir)

    def test_repr(self):
        """Check string representation."""
        ds = YOLODataset(self._tmpdir, image_size=64, split="train", max_boxes=10)
        r = repr(ds)
        self.assertIn("YOLODataset(", r)
        self.assertIn("train", r)


# ──────────────────────────────────────────────────────────────────────────────
# Test YOLODataset __getitem__
# ──────────────────────────────────────────────────────────────────────────────


class TestYOLODatasetGetItem(unittest.TestCase):
    """Verify sample loading correctness."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        _create_synthetic_dataset(
            self._tmpdir, num_images=5,
            img_size=(120, 90),   # landscape
            boxes_per_image=4,
        )
        self._ds = YOLODataset(
            self._tmpdir, image_size=64,
            split="train", max_boxes=10,
        )

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_returns_sample(self):
        """__getitem__ should return a Sample."""
        sample = self._ds[0]
        self.assertIsInstance(sample, Sample)

    def test_image_shape(self):
        """Image should be [3, 64, 64]."""
        sample = self._ds[0]
        self.assertEqual(sample.image.shape, (3, 64, 64))

    def test_image_range(self):
        """Image values should be in [-1, 1]."""
        sample = self._ds[0]
        self.assertGreaterEqual(sample.image.min().item(), -1.0)
        self.assertLessEqual(sample.image.max().item(), 1.0)

    def test_boxes_shape(self):
        """Boxes should be [max_boxes, 4]."""
        sample = self._ds[0]
        self.assertEqual(sample.boxes.shape, (10, 4))  # max_boxes=10

    def test_boxes_range_valid(self):
        """Valid box coordinates should be in [0, 1]."""
        sample = self._ds[0]
        valid = sample.valid_mask
        if valid.any():
            valid_boxes = sample.boxes[valid]
            self.assertGreaterEqual(valid_boxes.min().item(), 0.0)
            self.assertLessEqual(valid_boxes.max().item(), 1.0)

    def test_padding_boxes_are_minus_one(self):
        """Invalid (padded) box entries should be -1.0."""
        sample = self._ds[0]
        invalid = ~sample.valid_mask
        if invalid.any():
            self.assertTrue((sample.boxes[invalid] == -1.0).all())

    def test_padding_labels_are_minus_one(self):
        """Invalid label entries should be -1."""
        sample = self._ds[0]
        invalid = ~sample.valid_mask
        if invalid.any():
            self.assertTrue((sample.labels[invalid] == -1).all())

    def test_labels_shape(self):
        """Labels should be [max_boxes]."""
        sample = self._ds[0]
        self.assertEqual(sample.labels.shape, (10,))

    def test_valid_mask_shape(self):
        """Valid mask should be [max_boxes]."""
        sample = self._ds[0]
        self.assertEqual(sample.valid_mask.shape, (10,))

    def test_padding_mask_in_metadata(self):
        """Padding mask should be stored in metadata."""
        sample = self._ds[0]
        self.assertIn("padding_mask", sample.metadata)
        pm = sample.metadata["padding_mask"]
        self.assertEqual(pm.shape, (1, 64, 64))
        self.assertEqual(pm.dtype, torch.bool)

    def test_orig_size_in_metadata(self):
        """Original image dimensions should be stored."""
        sample = self._ds[0]
        self.assertIn("orig_size", sample.metadata)
        self.assertEqual(sample.metadata["orig_size"], (120, 90))

    def test_image_path(self):
        """image_path should point to an existing file."""
        sample = self._ds[0]
        self.assertTrue(os.path.isfile(sample.image_path))

    def test_num_boxes_property(self):
        """num_boxes should reflect valid boxes in the sample."""
        sample = self._ds[0]
        self.assertGreaterEqual(sample.num_boxes, 0)
        self.assertLessEqual(sample.num_boxes, 10)
        self.assertEqual(sample.num_boxes, sample.valid_mask.sum().item())

    def test_max_boxes_truncation(self):
        """If a sample has more boxes than max_boxes, truncation should
        occur."""
        # Use a very small max_boxes to force truncation
        ds = YOLODataset(
            self._tmpdir, image_size=64,
            split="train", max_boxes=2,
        )
        sample = ds[0]
        self.assertLessEqual(sample.num_boxes, 2)
        self.assertEqual(sample.boxes.shape[0], 2)
        # There might be more boxes in the original labels (up to 4),
        # so valid_mask sum should be <= 2
        self.assertLessEqual(sample.valid_mask.sum().item(), 2)

    def test_empty_label_file(self):
        """If a label file is missing, boxes/labels should be all padding."""
        # Create an image with no corresponding label file
        img_dir = Path(self._tmpdir) / "images"
        label_dir = Path(self._tmpdir) / "labels"

        # Remove a label file manually
        label_file = label_dir / "img_0000.txt"
        if label_file.is_file():
            os.remove(label_file)

        ds = YOLODataset(
            self._tmpdir, image_size=64,
            split="train", max_boxes=10,
        )
        # The sample with missing labels should still be loaded
        for i in range(len(ds)):
            sample = ds[i]
            self.assertIsInstance(sample, Sample)
            self.assertEqual(sample.image.shape, (3, 64, 64))


# ──────────────────────────────────────────────────────────────────────────────
# Test integration with structures
# ──────────────────────────────────────────────────────────────────────────────


class TestDatasetBatchCollation(unittest.TestCase):
    """Verify that samples from YOLODataset can be collated into a Batch."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        _create_synthetic_dataset(
            self._tmpdir, num_images=8,
            img_size=(100, 100),
            boxes_per_image=3,
        )
        self._ds = YOLODataset(
            self._tmpdir, image_size=64,
            split="train", max_boxes=10,
        )

    def tearDown(self):
        shutil.rmtree(self._tmpdir)

    def test_collate(self):
        """Samples from the dataset should be collatable."""
        from ilgan.data.structures import Batch

        samples = [self._ds[i] for i in range(min(4, len(self._ds)))]
        batch = Batch.collate(samples, global_max_boxes=10)
        self.assertEqual(batch.batch_size, len(samples))
        self.assertEqual(batch.images.shape, (len(samples), 3, 64, 64))
        self.assertEqual(batch.boxes.shape, (len(samples), 10, 4))
        self.assertEqual(batch.labels.shape, (len(samples), 10))
        self.assertEqual(batch.valid_mask.shape, (len(samples), 10))


if __name__ == "__main__":
    unittest.main(verbosity=2)