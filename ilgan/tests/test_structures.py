"""
Tests for ``ilgan.data.structures``.

Verifies:
- ``Sample`` creation, validation, property access, and device movement.
- ``Batch.collate()`` with varying box counts, padding correctness, and edge
  cases (empty boxes, truncation via ``global_max_boxes``).
- ``parse_yolo_label`` on synthetic label files (normal, empty, missing,
  malformed).
- ``DatasetMetadata`` construction and validation.
"""

import os
import tempfile
import unittest

import torch

from ilgan.data.structures import (
    Sample,
    Batch,
    DatasetMetadata,
    parse_yolo_label,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_image(h: int = 64, w: int = 64) -> torch.Tensor:
    """Create a dummy image tensor in [-1, 1]."""
    return torch.rand(3, h, w) * 2.0 - 1.0


def _make_boxes(n: int) -> torch.Tensor:
    """Create *n* random valid bounding boxes in [0, 1]."""
    return torch.rand(n, 4).float()


def _make_labels(n: int, num_classes: int = 10) -> torch.Tensor:
    """Create *n* random class IDs."""
    return torch.randint(0, num_classes, (n,))


def _make_sample(n: int) -> Sample:
    """Create a Sample with *n* random boxes."""
    return Sample(
        image=_make_image(),
        boxes=_make_boxes(n),
        labels=_make_labels(n),
        valid_mask=torch.ones(n, dtype=torch.bool),
        image_path="/fake/path.png",
        metadata={"split": "train"},
    )


def _write_yolo(path: str, lines: list) -> None:
    """Write *lines* (list of strings) to *path*."""
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Test Sample
# ──────────────────────────────────────────────────────────────────────────────


class TestSampleCreation(unittest.TestCase):
    """Verify Sample creation and validation."""

    def test_valid_sample(self):
        """A default valid Sample should not raise."""
        s = _make_sample(3)
        self.assertEqual(s.num_boxes, 3)
        self.assertEqual(s.height, 64)
        self.assertEqual(s.width, 64)

    def test_empty_boxes(self):
        """A Sample with zero boxes is valid."""
        s = _make_sample(0)
        self.assertEqual(s.num_boxes, 0)
        self.assertEqual(s.boxes.shape, (0, 4))

    def test_invalid_image_channels(self):
        """Image must have 3 channels."""
        with self.assertRaises(ValueError):
            Sample(
                image=torch.rand(1, 64, 64),
                boxes=torch.zeros((0, 4)),
                labels=torch.zeros(0, dtype=torch.long),
                valid_mask=torch.zeros(0, dtype=torch.bool),
            )

    def test_invalid_image_range(self):
        """Image values must be in [-1, 1]."""
        with self.assertRaises(ValueError):
            Sample(
                image=torch.ones(3, 64, 64) * 2.0,
                boxes=torch.zeros((0, 4)),
                labels=torch.zeros(0, dtype=torch.long),
                valid_mask=torch.zeros(0, dtype=torch.bool),
            )

    def test_boxes_wrong_dim(self):
        """Boxes must be [N, 4]."""
        with self.assertRaises(ValueError):
            Sample(
                image=_make_image(),
                boxes=torch.rand(3, 5),
                labels=torch.zeros(3, dtype=torch.long),
                valid_mask=torch.ones(3, dtype=torch.bool),
            )

    def test_labels_wrong_size(self):
        """Labels must have same N as boxes."""
        with self.assertRaises(ValueError):
            Sample(
                image=_make_image(),
                boxes=torch.rand(3, 4),
                labels=torch.zeros(4, dtype=torch.long),
                valid_mask=torch.ones(3, dtype=torch.bool),
            )

    def test_valid_mask_not_bool(self):
        """valid_mask must be bool."""
        with self.assertRaises(ValueError):
            Sample(
                image=_make_image(),
                boxes=torch.rand(2, 4),
                labels=torch.zeros(2, dtype=torch.long),
                valid_mask=torch.ones(2, dtype=torch.float32),
            )

    def test_box_coordinates_out_of_range(self):
        """Box coordinates must be in [0, 1] for valid entries."""
        boxes = torch.rand(2, 4)
        boxes[0, 0] = 1.5  # invalid
        with self.assertRaises(ValueError):
            Sample(
                image=_make_image(),
                boxes=boxes,
                labels=torch.zeros(2, dtype=torch.long),
                valid_mask=torch.ones(2, dtype=torch.bool),
            )

    def test_to_device(self):
        """Sample.to() moves tensors to the target device."""
        s = _make_sample(3)
        if torch.cuda.is_available():
            s2 = s.to(torch.device("cuda:0"))
            self.assertTrue(s2.image.is_cuda)
            self.assertTrue(s2.boxes.is_cuda)
        else:
            s2 = s.to(torch.device("cpu"))
            self.assertFalse(s2.image.is_cuda)

    def test_repr(self):
        """String representation should contain key info."""
        s = _make_sample(3)
        r = repr(s)
        self.assertIn("Sample(", r)
        self.assertIn("(3,", r)  # 3 boxes

    def test_num_boxes_property(self):
        """num_boxes returns count of valid entries."""
        s = _make_sample(5)
        self.assertEqual(s.num_boxes, 5)
        # Manually mask one entry
        s.valid_mask[2] = False
        self.assertEqual(s.num_boxes, 4)


# ──────────────────────────────────────────────────────────────────────────────
# Test Batch
# ──────────────────────────────────────────────────────────────────────────────


class TestBatchCollation(unittest.TestCase):
    """Verify Batch.collate() correctness."""

    def test_basic_collation(self):
        """Collate 4 samples with varying box counts."""
        samples = [_make_sample(i) for i in [2, 5, 0, 3]]
        batch = Batch.collate(samples)
        self.assertEqual(batch.batch_size, 4)
        # max_boxes = 5
        self.assertEqual(batch.max_boxes_in_batch, 5)
        self.assertEqual(batch.images.shape, (4, 3, 64, 64))
        self.assertEqual(batch.boxes.shape, (4, 5, 4))
        self.assertEqual(batch.labels.shape, (4, 5))
        self.assertEqual(batch.valid_mask.shape, (4, 5))

        # Verify padding for sample at index 0 (2 boxes → first 2 valid, rest -1)
        # Boxes: first 2 should not be -1, last 3 should be -1
        for idx_in_batch, n in enumerate([2, 5, 0, 3]):
            masks = batch.valid_mask[idx_in_batch]
            self.assertEqual(masks[:n].sum().item(), n,
                             f"Sample {idx_in_batch} with {n} boxes")
            if n < 5:
                self.assertEqual(masks[n:].sum().item(), 0,
                                 f"Sample {idx_in_batch} padding check")

    def test_invalid_boxes_are_minus_one(self):
        """Padding entries in boxes should be -1.0."""
        samples = [_make_sample(2), _make_sample(4)]
        batch = Batch.collate(samples)
        # Sample 0 has 2 boxes → indices 2,3 should be -1
        self.assertTrue((batch.boxes[0, 2:] == -1.0).all())
        # Sample 1 has 4 boxes → all valid
        self.assertTrue((batch.boxes[1] != -1.0).all())

    def test_invalid_labels_are_minus_one(self):
        """Padding entries in labels should be -1."""
        samples = [_make_sample(2), _make_sample(4)]
        batch = Batch.collate(samples)
        self.assertTrue((batch.labels[0, 2:] == -1).all())

    def test_empty_list_raises(self):
        """Collation of an empty list should raise."""
        with self.assertRaises(ValueError):
            Batch.collate([])

    def test_inconsistent_image_sizes_raises(self):
        """Samples with different spatial sizes should raise."""
        s1 = _make_sample(1)  # 64x64
        s2 = Sample(
            image=torch.rand(3, 128, 128) * 2.0 - 1.0,
            boxes=torch.rand(1, 4),
            labels=torch.zeros(1, dtype=torch.long),
            valid_mask=torch.ones(1, dtype=torch.bool),
        )
        with self.assertRaises(ValueError):
            Batch.collate([s1, s2])

    def test_global_max_boxes_truncates(self):
        """global_max_boxes limits per-sample box count."""
        samples = [_make_sample(10), _make_sample(5)]
        batch = Batch.collate(samples, global_max_boxes=6)
        self.assertEqual(batch.max_boxes_in_batch, 6)
        # Sample 0 had 10 boxes, truncated to 6
        # The first 6 entries are valid in sample 0
        self.assertEqual(batch.valid_mask[0].sum().item(), 6)
        # Sample 1 had 5 boxes, all kept
        self.assertEqual(batch.valid_mask[1].sum().item(), 5)

    def test_image_paths_preserved(self):
        """image_paths list should match input order."""
        samples = [
            _make_sample(1),
            _make_sample(2),
        ]
        samples[0].image_path = "/path/a.png"
        samples[1].image_path = "/path/b.png"
        batch = Batch.collate(samples)
        self.assertEqual(batch.image_paths, ["/path/a.png", "/path/b.png"])

    def test_metadata_preserved(self):
        """Metadata dicts should match input order."""
        samples = [
            _make_sample(1),
            _make_sample(2),
        ]
        samples[0].metadata = {"key": "first"}
        samples[1].metadata = {"key": "second"}
        batch = Batch.collate(samples)
        self.assertEqual(batch.metadata[0]["key"], "first")
        self.assertEqual(batch.metadata[1]["key"], "second")

    def test_batch_repr(self):
        """String representation of Batch."""
        samples = [_make_sample(2), _make_sample(3)]
        batch = Batch.collate(samples)
        r = repr(batch)
        self.assertIn("Batch(", r)
        self.assertIn("B=2", r)

    def test_batch_len(self):
        """len(batch) returns batch_size."""
        samples = [_make_sample(1) for _ in range(7)]
        batch = Batch.collate(samples)
        self.assertEqual(len(batch), 7)

    def test_batch_to_device(self):
        """Batch.to() moves all tensors."""
        samples = [_make_sample(2), _make_sample(3)]
        batch = Batch.collate(samples)
        if torch.cuda.is_available():
            batch2 = batch.to(torch.device("cuda:0"))
            self.assertTrue(batch2.images.is_cuda)
            self.assertTrue(batch2.boxes.is_cuda)
        else:
            batch2 = batch.to(torch.device("cpu"))
            self.assertFalse(batch2.images.is_cuda)


# ──────────────────────────────────────────────────────────────────────────────
# Test parse_yolo_label
# ──────────────────────────────────────────────────────────────────────────────


class TestParseYOLOLabel(unittest.TestCase):
    """Verify YOLO label parsing."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        for f in os.listdir(self._tmpdir):
            os.remove(os.path.join(self._tmpdir, f))
        os.rmdir(self._tmpdir)

    def _path(self, name: str = "labels.txt") -> str:
        return os.path.join(self._tmpdir, name)

    def test_three_boxes(self):
        """Parse a file with 3 valid boxes."""
        lines = [
            "0 0.5 0.5 0.2 0.3",
            "1 0.1 0.2 0.05 0.1",
            "2 0.9 0.8 0.15 0.25",
        ]
        _write_yolo(self._path(), lines)
        boxes, labels, valid_mask = parse_yolo_label(self._path())
        self.assertEqual(boxes.shape, (3, 4))
        self.assertEqual(labels.shape, (3,))
        self.assertEqual(valid_mask.shape, (3,))
        self.assertTrue(valid_mask.all())
        # Verify values
        self.assertAlmostEqual(boxes[0, 0].item(), 0.5)
        self.assertAlmostEqual(boxes[0, 1].item(), 0.5)
        self.assertAlmostEqual(boxes[0, 2].item(), 0.2)
        self.assertAlmostEqual(boxes[0, 3].item(), 0.3)
        self.assertEqual(labels[0].item(), 0)
        self.assertEqual(labels[1].item(), 1)
        self.assertEqual(labels[2].item(), 2)

    def test_file_not_found(self):
        """Missing file returns empty tensors."""
        boxes, labels, valid_mask = parse_yolo_label("/nonexistent/path.txt")
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(labels.shape, (0,))
        self.assertEqual(valid_mask.shape, (0,))

    def test_empty_file(self):
        """Empty file returns empty tensors."""
        _write_yolo(self._path(), [])
        boxes, labels, valid_mask = parse_yolo_label(self._path())
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(labels.shape, (0,))

    def test_file_with_blank_lines(self):
        """Blank lines are skipped."""
        lines = [
            "0 0.5 0.5 0.2 0.3",
            "",
            "   ",
            "1 0.1 0.2 0.05 0.1",
        ]
        _write_yolo(self._path(), lines)
        boxes, labels, valid_mask = parse_yolo_label(self._path())
        self.assertEqual(boxes.shape, (2, 4))

    def test_malformed_line_raises(self):
        """A line with wrong number of fields raises ValueError."""
        lines = ["0 0.5 0.5 0.2"]  # only 4 fields
        _write_yolo(self._path(), lines)
        with self.assertRaises(ValueError):
            parse_yolo_label(self._path())

    def test_non_numeric_raises(self):
        """A non-numeric token raises ValueError."""
        lines = ["0 abc 0.5 0.2 0.3"]
        _write_yolo(self._path(), lines)
        with self.assertRaises(ValueError):
            parse_yolo_label(self._path())

    def test_negative_class_id_raises(self):
        """Negative class_id raises ValueError."""
        lines = ["-1 0.5 0.5 0.2 0.3"]
        _write_yolo(self._path(), lines)
        with self.assertRaises(ValueError):
            parse_yolo_label(self._path())

    def test_coordinate_out_of_range_raises(self):
        """x_center > 1.0 raises ValueError."""
        lines = ["0 1.5 0.5 0.2 0.3"]
        _write_yolo(self._path(), lines)
        with self.assertRaises(ValueError):
            parse_yolo_label(self._path())

    def test_negative_width_raises(self):
        """Negative width raises ValueError."""
        lines = ["0 0.5 0.5 -0.2 0.3"]
        _write_yolo(self._path(), lines)
        with self.assertRaises(ValueError):
            parse_yolo_label(self._path())

    def test_zero_width_raises(self):
        """Zero width raises ValueError."""
        lines = ["0 0.5 0.5 0.0 0.3"]
        _write_yolo(self._path(), lines)
        with self.assertRaises(ValueError):
            parse_yolo_label(self._path())


# ──────────────────────────────────────────────────────────────────────────────
# Test DatasetMetadata
# ──────────────────────────────────────────────────────────────────────────────


class TestDatasetMetadata(unittest.TestCase):
    """Verify DatasetMetadata construction and validation."""

    def test_valid_metadata(self):
        """A standard metadata object works."""
        meta = DatasetMetadata(
            class_names=["person", "car", "dog"],
            class_to_idx={"person": 0, "car": 1, "dog": 2},
            num_classes=3,
            train_size=1000,
            val_size=200,
            image_size=(128, 128),
        )
        self.assertEqual(meta.num_classes, 3)
        self.assertEqual(meta.train_size, 1000)
        self.assertEqual(meta.val_size, 200)
        self.assertEqual(meta.total_size, 1200)
        self.assertEqual(meta.image_size, (128, 128))
        self.assertEqual(meta.idx_to_name(1), "car")
        self.assertEqual(meta.name_to_idx("dog"), 2)

    def test_class_names_length_mismatch(self):
        """Mismatch between class_names length and num_classes raises."""
        with self.assertRaises(ValueError):
            DatasetMetadata(
                class_names=["a", "b"],
                class_to_idx={"a": 0, "b": 1},
                num_classes=3,  # mismatch
                train_size=100,
                val_size=20,
                image_size=(64, 64),
            )

    def test_class_to_idx_size_mismatch(self):
        """Mismatch between class_to_idx size and num_classes raises."""
        with self.assertRaises(ValueError):
            DatasetMetadata(
                class_names=["a", "b"],
                class_to_idx={"a": 0},  # only 1
                num_classes=2,
                train_size=100,
                val_size=20,
                image_size=(64, 64),
            )

    def test_negative_train_size_raises(self):
        """Negative train_size raises."""
        with self.assertRaises(ValueError):
            DatasetMetadata(
                class_names=["a"],
                class_to_idx={"a": 0},
                num_classes=1,
                train_size=-1,
                val_size=0,
                image_size=(64, 64),
            )

    def test_image_size_must_be_tuple_of_two(self):
        """image_size must be a 2-tuple."""
        with self.assertRaises(ValueError):
            DatasetMetadata(
                class_names=["a"],
                class_to_idx={"a": 0},
                num_classes=1,
                train_size=10,
                val_size=0,
                image_size=(64, 64, 3),  # wrong length
            )


if __name__ == "__main__":
    unittest.main()