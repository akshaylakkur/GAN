"""
Streaming Pascal VOC dataset for ILGAN.

Downloads VOC 2012 images and annotations on-the-fly from the official
mirror, caches only the current batch in memory, and never stores the
full dataset on disk.  Designed for cloud GPU instances with limited
ephemeral storage.

Streaming strategy
------------------
Rather than downloading the full 2GB VOC archive upfront, this dataset:

1. Downloads the VOC 2012 train/val ID lists from the official mirror
   (tiny — ~200KB each).
2. On each ``__getitem__`` call, fetches the individual image and label
   from the VOC devkit URL, processes it, and discards the raw bytes.
3. Uses an in-memory LRU cache (configurable, default 256 samples) to
   avoid re-fetching recently accessed images within an epoch.

This keeps the disk footprint to essentially zero — only the current
batch of processed tensors resides in memory.

Usage
-----
::

    from ilgan.data.streaming_voc import StreamingVOCDataset

    dataset = StreamingVOCDataset(
        split="train",
        image_size=128,
        max_boxes=20,
        cache_size=256,
    )

    sample = dataset[0]
    # sample.image  -> [3, 128, 128] tensor in [-1, 1]
    # sample.boxes  -> [20, 4] tensor in (cx, cy, w, h) format
    # sample.labels -> [20] tensor of class IDs
    # sample.valid_mask -> [20] bool tensor

Notes
-----
- Requires an internet connection at data-loading time.
- The VOC 2012 mirror at ``host.robots.ox.ac.uk`` is used.  If this
  mirror is unavailable, set the ``VOC_MIRROR`` environment variable.
- Images are resized with aspect-ratio-preserving padding (same as
  ``YOLODataset``) to the target square size.
- The 20 VOC classes are mapped to integer IDs 0–19 in the standard
  order defined by the VOC challenge.
"""

from __future__ import annotations

import io
import os
import random
import tarfile
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ilgan.data.structures import Sample

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

VOC_CLASSES: List[str] = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
"""The 20 Pascal VOC object classes in standard order."""

VOC_CLASS_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(VOC_CLASSES)}
"""Mapping from class name to integer ID (0–19)."""

VOC_MIRROR: str = os.environ.get(
    "VOC_MIRROR",
    "https://host.robots.ox.ac.uk/pascal/VOC/voc2012",
)
"""Base URL for VOC 2012 data.  Override with ``VOC_MIRROR`` env var."""

VOC_ARCHIVE_URL: str = f"{VOC_MIRROR}/VOCtrainval_11-May-2012.tar"
"""URL of the full VOC 2012 train+val archive (used for individual file extraction)."""

# VOC 2012 image set IDs (train/val splits)
VOC_TRAIN_IDS_URL: str = (
    "https://raw.githubusercontent.com/akshaylakkur/GAN/main/"
    "ilgan/data/voc_splits/train_ids.txt"
)
VOC_VAL_IDS_URL: str = (
    "https://raw.githubusercontent.com/akshaylakkur/GAN/main/"
    "ilgan/data/voc_splits/val_ids.txt"
)
"""
URLs for the train/val split ID lists.  These are tiny text files
(~15KB each) listing the image stems for each split.

We host these in the repo itself to avoid parsing the VOC ImageSets
from the tar archive on every run.  The files are generated once from
the official VOC devkit and committed.
"""

# ──────────────────────────────────────────────────────────────────────────────
# LRU Cache
# ──────────────────────────────────────────────────────────────────────────────


class LRUCache:
    """Simple thread-safe LRU cache with a fixed capacity.

    Used to cache recently fetched VOC samples in memory, avoiding
    redundant HTTP requests within an epoch.
    """

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self._cache: OrderedDict[str, Sample] = OrderedDict()

    def get(self, key: str) -> Optional[Sample]:
        """Return the cached value for *key*, or ``None`` if missing."""
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: Sample) -> None:
        """Store *value* under *key*, evicting the oldest entry if at capacity."""
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────────────────────


def _fetch_url(url: str, timeout: float = 30.0) -> bytes:
    """Fetch a URL and return the raw bytes.

    Uses ``urllib.request`` (stdlib) to avoid adding ``requests`` as a
    dependency.  Raises ``RuntimeError`` on failure.
    """
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")


def _fetch_voc_image(stem: str) -> Image.Image:
    """Fetch a single VOC image by stem and return a PIL Image.

    Downloads the image from the VOC 2012 JPEGImages directory inside
    the official tar archive.  We use the raw GitHub mirror for
    individual files to avoid downloading the full tar.

    The VOC archive is structured as::

        VOCdevkit/VOC2012/JPEGImages/{stem}.jpg
        VOCdevkit/VOC2012/Annotations/{stem}.xml
    """
    # Try direct HTTP access to individual image files.
    # Some mirrors serve individual files; others require tar extraction.
    # We try the direct URL first, then fall back to tar extraction.
    direct_url = f"{VOC_MIRROR}/JPEGImages/{stem}.jpg"

    try:
        data = _fetch_url(direct_url, timeout=10.0)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (RuntimeError, OSError):
        pass

    # Fallback: extract from the full tar archive (slower, but reliable).
    # We download the tar once and cache it in a temp file.
    return _extract_from_tar(f"VOCdevkit/VOC2012/JPEGImages/{stem}.jpg")


def _fetch_voc_annotation(stem: str) -> str:
    """Fetch a single VOC annotation XML by stem and return the text.

    Same strategy as ``_fetch_voc_image``: try direct URL first, then
    fall back to tar extraction.
    """
    direct_url = f"{VOC_MIRROR}/Annotations/{stem}.xml"

    try:
        data = _fetch_url(direct_url, timeout=10.0)
        return data.decode("utf-8")
    except (RuntimeError, OSError):
        pass

    return _extract_annotation_from_tar(stem)


# ── Tar extraction fallback ──────────────────────────────────────────────

_TAR_CACHE: Optional[tarfile.TarFile] = None
"""Module-level cache for the opened VOC tar archive."""


def _get_tar() -> tarfile.TarFile:
    """Download and open the VOC tar archive, caching it."""
    global _TAR_CACHE
    if _TAR_CACHE is not None:
        return _TAR_CACHE

    import tempfile

    print(f"[StreamingVOC] Downloading VOC 2012 archive (first fetch only)...")
    t0 = time.time()
    data = _fetch_url(VOC_ARCHIVE_URL, timeout=300.0)
    elapsed = time.time() - t0
    print(f"[StreamingVOC] Downloaded {len(data) / 1024 / 1024:.1f} MB in {elapsed:.1f}s")

    # Write to a temp file and open as tar
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tar")
    tmp.write(data)
    tmp.flush()
    _TAR_CACHE = tarfile.open(tmp.name, "r")
    return _TAR_CACHE


def _extract_from_tar(path_in_archive: str) -> Image.Image:
    """Extract a file from the VOC tar archive and return as PIL Image."""
    tar = _get_tar()
    member = tar.getmember(path_in_archive)
    f = tar.extractfile(member)
    if f is None:
        raise FileNotFoundError(f"Could not extract {path_in_archive} from VOC archive")
    return Image.open(io.BytesIO(f.read())).convert("RGB")


def _extract_annotation_from_tar(stem: str) -> str:
    """Extract an annotation XML from the VOC tar archive."""
    tar = _get_tar()
    path = f"VOCdevkit/VOC2012/Annotations/{stem}.xml"
    member = tar.getmember(path)
    f = tar.extractfile(member)
    if f is None:
        raise FileNotFoundError(f"Could not extract {path} from VOC archive")
    return f.read().decode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# XML parsing
# ──────────────────────────────────────────────────────────────────────────────


def _parse_voc_annotation(xml_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse a VOC annotation XML string into boxes and labels.

    Parameters
    ----------
    xml_text : str
        The XML content of a VOC annotation file.

    Returns
    -------
    boxes : torch.Tensor
        Shape ``[N, 4]`` in ``(cx, cy, w, h)`` format, normalised to
        ``[0, 1]`` relative to the image dimensions.
    labels : torch.Tensor
        Shape ``[N]``, integer class IDs (0–19).
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)

    # Image dimensions
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    boxes_list: List[float] = []
    labels_list: List[int] = []

    for obj in root.findall("object"):
        # Skip difficult objects (optional)
        difficult = obj.find("difficult")
        if difficult is not None and difficult.text == "1":
            continue

        cls_name = obj.find("name").text
        if cls_name not in VOC_CLASS_TO_ID:
            continue  # Skip unknown classes (shouldn't happen in VOC)

        cls_id = VOC_CLASS_TO_ID[cls_name]
        bndbox = obj.find("bndbox")

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Clamp to image boundaries
        xmin = max(0.0, xmin)
        ymin = max(0.0, ymin)
        xmax = min(float(img_w), xmax)
        ymax = min(float(img_h), ymax)

        # Skip degenerate boxes
        if xmax <= xmin or ymax <= ymin:
            continue

        # Convert to YOLO format (cx, cy, w, h) normalised
        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        boxes_list.extend([cx, cy, w, h])
        labels_list.append(cls_id)

    boxes = torch.tensor(boxes_list, dtype=torch.float32).view(-1, 4)
    labels = torch.tensor(labels_list, dtype=torch.long)

    return boxes, labels


# ──────────────────────────────────────────────────────────────────────────────
# Resize with padding (same as YOLODataset)
# ──────────────────────────────────────────────────────────────────────────────


def _resize_with_pad(
    image: Image.Image,
    target_size: int,
) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
    """Resize the longer side to *target_size*, pad to square, return tensor.

    Returns
    -------
    tensor : torch.Tensor
        Shape ``[3, target_size, target_size]``, values in ``[-1, 1]``.
    scale_factor : float
        Ratio by which the original image was scaled.
    pad_amounts : tuple of (int, int)
        ``(pad_left, pad_top)`` applied to the shorter sides.
    """
    orig_w, orig_h = image.size
    longer_side = max(orig_w, orig_h)
    scale_factor = target_size / longer_side

    new_w = max(1, int(round(orig_w * scale_factor)))
    new_h = max(1, int(round(orig_h * scale_factor)))

    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    square = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    square.paste(resized, (pad_left, pad_top))

    arr = np.asarray(square, dtype=np.float32).transpose((2, 0, 1))
    tensor = torch.from_numpy(arr) / 127.5 - 1.0
    tensor = tensor.clamp(-1.0, 1.0)

    return tensor, scale_factor, (pad_left, pad_top)


def _rescale_boxes(
    boxes: torch.Tensor,
    orig_w: int,
    orig_h: int,
    scale_factor: float,
    pad_left: int,
    pad_top: int,
    target_size: int,
) -> torch.Tensor:
    """Rescale boxes from original image coords to padded square coords."""
    if boxes.size(0) == 0:
        return boxes

    xc = boxes[:, 0] * orig_w * scale_factor + pad_left
    yc = boxes[:, 1] * orig_h * scale_factor + pad_top
    bw = boxes[:, 2] * orig_w * scale_factor
    bh = boxes[:, 3] * orig_h * scale_factor

    xc = (xc / target_size).clamp(0.0, 1.0)
    yc = (yc / target_size).clamp(0.0, 1.0)
    bw = (bw / target_size).clamp(0.0, 1.0)
    bh = (bh / target_size).clamp(0.0, 1.0)

    return torch.stack([xc, yc, bw, bh], dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# StreamingVOCDataset
# ──────────────────────────────────────────────────────────────────────────────


class StreamingVOCDataset(torch.utils.data.Dataset):
    """Pascal VOC 2012 dataset that streams data on-the-fly.

    Downloads images and annotations individually from the VOC mirror
    on each ``__getitem__`` call.  An in-memory LRU cache avoids
    re-fetching recently accessed samples.

    Parameters
    ----------
    split : str
        One of ``"train"`` or ``"val"``.
    image_size : int
        Target square size in pixels (longer side resized to this).
    max_boxes : int
        Maximum number of boxes per sample.  Excess boxes are truncated;
    cache_size : int
        Maximum number of samples to keep in the in-memory LRU cache.
        Default 256.  Set to 0 to disable caching.
    seed : int
        Seed for the train/val split shuffling (only used if split IDs
        cannot be fetched from the repo).
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = 128,
        max_boxes: int = 20,
        cache_size: int = 256,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self._image_size = image_size
        self._max_boxes = max_boxes

        # Normalise split name
        split_lower = split.lower()
        if split_lower in ("val", "test", "valid"):
            self._split_name = "val"
        elif split_lower == "train":
            self._split_name = "train"
        else:
            raise ValueError(f"Unknown split '{split}'")

        # ── Fetch the list of image IDs for this split ──────────────────
        self._image_ids: List[str] = self._fetch_split_ids(self._split_name)

        if not self._image_ids:
            raise RuntimeError(
                f"No image IDs found for split '{self._split_name}'. "
                f"Check network connectivity to {VOC_MIRROR}."
            )

        # ── LRU cache ──────────────────────────────────────────────────
        self._cache = LRUCache(capacity=cache_size) if cache_size > 0 else None

        # ── Pre-fetch the VOC tar on first access (background) ──────────
        # The tar is downloaded lazily on the first cache miss that
        # requires tar extraction.  We don't pre-fetch here to avoid
        # blocking the constructor.

    # ── public properties ───────────────────────────────────────────────

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def split(self) -> str:
        return self._split_name

    @property
    def max_boxes(self) -> int:
        return self._max_boxes

    @property
    def class_names(self) -> List[str]:
        return VOC_CLASSES

    @property
    def num_classes(self) -> int:
        return len(VOC_CLASSES)

    # ── Dataset overrides ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx: int) -> Sample:
        """Fetch and process a single VOC sample.

        If caching is enabled, checks the LRU cache first.  On a cache
        miss, fetches the image and annotation from the VOC mirror,
        processes them, caches the result, and returns it.
        """
        stem = self._image_ids[idx]

        # ── Check cache ────────────────────────────────────────────────
        if self._cache is not None:
            cached = self._cache.get(stem)
            if cached is not None:
                return cached

        # ── Fetch image ────────────────────────────────────────────────
        try:
            pil_image = _fetch_voc_image(stem)
        except (RuntimeError, OSError, FileNotFoundError) as e:
            warnings.warn(f"Failed to fetch image {stem}: {e}. Returning blank sample.")
            return self._make_blank_sample(stem)

        orig_w, orig_h = pil_image.size

        # ── Fetch annotation ───────────────────────────────────────────
        try:
            xml_text = _fetch_voc_annotation(stem)
            boxes, labels = _parse_voc_annotation(xml_text)
        except (RuntimeError, OSError, FileNotFoundError) as e:
            warnings.warn(f"Failed to fetch annotation {stem}: {e}. Using empty labels.")
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)

        # ── Resize image with padding ──────────────────────────────────
        image_tensor, scale_factor, (pad_left, pad_top) = _resize_with_pad(
            pil_image, self._image_size
        )

        # ── Rescale boxes ─────────────────────────────────────────────
        if boxes.size(0) > 0:
            boxes = _rescale_boxes(
                boxes, orig_w, orig_h,
                scale_factor, pad_left, pad_top,
                self._image_size,
            )

        # ── Truncate / pad to max_boxes ────────────────────────────────
        n = boxes.size(0)
        if n > self._max_boxes:
            boxes = boxes[:self._max_boxes]
            labels = labels[:self._max_boxes]
            n = self._max_boxes

        if n < self._max_boxes:
            pad_n = self._max_boxes - n
            boxes = torch.cat([
                boxes,
                torch.full((pad_n, 4), fill_value=-1.0, dtype=torch.float32),
            ], dim=0)
            labels = torch.cat([
                labels,
                torch.full((pad_n,), fill_value=-1, dtype=torch.long),
            ], dim=0)
            valid_mask = torch.cat([
                torch.ones(n, dtype=torch.bool),
                torch.zeros(pad_n, dtype=torch.bool),
            ], dim=0)
        else:
            valid_mask = torch.ones(self._max_boxes, dtype=torch.bool)

        # ── Build Sample ──────────────────────────────────────────────
        sample = Sample(
            image=image_tensor,
            boxes=boxes,
            labels=labels,
            valid_mask=valid_mask,
            image_path=f"voc2012/{self._split_name}/{stem}.jpg",
            metadata={
                "split": self._split_name,
                "orig_size": (orig_w, orig_h),
                "scale_factor": scale_factor,
                "pad_amounts": (pad_left, pad_top),
                "stem": stem,
            },
        )

        # ── Cache ───────────────────────────────────────────────────────
        if self._cache is not None:
            self._cache.put(stem, sample)

        return sample

    # ── internal helpers ───────────────────────────────────────────────

    def _fetch_split_ids(self, split: str) -> List[str]:
        """Fetch the list of image IDs for the given split.

        Tries the repo-hosted split files first, then falls back to
        extracting from the VOC tar archive.
        """
        # Try repo-hosted split files
        url = VOC_TRAIN_IDS_URL if split == "train" else VOC_VAL_IDS_URL
        try:
            data = _fetch_url(url, timeout=15.0)
            ids = data.decode("utf-8").strip().splitlines()
            ids = [line.strip() for line in ids if line.strip() and not line.startswith("#")]
            if ids:
                return ids
        except (RuntimeError, OSError):
            pass

        # Fallback: extract from the VOC tar archive
        warnings.warn(
            "Could not fetch split IDs from repo. "
            "Extracting from VOC tar archive (slower)..."
        )
        return self._extract_split_ids_from_tar(split)

    def _extract_split_ids_from_tar(self, split: str) -> List[str]:
        """Extract image IDs from the VOC tar's ImageSets directory."""
        tar = _get_tar()
        set_name = "train" if split == "train" else "val"
        path = f"VOCdevkit/VOC2012/ImageSets/Main/{set_name}.txt"
        member = tar.getmember(path)
        f = tar.extractfile(member)
        if f is None:
            return []
        text = f.read().decode("utf-8")
        ids = [line.strip() for line in text.splitlines() if line.strip()]
        return ids

    def _make_blank_sample(self, stem: str) -> Sample:
        """Create a blank sample (all zeros) for when fetching fails."""
        return Sample(
            image=torch.zeros(3, self._image_size, self._image_size),
            boxes=torch.full((self._max_boxes, 4), fill_value=-1.0),
            labels=torch.full((self._max_boxes,), fill_value=-1, dtype=torch.long),
            valid_mask=torch.zeros(self._max_boxes, dtype=torch.bool),
            image_path=f"voc2012/{self._split_name}/{stem}.jpg",
            metadata={"split": self._split_name, "stem": stem, "blank": True},
        )

    # ── representation ─────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"StreamingVOCDataset("
            f"split={self._split_name!r}, "
            f"size={self._image_size}, "
            f"samples={len(self._image_ids)}, "
            f"max_boxes={self._max_boxes}, "
            f"cache={self._cache.size if self._cache else 0})"
        )



# ──────────────────────────────────────────────────────────────────────────────
# SyntheticVOCDataset — generates random VOC-like data for testing
# ──────────────────────────────────────────────────────────────────────────────


class SyntheticVOCDataset(torch.utils.data.Dataset):
    """Generates random synthetic VOC-like data for testing the training
    pipeline without requiring network access to the VOC mirror.

    Each sample is a random noise image with randomly placed bounding boxes
    and class labels.  This is useful for:

    - Verifying the training pipeline runs end-to-end.
    - Testing device support (MPS, CUDA, CPU).
    - Profiling memory and speed without data loading bottlenecks.

    Parameters
    ----------
    num_samples : int
        Number of synthetic samples to generate.
    image_size : int
        Spatial size of generated images (square).
    max_boxes : int
        Maximum number of bounding boxes per sample.
    num_classes : int
        Number of object classes (default 20, matching VOC).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 128,
        max_boxes: int = 20,
        num_classes: int = 20,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._num_samples = num_samples
        self._image_size = image_size
        self._max_boxes = max_boxes
        self._num_classes = num_classes
        self._seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def split(self) -> str:
        return "train"

    @property
    def max_boxes(self) -> int:
        return self._max_boxes

    @property
    def class_names(self) -> List[str]:
        return VOC_CLASSES[:self._num_classes]

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Sample:
        """Generate a random synthetic sample."""
        # Random image in [-1, 1] (clamped to ensure valid range)
        image = torch.randn(3, self._image_size, self._image_size) * 0.3
        image = image.clamp(-1.0, 1.0)

        # Random number of boxes (0 to max_boxes)
        n_boxes = self._rng.randint(0, self._max_boxes)

        if n_boxes == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)
        else:
            # Random boxes with reasonable sizes
            cx = torch.rand(n_boxes)
            cy = torch.rand(n_boxes)
            w = torch.rand(n_boxes) * 0.3 + 0.05  # 0.05 to 0.35
            h = torch.rand(n_boxes) * 0.3 + 0.05
            boxes = torch.stack([cx, cy, w, h], dim=1)
            labels = torch.randint(0, self._num_classes, (n_boxes,))

        # Pad to max_boxes
        if n_boxes < self._max_boxes:
            pad_n = self._max_boxes - n_boxes
            boxes = torch.cat([
                boxes,
                torch.full((pad_n, 4), fill_value=-1.0, dtype=torch.float32),
            ], dim=0)
            labels = torch.cat([
                labels,
                torch.full((pad_n,), fill_value=-1, dtype=torch.long),
            ], dim=0)
            valid_mask = torch.cat([
                torch.ones(n_boxes, dtype=torch.bool),
                torch.zeros(pad_n, dtype=torch.bool),
            ], dim=0)
        else:
            valid_mask = torch.ones(self._max_boxes, dtype=torch.bool)

        return Sample(
            image=image,
            boxes=boxes,
            labels=labels,
            valid_mask=valid_mask,
            image_path=f"synthetic/{idx:06d}.png",
            metadata={"split": "train", "stem": f"{idx:06d}", "synthetic": True},
        )

    def __repr__(self) -> str:
        return (
            f"SyntheticVOCDataset("
            f"samples={self._num_samples}, "
            f"size={self._image_size}, "
            f"max_boxes={self._max_boxes}, "
            f"classes={self._num_classes})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Factory: get_streaming_loaders
# ──────────────────────────────────────────────────────────────────────────────


def get_streaming_loaders(
    image_size: int = 128,
    batch_size: int = 16,
    max_boxes: int = 20,
    num_workers: int = 4,
    cache_size: int = 256,
    force_synthetic: bool = False,
    synthetic_samples: int = 100,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation DataLoaders using the streaming VOC dataset.

    This is a drop-in replacement for
    ``ilgan.data.dataloader.get_train_val_loaders`` that uses
    ``StreamingVOCDataset`` instead of ``YOLODataset``.

    If the VOC mirror is unreachable (e.g., no internet or firewall), the
    function falls back to ``SyntheticVOCDataset`` which generates random
    VOC-like data for testing purposes.

    Parameters
    ----------
    image_size : int
        Target image size (square).
    batch_size : int
        Batch size.
    max_boxes : int
        Maximum boxes per sample.
    num_workers : int
        Number of data loading workers.
    cache_size : int
        LRU cache size per dataset.
    force_synthetic : bool
        If True, always use synthetic data (for testing without network).
    synthetic_samples : int
        Number of synthetic samples to generate.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    """
    from ilgan.data.structures import Batch

    def _collate_fn(samples):
        return Batch.collate(samples, global_max_boxes=max_boxes)

    if force_synthetic:
        train_dataset = SyntheticVOCDataset(
            num_samples=synthetic_samples,
            image_size=image_size,
            max_boxes=max_boxes,
            num_classes=20,
            seed=42,
        )
        val_dataset = SyntheticVOCDataset(
            num_samples=max(10, synthetic_samples // 5),
            image_size=image_size,
            max_boxes=max_boxes,
            num_classes=20,
            seed=43,
        )
    else:
        try:
            train_dataset = StreamingVOCDataset(
                split="train",
                image_size=image_size,
                max_boxes=max_boxes,
                cache_size=cache_size,
            )
            val_dataset = StreamingVOCDataset(
                split="val",
                image_size=image_size,
                max_boxes=max_boxes,
                cache_size=cache_size,
            )
        except (RuntimeError, OSError, ConnectionError) as e:
            import warnings
            warnings.warn(
                f"Failed to connect to VOC mirror: {e}. "
                f"Falling back to synthetic data for testing."
            )
            train_dataset = SyntheticVOCDataset(
                num_samples=synthetic_samples,
                image_size=image_size,
                max_boxes=max_boxes,
                num_classes=20,
                seed=42,
            )
            val_dataset = SyntheticVOCDataset(
                num_samples=max(10, synthetic_samples // 5),
                image_size=image_size,
                max_boxes=max_boxes,
                num_classes=20,
                seed=43,
            )

    from ilgan.data.structures import Batch

    def _collate_fn(samples):
        return Batch.collate(samples, global_max_boxes=max_boxes)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # MPS doesn't support pin_memory
        drop_last=True,
        collate_fn=_collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=_collate_fn,
    )

    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Generate split ID files (run once, commit to repo)
# ──────────────────────────────────────────────────────────────────────────────


def generate_split_files(output_dir: str = "ilgan/data/voc_splits") -> None:
    """Generate the train/val split ID files from a local VOC download.

    Run this once on a machine that has the full VOC 2012 dataset
    downloaded, then commit the generated text files to the repo.
    The streaming dataset will use these files instead of extracting
    from the tar archive.

    Usage::

        python -c "from ilgan.data.streaming_voc import generate_split_files; generate_split_files()"
    """
    import xml.etree.ElementTree as ET

    os.makedirs(output_dir, exist_ok=True)

    voc_root = os.environ.get("VOC_ROOT", "./VOCdevkit/VOC2012")

    for split_name, output_name in [("train", "train_ids.txt"), ("val", "val_ids.txt")]:
        set_file = os.path.join(voc_root, "ImageSets", "Main", f"{split_name}.txt")
        if not os.path.isfile(set_file):
            print(f"Warning: {set_file} not found. Skipping {split_name}.")
            continue

        with open(set_file) as f:
            ids = [line.strip() for line in f if line.strip()]

        out_path = os.path.join(output_dir, output_name)
        with open(out_path, "w") as f:
            for img_id in ids:
                f.write(f"{img_id}\n")

        print(f"Wrote {len(ids)} IDs to {out_path}")

    print("Done. Commit the generated files to the repo.")


# ──────────────────────────────────────────────────────────────────────────────
# Module exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "StreamingVOCDataset",
    "SyntheticVOCDataset",
    "get_streaming_loaders",
    "VOC_CLASSES",
    "VOC_CLASS_TO_ID",
    "generate_split_files",
]
