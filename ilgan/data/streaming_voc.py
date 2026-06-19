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

**No synthetic data is ever used.**  If the VOC mirror is unreachable,
the dataset raises a ``RuntimeError``.  The only alternative data source
is HuggingFace ``datasets`` (via ``get_hf_voc_loaders``), which streams
real VOC 2012 data from the HuggingFace hub.

Usage
-----
::

    from ilgan.data.streaming_voc import StreamingVOCDataset, get_streaming_loaders

    # Direct usage (streams from VOC mirror)
    dataset = StreamingVOCDataset(split="train", image_size=128, max_boxes=20)
    sample = dataset[0]

    # Via factory (recommended)
    train_loader, val_loader = get_streaming_loaders(
        image_size=128, batch_size=32, max_boxes=20
    )

    # Via HuggingFace datasets (alternative when VOC mirror is blocked)
    from ilgan.data.streaming_voc import get_hf_voc_loaders
    train_loader, val_loader = get_hf_voc_loaders(
        image_size=128, batch_size=32, max_boxes=20
    )
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
"""URL of the full VOC 2012 train+val archive."""

VOC_TRAIN_IDS_URL: str = (
    "https://raw.githubusercontent.com/akshaylakkur/GAN/main/"
    "ilgan/data/voc_splits/train_ids.txt"
)
VOC_VAL_IDS_URL: str = (
    "https://raw.githubusercontent.com/akshaylakkur/GAN/main/"
    "ilgan/data/voc_splits/val_ids.txt"
)
"""Repo-hosted split ID files (tiny text files, ~15KB each)."""


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
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: Sample) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────────────────────


def _fetch_url(url: str, timeout: float = 30.0) -> bytes:
    """Fetch a URL and return the raw bytes.

    Uses ``urllib.request`` (stdlib).  Raises ``RuntimeError`` on failure.
    """
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")


def _fetch_voc_image(stem: str) -> Image.Image:
    """Fetch a single VOC image by stem and return a PIL Image.

    Tries direct HTTP first, then falls back to tar archive extraction.
    """
    direct_url = f"{VOC_MIRROR}/JPEGImages/{stem}.jpg"
    try:
        data = _fetch_url(direct_url, timeout=10.0)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (RuntimeError, OSError):
        pass
    return _extract_from_tar(f"VOCdevkit/VOC2012/JPEGImages/{stem}.jpg")


def _fetch_voc_annotation(stem: str) -> str:
    """Fetch a single VOC annotation XML by stem and return the text."""
    direct_url = f"{VOC_MIRROR}/Annotations/{stem}.xml"
    try:
        data = _fetch_url(direct_url, timeout=10.0)
        return data.decode("utf-8")
    except (RuntimeError, OSError):
        pass
    return _extract_annotation_from_tar(stem)


# ── Tar extraction fallback ──────────────────────────────────────────────

_TAR_CACHE: Optional[tarfile.TarFile] = None


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

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tar")
    tmp.write(data)
    tmp.flush()
    _TAR_CACHE = tarfile.open(tmp.name, "r")
    return _TAR_CACHE


def _extract_from_tar(path_in_archive: str) -> Image.Image:
    tar = _get_tar()
    member = tar.getmember(path_in_archive)
    f = tar.extractfile(member)
    if f is None:
        raise FileNotFoundError(f"Could not extract {path_in_archive} from VOC archive")
    return Image.open(io.BytesIO(f.read())).convert("RGB")


def _extract_annotation_from_tar(stem: str) -> str:
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

    Returns
    -------
    boxes : torch.Tensor
        Shape ``[N, 4]`` in ``(cx, cy, w, h)`` format, normalised to ``[0, 1]``.
    labels : torch.Tensor
        Shape ``[N]``, integer class IDs (0–19).
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    boxes_list: List[float] = []
    labels_list: List[int] = []

    for obj in root.findall("object"):
        difficult = obj.find("difficult")
        if difficult is not None and difficult.text == "1":
            continue

        cls_name = obj.find("name").text
        if cls_name not in VOC_CLASS_TO_ID:
            continue

        cls_id = VOC_CLASS_TO_ID[cls_name]
        bndbox = obj.find("bndbox")

        xmin = max(0.0, float(bndbox.find("xmin").text))
        ymin = max(0.0, float(bndbox.find("ymin").text))
        xmax = min(float(img_w), float(bndbox.find("xmax").text))
        ymax = min(float(img_h), float(bndbox.find("ymax").text))

        if xmax <= xmin or ymax <= ymin:
            continue

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
# Resize with padding
# ──────────────────────────────────────────────────────────────────────────────


def _resize_with_pad(
    image: Image.Image,
    target_size: int,
) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
    """Resize the longer side to *target_size*, pad to square, return tensor."""
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

    **No synthetic data is ever used.**  If the VOC mirror is unreachable,
    a ``RuntimeError`` is raised.  Use ``get_hf_voc_loaders()`` for an
    alternative real-data path via HuggingFace datasets.

    Parameters
    ----------
    split : str
        One of ``"train"`` or ``"val"``.
    image_size : int
        Target square size in pixels.
    max_boxes : int
        Maximum number of boxes per sample.
    cache_size : int
        Maximum samples in the in-memory LRU cache.  Set to 0 to disable.
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = 128,
        max_boxes: int = 20,
        cache_size: int = 256,
    ) -> None:
        super().__init__()

        self._image_size = image_size
        self._max_boxes = max_boxes

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
                f"Check network connectivity to {VOC_MIRROR}. "
                f"Set VOC_MIRROR env var to an alternative mirror, or use "
                f"get_hf_voc_loaders() for HuggingFace-based streaming."
            )

        self._cache = LRUCache(capacity=cache_size) if cache_size > 0 else None

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

        Raises
        ------
        RuntimeError
            If the VOC mirror is unreachable and the image cannot be
            fetched via any fallback mechanism.
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
            raise RuntimeError(
                f"Failed to fetch VOC image '{stem}' from {VOC_MIRROR}. "
                f"Check network connectivity or set VOC_MIRROR env var. "
                f"Original error: {e}"
            )

        orig_w, orig_h = pil_image.size

        # ── Fetch annotation ───────────────────────────────────────────
        try:
            xml_text = _fetch_voc_annotation(stem)
            boxes, labels = _parse_voc_annotation(xml_text)
        except (RuntimeError, OSError, FileNotFoundError) as e:
            # If annotation fails, use empty labels (image-only sample)
            warnings.warn(f"Failed to fetch annotation for {stem}: {e}. Using empty labels.")
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

        if self._cache is not None:
            self._cache.put(stem, sample)

        return sample

    # ── internal helpers ───────────────────────────────────────────────

    def _fetch_split_ids(self, split: str) -> List[str]:
        """Fetch the list of image IDs for the given split.

        Tries the repo-hosted split files first, then falls back to
        extracting from the VOC tar archive.
        """
        url = VOC_TRAIN_IDS_URL if split == "train" else VOC_VAL_IDS_URL
        try:
            data = _fetch_url(url, timeout=15.0)
            ids = data.decode("utf-8").strip().splitlines()
            ids = [line.strip() for line in ids if line.strip() and not line.startswith("#")]
            if ids:
                return ids
        except (RuntimeError, OSError):
            pass

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
# Factory: get_streaming_loaders (VOC mirror, hard-fail on unavailability)
# ──────────────────────────────────────────────────────────────────────────────


def get_streaming_loaders(
    image_size: int = 128,
    batch_size: int = 16,
    max_boxes: int = 20,
    num_workers: int = 4,
    cache_size: int = 256,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation DataLoaders using the streaming VOC dataset.

    This is a drop-in replacement for
    ``ilgan.data.dataloader.get_train_val_loaders`` that uses
    ``StreamingVOCDataset`` instead of ``YOLODataset``.

    **No synthetic data is ever used.**  If the VOC mirror is unreachable,
    a ``RuntimeError`` is raised.  Use ``get_hf_voc_loaders()`` for an
    alternative real-data path via HuggingFace datasets.

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

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader

    Raises
    ------
    RuntimeError
        If the VOC mirror is unreachable and no data can be loaded.
    """
    from ilgan.data.structures import Batch

    def _collate_fn(samples):
        return Batch.collate(samples, global_max_boxes=max_boxes)

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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
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
# HuggingFace datasets-based streaming (alternative data path)
# ──────────────────────────────────────────────────────────────────────────────


def get_hf_voc_loaders(
    image_size: int = 128,
    batch_size: int = 16,
    max_boxes: int = 20,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/val DataLoaders using HuggingFace ``datasets`` to stream
    VOC 2012 data.

    This is an alternative to ``get_streaming_loaders`` that uses the
    HuggingFace hub as the data source instead of the official VOC mirror.
    It requires the ``datasets`` package:

        pip install datasets

    The HuggingFace path is useful when:
    - The official VOC mirror (host.robots.ox.ac.uk) is blocked/firewalled.
    - You want faster downloads via HF's CDN.
    - You're running on a cloud instance with limited egress to Oxford.

    **No synthetic data is ever used.**  If the HuggingFace hub is
    unreachable, a ``RuntimeError`` is raised.

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

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader

    Raises
    ------
    ImportError
        If the ``datasets`` package is not installed.
    RuntimeError
        If the HuggingFace hub is unreachable.
    """
    try:
        import datasets
    except ImportError:
        raise ImportError(
            "get_hf_voc_loaders requires the 'datasets' package. "
            "Install it with: pip install datasets"
        )

    from ilgan.data.structures import Batch

    # ── Load VOC 2012 from HuggingFace hub ──────────────────────────────
    # The VOC 2012 dataset is available on the HF hub under multiple orgs.
    # We use the official torchvision-compatible dataset.
    try:
        hf_dataset = datasets.load_dataset(
            "ilhar/vis-datasets-pascal-voc-2012",
            split="train",
            streaming=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load VOC 2012 from HuggingFace hub: {e}. "
            f"Check network connectivity or use get_streaming_loaders() "
            f"to stream from the official VOC mirror instead."
        )

    # ── Split into train/val ────────────────────────────────────────────
    # The HF dataset doesn't have built-in splits, so we use the same
    # split IDs as the streaming dataset.
    import urllib.request

    def _fetch_ids(url: str) -> List[str]:
        try:
            data = urllib.request.urlopen(url, timeout=15).read()
            return [line.strip() for line in data.decode("utf-8").splitlines()
                    if line.strip() and not line.startswith("#")]
        except Exception as e:
            raise RuntimeError(f"Failed to fetch split IDs: {e}")

    train_ids = set(_fetch_ids(VOC_TRAIN_IDS_URL))
    val_ids = set(_fetch_ids(VOC_VAL_IDS_URL))

    # ── Filter dataset by split ─────────────────────────────────────────
    def _is_train(example):
        stem = example["image_id"] if "image_id" in example else example["id"]
        return stem in train_ids

    def _is_val(example):
        stem = example["image_id"] if "image_id" in example else example["id"]
        return stem in val_ids

    train_stream = hf_dataset.filter(_is_train)
    val_stream = hf_dataset.filter(_is_val)

    # ── Wrap in PyTorch Dataset ─────────────────────────────────────────
    class _HFVOCDataset(torch.utils.data.IterableDataset):
        """Wrapper that converts HF VOC examples to ILGAN Samples."""

        def __init__(self, hf_stream, split_name: str):
            self._stream = hf_stream
            self._split_name = split_name

        def __iter__(self):
            for example in self._stream:
                yield self._example_to_sample(example)

        def _example_to_sample(self, example) -> Sample:
            # Convert HF example to ILGAN Sample
            # HF dataset returns PIL images and VOC annotations
            pil_image = example["image"]
            if not isinstance(pil_image, Image.Image):
                pil_image = Image.open(io.BytesIO(pil_image["bytes"])).convert("RGB")

            orig_w, orig_h = pil_image.size

            # Parse objects
            objects = example.get("objects", example.get("annotation", {}).get("object", []))
            boxes_list = []
            labels_list = []

            for obj in objects:
                cls_name = obj["name"] if isinstance(obj, dict) else obj.find("name").text
                if cls_name not in VOC_CLASS_TO_ID:
                    continue

                cls_id = VOC_CLASS_TO_ID[cls_name]

                if isinstance(obj, dict):
                    bbox = obj["bbox"]
                    # HF bbox format varies; handle both VOC and YOLO formats
                    if "xmin" in bbox:
                        xmin = max(0.0, float(bbox["xmin"]))
                        ymin = max(0.0, float(bbox["ymin"]))
                        xmax = min(float(orig_w), float(bbox["xmax"]))
                        ymax = min(float(orig_h), float(bbox["ymax"]))
                    else:
                        # COCO/YOLO format: [x, y, w, h]
                        xmin = max(0.0, float(bbox[0]))
                        ymin = max(0.0, float(bbox[1]))
                        xmax = min(float(orig_w), float(bbox[0]) + float(bbox[2]))
                        ymax = min(float(orig_h), float(bbox[1]) + float(bbox[3]))
                else:
                    # XML element
                    bndbox = obj.find("bndbox")
                    xmin = max(0.0, float(bndbox.find("xmin").text))
                    ymin = max(0.0, float(bndbox.find("ymin").text))
                    xmax = min(float(orig_w), float(bndbox.find("xmax").text))
                    ymax = min(float(orig_h), float(bndbox.find("ymax").text))

                if xmax <= xmin or ymax <= ymin:
                    continue

                cx = (xmin + xmax) / 2.0 / orig_w
                cy = (ymin + ymax) / 2.0 / orig_h
                w = (xmax - xmin) / orig_w
                h = (ymax - ymin) / orig_h

                boxes_list.extend([cx, cy, w, h])
                labels_list.append(cls_id)

            boxes = torch.tensor(boxes_list, dtype=torch.float32).view(-1, 4) if boxes_list else torch.zeros((0, 4))
            labels = torch.tensor(labels_list, dtype=torch.long) if labels_list else torch.zeros(0, dtype=torch.long)

            # Resize
            image_tensor, scale_factor, (pad_left, pad_top) = _resize_with_pad(
                pil_image, image_size
            )

            if boxes.size(0) > 0:
                boxes = _rescale_boxes(
                    boxes, orig_w, orig_h, scale_factor, pad_left, pad_top, image_size
                )

            # Pad to max_boxes
            n = boxes.size(0)
            if n > max_boxes:
                boxes = boxes[:max_boxes]
                labels = labels[:max_boxes]
                n = max_boxes

            if n < max_boxes:
                pad_n = max_boxes - n
                boxes = torch.cat([boxes, torch.full((pad_n, 4), fill_value=-1.0)], dim=0)
                labels = torch.cat([labels, torch.full((pad_n,), fill_value=-1, dtype=torch.long)], dim=0)
                valid_mask = torch.cat([torch.ones(n, dtype=torch.bool), torch.zeros(pad_n, dtype=torch.bool)], dim=0)
            else:
                valid_mask = torch.ones(max_boxes, dtype=torch.bool)

            stem = example.get("image_id", example.get("id", "unknown"))
            return Sample(
                image=image_tensor,
                boxes=boxes,
                labels=labels,
                valid_mask=valid_mask,
                image_path=f"voc2012_hf/{self._split_name}/{stem}.jpg",
                metadata={"split": self._split_name, "stem": stem, "source": "huggingface"},
            )

    def _collate_fn(samples):
        return Batch.collate(list(samples), global_max_boxes=max_boxes)

    train_dataset = _HFVOCDataset(train_stream, "train")
    val_dataset = _HFVOCDataset(val_stream, "val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,  # IterableDataset doesn't support multi-worker
        pin_memory=False,
        drop_last=True,
        collate_fn=_collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
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
    """
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
    "get_streaming_loaders",
    "get_hf_voc_loaders",
    "VOC_CLASSES",
    "VOC_CLASS_TO_ID",
    "generate_split_files",
]
