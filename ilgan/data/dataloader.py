"""
DataLoader for ILGAN training and validation.

Provides ``GANDataloader`` — a wrapper around ``torch.utils.data.DataLoader``
that integrates the augmentation pipeline from ``ilgan.data.augmentation``
with ``YOLODataset``.  Also exposes a ``get_train_val_loaders`` factory
function for convenient dataset construction.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ilgan.data.augmentation import (
    Augmentation,
    Compose,
    build_default_augmentation_pipeline,
)
from ilgan.data.dataset import YOLODataset
from ilgan.data.structures import Batch, Sample

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_EPOCH_SEED_OFFSET = 1_000_000
"""Offset added to the epoch number when computing the per-sample / per-epoch
RNG seed.  This avoids collisions with other seed spaces."""


# ──────────────────────────────────────────────────────────────────────────────
# GANDataloader
# ──────────────────────────────────────────────────────────────────────────────


class GANDataloader:
    """GAN-specific dataloader that wraps ``YOLODataset`` + augmentations.

    The dataloader is designed to be iterated over for multiple epochs,
    with deterministic augmentation seeds per sample per epoch for
    reproducibility.

    Parameters
    ----------
    dataset : YOLODataset
        The underlying dataset (training split).
    augmentations : Compose, optional
        Augmentation pipeline.  If ``None``, no augmentations are applied
        (identity transform).
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of subprocesses for data loading.
    shuffle : bool
        Whether to shuffle the dataset each epoch.
    pin_memory : bool
        Whether to use pinned memory for faster GPU transfers.
    drop_last : bool
        Whether to drop the last incomplete batch.
    global_max_boxes : int, optional
        Hard upper bound on boxes per sample passed to ``Batch.collate``.
        Defaults to ``dataset.max_boxes``.
    collate_fn : callable, optional
        Custom collation function.  Defaults to ``Batch.collate``.
    """

    def __init__(
        self,
        dataset: YOLODataset,
        augmentations: Optional[Compose] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        global_max_boxes: Optional[int] = None,
        collate_fn: Optional[Callable[[List[Sample]], Batch]] = None,
    ) -> None:
        self._dataset = dataset
        self._augmentations = augmentations
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle
        self._pin_memory = pin_memory
        self._drop_last = drop_last
        self._global_max_boxes = global_max_boxes or dataset.max_boxes
        self._custom_collate_fn = collate_fn

        # Current epoch (0-indexed); used for deterministic seeding
        self._current_epoch: int = 0

        # Build the underlying DataLoader
        self._loader = self._build_loader()

    # ── public API ─────────────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch number for deterministic augmentation.

        The seed for each sample is derived from:
            ``seed = epoch * _EPOCH_SEED_OFFSET + sample_index``

        This ensures that across training runs, the same sample in the same
        epoch receives the same augmentation, provided the dataset order is
        consistent (controlled by the worker seed).

        Parameters
        ----------
        epoch : int
            The (0-indexed) epoch number.
        """
        self._current_epoch = epoch

        # If using a distributed / seed-aware sampler, set the epoch on it
        if hasattr(self._loader.sampler, "set_epoch"):
            self._loader.sampler.set_epoch(epoch)

        # Seed the worker-init function for deterministic shuffling
        worker_seed = epoch * _EPOCH_SEED_OFFSET
        self._loader.worker_init_fn = (
            lambda worker_id: self._worker_init_fn(worker_id, worker_seed)
        )

    @property
    def dataset(self) -> YOLODataset:
        """Underlying dataset."""
        return self._dataset

    @property
    def batch_size(self) -> int:
        """Configured batch size."""
        return self._batch_size

    @property
    def num_batches(self) -> int:
        """Number of batches per epoch."""
        n = len(self._dataset)
        if self._drop_last:
            return n // self._batch_size
        return math.ceil(n / self._batch_size)

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.num_batches

    def __iter__(self) -> torch.utils.data.DataLoader:
        """Return an iterator over batches.

        Before iterating, the epoch seed is applied for deterministic
        augmentation.  Each sample in the batch receives a unique
        ``rng_seed`` derived from its index.
        """
        epoch_seed = self._current_epoch * _EPOCH_SEED_OFFSET

        # If augmentations are active, wrap the DataLoader's collate_fn
        # to inject per-sample RNG seeds
        collate_fn = self._get_collate_fn_with_augmentation(epoch_seed)

        # Rebuild the loader with the augmentation-aware collate
        self._loader = self._build_loader(collate_fn_override=collate_fn)

        return iter(self._loader)

    # ── internal helpers ───────────────────────────────────────────────

    def _build_loader(
        self,
        collate_fn_override: Optional[Callable[[List[Sample]], Batch]] = None,
    ) -> DataLoader:
        """Construct (or reconstruct) the underlying ``DataLoader``."""
        worker_seed = self._current_epoch * _EPOCH_SEED_OFFSET
        return DataLoader(
            dataset=self._dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
            collate_fn=collate_fn_override or self._default_collate_fn,
            worker_init_fn=lambda wid: self._worker_init_fn(wid, worker_seed),
        )

    def _default_collate_fn(self, samples: List[Sample]) -> Batch:
        """Default collation (no augmentation)."""
        return Batch.collate(samples, global_max_boxes=self._global_max_boxes)

    def _get_collate_fn_with_augmentation(
        self,
        epoch_seed: int,
    ) -> Callable[[List[Sample]], Batch]:
        """Return a collation function that applies augmentations.

        Each sample gets a deterministic seed derived from its position in
        the dataset and the current epoch.
        """

        def _collate(samples: List[Sample]) -> Batch:
            augmented: List[Sample] = []
            for sample in samples:
                # Derive per-sample seed: epoch_seed + sample index
                # We use the image_path as a proxy for sample identity
                sample_idx = hash(sample.image_path) & 0xFFFFFFFF
                rng_seed = epoch_seed + sample_idx

                if self._augmentations is not None:
                    aug_sample = self._augmentations.apply_to_sample(
                        sample, rng_seed=rng_seed,
                    )
                else:
                    aug_sample = sample

                augmented.append(aug_sample)

            return Batch.collate(augmented, global_max_boxes=self._global_max_boxes)

        return _collate

    @staticmethod
    def _worker_init_fn(worker_id: int, base_seed: int) -> None:
        """Seed the PyTorch and Python RNGs for a DataLoader worker.

        This ensures deterministic shuffling and augmentation RNG for each
        worker across epochs.
        """
        worker_seed = base_seed + worker_id
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)
        import numpy as np
        np.random.seed(worker_seed)

    def __repr__(self) -> str:
        aug = "augmented" if self._augmentations is not None else "no-aug"
        return (
            f"GANDataloader("
            f"dataset={self._dataset!r}, "
            f"batch_size={self._batch_size}, "
            f"epoch={self._current_epoch}, "
            f"{aug})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Factory: get_train_val_loaders
# ──────────────────────────────────────────────────────────────────────────────


def get_train_val_loaders(
    root_dir: str,
    image_size: int = 128,
    batch_size: int = 16,
    num_workers: int = 4,
    val_split: float = 0.2,
    augment: bool = True,
    global_max_boxes: Optional[int] = None,
    train_max_boxes: int = 50,
    val_max_boxes: int = 50,
    augmentation_kwargs: Optional[Dict[str, Any]] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> Tuple[GANDataloader, GANDataloader]:
    """Create training and validation :class:`GANDataloader` instances.

    Parameters
    ----------
    root_dir : str
        Path to the dataset root (containing ``images/``, ``labels/``,
        and optional ``train.txt`` / ``val.txt`` split files).
    image_size : int
        Target size for the longer side of the image in pixels (square).
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of data-loading subprocesses.
    val_split : float
        Fraction of data to reserve for validation (used only when split
        files ``train.txt`` / ``val.txt`` do NOT exist).  Default 0.2.
    augment : bool
        If ``True``, the training loader applies the default augmentation
        pipeline.  The validation loader never uses augmentations (only
        resize + pad).
    global_max_boxes : int, optional
        Hard upper bound on boxes per sample for collation (applies to
        both loaders).  If ``None``, uses the per-dataset ``max_boxes``.
    train_max_boxes : int
        Max boxes for training dataset.
    val_max_boxes : int
        Max boxes for validation dataset.
    augmentation_kwargs : dict, optional
        Keyword arguments forwarded to
        :func:`build_default_augmentation_pipeline`.
    pin_memory : bool
        Whether to use pinned memory.
    drop_last : bool
        Whether to drop the last incomplete batch.

    Returns
    -------
    train_loader : GANDataloader
        Augmented training data loader.
    val_loader : GANDataloader
        Non-augmented validation data loader (resize + pad only).
    """
    # ── augmentations ──────────────────────────────────────────────────
    augmentations: Optional[Compose] = None
    if augment:
        kwargs = augmentation_kwargs or {}
        augmentations = build_default_augmentation_pipeline(**kwargs)

    # ── training dataset ───────────────────────────────────────────────
    train_dataset = YOLODataset(
        root_dir=root_dir,
        image_size=image_size,
        split="train",
        max_boxes=train_max_boxes,
        transform=None,
    )

    # ── validation dataset ─────────────────────────────────────────────
    val_dataset = YOLODataset(
        root_dir=root_dir,
        image_size=image_size,
        split="val",
        max_boxes=val_max_boxes,
        transform=None,
    )

    # ── dataloaders ─────────────────────────────────────────────────────
    train_loader = GANDataloader(
        dataset=train_dataset,
        augmentations=augmentations,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=drop_last,
        global_max_boxes=global_max_boxes,
    )

    val_loader = GANDataloader(
        dataset=val_dataset,
        augmentations=None,  # validation never uses augmentation
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # no shuffling for validation
        pin_memory=pin_memory,
        drop_last=False,
        global_max_boxes=global_max_boxes,
    )

    return train_loader, val_loader


__all__ = [
    "GANDataloader",
    "get_train_val_loaders",
]