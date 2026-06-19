"""Data loading, preprocessing, and core data structures for ILGAN."""

from ilgan.data.structures import (
    Sample,
    Batch,
    DatasetMetadata,
    parse_yolo_label,
)
from ilgan.data.dataset import (
    YOLODataset,
    resize_with_pad,
)
from ilgan.data.augmentation import (
    Augmentation,
    RandomHorizontalFlip,
    RandomColorJitter,
    RandomAffine,
    Cutout,
    Compose,
    build_default_augmentation_pipeline,
)
from ilgan.data.dataloader import (
    GANDataloader,
    get_train_val_loaders,
)

__all__ = [
    # Structures
    "Sample",
    "Batch",
    "DatasetMetadata",
    "parse_yolo_label",
    # Dataset
    "YOLODataset",
    "resize_with_pad",
    # Augmentation
    "Augmentation",
    "RandomHorizontalFlip",
    "RandomColorJitter",
    "RandomAffine",
    "Cutout",
    "Compose",
    "build_default_augmentation_pipeline",
    # Dataloader
    "GANDataloader",
    "get_train_val_loaders",
]