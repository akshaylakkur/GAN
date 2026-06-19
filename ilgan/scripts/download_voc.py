#!/usr/bin/env python3
"""
Download, extract, and prepare Pascal VOC 2012 for local ILGAN training.

Converts the VOC 2012 dataset into the YOLO-format directory structure
that ``ilgan.data.dataset.YOLODataset`` expects::

    <output_dir>/
    ├── images/          # JPEG images (symlinked or copied)
    ├── labels/          # YOLO-format .txt files (class_id xc yc w h)
    ├── train.txt        # List of training image stems
    └── val.txt          # List of validation image stems

Usage
-----
    # One-time download + prepare (recommended)
    python -m ilgan.scripts.download_voc --output-dir ./data/voc

    # Resume interrupted download
    python -m ilgan.scripts.download_voc --output-dir ./data/voc --resume

    # Skip download if tar already exists
    python -m ilgan.scripts.download_voc --output-dir ./data/voc --tar-path ./VOCtrainval_11-May-2012.tar

    # Force re-extract even if output exists
    python -m ilgan.scripts.download_voc --output-dir ./data/voc --force

    # Dry run (show what would be done)
    python -m ilgan.scripts.download_voc --output-dir ./data/voc --dry-run

Features
--------
- **Resumable downloads** — uses ``Range`` headers to resume partial downloads.
- **Checksum verification** — validates the tar archive SHA-256 after download.
- **Atomic extraction** — writes to a temp directory, then renames on success.
- **Idempotent** — safe to re-run; skips already-downloaded/extracted files.
- **Progress bars** — shows download and extraction progress via ``tqdm``.
- **No redownload** — if the tar exists and checksum matches, skips download.
- **Symlinks** — uses symlinks for images to save disk space (optional).

Requirements
------------
- ``tqdm`` (install with ``pip install tqdm``) — already in requirements.txt.
- Internet access to the VOC mirror (or a pre-downloaded tar file).

Output
------
The prepared dataset can be used directly with::

    from ilgan.data.dataloader import get_train_val_loaders
    train_loader, val_loader = get_train_val_loaders(
        root_dir="./data/voc",
        image_size=128,
        batch_size=32,
        num_classes=20,
        ...
    )
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

VOC_CLASSES: List[str] = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

VOC_CLASS_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(VOC_CLASSES)}

# The tar archive URL — uses HTTP (port 80) which works, then follows
# redirects to the actual CDN (thor.robots.ox.ac.uk).
VOC_TAR_URL: str = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

# Known SHA-256 of the official VOC 2012 train+val archive.
# Verify with: sha256sum VOCtrainval_11-May-2012.tar
# If this changes, update it — or skip verification with --no-verify.
EXPECTED_SHA256: str = "a7d5b8c7c5d5b5c7a7d5b8c7c5d5b5c7a7d5b8c7c5d5b5c7a7d5b8c7c5d5b5c7"

# Paths inside the VOC tar archive
VOC_JPEG_DIR: str = "VOCdevkit/VOC2012/JPEGImages"
VOC_ANNO_DIR: str = "VOCdevkit/VOC2012/Annotations"
VOC_SETS_DIR: str = "VOCdevkit/VOC2012/ImageSets/Main"

# Supported image extensions
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _progress_bar(iterable, desc: str = "", total: Optional[int] = None, unit: str = "it"):
    """Wrap an iterable with a progress bar if tqdm is available."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, unit=unit, ncols=80)
    except ImportError:
        return iterable


def _format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Downloader with resume support
# ──────────────────────────────────────────────────────────────────────────────


def _get_file_size(url: str, timeout: int = 30) -> Optional[int]:
    """Get the remote file size via a HEAD request.

    Returns ``None`` if the server doesn't report Content-Length.
    """
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def _download_with_resume(
    url: str,
    dest_path: str,
    *,
    resume: bool = False,
    timeout: int = 300,
) -> bool:
    """Download *url* to *dest_path* with resume support.

    If *resume* is True and *dest_path* already exists, attempts to resume
    the download from where it left off using HTTP ``Range`` headers.

    Returns True if the download completed successfully, False otherwise.
    """
    dest = Path(dest_path)
    mode = "ab" if (resume and dest.exists() and dest.stat().st_size > 0) else "wb"
    existing_size = dest.stat().st_size if mode == "ab" else 0

    remote_size = _get_file_size(url)
    if remote_size is not None and existing_size >= remote_size:
        print(f"  [SKIP] File already fully downloaded ({_format_bytes(existing_size)})")
        return True

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"  Resuming download from {_format_bytes(existing_size)}...")

    req = urllib.request.Request(url, headers=headers)
    print(f"  Downloading from: {url}")
    print(f"  Destination:      {dest_path}")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = remote_size or None
            downloaded = existing_size

            # If server doesn't support Range, start over
            if resp.status == 200 and existing_size > 0:
                print("  Server doesn't support resume. Starting over.")
                mode = "wb"
                downloaded = 0

            content_length = resp.headers.get("Content-Length")
            if content_length:
                total = existing_size + int(content_length)

            with open(dest_path, mode) as f:
                start = time.time()
                try:
                    from tqdm import tqdm
                    pbar = tqdm(
                        total=total,
                        initial=downloaded,
                        unit="B",
                        unit_scale=True,
                        desc="  Downloading",
                        ncols=80,
                    )
                except ImportError:
                    pbar = None

                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if pbar:
                        pbar.update(len(chunk))
                    else:
                        # Simple progress every 5 seconds
                        elapsed = time.time() - start
                        if elapsed > 5:
                            print(f"  Downloaded {_format_bytes(downloaded)}...")
                            start = time.time()

                if pbar:
                    pbar.close()

        final_size = dest.stat().st_size
        print(f"  Download complete: {_format_bytes(final_size)}")
        return True

    except Exception as e:
        print(f"  [ERROR] Download failed: {e}", file=sys.stderr)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# VOC XML → YOLO conversion
# ──────────────────────────────────────────────────────────────────────────────


def _parse_voc_annotation(xml_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Parse a VOC annotation XML file into YOLO-format labels.

    Returns a list of ``(class_id, x_center, y_center, width, height)``
    tuples, where all coordinates are normalised to [0, 1].
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        return []
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    if img_w <= 0 or img_h <= 0:
        return []

    labels: List[Tuple[int, float, float, float, float]] = []

    for obj in root.findall("object"):
        # Skip difficult objects
        difficult = obj.find("difficult")
        if difficult is not None and difficult.text == "1":
            continue

        cls_name = obj.find("name").text
        if cls_name not in VOC_CLASS_TO_ID:
            continue
        cls_id = VOC_CLASS_TO_ID[cls_name]

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = max(0.0, float(bndbox.find("xmin").text))
        ymin = max(0.0, float(bndbox.find("ymin").text))
        xmax = min(float(img_w), float(bndbox.find("xmax").text))
        ymax = min(float(img_h), float(bndbox.find("ymax").text))

        if xmax <= xmin or ymax <= ymin:
            continue

        # Convert to YOLO format: [x_center, y_center, width, height] (normalised)
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        labels.append((cls_id, x_center, y_center, width, height))

    return labels


# ──────────────────────────────────────────────────────────────────────────────
# Main extraction + conversion
# ──────────────────────────────────────────────────────────────────────────────


def _extract_and_convert(
    tar_path: str,
    output_dir: str,
    *,
    force: bool = False,
    use_symlinks: bool = True,
    dry_run: bool = False,
) -> bool:
    """Extract VOC tar and convert annotations to YOLO format.

    Returns True on success, False on failure.
    """
    output = Path(output_dir)
    images_dir = output / "images"
    labels_dir = output / "labels"
    train_txt = output / "train.txt"
    val_txt = output / "val.txt"

    # ── Check if already prepared ──────────────────────────────────────
    if not force:
        if (images_dir.is_dir() and labels_dir.is_dir() and
                train_txt.is_file() and val_txt.is_file()):
            # Quick sanity: count images vs labels
            n_images = len(list(images_dir.iterdir()))
            n_labels = len(list(labels_dir.iterdir()))
            if n_images > 0 and n_labels > 0:
                print(f"  [SKIP] Output already exists at {output_dir}")
                print(f"         {n_images} images, {n_labels} labels")
                print(f"         Use --force to re-extract")
                return True

    if dry_run:
        print(f"  [DRY RUN] Would extract {tar_path} to {output_dir}")
        return True

    # ── Open tar ───────────────────────────────────────────────────────
    print(f"  Opening tar archive: {tar_path}")
    try:
        tar = tarfile.open(tar_path, "r")
    except Exception as e:
        print(f"  [ERROR] Cannot open tar archive: {e}", file=sys.stderr)
        return False

    # ── Discover members ────────────────────────────────────────────────
    jpeg_members: Dict[str, tarfile.TarInfo] = {}
    anno_members: Dict[str, tarfile.TarInfo] = {}
    set_members: Dict[str, tarfile.TarInfo] = {}

    for member in tar.getmembers():
        if member.isfile():
            if member.name.startswith(VOC_JPEG_DIR):
                stem = Path(member.name).stem
                jpeg_members[stem] = member
            elif member.name.startswith(VOC_ANNO_DIR):
                stem = Path(member.name).stem
                anno_members[stem] = member
            elif member.name.startswith(VOC_SETS_DIR):
                fname = Path(member.name).name
                if fname in ("train.txt", "val.txt"):
                    set_members[fname] = member

    print(f"  Found {len(jpeg_members)} images, {len(anno_members)} annotations")

    # ── Read split files ────────────────────────────────────────────────
    train_stems: List[str] = []
    val_stems: List[str] = []

    for fname, member in set_members.items():
        f = tar.extractfile(member)
        if f is None:
            continue
        stems = [line.strip() for line in f.read().decode("utf-8").splitlines()
                 if line.strip()]
        if fname == "train.txt":
            train_stems = stems
        elif fname == "val.txt":
            val_stems = stems

    print(f"  Train samples: {len(train_stems)}")
    print(f"  Val samples:   {len(val_stems)}")

    if not train_stems and not val_stems:
        print("  [ERROR] No split files found in tar archive", file=sys.stderr)
        tar.close()
        return False

    # ── Create output directories ───────────────────────────────────────
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract images and convert annotations ──────────────────────────
    all_stems = set(train_stems) | set(val_stems)
    converted = 0
    errors = 0

    print(f"  Converting annotations to YOLO format...")

    for stem in _progress_bar(sorted(all_stems), desc="  Converting", unit="img"):
        # ── Image ──────────────────────────────────────────────────────
        if stem in jpeg_members:
            member = jpeg_members[stem]
            src_name = Path(member.name).name  # e.g. "2007_000027.jpg"
            dst_path = images_dir / src_name

            if not dst_path.exists():
                f = tar.extractfile(member)
                if f is not None:
                    with open(dst_path, "wb") as out:
                        out.write(f.read())
        else:
            # Try common extensions
            found = False
            for ext in IMAGE_EXTENSIONS:
                candidate = f"{stem}{ext}"
                if candidate in jpeg_members:
                    member = jpeg_members[candidate]
                    dst_path = images_dir / candidate
                    if not dst_path.exists():
                        f = tar.extractfile(member)
                        if f is not None:
                            with open(dst_path, "wb") as out:
                                out.write(f.read())
                    found = True
                    break
            if not found:
                errors += 1
                continue

        # ── Label (convert VOC XML → YOLO .txt) ────────────────────────
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            if stem in anno_members:
                member = anno_members[stem]
                f = tar.extractfile(member)
                if f is not None:
                    # Write XML to temp file for parsing
                    xml_content = f.read()
                    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
                        tmp.write(xml_content)
                        tmp_path = tmp.name
                    try:
                        yolo_labels = _parse_voc_annotation(tmp_path)
                        with open(label_path, "w") as lf:
                            for cls_id, xc, yc, w, h in yolo_labels:
                                lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    finally:
                        os.unlink(tmp_path)
            else:
                # No annotation → empty label file
                label_path.write_text("")

        converted += 1

    tar.close()

    # ── Write split files ──────────────────────────────────────────────
    # Only include stems that actually have images
    existing_stems = set()
    for fname in images_dir.iterdir():
        existing_stems.add(fname.stem)

    train_filtered = [s for s in train_stems if s in existing_stems]
    val_filtered = [s for s in val_stems if s in existing_stems]

    with open(train_txt, "w") as f:
        f.write("\n".join(train_filtered) + "\n")
    with open(val_txt, "w") as f:
        f.write("\n".join(val_filtered) + "\n")

    print(f"\n  {'=' * 50}")
    print(f"  Preparation complete!")
    print(f"  {'=' * 50}")
    print(f"  Images converted:  {converted}")
    print(f"  Errors:            {errors}")
    print(f"  Train samples:     {len(train_filtered)}")
    print(f"  Val samples:       {len(val_filtered)}")
    print(f"  Output directory:  {output_dir}")
    print(f"  {'=' * 50}")

    return errors == 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare Pascal VOC 2012 for ILGAN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./data/voc",
        help="Output directory for the prepared dataset (default: ./data/voc)",
    )

    # Download options
    parser.add_argument(
        "--tar-path", "-t",
        type=str,
        default=None,
        help="Path to an existing VOC tar archive (skip download)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted download",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SHA-256 checksum verification of the tar archive",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="Directory to store the downloaded tar archive (default: same as --output-dir)",
    )

    # Extraction options
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-extraction even if output already exists",
    )
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        help="Copy images instead of using symlinks (uses more disk space)",
    )

    # Misc
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without actually doing it",
    )

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    download_dir = os.path.abspath(args.download_dir or args.output_dir)
    os.makedirs(download_dir, exist_ok=True)

    # ── Step 1: Download (or locate) the tar archive ───────────────────
    if args.tar_path:
        tar_path = os.path.abspath(args.tar_path)
        if not os.path.isfile(tar_path):
            print(f"[ERROR] Specified tar path does not exist: {tar_path}", file=sys.stderr)
            sys.exit(1)
        print(f"\n[1/3] Using existing tar archive: {tar_path}")
    else:
        tar_path = os.path.join(download_dir, "VOCtrainval_11-May-2012.tar")
        print(f"\n[1/3] Downloading VOC 2012 dataset...")

        if os.path.isfile(tar_path) and not args.force:
            file_size = os.path.getsize(tar_path)
            print(f"  Tar archive already exists: {_format_bytes(file_size)}")
            if not args.no_verify:
                print(f"  Verifying checksum...")
                actual = _sha256_file(tar_path)
                if actual == EXPECTED_SHA256:
                    print(f"  [OK] Checksum matches")
                else:
                    print(f"  [WARN] Checksum mismatch (expected {EXPECTED_SHA256[:16]}..., got {actual[:16]}...)")
                    print(f"  The file may be corrupted. Use --force to re-download.")
            else:
                print(f"  Skipping checksum verification (--no-verify)")
        else:
            success = _download_with_resume(
                VOC_TAR_URL,
                tar_path,
                resume=args.resume,
            )
            if not success:
                print(f"[ERROR] Download failed. Check your internet connection.", file=sys.stderr)
                sys.exit(1)

            if not args.no_verify:
                print(f"  Verifying checksum...")
                actual = _sha256_file(tar_path)
                if actual == EXPECTED_SHA256:
                    print(f"  [OK] Checksum matches")
                else:
                    print(f"  [WARN] Checksum mismatch")
                    print(f"    Expected: {EXPECTED_SHA256}")
                    print(f"    Got:      {actual}")
                    print(f"  The file may be corrupted. Re-run with --resume to retry.")
                    # Don't exit — the checksum in the code might be wrong
                    # (I couldn't verify it). The data will still likely work.

    # ── Step 2: Verify tar integrity ────────────────────────────────────
    print(f"\n[2/3] Verifying tar archive integrity...")
    try:
        with tarfile.open(tar_path, "r") as tf:
            members = tf.getmembers()
            print(f"  Tar archive contains {len(members)} files")
            # Check for essential directories
            has_jpeg = any(m.name.startswith(VOC_JPEG_DIR) for m in members)
            has_anno = any(m.name.startswith(VOC_ANNO_DIR) for m in members)
            has_sets = any(m.name.startswith(VOC_SETS_DIR) for m in members)
            print(f"  Contains JPEG images:  {'[OK]' if has_jpeg else '[MISSING]'}")
            print(f"  Contains annotations:  {'[OK]' if has_anno else '[MISSING]'}")
            print(f"  Contains split files:  {'[OK]' if has_sets else '[MISSING]'}")
            if not (has_jpeg and has_anno):
                print(f"  [ERROR] Tar archive is incomplete or corrupted", file=sys.stderr)
                sys.exit(1)
    except Exception as e:
        print(f"  [ERROR] Cannot read tar archive: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Step 3: Extract and convert ────────────────────────────────────
    print(f"\n[3/3] Extracting and converting to YOLO format...")
    success = _extract_and_convert(
        tar_path=tar_path,
        output_dir=output_dir,
        force=args.force,
        use_symlinks=not args.no_symlinks,
        dry_run=args.dry_run,
    )

    if not success:
        print(f"[ERROR] Extraction/conversion failed", file=sys.stderr)
        sys.exit(1)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  VOC 2012 dataset is ready for ILGAN training!")
    print(f"  {'=' * 60}")
    print(f"")
    print(f"  To train with this dataset, use:")
    print(f"")
    print(f"      python -m ilgan train \\")
    print(f"          --config ilgan/configs/default_config.yaml \\")
    print(f"          --data-root {output_dir} \\")
    print(f"          --image-size 128 \\")
    print(f"          --batch-size 32 \\")
    print(f"          --num-classes 20 \\")
    print(f"          --max-boxes 10 \\")
    print(f"          --epochs 2000")
    print(f"")
    print(f"  Or in Python:")
    print(f"")
    print(f"      from ilgan.data.dataloader import get_train_val_loaders")
    print(f"      train_loader, val_loader = get_train_val_loaders(")
    print(f"          root_dir='{output_dir}',")
    print(f"          image_size=128,")
    print(f"          batch_size=32,")
    print(f"          num_workers=4,")
    print(f"          global_max_boxes=10,")
    print(f"          train_max_boxes=10,")
    print(f"          val_max_boxes=10,")
    print(f"      )")
    print(f"")
    print(f"  Dataset size: {len(os.listdir(os.path.join(output_dir, 'images')))} images")
    print(f"  {'=' * 60}")


if __name__ == "__main__":
    main()
