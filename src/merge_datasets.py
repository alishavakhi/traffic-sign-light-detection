"""Merge the two converted datasets into one YOLO training set.

The original project performed this step (its report describes merging the two
normalized datasets into a single one) but no script for it was included in the
submission. This module is a **reconstruction** written for this repository so the
pipeline is reproducible end to end; it implements the merge as described, and is
not the exact code that produced the original training set. See CHANGES.md.

Both converters already emit the unified class indices defined in
``src/classes.py`` (traffic signs 0-19, traffic light states 20-32), so merging is
a matter of collecting image/label pairs into one directory tree and emitting a
``data.yaml`` for Ultralytics YOLO.

Usage:
    python -m src.merge_datasets \
        --source signs=path/to/signs \
        --source lights=path/to/lights \
        --output data/unified --width 1280 --height 960
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

from . import classes

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".pgm")

DATA_YAML = """# Unified traffic sign + traffic light dataset, in YOLO format.
#
# Class indices are frozen and defined in src/classes.py:
#   0-19   traffic signs
#   20-32  traffic light states
#
# Paths are relative to this file's location.

path: {root}
train: images/train
val: images/val

names:
{names}
"""


def _files_with_suffixes(directory: Path) -> list[Path]:
    out = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return out


def _unique_name(path: Path, seen: dict[str, int]) -> str:
    """Return a collision-free filename for ``path``.

    Two source datasets can perfectly well contain ``0001.jpg``. Without this,
    the second one silently overwrites the first and the training set quietly
    shrinks.
    """
    name = path.name
    if name in seen:
        digest = hashlib.sha1(str(path).encode()).hexdigest()[:8]
        name = f"{path.stem}_{digest}{path.suffix}"
    seen[name] = seen.get(name, 0) + 1
    return name


def merge(
    sources: list[Path],
    output: Path,
    val_fraction: float = 0.0,
    seed: int = 0,
) -> dict:
    """Collect image/label pairs from ``sources`` into ``output``.

    Args:
        sources: directories searched recursively for images.
        output: destination root; ``images/`` and ``labels/`` are created inside.
        val_fraction: if non-zero, hold out this fraction of pairs as validation.
        seed: RNG seed for the split, so the result is reproducible.

    Returns:
        Summary counts.
    """
    import random

    img_dir = output / "images" / "train"
    lbl_dir = output / "labels" / "train"
    val_img = output / "images" / "val"
    val_lbl = output / "labels" / "val"
    for d in (img_dir, lbl_dir, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    seen: dict[str, int] = {}
    pairs: list[tuple[Path, Path | None, str]] = []
    unlabelled = 0

    for src in sources:
        if not src.is_dir():
            raise NotADirectoryError(f"source directory not found: {src}")
        for image in _files_with_suffixes(src):
            label = image.with_suffix(".txt")
            has_label = label.is_file()
            if not has_label:
                unlabelled += 1
            pairs.append((image, label if has_label else None, _unique_name(image, seen)))

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = int(len(pairs) * val_fraction)
    val_set = set(range(n_val))

    counts = {"train": 0, "val": 0, "boxes": 0, "empty_labels": 0}
    for i, (image, label, name) in enumerate(pairs):
        to_val = i in val_set
        dst_img = (val_img if to_val else img_dir) / name
        dst_lbl = (val_lbl if to_val else lbl_dir) / (Path(name).stem + ".txt")

        shutil.copy2(image, dst_img)
        if label is not None:
            text = label.read_text()
            dst_lbl.write_text(text)
            counts["boxes"] += sum(1 for ln in text.splitlines() if ln.strip())
            if not text.strip():
                counts["empty_labels"] += 1
        else:
            dst_lbl.write_text("")
            counts["empty_labels"] += 1

        counts["val" if to_val else "train"] += 1

    names = "\n".join(
        f"  {idx}: {label}" for label, idx in sorted(classes.CLASSES.items(), key=lambda kv: kv[1])
    )
    (output / "data.yaml").write_text(
        DATA_YAML.format(root=os.path.abspath(output), names=names)
    )

    return {**counts, "total": len(pairs), "unlabelled_images": unlabelled}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge converted datasets for YOLO.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a converted dataset root; repeatable",
    )
    parser.add_argument("--output", required=True, help="destination root")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="fraction held out for validation (default: none)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    sources = []
    for item in args.source:
        if "=" not in item:
            parser.error(f"--source expects NAME=PATH, got {item!r}")
        _name, path = item.split("=", 1)
        sources.append(Path(path))

    summary = merge(sources, Path(args.output), args.val_fraction, args.seed)

    print(f"images total        : {summary['total']}")
    print(f"  train / val       : {summary['train']} / {summary['val']}")
    print(f"bounding boxes      : {summary['boxes']}")
    print(f"empty label files   : {summary['empty_labels']}"
          "   (background images; expected and useful)")
    if summary["unlabelled_images"]:
        print(
            f"images with no label file: {summary['unlabelled_images']}"
            "   (copied as background)",
            file=sys.stderr,
        )
    print(f"data.yaml           : {os.path.join(args.output, 'data.yaml')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
