#!/usr/bin/env python3
"""Converter tests — normalization arithmetic and malformed-input handling.

These exercise the two annotation converters on small hand-built fixtures in the
documented input formats. They verify the **conversion arithmetic** and the
corrections the converters claim to apply. They do not test detection accuracy and
do not run the model.

Run with either:
    python -m pytest tests/ -v
    python tests/test_converters.py          # no pytest needed
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import bstld_to_yolo, classes, signs_to_yolo  # noqa: E402

TOL = 1e-6


def approx(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


def parse_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        assert len(parts) == 5, f"expected 5 fields, got {line!r}"
        out.append((int(parts[0]), *(float(v) for v in parts[1:])))
    return out


# --------------------------------------------------------------------------
# BSTLD converter
# --------------------------------------------------------------------------

def test_bstld_normalisation() -> None:
    """centre = (corner sum)/2/dim, size = extent/dim, class = unified index."""
    tmp = Path(tempfile.mkdtemp())
    try:
        for name in ("a.png", "b.png"):
            (tmp / name).write_bytes(b"")
        (tmp / "train.yaml").write_text(
            "- path: a.png\n"
            "  boxes:\n"
            "    - {x_min: 100, x_max: 300, y_min: 200, y_max: 400, label: Green}\n"
        )
        labels = bstld_to_yolo.get_all_labels(str(tmp / "train.yaml"))
        written, empty, _, unmapped = bstld_to_yolo.write_yolo_labels(labels)

        assert written == 1, written
        assert not unmapped, unmapped
        rows = parse_label_file(tmp / "a.txt")
        assert len(rows) == 1, rows
        cls, cx, cy, w, h = rows[0]
        assert cls == classes.CLASSES["Green"] == 24, cls
        assert approx(cx, 200 / 1280), cx
        assert approx(cy, 300 / 720), cy
        assert approx(w, 200 / 1280), w
        assert approx(h, 200 / 720), h
    finally:
        shutil.rmtree(tmp)


def test_bstld_swapped_corners_are_corrected() -> None:
    """x_min > x_max must not produce a negative width."""
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "a.png").write_bytes(b"")
        (tmp / "train.yaml").write_text(
            "- path: a.png\n"
            "  boxes:\n"
            "    - {x_min: 300, x_max: 100, y_min: 400, y_max: 200, label: Red}\n"
        )
        labels = bstld_to_yolo.get_all_labels(str(tmp / "train.yaml"))
        # the swap happens in get_all_labels
        box = labels[0]["boxes"][0]
        assert box["x_min"] == 100 and box["x_max"] == 300, box
        assert box["y_min"] == 200 and box["y_max"] == 400, box

        bstld_to_yolo.write_yolo_labels(labels)
        cls, cx, cy, w, h = parse_label_file(tmp / "a.txt")[0]
        assert cls == classes.CLASSES["Red"] == 22, cls
        assert w > 0 and h > 0, (w, h)
        assert approx(cx, 200 / 1280) and approx(cy, 300 / 720), (cx, cy)
    finally:
        shutil.rmtree(tmp)


def test_bstld_out_of_bounds_is_clipped() -> None:
    """A box past the right edge is clamped, keeping the centre <= 1."""
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "a.png").write_bytes(b"")
        (tmp / "train.yaml").write_text(
            "- path: a.png\n"
            "  boxes:\n"
            "    - {x_min: 1200, x_max: 1400, y_min: 100, y_max: 200, label: Yellow}\n"
        )
        labels = bstld_to_yolo.get_all_labels(str(tmp / "train.yaml"))
        bstld_to_yolo.write_yolo_labels(labels)
        cls, cx, cy, w, h = parse_label_file(tmp / "a.txt")[0]
        assert cls == classes.CLASSES["Yellow"] == 20, cls
        assert 0 <= cx <= 1 and 0 <= cy <= 1, (cx, cy)
        assert cx + w / 2 <= 1.0 + TOL, f"box extends past the frame: {cx + w/2}"
    finally:
        shutil.rmtree(tmp)


def test_bstld_unmapped_label_is_reported_not_guessed() -> None:
    """A label outside the unified index is reported, never silently remapped."""
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "a.png").write_bytes(b"")
        (tmp / "train.yaml").write_text(
            "- path: a.png\n"
            "  boxes:\n"
            "    - {x_min: 10, x_max: 20, y_min: 10, y_max: 20, label: BlueFlashing}\n"
        )
        labels = bstld_to_yolo.get_all_labels(str(tmp / "train.yaml"))
        written, _empty, _missing, unmapped = bstld_to_yolo.write_yolo_labels(labels)
        assert unmapped == ["BlueFlashing"], unmapped
        assert (tmp / "a.txt").read_text() == "", "unmapped label must not be written"
    finally:
        shutil.rmtree(tmp)


def test_bstld_unlabelled_images_kept_by_default() -> None:
    """An image with no boxes is not deleted unless asked."""
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "a.png").write_bytes(b"")
        (tmp / "train.yaml").write_text("- path: a.png\n  boxes: []\n")
        labels = bstld_to_yolo.get_all_labels(str(tmp / "train.yaml"))
        written, empty, _, _ = bstld_to_yolo.write_yolo_labels(
            labels, drop_unlabelled=False
        )
        assert written == 0 and empty == 1, (written, empty)
        assert (tmp / "a.png").exists(), "image must survive the default path"
    finally:
        shutil.rmtree(tmp)


# --------------------------------------------------------------------------
# Traffic sign converter
# --------------------------------------------------------------------------

def test_signs_normalisation() -> None:
    """Field order is visibility, lrx, lry, ulx, uly, -, name."""
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "img1.jpg").write_bytes(b"")
        (tmp / "annotations.txt").write_text(
            "img1.jpg:3, 100, 200, 50, 100, 1, STOP;\n"
        )
        parsed = signs_to_yolo.parse_sign_annotations(str(tmp / "annotations.txt"))
        assert len(parsed) == 1
        written, skipped, unmapped = signs_to_yolo.write_labels(
            parsed, str(tmp), 1280, 960
        )
        assert written == 1 and skipped == 0, (written, skipped)
        assert not unmapped, unmapped

        cls, cx, cy, w, h = parse_label_file(tmp / "img1.txt")[0]
        assert cls == classes.CLASSES["STOP"] == 18, cls
        # signBB = [ulx, uly, lrx, lry] = [50, 100, 100, 200]; centre = [75, 150]
        assert approx(cx, 75 / 1280), cx
        assert approx(cy, 150 / 960), cy
        assert approx(w, 50 / 1280), w
        assert approx(h, 100 / 960), h
    finally:
        shutil.rmtree(tmp)


def test_signs_misc_signs_record_is_skipped() -> None:
    """A MISC_SIGNS record has no usable geometry and must yield no box."""
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "img2.jpg").write_bytes(b"")
        (tmp / "annotations.txt").write_text(
            "img2.jpg:3, 300, 400, 100, 200, 1, MISC_SIGNS;\n"
        )
        parsed = signs_to_yolo.parse_sign_annotations(str(tmp / "annotations.txt"))
        assert signs_to_yolo.has_no_label(parsed[0]) is True
        written, skipped, _ = signs_to_yolo.write_labels(parsed, str(tmp), 1280, 960)
        assert written == 0 and skipped == 1, (written, skipped)
        assert not (tmp / "img2.txt").exists(), "no label file for a MISC_SIGNS image"
    finally:
        shutil.rmtree(tmp)


def test_signs_unmapped_class_is_reported() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "img3.jpg").write_bytes(b"")
        (tmp / "annotations.txt").write_text(
            "img3.jpg:3, 10, 20, 5, 10, 1, NOT_A_REAL_CLASS;\n"
        )
        parsed = signs_to_yolo.parse_sign_annotations(str(tmp / "annotations.txt"))
        _written, _skipped, unmapped = signs_to_yolo.write_labels(
            parsed, str(tmp), 1280, 960
        )
        assert unmapped == ["NOT_A_REAL_CLASS"], unmapped
    finally:
        shutil.rmtree(tmp)


def test_signs_destructive_delete_is_opt_in() -> None:
    """--delete-unlabelled removes files; the default path must not."""
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "img2.jpg").write_bytes(b"")
        (tmp / "annotations.txt").write_text(
            "img2.jpg:3, 300, 400, 100, 200, 1, MISC_SIGNS;\n"
        )
        parsed = signs_to_yolo.parse_sign_annotations(str(tmp / "annotations.txt"))
        signs_to_yolo.write_labels(parsed, str(tmp), 1280, 960)
        assert (tmp / "img2.jpg").exists(), "default path must not delete images"
        removed = signs_to_yolo.delete_unlabelled(parsed, str(tmp))
        assert removed == 1 and not (tmp / "img2.jpg").exists()
    finally:
        shutil.rmtree(tmp)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    failures = []
    for fn in TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failures.append(fn.__name__)
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failures.append(fn.__name__)
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
