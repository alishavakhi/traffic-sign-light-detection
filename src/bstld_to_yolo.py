"""Convert Bosch Small Traffic Lights Dataset (BSTLD) annotations to YOLO format.

Reads the dataset's YAML label file and writes one normalized
``<class> <cx> <cy> <w> <h>`` label file per image, alongside it. Images with no
annotated boxes are reported (and optionally removed) because an empty label file
is indistinguishable from an unlabelled image to the YOLO loader.

Derived from the BSTLD sample script ``read_label_file.py``
(https://github.com/bosch-ros-pkg/bstld), MIT License,
Copyright (c) 2018 Robert Bosch LLC, Karsten Behrendt. The changes here are the
YOLO emission and the command-line interface; the parsing, the x/y ordering fix
and the clipping logic are the original authors' work. See
THIRD_PARTY_NOTICES.md.

The BSTLD *dataset* is released for non-commercial use only. This script only
reads it; no dataset content is redistributed with this repository.

Usage:
    python -m src.bstld_to_yolo dataset_train_rgb/train.yaml
    python -m src.bstld_to_yolo dataset_train_rgb/train.yaml --drop-unlabelled
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

from . import classes

#: BSTLD RGB frames are 1280x720; the RIIB variant is 1280x736. Overridable,
#: because normalizing against the wrong dimension silently corrupts every box.
WIDTH = 1280
HEIGHT = 720


def get_all_labels(input_yaml: str, riib: bool = False, clip: bool = True) -> list[dict]:
    """Load BSTLD labels from ``input_yaml``.

    Args:
        input_yaml: path to the dataset's ``train.yaml`` / ``test.yaml``.
        riib: set for the RIIB (raw) images, which are 1280x736 and carry an
            8-pixel header, so paths and y coordinates are shifted.
        clip: clamp boxes to the image bounds.

    Returns:
        The parsed label records, with absolute image paths.

    Raises:
        FileNotFoundError: if ``input_yaml`` does not exist.
        ValueError: if the file does not look like a BSTLD label file.
    """
    if not os.path.isfile(input_yaml):
        raise FileNotFoundError(f"Input yaml {input_yaml} does not exist")

    with open(input_yaml, "rb") as iy_handle:
        images = yaml.safe_load(iy_handle)

    if not images or not isinstance(images[0], dict) or "path" not in images[0]:
        raise ValueError(f"Something seems wrong with this label-file: {input_yaml}")

    for i in range(len(images)):
        images[i]["path"] = os.path.abspath(
            os.path.join(os.path.dirname(input_yaml), images[i]["path"])
        )

        # At least one annotation in the dataset has xmin > xmax / ymin > ymax.
        for j, box in enumerate(images[i]["boxes"]):
            if box["x_min"] > box["x_max"]:
                images[i]["boxes"][j]["x_min"], images[i]["boxes"][j]["x_max"] = (
                    images[i]["boxes"][j]["x_max"],
                    images[i]["boxes"][j]["x_min"],
                )
            if box["y_min"] > box["y_max"]:
                images[i]["boxes"][j]["y_min"], images[i]["boxes"][j]["y_max"] = (
                    images[i]["boxes"][j]["y_max"],
                    images[i]["boxes"][j]["y_min"],
                )

        # At least one annotation extends past the right edge.
        if clip:
            for j, box in enumerate(images[i]["boxes"]):
                images[i]["boxes"][j]["x_min"] = max(min(box["x_min"], WIDTH - 1), 0)
                images[i]["boxes"][j]["x_max"] = max(min(box["x_max"], WIDTH - 1), 0)
                images[i]["boxes"][j]["y_min"] = max(min(box["y_min"], HEIGHT - 1), 0)
                images[i]["boxes"][j]["y_max"] = max(min(box["y_max"], HEIGHT - 1), 0)

        # RIIB images carry an extra 8-pixel header and live in a parallel tree.
        if riib:
            images[i]["path"] = images[i]["path"].replace(".png", ".pgm")
            images[i]["path"] = images[i]["path"].replace("rgb/train", "riib/train")
            images[i]["path"] = images[i]["path"].replace("rgb/test", "riib/test")
            for box in images[i]["boxes"]:
                box["y_max"] = box["y_max"] + 8
                box["y_min"] = box["y_min"] + 8

    return images


def write_yolo_labels(
    labels: list[dict],
    drop_unlabelled: bool = False,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> tuple[int, int, int, list[str]]:
    """Write one YOLO label file per image.

    Args:
        labels: output of :func:`get_all_labels`.
        drop_unlabelled: delete images that have no boxes. Off by default —
            this deletes files from the dataset tree.
        width: normalization width.
        height: normalization height.

    Returns:
        ``(written, empty, missing, unmapped)`` where ``unmapped`` lists labels
        that are not present in the unified class set.
    """
    written = empty = missing = 0
    unmapped: list[str] = []

    for label_info in labels:
        if len(label_info["boxes"]) == 0:
            empty += 1
            if drop_unlabelled:
                try:
                    os.remove(label_info["path"])
                except OSError:
                    missing += 1
            continue

        with open(label_info["path"].replace(".png", ".txt"), "w") as file:
            for box in label_info["boxes"]:
                label = box["label"]
                if label not in classes.CLASSES:
                    unmapped.append(label)
                    continue
                class_inx = classes.CLASSES[label]
                ulx, uly = box["x_min"], box["y_min"]
                lrx, lry = box["x_max"], box["y_max"]
                center_x = (ulx + lrx) / 2 / width
                center_y = (uly + lry) / 2 / height
                box_w = (lrx - ulx) / width
                box_h = (lry - uly) / height
                file.write(f"{class_inx} {center_x} {center_y} {box_w} {box_h}\n")
        written += 1

    return written, empty, missing, sorted(set(unmapped))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert BSTLD YAML annotations to YOLO label files."
    )
    parser.add_argument("input_yaml", help="path to train.yaml / test.yaml")
    parser.add_argument(
        "--riib", action="store_true", help="labels are for the RIIB (raw) images"
    )
    parser.add_argument(
        "--no-clip", action="store_true", help="do not clamp boxes to image bounds"
    )
    parser.add_argument(
        "--drop-unlabelled",
        action="store_true",
        help="delete images with no boxes (destructive; off by default)",
    )
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    args = parser.parse_args(argv)

    labels = get_all_labels(args.input_yaml, riib=args.riib, clip=not args.no_clip)
    written, empty, missing, unmapped = write_yolo_labels(
        labels,
        drop_unlabelled=args.drop_unlabelled,
        width=args.width,
        height=args.height,
    )

    print(f"images with labels written : {written}")
    print(f"images with no boxes       : {empty}"
          + ("  (deleted)" if args.drop_unlabelled else "  (kept)"))
    if missing:
        print(f"could not delete           : {missing}")
    if unmapped:
        print(f"labels not in the unified class set: {unmapped}", file=sys.stderr)
        print("add them to src/classes.py or they will be skipped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
