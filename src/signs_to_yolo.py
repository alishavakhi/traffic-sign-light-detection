"""Convert a text-annotated traffic sign dataset to YOLO format.

The dataset this reads is a directory of images plus a single ``annotations.txt``
whose lines have the form::

    <image name>:<sign record>;<sign record>;...

where each sign record is seven comma-separated fields::

    <visibility>, <lrx>, <lry>, <ulx>, <uly>, <sign type>, <sign name>

A record whose first field is ``MISC_SIGNS`` carries no usable geometry and is
skipped. Images with no usable record are left alone (the dataset also has a mode
that deletes them) because an empty label file is indistinguishable from an
unlabelled image to the YOLO loader.

Provenance: this parser was written against the annotation format of the traffic
sign dataset used in the original project. The dataset is not redistributed with
this repository and its exact source is not recorded in the original work, so no
attribution is claimed here — see THIRD_PARTY_NOTICES.md, which flags this as an
open attribution item.

Usage:
    python -m src.signs_to_yolo --dataset-root Set2Part0 --width 1280 --height 960
    python -m src.signs_to_yolo --dataset-root Set2Part0 --delete-unlabelled
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from . import classes

#: Field separator offsets: each field is preceded by ", ".
_FIELD_OFFSET = 2


def parse_sign_annotations(file_name: str) -> list[dict]:
    """Parse the dataset's annotation file.

    Args:
        file_name: path to ``annotations.txt``.

    Returns:
        One record per image: ``{'name': str, 'signs': [ {...}, ... ]}``, in file
        order. Each sign carries ``signTypes``, ``signStatus``, ``signBB``
        (``[ulx, uly, lrx, lry]``), ``signC`` (centre), ``signSize`` and
        ``aspectRatio``.

    Raises:
        FileNotFoundError: if the annotation file does not exist.
    """
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"annotation file not found: {file_name}")

    content: list[dict] = []

    with open(file_name, "r") as fid:
        while True:
            line = fid.readline().strip()

            if not line:
                break  # end of file

            # The colon separates the image name from its annotations.
            name_end = line.find(":")
            if name_end == -1:
                print(f"ERROR: malformed annotation line, stopping: {line[:80]}")
                break

            im_name = line[:name_end]
            image_info: dict = {"name": im_name, "signs": []}

            if len(line) == name_end:
                # No signs annotated in this image.
                content.append(image_info)
                continue

            sign_end = line.find(";", name_end)

            while sign_end != -1:
                if line[name_end + 1:sign_end] == "MISC_SIGNS":
                    # A miscelaneous sign carries no box; the number of these is
                    # why some images are dropped rather than labelled.
                    image_info["signs"].append(
                        {
                            "signTypes": "MISC_SIGNS",
                            "signStatus": "N/A",
                            "signBB": [-1, -1, -1, -1],
                            "signC": [-1, -1],
                            "signSize": 0,
                            "aspectRatio": 0,
                        }
                    )
                    line = line[sign_end + 1:]
                    sign_end = line.find(";")
                else:
                    commas = [m.start() for m in re.finditer(",", line)]

                    visibility = line[:commas[0]].split(":")[-1]

                    lrx = float(line[commas[0] + _FIELD_OFFSET:commas[1]])
                    lry = float(line[commas[1] + _FIELD_OFFSET:commas[2]])
                    ulx = float(line[commas[2] + _FIELD_OFFSET:commas[3]])
                    uly = float(line[commas[3] + _FIELD_OFFSET:commas[4]])
                    sign_name = line[commas[5] + _FIELD_OFFSET:sign_end]

                    image_info["signs"].append(
                        {
                            "signTypes": sign_name,
                            "signStatus": visibility,
                            "signBB": [ulx, uly, lrx, lry],
                            "signC": [(ulx + lrx) / 2, (uly + lry) / 2],
                            # NOTE: preserved from the original implementation.
                            # ulx <= lrx, so this is a squared negative number and
                            # is not a usable area. The field is unused by the YOLO
                            # export; it is kept so the parsed record matches the
                            # original, and flagged here rather than silently
                            # changed.
                            "signSize": min((ulx - lrx), (uly - lry)) ** 2,
                            "aspectRatio": max(
                                (ulx - lrx) / (uly - lry + 1e-5),
                                (uly - lry) / (ulx - lrx + 1e-5),
                            ),
                        }
                    )

                    line = line[sign_end + 1:]
                    sign_end = line.find(";")

            content.append(image_info)

    return content


def has_no_label(image_info: dict) -> bool:
    """True if the image has no usable sign annotation."""
    return image_info["signs"] == [] or (
        len(image_info["signs"]) == 1
        and image_info["signs"][0]["signTypes"] == "MISC_SIGNS"
    )


def delete_unlabelled(parsed_data: list[dict], dataset_root: str) -> int:
    """Delete images that carry no usable annotation. Destructive.

    Returns:
        Number of images removed.
    """
    removed = 0
    for image_info in parsed_data:
        if has_no_label(image_info):
            try:
                os.remove(os.path.join(dataset_root, image_info["name"]))
                removed += 1
            except OSError:
                pass
    return removed


def write_labels(
    parsed_data: list[dict],
    dataset_root: str,
    width: int,
    height: int,
) -> tuple[int, int, list[str]]:
    """Write one YOLO label file per image.

    Args:
        parsed_data: output of :func:`parse_sign_annotations`.
        dataset_root: directory holding the images.
        width: normalization width, in pixels.
        height: normalization height, in pixels.

    Returns:
        ``(written, skipped, unmapped)`` where ``unmapped`` lists dataset labels
        absent from the unified class set.
    """
    written = skipped = 0
    unmapped: list[str] = []

    for image_info in parsed_data:
        if has_no_label(image_info):
            skipped += 1
            continue

        label_path = os.path.join(
            dataset_root, os.path.splitext(image_info["name"])[0] + ".txt"
        )
        with open(label_path, "w") as file:
            for sign in image_info["signs"]:
                if sign["signTypes"] in ("MISC_SIGNS", ""):
                    continue
                if sign["signTypes"] not in classes.CLASSES:
                    unmapped.append(sign["signTypes"])
                    continue
                class_inx = classes.CLASSES[sign["signTypes"]]
                center_x = sign["signC"][0] / width
                center_y = sign["signC"][1] / height
                box_w = (sign["signBB"][2] - sign["signBB"][0]) / width
                box_h = (sign["signBB"][3] - sign["signBB"][1]) / height
                file.write(f"{class_inx} {center_x} {center_y} {box_w} {box_h}\n")
        written += 1

    return written, skipped, sorted(set(unmapped))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a text-annotated traffic sign dataset to YOLO labels."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="directory containing the images and annotations.txt",
    )
    parser.add_argument("--annotations", default="annotations.txt")
    parser.add_argument("--width", type=int, required=True, help="image width in px")
    parser.add_argument("--height", type=int, required=True, help="image height in px")
    parser.add_argument(
        "--delete-unlabelled",
        action="store_true",
        help="delete images with no usable annotation (destructive; off by default)",
    )
    args = parser.parse_args(argv)

    parsed = parse_sign_annotations(os.path.join(args.dataset_root, args.annotations))

    if args.delete_unlabelled:
        removed = delete_unlabelled(parsed, args.dataset_root)
        print(f"images deleted (no usable annotation): {removed}")

    written, skipped, unmapped = write_labels(
        parsed, args.dataset_root, args.width, args.height
    )

    print(f"label files written : {written}")
    print(f"images skipped      : {skipped}")
    if unmapped:
        print(f"labels not in the unified class set: {unmapped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
