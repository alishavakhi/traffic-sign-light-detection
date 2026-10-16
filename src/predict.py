"""Run the trained detector over images and write annotated copies.

Two backends:

``--backend roboflow``
    Query the hosted Roboflow model over HTTP. This is what ``predict.py`` /
    ``test.py`` in the original submission did, and it is how the images in
    ``samples/predictions/`` were produced. Needs ``ROBOFLOW_API_KEY``.

``--backend local``
    Run a local ``.pt`` checkpoint with Ultralytics. No account or network
    needed — use this if you trained with ``--route local``.

Boxes are drawn in the same style as the original (1 px green rectangle, label in
green above the box) so new output is directly comparable with the recorded
results.

Usage:
    export ROBOFLOW_API_KEY=...
    python -m src.predict --backend roboflow --split easy
    python -m src.predict --backend local --weights runs/detect/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from . import classes

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

BOX_COLOUR = (0, 255, 0)  # BGR green, as in the original implementation
BOX_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
LABEL_OFFSET = 10


def draw_detections(image, detections):
    """Draw boxes and labels onto ``image`` in place.

    Args:
        image: BGR array from ``cv2.imread``.
        detections: iterable of ``(x0, y0, x1, y1, label)``.

    Returns:
        The annotated image.
    """
    for x0, y0, x1, y1, label in detections:
        cv2.rectangle(image, (x0, y0), (x1, y1), BOX_COLOUR, BOX_THICKNESS)
        cv2.putText(
            image,
            label,
            (x0, y0 - LABEL_OFFSET),
            fontFace=FONT,
            fontScale=FONT_SCALE,
            color=BOX_COLOUR,
            thickness=FONT_THICKNESS,
        )
    return image


def _iter_images(directory: Path):
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            yield p


def _detections_roboflow(model, image_path: Path, confidence: int, overlap: int):
    """Query the hosted Roboflow model and yield boxes in xyxy + label form."""
    prediction = model.predict(
        str(image_path), confidence=confidence, overlap=overlap
    ).json()
    for box in prediction.get("predictions", []):
        x0 = box["x"] - box["width"] / 2
        x1 = box["x"] + box["width"] / 2
        y0 = box["y"] - box["height"] / 2
        y1 = box["y"] + box["height"] / 2
        # The class index arrives as a string from the API.
        yield (int(x0), int(y0), int(x1), int(y1), classes.name_for(box["class"]))


def _detections_local(model, image_path: Path, conf: float):
    """Run a local checkpoint and yield boxes in the same form."""
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    for result in results:
        names = result.names
        for box in result.boxes:
            x0, y0, x1, y1 = (int(v) for v in box.xyxy[0].tolist())
            label = names.get(int(box.cls[0]), str(int(box.cls[0])))
            yield (x0, y0, x1, y1, label)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect traffic signs and lights and write annotated images."
    )
    parser.add_argument("--backend", choices=("roboflow", "local"), default="roboflow")
    parser.add_argument("--input", help="input directory (overrides --split)")
    parser.add_argument("--output", help="output directory (overrides --split)")
    parser.add_argument(
        "--split",
        default="easy",
        help="convenience preset: reads samples/inputs/<split>, writes samples/predictions/<split>",
    )
    parser.add_argument("--confidence", type=int, default=40, help="Roboflow: percent")
    parser.add_argument("--overlap", type=int, default=30, help="Roboflow: percent")
    parser.add_argument("--conf", type=float, default=0.25, help="local: 0-1")
    parser.add_argument("--weights", help="local .pt checkpoint")
    parser.add_argument("--workspace", default="main-hksvg")
    parser.add_argument("--project-name", default="traffic-sign-light-detector")
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent.parent
    src_dir = Path(args.input) if args.input else here / "samples" / "inputs" / args.split
    dst_dir = Path(args.output) if args.output else here / "samples" / "predictions" / args.split

    if not src_dir.is_dir():
        print(f"input directory not found: {src_dir}", file=sys.stderr)
        return 2
    dst_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "roboflow":
        import os

        from roboflow import Roboflow

        key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
        if not key:
            print(
                "ROBOFLOW_API_KEY is not set. Get one from app.roboflow.com and\n"
                "    export ROBOFLOW_API_KEY=your_key_here\n"
                "or use --backend local with a .pt checkpoint.",
                file=sys.stderr,
            )
            return 2
        rf = Roboflow(api_key=key)
        model = (
            rf.workspace(args.workspace)
            .project(args.project_name)
            .version(args.version)
            .model
        )
        detect = lambda p: _detections_roboflow(model, p, args.confidence, args.overlap)
    else:
        if not args.weights:
            print("--backend local requires --weights path/to/best.pt", file=sys.stderr)
            return 2
        from ultralytics import YOLO

        model = YOLO(args.weights)
        detect = lambda p: _detections_local(model, p, args.conf)

    n_images = n_boxes = 0
    for image_path in _iter_images(src_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"could not read {image_path}", file=sys.stderr)
            continue
        boxes = list(detect(image_path))
        draw_detections(image, boxes)
        cv2.imwrite(str(dst_dir / image_path.name), image)
        n_images += 1
        n_boxes += len(boxes)
        print(f"{image_path.name}: {len(boxes)} detection(s)")

    print(f"\n{n_images} image(s) processed, {n_boxes} detection(s) total")
    print(f"written to {dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
