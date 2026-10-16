"""Train the unified detector with Ultralytics YOLO.

Two routes are supported, matching what the original project did:

``--route roboflow`` (default)
    Pull the dataset from Roboflow in YOLOv11 format and train locally. This is
    what ``train.py`` in the original submission did, and it needs a Roboflow
    account and an API key.

``--route local``
    Train on an existing ``data.yaml`` produced by ``src.merge_datasets``. No
    Roboflow account needed — use this to reproduce the training from datasets
    you downloaded yourself.

Note on the original: the project report says the training script "dynamically
adjusts hyperparameters and batch sizes" between the easy and hard splits. The
script as submitted did not do that — it always trained ``yolo11n.pt`` for 100
epochs at 640 px. This version exposes those as flags rather than inventing
behaviour that was not there.

Usage:
    export ROBOFLOW_API_KEY=...
    python -m src.train --route roboflow --epochs 100 --imgsz 640
    python -m src.train --route local --data data/unified/data.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_MODEL = "yolo11n.pt"


def _require_api_key() -> str:
    """Return the Roboflow API key from the environment.

    The key must never be committed: an earlier revision of this project had it
    hard-coded in the source. See CHANGES.md.
    """
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "ROBOFLOW_API_KEY is not set.\n"
            "Get a key from https://app.roboflow.com/ (Settings -> API Keys), then:\n"
            "    export ROBOFLOW_API_KEY=your_key_here\n"
            "or put it in a .env file (see .env.example) and source that."
        )
    return key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the unified detector.")
    parser.add_argument("--route", choices=("roboflow", "local"), default="roboflow")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO weights to start from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 = auto")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="train")
    parser.add_argument(
        "--data", help="path to data.yaml (required for --route local)"
    )
    # Roboflow route options
    parser.add_argument("--workspace", default="main-hksvg")
    parser.add_argument("--project-name", default="traffic-sign-light-detector")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--dataset-format", default="yolov11")
    args = parser.parse_args(argv)

    if args.route == "local" and not args.data:
        parser.error("--route local requires --data path/to/data.yaml")

    from ultralytics import YOLO

    if args.route == "roboflow":
        from roboflow import Roboflow

        rf = Roboflow(api_key=_require_api_key())
        project = rf.workspace(args.workspace).project(args.project_name)
        dataset = project.version(args.version).download(args.dataset_format)
        data_yaml = os.path.join(dataset.location, "data.yaml")
        print(f"dataset downloaded to {dataset.location}")
    else:
        data_yaml = args.data

    model = YOLO(args.model)
    print(f"training on {data_yaml} for {args.epochs} epochs at {args.imgsz}px ...")
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    print(f"done. best weights: {results.save_dir}/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
