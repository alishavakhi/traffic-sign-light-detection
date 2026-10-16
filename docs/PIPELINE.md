# Processing pipeline

How the modules fit together, and what each one is responsible for.

## Overview

```
                    TRAFFIC SIGNS                          TRAFFIC LIGHTS
        images + annotations.txt                frames + per-frame YAML labels
                    │                                        │
                    ▼                                        ▼
          src/signs_to_yolo.py                     src/bstld_to_yolo.py
      parse "<img>:<record>;<record>;"          parse YAML, fix corner order,
      skip MISC_SIGNS, emit normalized          clip to frame, emit normalized
      "cls cx cy w h"                           "cls cx cy w h"
                    │                                        │
                    └───────────────┬────────────────────────┘
                                    ▼
                          src/classes.py  (unified index)
                          signs 0-19 · lights 20-32
                                    │
                                    ▼
                        src/merge_datasets.py
                collect pairs · de-collide filenames · data.yaml
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          src/train.py --route local      src/train.py --route roboflow
            (own data.yaml)                 (dataset uploaded to the platform)
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                             yolo11n fine-tune
                                    │
                                    ▼
                            src/predict.py
            draws boxes in the original style, writes annotated copies
```

## Module responsibilities

| Module | Role |
|---|---|
| `src/classes.py` | The unified 33-class index. The single source of truth for class numbering; `CLASS_NAMES` is derived from `CLASSES` so the two cannot drift. |
| `src/bstld_to_yolo.py` | BSTLD YAML → YOLO labels. Derived from the dataset's MIT-licensed sample script. |
| `src/signs_to_yolo.py` | Sign `annotations.txt` → YOLO labels. Original code written against the observed format. |
| `src/merge_datasets.py` | Combines converted datasets into one tree plus a `data.yaml`. Reconstructed for this repository — see `docs/CHANGES.md`. |
| `src/train.py` | Fine-tunes `yolo11n` for 100 epochs at 640 px by default, either from Roboflow or from a local `data.yaml`. |
| `src/predict.py` | Runs inference and writes annotated images. Local checkpoint or hosted Roboflow model. |
| `tools/static_checks.py` | Twelve checks that need no GPU, weights or dataset. |

## Stage 1 — Normalisation to YOLO format

YOLO does not take pixel coordinates. It takes, per box, one line of:

```
<class_index> <centre_x> <centre_y> <width> <height>
```

with all four geometry values divided by the image dimensions, so they land in
`[0, 1]`. Two separate converters are needed because the two source datasets use
incompatible annotation formats; both are in `src/`.

Three corrections happen on load, and each of them matters:

1. **Corner order.** At least one BSTLD annotation has `x_min > x_max` or
   `y_min > y_max`. Averaging the corners still gives the right centre, but the
   computed width comes out negative and the label is unusable. The converter
   swaps them.
2. **Clipping.** At least one box extends past the right edge. Unclipped, the
   normalized centre can exceed 1.0 and the sample trains as noise.
3. **Correct dimensions.** Normalisation is valid only against the frame size the
   annotation was made on. Both converters therefore take width and height as
   arguments instead of hard-coding them — a wrong value corrupts every box
   silently, with no error to notice.

The RIIB variant of BSTLD is supported through `--riib`, which also has to correct
the 8-pixel header offset on `y`.

## Stage 2 — The unified class index

This is the part the project exists for. Two datasets with unrelated taxonomies
are placed in one 33-entry index (`src/classes.py`):

- **0-19** traffic signs
- **20-32** traffic light states

The numbering is **frozen**. It is written into every generated `.txt` label and
into the trained model's output layer, so `classes.py` is a compatibility
contract, not a preference. `CLASS_NAMES` is derived from `CLASSES` rather than
written out a second time, which removes the class of bug where the forward and
reverse maps disagree.

A label absent from the index is **skipped and reported**, not silently mapped to
`OTHER`; a converter that guesses would quietly poison the training set.

## Stage 3 — Merge

`src/merge_datasets.py` collects image/label pairs from both converted trees into
one, and writes the Ultralytics `data.yaml` describing the class names.

Filenames are de-collided: two datasets can both contain `0001.jpg`, and a
straight copy would silently lose one. Colliding names get a short content hash.

Images with no label file are copied as **backgrounds** with an empty `.txt`. That
is deliberate — YOLO uses them as negatives, which is what stops a detector from
firing on empty road.

## Stage 4 — Training

Fine-tuning from COCO-pretrained `yolo11n` weights, 100 epochs, 640 px, default
augmentation. `--route local` trains on a `data.yaml` you produced yourself;
`--route roboflow` pulls the dataset from the platform first.

The original submission trained this way in a Google Colab notebook, which is
where the usable model came from. See `docs/TEST_PLAN.md` for what was and was not
run.

## Stage 5 — Inference

`src/predict.py` runs detection and renders the result in the same style as the
recorded outputs — a 1 px green rectangle with the class name above it — so new
output can be compared side by side with `samples/predictions/`.

Two backends:

- `--backend roboflow` queries the hosted model. This is what produced the
  recorded sample predictions, and needs `ROBOFLOW_API_KEY`.
- `--backend local` runs a `.pt` checkpoint you trained yourself. No account.

Confidence thresholds differ between them (`--confidence` 40 for Roboflow,
`--conf` 0.25 for local) because the two backends scale the score differently.

## Design decisions worth naming

**Why one model instead of two.** Signs and lights are different objects at
different scales, and separating them would let each be tuned independently. They
are combined here because a single pass over a frame is cheaper than two, and
because the shared early layers see the same cues — edges, saturated colour,
circular and rectangular shapes.

**Why the class index is not reordered for tidiness.** Every published weight and
every existing label file depends on the current numbering. Ordering the signs
alphabetically would look neater and invalidate the trained model.

**Why the converters are separate.** One merged converter would need a format
sniff per input, and the two formats have nothing in common. Two small, explicit
converters are easier to verify than one that branches.

**Why `MISC_SIGNS` records are skipped.** The sign dataset uses it for records
with no usable geometry; emitting a box from one produces a label pointing
nowhere.
