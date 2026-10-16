# Test and reproducibility plan

What has been verified, what was demonstrated by the original project, and what
remains unverified. The three are kept separate deliberately — a portfolio loses
its value the moment a reader cannot tell which is which.

---

## 1. What the original project demonstrated

These are results the **original implementation produced in its original
environment**, in 2025, before this repository existed. They are evidence of the
project working, not evidence that this prepared copy runs.

| Claim | Evidence |
|---|---|
| The detector detects traffic signs on real road scenes | `samples/predictions/easy/rs*.jpg` — recorded model output |
| The detector detects traffic lights and distinguishes their state | `samples/predictions/easy/tl*.jpg`, `samples/predictions/hard/tl*.jpg` — recorded output, labels such as `Green` |
| The pipeline was trained to a working state | `docs/report/Ali-Shavakhi-Computer-Vision-Project-Report.pdf`, which states the usable model came from a Google Colab training run |
| Both datasets were converted and merged | The converters themselves, plus the report's description of the merge and the 87/8/4 split |
| Training converged with a positive mAP | `docs/figures/roboflow_image1.png` — the platform's training graphs, in the report |
| The merged dataset is 14,285 images | `docs/figures/roboflow_image2.png` — the platform's dataset summary, in the report |

### Training metrics recorded by the original run

Read off `docs/figures/roboflow_image1.png`, which the report includes. These are
**the original project's recorded numbers**, not measurements taken here:

| Series | Reading at the right-hand end of the curve (~280 epochs) |
|---|---|
| mAP | ≈ 0.72 |
| mAP@50:95 | ≈ 0.48 |
| Box loss | 2.0 → ≈ 1.10 |
| Class loss | 2.2 → ≈ 0.60 |
| Object loss | 1.08 → ≈ 0.88 |

Two caveats, stated because the chart does not state them:

- **The evaluated split is not labelled on the chart.** Roboflow's training graphs
  are normally computed on the validation split (1,199 images, 8%), and that is the
  most likely reading, but it is an inference rather than something the figure says.
- **The curves run to ~280 epochs**, which is more than the 100 epochs
  `src/train.py` defaults to. The platform-hosted training therefore used different
  settings from the script's defaults, and the script's defaults should not be
  assumed to reproduce this curve.

No metric was measured on this host, and none of the above was independently
re-derived.

The recorded outputs in `samples/predictions/` are **the original project's own
results**, shipped unmodified apart from downscaling for repository size. They
were produced by querying the hosted Roboflow model — the checkpoint itself was
never in the submission and is not here.

### What those outputs show

- **Easy split, traffic sign** — a `NO_PARKING` sign detected with a tight box.
- **Easy split, traffic light** — a `Green` state detected on a vehicle-mounted
  signal in an urban scene.
- **Hard split** — a `Green` state detected in low-contrast, wet, heavy-traffic
  conditions, which is the case the split exists to stress.

The samples also contain **misses and at least one apparent false positive** on a
vehicle body. These are left in rather than curated away: the output is the output,
and the limitations section says so.

---

## 2. What was verified while preparing this repository

Run on an Ubuntu 24.04 host with no GPU, no dataset, no Roboflow account and no
trained weights. `python tools/static_checks.py` — **12 checks, all passing**:

| Check | What it establishes |
|---|---|
| layout | every file the README names exists |
| syntax | every `.py` parses (`ast.parse`) |
| classes | the 33-class index is contiguous 0-32, unique, round-trips through `name_for`, and splits cleanly into signs 0-19 / lights 20-32 |
| secrets-present | the hard-coded API key of the original submission is gone from the tree |
| secrets-env | `train.py` and `predict.py` read the key from `ROBOFLOW_API_KEY` |
| samples | all 38 recorded files pair up: every prediction has a matching input |
| docs-links | every relative link in every markdown file resolves |
| readme-figures | README figures are `![]()` embeds, and each file exists |
| portability | no absolute local path (`/home/…`, `C:\…`) in `src/` |
| cleanliness | no `.DS_Store`, `__MACOSX`, `__pycache__` committed |
| gitignore | `.env`, `runs/`, `*.pt`, `data/`, `__pycache__` are ignored |
| size | no committed file exceeds 5 MB |

Additionally performed by hand:

- **Converter logic exercised on synthetic input.** Both converters were run
  against small hand-built fixtures in the documented formats, confirming that
  boxes are normalized correctly (`centre = (upper_left + lower_right) / 2 / dim`),
  that the corner-order and clipping corrections fire on malformed records, and
  that an unmapped class label is reported rather than silently dropped.
  *This tests the conversion arithmetic. It is not a test of detection accuracy.*
- **Report PDF sanitised and re-verified** — the author's student ID was removed
  at the text level; page count, the author's name and the surrounding prose are
  unchanged; 0 hits in extracted text, raw bytes, decompressed streams and
  metadata.
- **Figure/asset integrity** — all committed images open, and each
  `samples/predictions` file has a matching `samples/inputs` file of the same
  native resolution.

---

## 3. What has NOT been run

Stated plainly. Each of these needs something this host does not have.

| Not run | Why | What it would take |
|---|---|---|
| `colcon`-style build | not applicable — pure Python | — |
| `src/train.py` end to end | needs the merged dataset (~GBs) and a GPU for 100 epochs | acceptable terms to both datasets, a CUDA host, `pip install -r requirements.txt` |
| `src/train.py --route roboflow` | needs a Roboflow account and API key | a key in `ROBOFLOW_API_KEY` |
| `src/predict.py --backend roboflow` | needs the key, and the hosted model must still exist | as above |
| `src/predict.py --backend local` | no `.pt` checkpoint exists in this repository | train first, or supply a checkpoint |
| Quantitative accuracy | no held-out metric was measured on this host | a labelled test split and an evaluation run here |
| Original sample outputs reproduced | would need the hosted model to return identical boxes | the key and a live model version |
| The recorded training curve reproduced | the platform-hosted run used ~280 epochs, not the script's 100 | a GPU run matched to the original settings |

**The only accuracy figures in this repository are the original project's, read off
the training chart it recorded** (see §1). They were not measured here, the
evaluated split is not stated on the chart, and no metric was invented to fill the
gap.

---

## 4. Reproducing on a machine with a GPU

```bash
git clone https://github.com/alishavakhi/traffic-sign-light-detection
cd traffic-sign-light-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. obtain both datasets yourself and accept their terms (see docs/DATASETS.md)
# 2. convert
python -m src.bstld_to_yolo path/to/dataset_train_rgb/train.yaml
python -m src.signs_to_yolo --dataset-root path/to/signs --width 1280 --height 960

# 3. merge
python -m src.merge_datasets \
    --source lights=path/to/dataset_train_rgb \
    --source signs=path/to/signs \
    --output data/unified --val-fraction 0.08

# 4. train
python -m src.train --route local --data data/unified/data.yaml --epochs 100 --imgsz 640

# 5. run it
python -m src.predict --backend local --weights runs/detect/train/weights/best.pt \
    --split easy
```

Expected on success: `runs/detect/train/weights/best.pt` exists, and step 5 writes
annotated images into `samples/predictions/easy/`. Compare them against the
recorded outputs already there — the original project's results are the reference.

### Checks that need no GPU

```bash
python tools/static_checks.py          # 12 checks, exit 0
python tools/static_checks.py --json   # machine-readable
```

---

## 5. Known limitations

Carried here from the report and from reading the code, not from testing:

1. **Small-object detection is the difficult case.** Traffic lights are ~10 px at
   1280 × 720. The hard split exists because of it, and the recorded outputs show
   the failure mode.
2. **False positives on vehicle-mounted signals.** A signal on a car is
   geometrically and chromatically similar to a traffic light. One apparent
   false positive is visible in the recorded easy-split output.
3. **The model has not been evaluated numerically.** No held-out metric exists.
4. **The sign dataset's licence is unknown** (`THIRD_PARTY_NOTICES.md` §2), which
   is a real constraint on redistributing anything derived from it.
5. **`ultralytics` is AGPL-3.0** — see `THIRD_PARTY_NOTICES.md` §3. Relevant to any
   closed-source reuse.
6. **No checkpoint is published.** Reproducing inference as-shipped requires either
   a Roboflow account or your own training run.
