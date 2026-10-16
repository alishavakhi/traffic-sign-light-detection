# Datasets

The detector is trained on a **merge of two independent public datasets** that
have nothing in common except the road scene: one annotates traffic signs, the
other annotates traffic light states. Neither is redistributed with this
repository.

| | Traffic signs | Traffic lights |
|---|---|---|
| Role | 20 classes | 13 classes |
| Class indices | 0-19 | 20-32 |
| Annotation format | single `annotations.txt`, `<image>:<record>;<record>;…` | per-frame YAML label files |
| Native resolution | 1280 × 960 | 1280 × 720 |
| Converter | `src/signs_to_yolo.py` | `src/bstld_to_yolo.py` |
| Source | see *Open item* below | Bosch Small Traffic Lights Dataset (BSTLD) |

## Traffic lights — Bosch Small Traffic Lights Dataset

Published by the Bosch research group at Heidelberg University:

```
https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset
```

It is the standard benchmark for small-object traffic light detection: lights are
typically only ~10 px wide at 1280 × 720, which is precisely what makes the task
hard. Frames also carry a `camera` field, so timestamps can be recovered from the
raw capture stream.

**Licence: non-commercial use only.** Accepted terms are required before
downloading. Nothing from it is redistributed here; the sample frames under
`samples/` are included for non-commercial illustration only (see
`THIRD_PARTY_NOTICES.md`).

Citation:

> K. Behrendt, L. Novak, R. Botros. *A Deep Learning Approach to Traffic Lights:
> Detection, Tracking, and Classification.* IEEE ICRA 2017.

The `read_label_file.py` sample script from the dataset's tooling repository
(MIT-licensed) is the basis of `src/bstld_to_yolo.py`, summarised in
`docs/CHANGES.md` and `THIRD_PARTY_NOTICES.md`.

### Traffic light classes

Thirteen states, covering both single-lamp and multi-lamp heads — the
left/right/straight variants describe *which* lamp in the head is lit, which is
the part that matters for a vehicle deciding whether it may proceed:

```
off                    Red                    RedLeft                RedRight
RedStraight            RedStraightLeft        Yellow                 Green
GreenLeft              GreenRight             GreenStraight          GreenStraightRight
GreenStraightLeft
```

## Traffic signs — source not recorded

The sign half comes from a text-annotated detection dataset. Its annotation format
is preserved in `src/signs_to_yolo.py`; the dataset's identity is documented in
that module's docstring and **is not known**, because the original project did not
record it. Twenty classes:

```
100_SIGN  110_SIGN  120_SIGN  30_SIGN  50_SIGN  60_SIGN  70_SIGN  80_SIGN
90_SIGN   GIVE_WAY  NO_PARKING  NO_STOPPING_NO_STANDING  OTHER
PASS_EITHER_SIDE  PASS_LEFT_SIDE  PASS_RIGHT_SIDE  PEDESTRIAN_CROSSING
PRIORITY_ROAD  STOP  URDBL
```

The class names are reproduced exactly as they appear in the dataset's own
annotation file and index. They are **not renamed here**: the integers 0-19 are
baked into every generated label file and into the trained model's output layer,
so renaming a class would desynchronise the code from the trained weights without
changing what the model does.

### Open item — attribution

Because the dataset is unidentified, this repository **claims no rights over it**
and its licence terms cannot be stated. Two consequences are handled elsewhere:

- no images from the sign dataset are published beyond the illustrative samples
  (see `THIRD_PARTY_NOTICES.md` §2 and §5);
- the `samples/` directory is explicitly excluded from this repository's MIT
  licence.

If you know the source, add the dataset name, URL and licence to
`THIRD_PARTY_NOTICES.md` §2.

## Why the merge is the interesting part

The two datasets disagree on almost everything:

- **Geometry** — signs carry `[ulx, uly, lrx, lry]`, traffic lights carry
  `x_min/x_max/y_min/y_max`. The sign set also contains at least one record with
  the corners the wrong way round, and one box extending past the image edge; the
  light set contains a box with `x_min > x_max`. Both are corrected on load, or
  the resulting normalized centre lands outside the frame and the training sample
  becomes noise.
- **Granularity** — a sign is one object; a traffic light may be one lamp or a
  whole head, and the class encodes which.
- **Storage** — one file of all annotations versus one file per frame.
- **Normalisation basis** — a *single* width/height per dataset, which must be
  right or every box is silently scaled wrong. This is why both converters take
  the dimensions as arguments rather than assuming them.

The unified index in `src/classes.py` is the artefact that resolves this. Getting
the merge right is what the project's engineering effort actually went into.

## Splits

The merged dataset was split **87% train / 8% validation / 4% test** by upload
configuration on Roboflow. `src/merge_datasets.py` reproduces a split locally and
takes `--val-fraction` so you can match or vary it.

## Reproducing the dataset

```bash
# traffic lights: download BSTLD from the publisher and accept its terms, then
python -m src.bstld_to_yolo path/to/dataset_train_rgb/train.yaml
python -m src.bstld_to_yolo path/to/dataset_train_rgb/test.yaml

# traffic signs: point at the directory holding images + annotations.txt
python -m src.signs_to_yolo --dataset-root path/to/signs --width 1280 --height 960

# merge into one YOLO dataset
python -m src.merge_datasets \
    --source lights=path/to/dataset_train_rgb \
    --source signs=path/to/signs \
    --output data/unified --val-fraction 0.08
```

Then train on `data/unified/data.yaml` with `src/train.py --route local`.
