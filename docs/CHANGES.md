# Changes made preparing this repository

What was changed from the original submission, why, and what was deliberately left
alone. The original files are preserved unmodified in the project archive; nothing
here destroys the original work.

## The engineering algorithm was not changed

No detection, conversion or normalisation logic was altered. The YOLO format, the
class ordering, the box arithmetic and the drawing style are as the original
author left them. Everything below is structure, portability, safety and
documentation.

---

## 1. Security — a credential was removed

**The original `train.py` and `test.py` contained a live Roboflow API key
hard-coded as a string literal.** It is removed from both files here.

- `src/train.py` and `src/predict.py` now read `ROBOFLOW_API_KEY` from the
  environment and exit with an instructive message if it is unset.
- `.env.example` documents the variable; `.env` is gitignored.
- `tools/static_checks.py` fails if a credential literal appears anywhere in the
  tree, matching **structurally** rather than by value, plus one digest check
  confirming that the specific key from the original submission has not returned.

  An earlier revision of that checker listed the literal key as its search
  pattern, which meant this repository published the credential it was testing
  for. Caught by verifying the live remote rather than the push. The pattern is
  now structural and the specific key is matched by SHA-256, so the check remains
  verifiable without republishing anything.

**A key that has been committed is compromised even after deletion.** The original
key was never published — this is the first public repository for this project —
but the key did exist in files that circulated, and it should be **rotated** in the
Roboflow dashboard.

## 2. Privacy — a student ID was removed from the report

The report PDF's title block read `<author name> / <student ID>`. The student ID
identifying number has been removed at the text level: the characters are gone from
the content stream, not covered by a rectangle. The author's name, the page count
(15) and the surrounding prose are unchanged.

Verified with 0 remaining occurrences in the extracted text, the raw file bytes,
the decompressed content streams and the document metadata. The original
unredacted PDF is preserved unmodified in the archive and is not published.

## 3. Structure — the flat submission became a package

The original was five scripts in one directory with no package structure, no
installable entry point and no dependency manifest. It is now:

| Original | Now | Change |
|---|---|---|
| `classes.py` | `src/classes.py` | Not a valid module name for a package import; content preserved, `CLASS_NAMES` now derived from `CLASSES` instead of duplicated |
| `traffic-light-set-labelizer.py` | `src/bstld_to_yolo.py` | Renamed to say what it converts and to which direction; the provenance of its BSD/MIT origin is now documented in the module |
| `traffic-sign-set-labelizer.py` | `src/signs_to_yolo.py` | Same |
| `train.py` | `src/train.py` | Moved; key from environment; training parameters exposed as flags |
| `test.py` | `src/predict.py` | Renamed — it is inference, not a test, and the old name collided with the idea of a test suite. Report also calls it `predict.py` |
| *(absent)* | `src/merge_datasets.py` | **New.** See §4 |
| `test_images/`, `test_results/` | `samples/inputs/`, `samples/predictions/` | Renamed to say what they are; downscaled to 1000 px wide to keep the repository small |

## 4. One module was written for this repository

The project report describes merging the two converted datasets into a single
training set, and the original submission contains **no script for that step** — it
was done by hand or through the platform UI.

`src/merge_datasets.py` is therefore a **reconstruction**, written here so the
pipeline runs end to end from a clean checkout. It implements the merge as
described. It is **not** the exact code that produced the original training set and
does not claim to be.

It also adds one behaviour the original had no need for: filename de-collision.
Two source datasets can both contain `0001.jpg`, and a plain copy silently loses
one of them.

## 5. Code quality and portability

- **Hard-coded paths removed.** The originals opened `train.yaml`, `test.yaml` and
  `Set2Part0/` by relative path from the working directory. All entry points now
  take arguments and use `pathlib`.
- **Destructive defaults inverted.** The sign converter deleted every image without
  a usable label as a side effect of loading. That is now `--delete-unlabelled`,
  off by default. A conversion script that removes source data when run is a
  footgun.
- **Unmapped labels are reported, not swallowed.** A class label absent from the
  unified index was previously skipped silently, which would have quietly produced a
  training set missing data. Both converters now collect and print them.
- **Module-level side effects removed.** The labelizer scripts executed on import;
  they now sit behind `main()` with `argparse`.
- **Image dimensions are arguments, not constants.** Normalisation is only correct
  against the frame size the annotation was made on. `--width` / `--height` are
  required for the sign converter.
- **Docstrings added** throughout, covering inputs, outputs, raised exceptions and
  the non-obvious arithmetic.
- **`classes.py` no longer stores the reverse map twice.** `CLASS_NAMES` is derived
  from `CLASSES`, so the two cannot disagree.
- **A latent bug is annotated rather than silently changed.** `signSize` computes
  `min(ulx - lrx, uly - lry) ** 2`, which is a squared negative and not a usable
  area. The field is unused by the YOLO export. It is left as-is with a comment
  explaining it, because changing it would alter the parsed record the original
  code produced for no functional gain.

## 6. Documentation added

`README.md`, `docs/PIPELINE.md`, `docs/DATASETS.md`, `docs/TEST_PLAN.md` and
`THIRD_PARTY_NOTICES.md` are new. The project report is included under
`docs/report/` — redacted, but otherwise as submitted.

## 7. Verification added

- `tests/test_converters.py` — 9 tests over the conversion arithmetic and the
  malformed-record handling, using hand-built fixtures. Runnable with or without
  pytest.
- `tools/static_checks.py` — 13 checks that need no GPU, dataset or weights.

## 8. What was deliberately NOT done

- **No algorithm was rewritten, "improved" or optimised.** The engineering is the
  author's.
- **No accuracy metric was invented.** None was measured, so none appears.
- **No class was renamed.** The indices 0-32 are baked into every label file and
  into the trained model's output layer; reordering them would invalidate the
  trained weights.
- **No dataset is redistributed.** Both are third-party research datasets with
  their own terms — see `THIRD_PARTY_NOTICES.md`.
- **The original Word document is not published.** It duplicates the PDF, is
  16.7 MB, and carried the same identifying number. The PDF is the published
  artefact.
- **The original 42 MB submission archive is not published.** It duplicates the
  extracted directory.
- **macOS archive junk was not carried over** — `__MACOSX/`, `.DS_Store` and a
  stale `__pycache__/` were present in the submission and are excluded.

## 9. Open item for the author

**The traffic sign dataset is unidentified.** The original project did not record
which dataset the sign annotations came from, and it is not determinable from the
annotation format. Since the licence cannot be stated, no images derived from it
beyond the illustrative samples are published, and that limitation is carried in
`THIRD_PARTY_NOTICES.md` §2 and `docs/TEST_PLAN.md`.

If the source is known, that section should be completed.
