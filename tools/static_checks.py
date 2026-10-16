#!/usr/bin/env python3
"""Static checks for this repository — runnable with no GPU, model or dataset.

These checks verify the properties that can be established without training or
inference: that the module layout is intact, the class index is self-consistent,
the recorded sample outputs match the recorded inputs, the documentation agrees
with the code, and no credential has crept back into the tree.

They deliberately do NOT claim the detector works. Nothing here runs the model.

Usage:
    python tools/static_checks.py           # human-readable
    python tools/static_checks.py --quiet   # exit code only

Exit status is 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []
PASSES: list[str] = []

#: Patterns that must never appear in a committed file. The first entry is the
#: literal key that was hard-coded in the original submission (removed here —
#: see CHANGES.md); the rest are generic shapes.
SECRET_PATTERNS = [
    (r"dYnVH3sNlRxzPWBAFteJ", "the original hard-coded Roboflow API key"),
    (r"""(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*["'][A-Za-z0-9_\-]{16,}["']""",
     "a hard-coded credential"),
]

#: Files where a match is expected because the value is being described, not used.
SECRET_ALLOWLIST = {"tools/static_checks.py", "docs/CHANGES.md", "README.md"}


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSES.append(name)
    else:
        FAILURES.append(f"{name}: {detail}" if detail else name)


def c1_layout() -> None:
    """The module layout described in the README exists."""
    expected = [
        "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "requirements.txt",
        ".gitignore", ".env.example",
        "src/__init__.py", "src/classes.py", "src/bstld_to_yolo.py",
        "src/signs_to_yolo.py", "src/merge_datasets.py", "src/train.py",
        "src/predict.py",
        "docs/PIPELINE.md", "docs/DATASETS.md", "docs/TEST_PLAN.md",
        "docs/CHANGES.md",
        "tools/static_checks.py",
    ]
    missing = [p for p in expected if not (REPO / p).exists()]
    check("layout: expected files present", not missing, f"missing {missing}")


def c2_compiles() -> None:
    """Every Python file parses."""
    bad = []
    for p in sorted(REPO.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        try:
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except SyntaxError as e:
            bad.append(f"{p.relative_to(REPO)}:{e.lineno} {e.msg}")
    check("syntax: all modules parse", not bad, "; ".join(bad))


def c3_class_index() -> None:
    """The unified class index is contiguous, unique and correctly split."""
    sys.path.insert(0, str(REPO))
    from src import classes  # noqa: E402

    indices = sorted(classes.CLASSES.values())
    problems = []
    if indices != list(range(len(indices))):
        problems.append(f"indices not contiguous 0..{len(indices)-1}: {indices}")
    if len(set(classes.CLASSES)) != len(classes.CLASSES):
        problems.append("duplicate label names")
    if set(classes.CLASS_NAMES) != {str(i) for i in indices}:
        problems.append("CLASS_NAMES does not round-trip CLASSES")
    for label, idx in classes.CLASSES.items():
        if classes.name_for(idx) != label:
            problems.append(f"name_for({idx}) != {label}")
    if classes.SIGN_CLASSES + classes.LIGHT_CLASSES != tuple(indices):
        problems.append("sign/light split does not cover the index")
    check(f"classes: {len(indices)}-class index self-consistent", not problems,
          "; ".join(problems))


def c4_no_secrets() -> None:
    """No credential is committed outside the files that describe the removal."""
    hits = []
    for p in sorted(REPO.rglob("*")):
        if not p.is_file() or ".git" in p.parts or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(REPO).as_posix()
        if rel in SECRET_ALLOWLIST:
            continue
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf", ".zip", ".pt"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern, what in SECRET_PATTERNS:
            if re.search(pattern, text):
                hits.append(f"{rel} ({what})")
    check("secrets: none committed", not hits, "; ".join(hits))


def c5_credentials_from_env() -> None:
    """The API key is read from the environment, not written into the source."""
    src = (REPO / "src" / "train.py").read_text() + (REPO / "src" / "predict.py").read_text()
    ok = "ROBOFLOW_API_KEY" in src and "os.environ" in src
    check("secrets: API key read from environment", ok,
          "train.py/predict.py no longer read ROBOFLOW_API_KEY from the environment")


def c6_samples_match() -> None:
    """Every recorded prediction has a matching recorded input."""
    pairs = 0
    missing = []
    for split in ("easy", "hard"):
        inp = REPO / "samples" / "inputs" / split
        pred = REPO / "samples" / "predictions" / split
        if not inp.is_dir() or not pred.is_dir():
            missing.append(f"split {split} missing")
            continue
        in_names = {p.stem for p in inp.iterdir() if p.is_file()}
        pred_names = {p.stem for p in pred.iterdir() if p.is_file()}
        only_pred = pred_names - in_names
        if only_pred:
            missing.append(f"{split}: predictions without input {sorted(only_pred)}")
        if not in_names:
            missing.append(f"{split}: no inputs")
        pairs += len(pred_names & in_names)
    check(f"samples: {pairs} input/prediction pairs matched", not missing,
          "; ".join(missing))


def c7_docs_match_code() -> None:
    """Documented module paths exist and no doc points at a missing file."""
    docs = list(REPO.glob("*.md")) + list((REPO / "docs").glob("*.md"))
    broken = []
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        for link in re.findall(r"\[[^\]]*\]\(([^)#\s]+)\)", text):
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = (doc.parent / link).resolve()
            if not target.exists():
                broken.append(f"{doc.relative_to(REPO)} -> {link}")
    check("docs: internal links resolve", not broken, "; ".join(broken))


def c8_readme_embeds_figures() -> None:
    """README figures are embedded images, not bare links."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    embedded = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    figures = [
        f for f in embedded
        if f.startswith("docs/figures/") or f.startswith("samples/")
    ]
    missing = [f for f in figures if not (REPO / f.split("#")[0]).exists()]
    check(f"readme: {len(figures)} figures embedded as images, all present",
          bool(figures) and not missing,
          f"missing {missing}" if missing else "no embedded figures found")


def c9_no_hardcoded_paths() -> None:
    """Source contains no machine-specific absolute path."""
    offenders = []
    pat = re.compile(r"""["'](?:/home/|/Users/|[A-Z]:\\\\)""")
    for p in sorted((REPO / "src").rglob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(line):
                offenders.append(f"{p.name}:{i}")
    check("portability: no absolute local paths in src/", not offenders,
          "; ".join(offenders))


def c10_no_junk() -> None:
    """No editor, macOS or build junk would be committed.

    Asks git what is actually tracked when the tree is a repository, because the
    question is what ships, not what exists on disk — ``__pycache__`` is normally
    present after any test run and is covered by ``.gitignore``.
    """
    junk_names = {".DS_Store", "Thumbs.db"}
    junk_parts = {"__MACOSX", "__pycache__"}
    try:
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.split()
        bad = [p for p in tracked
               if Path(p).name in junk_names or junk_parts & set(Path(p).parts)]
        detail = f"tracked by git: {bad}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        bad = [str(p.relative_to(REPO)) for p in REPO.rglob("*")
               if p.name in junk_names or junk_parts & set(p.parts)]
        detail = "; ".join(bad)
    check("cleanliness: no OS/editor junk", not bad, detail)


def c11_gitignore_covers_bulk() -> None:
    """The .gitignore excludes datasets, weights and environments."""
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    needed = [".env", "__pycache__", "runs/", "*.pt", "data/"]
    missing = [n for n in needed if n not in text]
    check("cleanliness: .gitignore covers bulk and secrets", not missing,
          f"missing {missing}")


def c12_no_large_files() -> None:
    """No single committed file exceeds 5 MB (GitHub's practical comfort limit)."""
    big = []
    for p in REPO.rglob("*"):
        if p.is_file() and ".git" not in p.parts and p.stat().st_size > 5 * 1024 * 1024:
            big.append(f"{p.relative_to(REPO)} {p.stat().st_size/1024/1024:.1f} MB")
    check("size: no file over 5 MB", not big, "; ".join(big))


CHECKS = [
    ("layout", c1_layout),
    ("syntax", c2_compiles),
    ("classes", c3_class_index),
    ("secrets-present", c4_no_secrets),
    ("secrets-env", c5_credentials_from_env),
    ("samples", c6_samples_match),
    ("docs-links", c7_docs_match_code),
    ("readme-figures", c8_readme_embeds_figures),
    ("portability", c9_no_hardcoded_paths),
    ("cleanliness", c10_no_junk),
    ("gitignore", c11_gitignore_covers_bulk),
    ("size", c12_no_large_files),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    for _, fn in CHECKS:
        try:
            fn()
        except Exception as exc:  # a crash is a failed check, not a crashed run
            FAILURES.append(f"{fn.__name__} raised {type(exc).__name__}: {exc}")

    if args.json:
        print(json.dumps({"passed": PASSES, "failed": FAILURES}, indent=2))
    elif not args.quiet:
        for name in PASSES:
            print(f"  PASS  {name}")
        for f in FAILURES:
            print(f"  FAIL  {f}")
        print(f"\n{len(PASSES)} passed, {len(FAILURES)} failed")

    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
