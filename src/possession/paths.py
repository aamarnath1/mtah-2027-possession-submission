"""Where preprocessing output lives, and how a verified run becomes `current/`.

    data/preprocessing/
      runs/<UTC timestamp>/     each rebuild gets its own fresh folder (never reused)
        first/ repeat/ status.json verification.json
      current/                  the last COMPLETE run's tables: what downstream code reads
        tracking.csv track_slots.json tags.csv aligned/...
        verification.json       hashes of exactly these files
        PUBLISHED.json          which run this is and when it was published

A failed or unfinished rebuild never touches `current/`. Downstream code should read from
CURRENT_DIR (or `current_file(...)`) instead of hard-coding a run folder.
"""

import json
import os
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSING_DIR = DATA_DIR / "preprocessing"
RUNS_DIR = PREPROCESSING_DIR / "runs"
CURRENT_DIR = PREPROCESSING_DIR / "current"


def current_file(name: str, root: Path = PREPROCESSING_DIR) -> Path:
    """Path to a published table, e.g. current_file("aligned/frames_with_tags.csv")."""
    path = Path(root) / "current" / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/rebuild_preprocessing.py` first.")
    return path


def new_run_dir(root: Path = PREPROCESSING_DIR) -> Path:
    """A fresh, not-yet-existing run folder named by UTC time."""
    runs = Path(root) / "runs"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    candidate, n = runs / stamp, 1
    while candidate.exists():
        n += 1
        candidate = runs / f"{stamp}-{n}"
    return candidate


def _remove_tree(path: Path) -> None:
    """Delete a folder using absolute paths only.

    shutil.rmtree works through relative names and directory descriptors, which the rebuild's
    local file guard (scripts/local_guard.py) cannot check against its allowed folders.
    """
    for folder, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(folder, name))
        for name in dirs:
            os.rmdir(os.path.join(folder, name))
    os.rmdir(path)


def _link_or_copy(source: str, target: str) -> None:
    """Hard-link when possible (instant, no extra disk); copy otherwise."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def publish_current(run_dir: Path, root: Path = PREPROCESSING_DIR) -> Path:
    """Make a COMPLETE run's `first/` tables the new `current/`, replacing the old one.

    The new folder is assembled beside `current/` and swapped in with renames, so readers see
    either the old complete set or the new complete set, never a mix.

    Raises:
        ValueError: the run is not COMPLETE or is missing its tables or verification.
    """
    run_dir, root = Path(run_dir), Path(root)
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "COMPLETE":
        raise ValueError(f"Only a COMPLETE run can be published (state: {status.get('state')})")
    source, verification = run_dir / "first", run_dir / "verification.json"
    if not source.is_dir() or not verification.is_file():
        raise ValueError("Run folder is missing first/ or verification.json")

    root.mkdir(parents=True, exist_ok=True)
    staging, previous, current = root / ".current-staging", root / ".current-previous", root / "current"
    for leftover in (staging, previous):  # from an interrupted earlier publish
        if leftover.exists():
            _remove_tree(leftover)
    shutil.copytree(source, staging, copy_function=_link_or_copy)
    shutil.copy2(verification, staging / "verification.json")
    record = {
        "run": os.path.relpath(run_dir, root),
        "published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mapping_review_required": status.get("mapping_review_required"),
    }
    (staging / "PUBLISHED.json").write_text(json.dumps(record, indent=2) + "\n",
                                            encoding="utf-8", newline="\n")
    if current.exists():
        current.rename(previous)
    try:
        staging.rename(current)
    except OSError:
        if previous.exists() and not current.exists():
            previous.rename(current)  # put the old set back
        raise
    if previous.exists():
        _remove_tree(previous)
    return current
