"""Rebuild tracking, tags and their alignment twice, with independent local checks."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import traceback

from check_alignment import check_alignment
from check_tag_table import check_tags
from check_tracking_table import check
from local_guard import install


def digest(path: Path) -> str:
    """Calculate a streaming SHA-256 digest.

    hashlib.file_digest only exists from Python 3.11. The fallback keeps the
    rebuild runnable on older interpreters, which matters because the organizers
    have to reproduce our submission on whatever they happen to have installed.
    """
    with path.open("rb") as handle:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(handle, "sha256").hexdigest()
        running = hashlib.sha256()
        for block in iter(lambda: handle.read(1 << 20), b""):
            running.update(block)
        return running.hexdigest()


def main() -> int:
    """Publish COMPLETE only after all values, memberships and hashes pass."""
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "vendor/mtah"))
    from games import asset_paths, source_frames
    from possession.alignment import build_alignment
    from possession.paths import PREPROCESSING_DIR, RAW_DIR, new_run_dir, publish_current
    from possession.tagging import build_tags
    from possession.tracking import build

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RAW_DIR,
                        help="folder that contains tactical/ (default: data/raw)")
    parser.add_argument("--out", type=Path, default=None,
                        help="fresh run folder (default: data/preprocessing/runs/<UTC time>)")
    parser.add_argument("--no-publish", action="store_true",
                        help="build and verify only; leave data/preprocessing/current/ as it is")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output = (args.out or new_run_dir()).resolve()
    if not output.is_relative_to((repo / "data").resolve()) or output.exists():
        parser.error("Choose a fresh folder inside the ignored repository data directory")
    data = args.data_dir.resolve()
    if output == data or data.is_relative_to(output) or output.is_relative_to(data):
        parser.error("Output must be separate from the raw data bundle")
    publish_root = PREPROCESSING_DIR.resolve()
    if not args.no_publish and (data.is_relative_to(publish_root) or publish_root.is_relative_to(data)):
        parser.error("The raw data bundle must not overlap data/preprocessing")
    output.mkdir(parents=True)
    if not args.no_publish:
        publish_root.mkdir(parents=True, exist_ok=True)
    paths = {key: Path(value).resolve() for key, value in asset_paths(data_dir=str(data)).items()
             if key in {"atd_meta", "atd_raw", "smart_tagging"}}
    install([Path(sys.base_prefix), Path(sys.prefix), repo / "src", repo / "scripts",
             repo / "vendor/mtah", *paths.values()],
            [output] + ([] if args.no_publish else [publish_root]))
    logging.info("Building into %s", output)
    started = time.monotonic()
    status = {"state": "RUNNING", "step": "hash-inputs"}

    def save() -> None:
        """Save only operational progress in the status file."""
        status["elapsed_seconds"] = round(time.monotonic() - started)
        (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    def step(name: str) -> None:
        """Record progress and refuse to advance beyond the bounded job budget."""
        if time.monotonic() - started > 1800:
            raise TimeoutError("Preprocessing job exceeded its 30-minute budget")
        status["step"] = name
        save()

    save()
    try:
        assert sys.flags.optimize == 0, "Checks require normal Python assertions"
        before = {key: digest(path) for key, path in paths.items()}
        n_frames = source_frames()
        file_names = ["tracking.csv", "track_slots.json", "tags.csv",
                      "aligned/frames_with_tags.csv", "aligned/tags_with_teams.csv",
                      "aligned/team_mapping.json"]
        hashes = []
        for attempt in ("first", "repeat"):
            folder = output / attempt
            step(f"{attempt}-tracking-build")
            build(paths["atd_meta"], paths["atd_raw"], folder, n_frames)
            step(f"{attempt}-tracking-independent-check")
            check(paths["atd_meta"], paths["atd_raw"], folder, n_frames,
                  deadline=started + 1800)
            step(f"{attempt}-tags-build-and-check")
            build_tags(paths["smart_tagging"], folder / "tags.csv", n_frames)
            tag_quality = check_tags(paths["smart_tagging"], folder / "tags.csv", n_frames)
            step(f"{attempt}-alignment-build")
            build_alignment(paths["atd_meta"], folder / "tracking.csv", folder / "tags.csv",
                            folder / "aligned", n_frames)
            step(f"{attempt}-alignment-independent-check")
            join_quality = check_alignment(paths["atd_meta"], folder / "tracking.csv",
                                           folder / "tags.csv", folder / "aligned", n_frames)
            hashes.append({name: digest(folder / name) for name in file_names})
        step("compare-builds-and-inputs")
        assert hashes[0] == hashes[1], "Repeated builds produced different files"
        assert before == {key: digest(path) for key, path in paths.items()}, "Input changed"
        evidence = {"checks_passed": ["every-tracking-value", "every-source-tag",
                    "every-tracking-cell-preserved", "every-active-tag-membership",
                    "team-map-source-provenance", "all-six-output-hashes-repeat",
                    "raw-inputs-unchanged"], "input_hashes": before,
                    "output_hashes": hashes[0], "tag_quality": tag_quality,
                    "join_quality": join_quality,
                    "source_hashes": {name: digest(repo / name) for name in (
                        "src/possession/tracking.py", "src/possession/tagging.py",
                        "src/possession/alignment.py", "scripts/check_tracking_table.py",
                        "scripts/check_tag_table.py", "scripts/check_alignment.py",
                        "scripts/rebuild_preprocessing.py", "vendor/mtah/games.py")}}
        (output / "verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        step("all-required-checks-passed")
        status.update(state="COMPLETE", mapping_review_required=bool(
            join_quality["unresolved_named_teams"]), teammate_reproduction="PENDING")
        save()
        logging.info("Full rebuild passed. Read the local handoff for remaining review.")
    except Exception as error:
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        status["state"] = "TIMED_OUT" if isinstance(error, TimeoutError) else "FAILED"
        save()
        logging.error("Rebuild stopped. Read status.json and failure.log locally. "
                      "data/preprocessing/current/ was not changed.")
        return 1
    if args.no_publish:
        logging.info("--no-publish: data/preprocessing/current/ was not changed.")
        return 0
    try:
        current = publish_current(output, publish_root)
    except Exception as error:  # the build itself is still COMPLETE and kept in its run folder
        status["published"] = False
        save()
        logging.error("Build passed but publishing to current/ failed: %s", error)
        return 1
    status["published"] = True
    save()
    logging.info("Published to %s", current)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
