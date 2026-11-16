"""Build the match viewer's data from the published preprocessing output (data/preprocessing/current/).

This is the viewer's data step. It reads the verified tables, so the viewer shows exactly what the
model reads:

    current/tracking.csv               ball + slot positions (x_norm, y_norm) and ball speed
    current/track_slots.json           slot -> H / A
    current/aligned/tags_with_teams.csv  Smart Tagging intervals with mapped teams
    current/aligned/team_mapping.json  team display names (for the legend)

The homography isn't part of the preprocessing, so it is still read from the raw CSV
(data/raw/tactical/BAR-ATH_tactical_homography.csv).

Writes the same files the viewer page already loads (default: data/viewer/):

    meta.json, positions.f32, ball_speed.f32, homography.f64, tags.json

Usage:
    python scripts/rebuild_preprocessing.py            # once, to publish current/
    python tools/viewer/prepare_viewer_from_current.py
"""

import argparse
import csv
import json
import math
import sys
from array import array
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from possession.paths import CURRENT_DIR, RAW_DIR  # noqa: E402

FPS = 25
PITCH = (105.3, 68.0)
MATRIX_COLUMNS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]


def read_slot_teams(current):
    """['H' | 'A', ...] in slot order, from track_slots.json."""
    slots = json.loads((current / "track_slots.json").read_text(encoding="utf-8"))
    slots = sorted(slots, key=lambda entry: int(entry["slot"]))
    if [int(entry["slot"]) for entry in slots] != list(range(1, len(slots) + 1)):
        raise ValueError("track_slots.json: slots are not numbered 1..n")
    teams = [entry["team"] for entry in slots]
    if not teams or set(teams) - {"H", "A"}:
        raise ValueError("track_slots.json: every slot must be H or A")
    return teams


def read_team_names(current):
    """Display names for the legend: the tag team names that mapped exactly to H and A."""
    names = {"H": "Home", "A": "Away"}
    path = current / "aligned" / "team_mapping.json"
    if path.is_file():
        mapping = json.loads(path.read_text(encoding="utf-8"))
        for raw, resolution in sorted(mapping.get("resolutions", {}).items()):
            if resolution.get("basis") == "metadata-exact" and resolution.get("team") in names:
                names[resolution["team"]] = raw
    return names


def read_tracking(current, n_slots):
    """Stream tracking.csv into (positions, ball_speed, n_frames). Ball first, then slots."""
    positions, ball_speed = array("f"), array("f")
    nan = float("nan")
    with open(current / "tracking.csv", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {name: i for i, name in enumerate(header)}
        prefixes = ["ball", *[f"slot_{s}" for s in range(1, n_slots + 1)]]
        try:
            columns = [(index[f"{p}_x_norm"], index[f"{p}_y_norm"]) for p in prefixes]
            frame_col, speed_col = index["frame"], index["ball_speed_mps"]
        except KeyError as missing:
            raise ValueError(f"tracking.csv: missing column {missing}") from None
        if f"slot_{n_slots + 1}_x_norm" in index:
            raise ValueError("tracking.csv has more slots than track_slots.json")
        n_frames = 0
        for row in reader:
            if int(row[frame_col]) != n_frames:
                raise ValueError(f"tracking.csv: expected frame {n_frames}, got {row[frame_col]}")
            for x_col, y_col in columns:
                x, y = float(row[x_col]), float(row[y_col])
                if not (math.isfinite(x) and math.isfinite(y)):
                    x = y = nan
                positions.append(x)
                positions.append(y)
            speed = float(row[speed_col])
            ball_speed.append(speed if math.isfinite(speed) else nan)
            n_frames += 1
            if n_frames % 20000 == 0:
                print(f"  tracking: {n_frames:,} frames", flush=True)
    return positions, ball_speed, n_frames


def read_tags(current, n_frames):
    """Smart Tagging intervals from tags_with_teams.csv, clipped to the video."""
    tags = []
    with open(current / "aligned" / "tags_with_teams.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tags.append({
                "index": int(row["tag_index"]),
                "code": row["code"].strip(),
                "start": max(0, int(row["start_frame"])),
                "end": min(n_frames, int(row["end_frame"])),
                "team": row["team"] if row["team"] in ("H", "A") else "?",
                "team_raw": row["team_raw"],
                "type": row["set_piece_type"] or ("" if row["type_raw"] == "UNKNOWN" else row["type_raw"]),
            })
    return tags


def read_homography(csv_path, n_frames):
    """Flat float64 array, n_frames x 9, NaN where the row is invalid (raw homography CSV)."""
    matrices = array("d")
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ["frame", "valid", *MATRIX_COLUMNS] if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{csv_path}: missing columns {missing}")
        count = 0
        for frame, row in enumerate(reader):
            if int(row["frame"]) != frame:
                raise ValueError(f"{csv_path}: expected frame {frame}, got {row['frame']}")
            valid = row["valid"].strip() in ("1", "True", "true")
            for column in MATRIX_COLUMNS:
                value = float(row[column]) if valid and row[column] not in ("", "NaN") else math.nan
                matrices.append(value)
            count += 1
    if count != n_frames:
        raise ValueError(f"{csv_path}: {count} rows, tracking has {n_frames} frames")
    return matrices


def graded_blocks(n_frames):
    """Graded video-frame blocks from the organizers' games.py, or [] for other inputs."""
    sys.path.insert(0, str(REPO / "vendor" / "mtah"))
    try:
        from games import GAMES, half_video_blocks
    except ImportError:
        return []
    if GAMES.get("BAR-ATH", {}).get("n_video_frames") != n_frames:
        return []
    return [[first, last] for _half, first, last in half_video_blocks("BAR-ATH")]


def build(current, homography_csv, out_dir):
    current, out = Path(current), Path(out_dir)
    if not (current / "tracking.csv").is_file():
        raise FileNotFoundError(
            f"No published preprocessing in {current}. Run `python scripts/rebuild_preprocessing.py` first.")
    if not Path(homography_csv).is_file():
        raise FileNotFoundError(f"Missing {homography_csv}")
    out.mkdir(parents=True, exist_ok=True)

    slot_teams = read_slot_teams(current)
    print(f"Reading {current / 'tracking.csv'} ({len(slot_teams)} slots)...", flush=True)
    positions, ball_speed, n_frames = read_tracking(current, len(slot_teams))
    print("Reading homography...", flush=True)
    matrices = read_homography(homography_csv, n_frames)
    tags = read_tags(current, n_frames)

    with open(out / "positions.f32", "wb") as handle:
        positions.tofile(handle)
    with open(out / "ball_speed.f32", "wb") as handle:
        ball_speed.tofile(handle)
    with open(out / "homography.f64", "wb") as handle:
        matrices.tofile(handle)
    (out / "tags.json").write_text(json.dumps(tags, ensure_ascii=False) + "\n",
                                   encoding="utf-8", newline="\n")

    published = current / "PUBLISHED.json"
    verification = current / "verification.json"
    meta = {
        "n_frames": n_frames,
        "fps": FPS,
        "pitch_m": list(PITCH),
        "teams": read_team_names(current),
        "slot_teams": slot_teams,
        "graded_blocks": graded_blocks(n_frames),
        "byte_order": sys.byteorder,
        "source": {
            "preprocessing": json.loads(published.read_text(encoding="utf-8")) if published.is_file() else None,
            "output_hashes": json.loads(verification.read_text(encoding="utf-8")).get("output_hashes")
            if verification.is_file() else None,
            "homography": Path(homography_csv).name,
        },
        "layout": {
            "positions.f32": ["n_frames", 1 + len(slot_teams), 2],
            "ball_speed.f32": ["n_frames"],
            "homography.f64": ["n_frames", 9],
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8", newline="\n")
    print(f"Wrote viewer data for {n_frames:,} frames to {out}")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current", default=str(CURRENT_DIR),
                        help="published preprocessing folder (default: data/preprocessing/current)")
    parser.add_argument("--homography", default=str(RAW_DIR / "tactical" / "BAR-ATH_tactical_homography.csv"),
                        help="raw homography CSV (default: data/raw/tactical/BAR-ATH_tactical_homography.csv)")
    parser.add_argument("--out", default=str(REPO / "data" / "viewer"),
                        help="output folder (default: data/viewer)")
    args = parser.parse_args()
    build(args.current, args.homography, args.out)


if __name__ == "__main__":
    main()
