"""NOT USED ANYMORE. Kept, unreferenced, in case we want to go back.

The viewer now builds its data from the verified preprocessing output in
data/preprocessing/current/ with `tools/viewer/prepare_viewer_from_current.py`, so it shows exactly
what the model reads. This older script reads the raw files directly and re-implements the same
alignment rules (frame_id - 1, round(seconds * 25), first XML team = H). It writes the same output
format, so it still works as a fallback or as an independent cross-check of the preprocessing.

---

Convert the raw tactical files into compact binaries the match viewer can load quickly.

Reads (never modifies) the ATD TXT + XML, the homography CSV and, if present, the Smart Tagging
XML. Writes to a local output folder (default: data/viewer, gitignored):

    meta.json         frame count, fps, pitch size, slot -> team, graded blocks, file layout
    positions.f32     float32, n_frames x (1 + n_slots) x 2  -- ball first, then slots; x, y in 0-1, NaN = unseen
    ball_speed.f32    float32, n_frames                       -- m/s, NaN = unseen
    homography.f64    float64, n_frames x 9                   -- row-major 3x3, NaN where invalid
    tags.json         Smart Tagging intervals as video frames (empty list if the XML is missing)

Frame numbering everywhere is the 0-based video frame (= ATD frame_id - 1 = homography `frame`).

Usage:
    python tools/viewer/prepare_viewer_data.py --data-dir data/raw --out data/viewer
"""

import argparse
import json
import math
import sys
import unicodedata
import xml.etree.ElementTree as ET
from array import array
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FPS = 25
PITCH = (105.3, 68.0)
MATRIX_COLUMNS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]


def _norm(text):
    return " ".join(unicodedata.normalize("NFKC", text or "").split()).casefold()


def read_metadata(xml_path):
    """(team names in file order, slot teams 'H'/'A' in slot order). First team is home."""
    root = ET.parse(xml_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError(f"{xml_path}: expected 2 teams, found {len(teams)}")
    ids = [team.get("id") for team in teams]
    names = [team.findtext("Name") or team.get("name") or team.get("id") or "" for team in teams]
    slot_teams = []
    for player in root.iter("Player"):
        team_id = player.get("teamId")
        if team_id not in ids:
            raise ValueError(f"{xml_path}: a track slot has an unknown teamId")
        slot_teams.append("H" if team_id == ids[0] else "A")
    if not slot_teams:
        raise ValueError(f"{xml_path}: no track slots found")
    return names, slot_teams


def read_tracking(txt_path, n_slots):
    """Stream the ATD TXT into flat float32 arrays. Returns (positions, ball_speed, n_frames)."""
    positions = array("f")
    ball_speed = array("f")
    nan = float("nan")
    expected = 1
    with open(txt_path, encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            frame_text, players_text, ball_text = line.split(":")
            if int(frame_text) != expected:
                raise ValueError(f"{txt_path}: expected frame_id {expected}, got {frame_text}")
            groups = [ball_text, *players_text.split(";")]
            if len(groups) != 1 + n_slots:
                raise ValueError(f"{txt_path}: frame {expected} has {len(groups) - 1} slots, "
                                 f"metadata says {n_slots}")
            for index, group in enumerate(groups):
                x, y, speed = (float(value) for value in group.split(","))
                if not (math.isfinite(x) and math.isfinite(y)):
                    x = y = nan
                positions.append(x)
                positions.append(y)
                if index == 0:
                    ball_speed.append(speed if math.isfinite(speed) else nan)
            expected += 1
            if expected % 20000 == 0:
                print(f"  tracking: {expected:,} frames", flush=True)
    return positions, ball_speed, expected - 1


def read_homography(csv_path, n_frames):
    """Flat float64 array, n_frames x 9, NaN where the row is invalid."""
    import csv

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


def read_tags(xml_path, team_names, n_frames):
    """Smart Tagging instances as [start_frame, end_frame) video-frame intervals."""
    lookup = {_norm(name): side for name, side in zip(team_names, ("H", "A")) if _norm(name)}
    tags = []
    for index, instance in enumerate(ET.parse(xml_path).getroot().iter("instance")):
        start, end = float(instance.findtext("start")), float(instance.findtext("end"))
        labels = {}
        for label in instance.findall("label"):
            group, text = (label.findtext("group") or "").strip(), (label.findtext("text") or "").strip()
            if group:
                labels[group] = text
        team_raw = next((v for k, v in labels.items() if k.casefold() == "team"), "")
        set_type = next((v for k, v in labels.items() if k.casefold() == "type"), "")
        first, stop = round(start * FPS), round(end * FPS)
        tags.append({
            "index": index,
            "code": (instance.findtext("code") or "").strip(),
            "start": max(0, first),
            "end": min(n_frames, stop),
            "team": lookup.get(_norm(team_raw), "?"),
            "team_raw": team_raw,
            "type": set_type,
        })
    return tags


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


def build(data_dir, out_dir, prefix="BAR-ATH_tactical"):
    source = Path(data_dir) / "tactical"
    paths = {
        "meta": source / f"{prefix}_atd.xml",
        "raw": source / f"{prefix}_atd.txt",
        "homography": source / f"{prefix}_homography.csv",
        "tags": source / f"{prefix}_smart_tagging.xml",
    }
    for key in ("meta", "raw", "homography"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"Missing {paths[key]}")
    out = Path(out_dir)
    if out.resolve() == source.resolve() or source.resolve() in out.resolve().parents:
        raise ValueError("Write the viewer data outside the raw data folder")
    out.mkdir(parents=True, exist_ok=True)

    team_names, slot_teams = read_metadata(paths["meta"])
    print(f"Reading tracking ({len(slot_teams)} slots)...", flush=True)
    positions, ball_speed, n_frames = read_tracking(paths["raw"], len(slot_teams))
    print("Reading homography...", flush=True)
    matrices = read_homography(paths["homography"], n_frames)
    tags = read_tags(paths["tags"], team_names, n_frames) if paths["tags"].is_file() else []

    with open(out / "positions.f32", "wb") as handle:
        positions.tofile(handle)
    with open(out / "ball_speed.f32", "wb") as handle:
        ball_speed.tofile(handle)
    with open(out / "homography.f64", "wb") as handle:
        matrices.tofile(handle)
    (out / "tags.json").write_text(json.dumps(tags, ensure_ascii=False) + "\n",
                                   encoding="utf-8", newline="\n")
    meta = {
        "n_frames": n_frames,
        "fps": FPS,
        "pitch_m": list(PITCH),
        "teams": {"H": team_names[0], "A": team_names[1]},
        "slot_teams": slot_teams,
        "graded_blocks": graded_blocks(n_frames),
        "byte_order": sys.byteorder,
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
    parser.add_argument("--data-dir", default=str(REPO / "data" / "raw"),
                        help="folder that contains tactical/ (default: data/raw)")
    parser.add_argument("--out", default=str(REPO / "data" / "viewer"),
                        help="output folder (default: data/viewer)")
    parser.add_argument("--prefix", default="BAR-ATH_tactical", help="raw file name prefix")
    args = parser.parse_args()
    build(args.data_dir, args.out, args.prefix)


if __name__ == "__main__":
    main()
