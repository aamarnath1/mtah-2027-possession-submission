"""IDSSE halves as the same numpy arrays our detectors read, cached on disk.

Parsing one 370 MB position file takes minutes, so each half is parsed once and
saved to data/cache/idsse/<match>_<half>.npz. Delete that folder to force a reparse.

The arrays follow possession.loading.load_arrays(), with two differences:
  * x is shifted so it runs 0..pitch_x from one goal line, like our own feed
    (IDSSE puts the origin at the centre spot), and `pitch_x_m` says how long
    the pitch is for this match.
  * `labels` holds the ground truth ('H' / 'A' / 'D' per frame).

The ball is present on every frame here. Hiding it to mimic our feed is the
caller's job (scripts/tune_external.py hide_ball, scripts/boundary_report.py
hide_live), so the same cached arrays serve every gap regime.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from possession.external import idsse
from possession.paths import DATA_DIR

IDSSE_DIR = DATA_DIR / "external" / "idsse"
CACHE_DIR = DATA_DIR / "cache" / "idsse"
HALVES = ("firstHalf", "secondHalf")


def find_files(match_id: str, folder: Path = IDSSE_DIR) -> tuple[Path, Path]:
    pos = sorted(folder.glob(f"DFL_04_03_positions_raw_observed_*{match_id}*.xml"))
    info = sorted(folder.glob(f"DFL_02_01_matchinformation_*{match_id}*.xml"))
    if not pos or not info:
        raise FileNotFoundError(f"no IDSSE files for {match_id} in {folder}")
    return pos[0], info[0]


def pitch_length(matchinfo_path: Path) -> float:
    """PitchX from the match information file, in metres."""
    for env in ET.parse(matchinfo_path).getroot().iter("Environment"):
        if env.get("PitchX"):
            return float(env.get("PitchX"))
    return 105.0


def to_arrays(match: idsse.ExternalMatch, length: float) -> dict:
    """Convert a parsed half to detector arrays, with x moved to 0..length."""
    def xy(points):
        return np.array([p if p is not None else (np.nan, np.nan) for p in points], dtype=float)

    tracks = match.players["H"] + match.players["A"]
    slot_xy = np.stack([xy(t) for t in tracks], axis=1)          # (n, slots, 2)
    ball_xy = xy(match.ball_xy)
    slot_xy[..., 0] += length / 2.0
    ball_xy[:, 0] += length / 2.0
    teams = np.array(["H"] * len(match.players["H"]) + ["A"] * len(match.players["A"]))
    return {
        "ball_xy": ball_xy,
        "ball_obs": np.isfinite(ball_xy).all(axis=1),
        "slot_xy": slot_xy,
        "slot_team": teams,
        "labels": np.array(match.labels),
        "pitch_x_m": np.float64(length),
    }


def load_half(match_id: str, half: str, folder: Path = IDSSE_DIR,
              cache_dir: Path = CACHE_DIR) -> dict:
    """Arrays for one half, from the cache if present, else parsed and cached."""
    cache = Path(cache_dir) / f"{match_id}_{half}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    pos, info = find_files(match_id, folder)
    arrays = to_arrays(idsse.load(pos, info, half=half), pitch_length(info))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)
    return arrays
