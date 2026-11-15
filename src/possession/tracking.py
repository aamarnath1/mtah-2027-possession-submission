"""Build an observation table without filling gaps or assigning possession labels."""

import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_observation(text: str) -> tuple[float, float, float]:
    """Read x, y and speed, preserving NaN and refusing malformed or infinite values."""
    parts = text.split(",")
    if len(parts) != 3:
        raise ValueError("Expected three observation values")
    x, y, speed = (float(part) for part in parts)
    if any(math.isinf(value) for value in (x, y, speed)):
        raise ValueError("Infinite observation")
    return x, y, speed


def build(meta_path: Path, raw_path: Path, out_dir: Path, n_frames: int) -> None:
    """Stream ATD into tracking.csv and track_slots.json.

    Args:
        meta_path: Companion EPTS XML describing teams and ordered player slots.
        raw_path: Raw frame:players:ball observations on the tactical timeline.
        out_dir: Local output directory. Raw files remain unchanged.
        n_frames: Expected number of consecutive, one-based tracking frames.

    Raises:
        ValueError: Metadata or observations violate the input contract.
    """
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 1:
        raise ValueError("Expected a positive integer frame count")
    root = ET.parse(meta_path).getroot()
    teams = [team.get("id") for team in root.iter("Team")]
    players = list(root.iter("Player"))
    if len(teams) != 2 or not all(teams) or len(set(teams)) != 2:
        raise ValueError("Expected two distinct team identifiers")
    if not players or any(player.get("teamId") not in teams for player in players):
        raise ValueError("Expected known team membership for every track slot")
    slots = [
        {"slot": index, "attributes": dict(player.attrib),
         "team": "H" if player.get("teamId") == teams[0] else "A"}
        for index, player in enumerate(players, 1)
    ]
    prefixes = ["ball", *[f"slot_{i}" for i in range(1, len(players) + 1)]]
    fields = ("x_norm", "y_norm", "speed_mps", "x_m", "y_m", "observed")
    header = ["frame", "source_frame_id", "time_s"]
    header.extend(f"{prefix}_{field}" for prefix in prefixes for field in fields)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with raw_path.open(encoding="utf-8-sig") as source, (out_dir / "tracking.csv").open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(header)
        for expected_id, line in enumerate(source, 1):
            parts = line.strip().split(":")
            if len(parts) != 3:
                raise ValueError("Malformed tracking record")
            frame_id = int(parts[0])
            if frame_id != expected_id or frame_id > n_frames:
                raise ValueError("Nonconsecutive or out-of-range tracking frame")
            groups = parts[1].split(";")
            if len(groups) != len(players):
                raise ValueError("Incorrect track-slot count")
            observations = [parse_observation(group) for group in [parts[2], *groups]]
            row: list[object] = [frame_id - 1, frame_id, (frame_id - 1) / 25]
            for x, y, speed in observations:
                values = (x, y, speed, x * 105.3, y * 68.0)
                row.extend("NaN" if math.isnan(value) else value for value in values)
                row.append(int(math.isfinite(x) and math.isfinite(y)))
            writer.writerow(row)
            count += 1
        if count != n_frames:
            raise ValueError("Missing tracking frames")
    (out_dir / "track_slots.json").write_text(
        json.dumps(slots, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
