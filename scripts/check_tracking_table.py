"""Independent table checks; never import the candidate implementation here."""

import csv
import itertools
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET


def columns(n_slots: int) -> list[str]:
    """Return the agreed column order."""
    result = ["frame", "source_frame_id", "time_s"]
    for prefix in ["ball", *[f"slot_{i}" for i in range(1, n_slots + 1)]]:
        result.extend(f"{prefix}_{field}" for field in (
            "x_norm", "y_norm", "speed_mps", "x_m", "y_m", "observed",
        ))
    return result


def check(
    meta: Path, raw: Path, output: Path, n_frames: int, deadline: float | None = None
) -> None:
    """Compare every output field against an independent reading of the raw input."""
    root = ET.parse(meta).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    players = list(root.iter("Player"))
    assert len(teams) == 2 and len(set(teams)) == 2 and None not in teams
    assert players and all(p.get("teamId") in teams for p in players)
    mapping = json.loads((output / "track_slots.json").read_text(encoding="utf-8"))
    assert mapping == [
        {"slot": i, "attributes": dict(p.attrib),
         "team": "H" if p.get("teamId") == teams[0] else "A"}
        for i, p in enumerate(players, 1)
    ], "Slot mapping mismatch"
    with raw.open(encoding="utf-8-sig") as source, (output / "tracking.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == columns(len(players)), "Column mismatch"
        count = 0
        for frame, pair in enumerate(itertools.zip_longest(source, reader)):
            if frame % 1000 == 0 and deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Independent check exceeded the job deadline")
            line, row = pair
            assert line is not None and row is not None, "Row count mismatch"
            pieces = line.strip().split(":")
            assert len(pieces) == 3 and int(pieces[0]) == frame + 1, "Input sequence mismatch"
            assert row["frame"] == str(frame) and row["source_frame_id"] == str(frame + 1)
            assert math.isclose(float(row["time_s"]), frame / 25, abs_tol=1e-12)
            groups = [pieces[2], *pieces[1].split(";")]
            assert len(groups) == len(players) + 1
            for index, group in enumerate(groups):
                values = [float(v) for v in group.split(",")]
                assert len(values) == 3 and not any(math.isinf(v) for v in values)
                x, y, speed = values
                prefix = "ball" if index == 0 else f"slot_{index}"
                expected = dict(zip(
                    ("x_norm", "y_norm", "speed_mps", "x_m", "y_m"),
                    (x, y, speed, x * 105.3, y * 68.0),
                ))
                for field, target in expected.items():
                    actual_text = row[f"{prefix}_{field}"]
                    actual = float(actual_text)
                    assert (math.isnan(target) and actual_text == "NaN") or (
                        math.isfinite(target)
                        and math.isclose(actual, target, rel_tol=1e-12, abs_tol=1e-12)
                    ), "Coordinate or speed mismatch"
                visible = math.isfinite(x) and math.isfinite(y)
                assert row[f"{prefix}_observed"] == str(int(visible))
            count += 1
        assert count == n_frames, "Wrong expected row count"


def fixture(folder: Path) -> tuple[Path, Path]:
    """Create synthetic metadata and tracking with partial, missing and outside positions."""
    folder.mkdir(parents=True, exist_ok=True)
    meta, raw = folder / "metadata.xml", folder / "tracking.txt"
    meta.write_text(
        '<root><Team id="home"/><Team id="away"/>'
        '<Player id="a" teamId="home"/><Player id="b" teamId="away"/></root>',
        encoding="utf-8",
    )
    raw.write_text(
        "1:0.1,0.2,3;0.7,0.8,4:0.3,0.4,5\n"
        "2:NaN,0.2,NaN;0.7,NaN,4:NaN,NaN,NaN\n"
        "3:-0.2,1.2,3;0.7,0.8,4:1.1,0.4,5\n", encoding="utf-8",
    )
    return meta, raw
