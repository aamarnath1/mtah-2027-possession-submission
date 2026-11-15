"""Check tag output independently of the builder, including lossless source content."""

import csv
import itertools
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def check_tags(source: Path, table: Path, n_frames: int) -> dict[str, int]:
    """Compare every source instance to its CSV row and return local quality counts."""
    expected_fields = [
        "tag_index", "source_id", "code", "start_s", "end_s", "start_frame", "end_frame",
        "team_raw", "half_raw", "type_raw", "set_piece_type", "labels_json", "instance_xml",
        "outside_video", "empty_after_rounding",
    ]
    counts = {"tags": 0, "unknown_teams": 0, "outside_video": 0, "empty_after_rounding": 0}
    with table.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        assert rows.fieldnames == expected_fields, "Tag columns mismatch"
        for index, (original, row) in enumerate(itertools.zip_longest(
            ET.parse(source).iter("instance"), rows,
        )):
            assert original is not None and row is not None, "Lost or extra tag row"
            assert None not in row and all(value is not None for value in row.values())
            assert row["tag_index"] == str(index)
            assert row["start_s"] == original.findtext("start")
            assert row["end_s"] == original.findtext("end")
            start, end = float(row["start_s"]), float(row["end_s"])
            assert math.isfinite(start) and math.isfinite(end) and end >= start
            assert row["start_frame"] == str(round(start * 25))
            assert row["end_frame"] == str(round(end * 25))
            assert row["code"] == (original.findtext("code") or "")
            assert row["source_id"] == (
                original.findtext("ID") or original.findtext("id") or original.get("id", "")
            )
            labels = [
                {"group": item.findtext("group"), "text": item.findtext("text")}
                for item in original.findall("label")
            ]
            assert json.loads(row["labels_json"]) == labels, "Labels changed or disappeared"
            for group in ("team", "half", "type"):
                values = {
                    item["text"] for item in labels
                    if (item["group"] or "").strip().casefold() == group
                    and item["text"] is not None and item["text"].strip()
                }
                expected = list(values)[0] if len(values) == 1 else "UNKNOWN"
                assert row[f"{group}_raw"] == expected
            assert row["set_piece_type"] == (
                row["type_raw"] if row["code"].strip().casefold() == "set pieces" else ""
            )
            assert row["instance_xml"] == ET.tostring(original, encoding="unicode")
            outside = int(start < 0 or end > n_frames / 25)
            empty = int(round(start * 25) == round(end * 25))
            assert row["outside_video"] == str(outside)
            assert row["empty_after_rounding"] == str(empty)
            counts["tags"] += 1
            counts["unknown_teams"] += int(row["team_raw"] == "UNKNOWN")
            counts["outside_video"] += outside
            counts["empty_after_rounding"] += empty
    assert counts["tags"] > 0, "No annotation instances checked"
    return counts
