"""Preserve SportsCode annotations on the tactical video timeline."""

import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


FIELDS = (
    "tag_index", "source_id", "code", "start_s", "end_s", "start_frame", "end_frame",
    "team_raw", "half_raw", "type_raw", "set_piece_type", "labels_json", "instance_xml",
    "outside_video", "empty_after_rounding",
)


def build_tags(source: Path, output: Path, n_frames: int) -> None:
    """Write annotations in source order, without assigning possession or team sides.

    Args:
        source: Original SportsCode XML, opened read-only.
        output: Destination CSV. The caller must choose a fresh local destination.
        n_frames: Number of tactical video frames, used only to flag outside intervals.

    Raises:
        ValueError: Times are absent, nonnumeric, nonfinite, or reversed.
    """
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 1:
        raise ValueError("Expected a positive integer frame count")
    if source.resolve() == output.resolve():
        raise ValueError("Output must not overwrite the source XML")
    instances = list(ET.parse(source).iter("instance"))
    if not instances:
        raise ValueError("No annotation instances found; check the XML format")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for index, instance in enumerate(instances):
            start_text, end_text = instance.findtext("start"), instance.findtext("end")
            if start_text is None or end_text is None:
                raise ValueError("Annotation lacks start or end time")
            start, end = float(start_text), float(end_text)
            if not (math.isfinite(start) and math.isfinite(end)) or end < start:
                raise ValueError("Annotation has nonfinite or reversed times")
            if not (math.isfinite(start * 25) and math.isfinite(end * 25)):
                raise ValueError("Annotation frame conversion overflows")
            first, stop = round(start * 25), round(end * 25)
            labels = [
                {"group": label.findtext("group"), "text": label.findtext("text")}
                for label in instance.findall("label")
            ]

            def single(group: str) -> str:
                """Project an unambiguous label while retaining every original label."""
                values = {
                    label["text"] for label in labels
                    if (label["group"] or "").strip().casefold() == group
                    and label["text"] is not None and label["text"].strip()
                }
                return next(iter(values)) if len(values) == 1 else "UNKNOWN"

            code = instance.findtext("code") or ""
            writer.writerow(dict(zip(FIELDS, (
                index, instance.findtext("ID") or instance.findtext("id")
                or instance.get("id", ""), code, start_text, end_text, first, stop,
                single("team"), single("half"), single("type"),
                single("type") if code.strip().casefold() == "set pieces" else "",
                json.dumps(labels, ensure_ascii=False, separators=(",", ":")),
                ET.tostring(instance, encoding="unicode"),
                int(start < 0 or end > n_frames / 25), int(first == stop),
            ))))
