"""Link tracking frames to original tags, retaining uncertainty about team names."""

import csv
import json
from pathlib import Path
import unicodedata
import xml.etree.ElementTree as ET


def normalize_name(value: str) -> str:
    """Normalize typography only; never remove words, accents or punctuation."""
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def team_mapping(metadata: Path, names: list[str]) -> dict:
    """Map unique exact metadata names/IDs to XML team order; retain unknowns.

    Args:
        metadata: Original ATD team metadata.
        names: Raw team labels from the separate tag table.

    Returns:
        Local provenance containing source team attributes, aliases and resolutions.
    """
    teams = list(ET.parse(metadata).iter("Team"))
    ids = [team.get("id") for team in teams]
    if len(teams) != 2 or not all(ids) or len(set(ids)) != 2:
        raise ValueError("Expected two distinct team identifiers")
    aliases: dict[str, set[str]] = {}
    for side, team in zip(("H", "A"), teams):
        values = [team.get(key) for key in ("id", "name", "teamName")]
        values.extend(team.findtext(key) for key in ("Name", "name", "TeamName"))
        for value in values:
            if value and normalize_name(value):
                aliases.setdefault(normalize_name(value), set()).add(side)
    resolutions = {}
    for name in sorted(set(names)):
        key = normalize_name(name)
        candidates = aliases.get(key, set()) if key not in {"", "unknown", "n/a"} else set()
        resolutions[name] = {
            "team": next(iter(candidates)) if len(candidates) == 1 else "UNKNOWN",
            "basis": "metadata-exact" if len(candidates) == 1 else
                     "ambiguous" if candidates else "unmatched",
        }
    return {"normalization": "NFKC, collapsed whitespace, casefold; no fuzzy matching",
            "teams": [{"team": side, "attributes": dict(team.attrib)}
                      for side, team in zip(("H", "A"), teams)],
            "aliases": {key: sorted(value) for key, value in sorted(aliases.items())},
            "resolutions": resolutions}


def build_alignment(
    metadata: Path, tracking: Path, tags_path: Path, output: Path, n_frames: int,
) -> None:
    """Append active tag IDs/sides while copying every original tracking cell unchanged.

    Args:
        metadata: Source ATD XML, read-only.
        tracking: Verified tracking CSV, read-only.
        tags_path: Verified tag CSV, read-only.
        output: Fresh directory for the three alignment outputs.
        n_frames: Expected consecutive video frames beginning at zero.

    Raises:
        ValueError: Frame/tag identities, intervals, or destination violate the contract.
    """
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 1:
        raise ValueError("Expected a positive integer frame count")
    output.mkdir(parents=True, exist_ok=False)
    with tags_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        tag_fields = reader.fieldnames
        tags = list(reader)
    required = {"tag_index", "start_frame", "end_frame", "team_raw"}
    if (not tag_fields or len(set(tag_fields)) != len(tag_fields)
            or not required.issubset(tag_fields) or not tags
            or any(key in tag_fields for key in ("team", "team_match_basis"))):
        raise ValueError("Expected the separate unmodified tag table")
    if any(None in row or any(value is None for value in row.values()) for row in tags):
        raise ValueError("Malformed tag row: every column must have a value")
    mapping = team_mapping(metadata, [row["team_raw"] for row in tags])
    (output / "team_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    starts: dict[int, list[int]] = {}
    stops: dict[int, list[int]] = {}
    sides = []
    with (output / "tags_with_teams.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*tag_fields, "team", "team_match_basis"],
                                lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(tags):
            if row["tag_index"] != str(index):
                raise ValueError("Expected unique sequential tag indices")
            start, stop = int(row["start_frame"]), int(row["end_frame"])
            if stop < start:
                raise ValueError("Reversed tag interval")
            match = mapping["resolutions"][row["team_raw"]]
            sides.append(match["team"])
            writer.writerow({**row, "team": match["team"], "team_match_basis": match["basis"]})
            # Original times stay intact; only the lookup is limited to existing video frames.
            first, last = max(0, start), min(n_frames, stop)
            if first < last:
                starts.setdefault(first, []).append(index)
                stops.setdefault(last, []).append(index)
    with tracking.open(newline="", encoding="utf-8") as source, (
        output / "frames_with_tags.csv"
    ).open("w", newline="", encoding="utf-8") as target:
        reader = csv.DictReader(source)
        extra = ["active_tag_ids_json", "active_tag_teams_json"]
        fields = reader.fieldnames
        if not fields or "frame" not in fields or len(set(fields)) != len(fields):
            raise ValueError("Invalid tracking columns")
        if any(field in fields for field in extra):
            raise ValueError("Tracking already contains alignment columns")
        writer = csv.DictWriter(target, fieldnames=[*fields, *extra], lineterminator="\n")
        writer.writeheader()
        active: set[int] = set()
        count = 0
        for frame, row in enumerate(reader):
            if row["frame"] != str(frame) or frame >= n_frames:
                raise ValueError("Missing, duplicate, unordered or extra tracking frame")
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed tracking row")
            active.difference_update(stops.get(frame, []))
            active.update(starts.get(frame, []))
            indices = sorted(active)
            writer.writerow({**row, extra[0]: json.dumps(indices, separators=(",", ":")),
                             extra[1]: json.dumps([sides[i] for i in indices],
                                                 separators=(",", ":"))})
            count += 1
        if count != n_frames:
            raise ValueError("Missing tracking frames")
