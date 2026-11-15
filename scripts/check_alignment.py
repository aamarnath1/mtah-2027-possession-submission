"""Independent membership and preservation checks; do not import the join builder."""

import csv
import itertools
import json
from pathlib import Path
import unicodedata
import xml.etree.ElementTree as ET


def check_alignment(
    metadata: Path, tracking: Path, tags_path: Path, output: Path, n_frames: int,
) -> dict[str, int]:
    """Check every cell and tag membership using direct interval containment."""
    teams = list(ET.parse(metadata).iter("Team"))
    assert len(teams) == 2
    ids = [team.get("id") for team in teams]
    assert all(ids) and len(set(ids)) == 2
    aliases: dict[str, set[str]] = {}
    for number, team in enumerate(teams):
        names = [team.get(key) for key in ("id", "name", "teamName")]
        names += [team.findtext(key) for key in ("Name", "name", "TeamName")]
        for name in names:
            if name:
                key = " ".join(unicodedata.normalize("NFKC", name).split()).casefold()
                if key:
                    aliases.setdefault(key, set()).add("H" if number == 0 else "A")
    with tags_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        tag_fields, tags = reader.fieldnames, list(reader)
    sides, matches = [], {}
    with (output / "tags_with_teams.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [*tag_fields, "team", "team_match_basis"]
        for index, pair in enumerate(itertools.zip_longest(tags, reader)):
            source, row = pair
            assert source is not None and row is not None
            assert source["tag_index"] == str(index)
            assert {key: row[key] for key in tag_fields} == source
            name = source["team_raw"]
            key = " ".join(unicodedata.normalize("NFKC", name).split()).casefold()
            options = aliases.get(key, set()) if key not in {"", "unknown", "n/a"} else set()
            side = list(options)[0] if len(options) == 1 else "UNKNOWN"
            basis = "metadata-exact" if len(options) == 1 else (
                "ambiguous" if options else "unmatched"
            )
            assert row["team"] == side and row["team_match_basis"] == basis
            sides.append(side)
            matches[name] = {"team": side, "basis": basis}
    mapping = json.loads((output / "team_mapping.json").read_text(encoding="utf-8"))
    assert mapping == {
        "normalization": "NFKC, collapsed whitespace, casefold; no fuzzy matching",
        "teams": [{"team": "H" if i == 0 else "A", "attributes": dict(team.attrib)}
                  for i, team in enumerate(teams)],
        "aliases": {key: sorted(value) for key, value in sorted(aliases.items())},
        "resolutions": matches,
    }
    intervals = [(int(tag["start_frame"]), int(tag["end_frame"])) for tag in tags]
    assert intervals and all(end >= start for start, end in intervals)
    extra = ["active_tag_ids_json", "active_tag_teams_json"]
    count, overlaps, no_tags = 0, 0, 0
    with tracking.open(newline="", encoding="utf-8") as raw, (
        output / "frames_with_tags.csv"
    ).open(newline="", encoding="utf-8") as joined:
        original, result = csv.DictReader(raw), csv.DictReader(joined)
        fields = original.fieldnames
        assert result.fieldnames == [*fields, *extra]
        for frame, pair in enumerate(itertools.zip_longest(original, result)):
            before, after = pair
            assert before is not None and after is not None, "Lost or extra frame"
            assert None not in before and None not in after
            assert all(value is not None for value in after.values())
            assert before["frame"] == after["frame"] == str(frame)
            assert {field: after[field] for field in fields} == before, "Tracking cell changed"
            # Different algorithm from the event-based writer: evaluate each interval directly.
            active = [i for i, (start, end) in enumerate(intervals) if start <= frame < end]
            assert after[extra[0]] == json.dumps(active, separators=(",", ":"))
            assert after[extra[1]] == json.dumps([sides[i] for i in active], separators=(",", ":"))
            overlaps += int(len(active) > 1)
            no_tags += int(not active)
            count += 1
    assert count == n_frames
    return {"frames": count, "overlapping_tag_frames": overlaps, "untagged_frames": no_tags,
            "unknown_team_tags": sides.count("UNKNOWN"),
            "unresolved_named_teams": sum(
                unicodedata.normalize("NFKC", name).strip().casefold() not in {"", "unknown", "n/a"}
                and match["team"] == "UNKNOWN" for name, match in matches.items()
            )}
