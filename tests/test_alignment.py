"""Synthetic tests for exact team matching and lossless frame/tag alignment."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from check_alignment import check_alignment
from possession.alignment import build_alignment, team_mapping
from possession.tagging import build_tags


def fixture(root: Path) -> tuple[Path, Path, Path]:
    """Make a tiny match with overlaps, unknowns, outside tags and a missing ball."""
    meta, tracking, tags = root / "meta.xml", root / "tracking.csv", root / "tags.csv"
    meta.write_text('<root><Team id="10" name="Alpha FC"/>'
                    '<Team id="20"><Name>Beta FC</Name></Team></root>', encoding="utf-8")
    tracking.write_text("frame,ball_x_norm,ball_observed\n0,NaN,0\n1,0.1,1\n2,1.2,1\n"
                        "3,NaN,0\n4,0.2,1\n5,0.3,1\n6,0.4,1\n", encoding="utf-8")
    source = root / "tags.xml"
    events = [(0, .08, " ALPHA   fc "), (.04, .12, "Beta FC"), (-1, .04, "Mystery"),
              (.12, .20, "Ｎ／Ａ"), (.20, .20, "Alpha FC"), (.20, .24, "Alpha FC"),
              (2, 3, "Beta FC")]
    xml = "<root>"
    for start, end, team in events:
        label = f"<label><group>Team</group><text>{team}</text></label>" if team else ""
        xml += (f"<instance><ID>duplicate</ID><start>{start}</start><end>{end}</end>"
                f"<code>Example phase</code>{label}</instance>")
    source.write_text(xml + "</root>", encoding="utf-8")
    build_tags(source, tags, 7)
    return meta, tracking, tags


class AlignmentTests(unittest.TestCase):
    """Use known memberships and deliberate corruptions, not only builder/checker agreement."""

    def test_boundaries_overlaps_unknowns_and_repeatability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta, tracking, tags = fixture(root)
            before = [path.read_bytes() for path in (meta, tracking, tags)]
            first, repeat = root / "first", root / "repeat"
            for out in (first, repeat):
                build_alignment(meta, tracking, tags, out, 7)
                result = check_alignment(meta, tracking, tags, out, 7)
                self.assertEqual(result["frames"], 7)
                self.assertEqual(result["unresolved_named_teams"], 1)
                with (out / "tags_with_teams.csv").open(newline="", encoding="utf-8") as handle:
                    placeholder = list(csv.DictReader(handle))[3]
                self.assertEqual((placeholder["team_raw"], placeholder["team"]), ("Ｎ／Ａ", "UNKNOWN"))
            self.assertEqual(before, [path.read_bytes() for path in (meta, tracking, tags)])
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (repeat / path.name).read_bytes())
            with (first / "frames_with_tags.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([json.loads(row["active_tag_ids_json"]) for row in rows],
                             [[0, 2], [0, 1], [1], [3], [3], [5], []])
            self.assertEqual(json.loads(rows[0]["active_tag_teams_json"]), ["H", "UNKNOWN"])
            self.assertEqual(json.loads(rows[1]["active_tag_teams_json"]), ["H", "A"])
            self.assertEqual(rows[0]["ball_x_norm"], "NaN")
            self.assertEqual(rows[2]["ball_x_norm"], "1.2")

    def test_checker_rejects_corrupted_membership_cells_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta, tracking, tags = fixture(root)
            out = root / "out"
            build_alignment(meta, tracking, tags, out, 7)
            table = out / "frames_with_tags.csv"
            with table.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            def write(changed: list[dict[str, str]]) -> None:
                """Replace only a synthetic output under test."""
                with table.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(changed)

            for field, wrong in (("active_tag_ids_json", "[0]"),
                                 ("active_tag_teams_json", '["A","UNKNOWN"]'),
                                 ("ball_x_norm", "0"), ("frame", "1")):
                changed = [dict(row) for row in rows]
                changed[0][field] = wrong
                write(changed)
                with self.assertRaises(AssertionError):
                    check_alignment(meta, tracking, tags, out, 7)
            for changed in (rows[:-1], [*rows, rows[-1]], [rows[0], *rows]):
                write(changed)
                with self.assertRaises(AssertionError):
                    check_alignment(meta, tracking, tags, out, 7)
            write(rows)
            mapping = out / "team_mapping.json"
            changed = json.loads(mapping.read_text(encoding="utf-8"))
            changed["resolutions"]["Beta FC"]["team"] = "H"
            mapping.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(AssertionError):
                check_alignment(meta, tracking, tags, out, 7)

    def test_ambiguous_names_do_not_guess_or_fuzzy_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.xml"
            path.write_text('<root><Team id="10" name="Example"/>'
                            '<Team id="20" name="EXAMPLE"/></root>', encoding="utf-8")
            choices = team_mapping(path, ["Example", "Example FC", "10", "UNKNOWN", " N/a "])
            self.assertEqual(choices["resolutions"]["Example"]["basis"], "ambiguous")
            self.assertEqual(choices["resolutions"]["Example FC"]["team"], "UNKNOWN")
            self.assertEqual(choices["resolutions"]["10"]["team"], "H")
            self.assertEqual(choices["resolutions"]["UNKNOWN"]["team"], "UNKNOWN")
            self.assertEqual(choices["resolutions"][" N/a "]["team"], "UNKNOWN")

    def test_invalid_frame_sequence_and_tag_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta, tracking, tags = fixture(root)
            tracking.write_text("frame\n0\n0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_alignment(meta, tracking, tags, root / "bad_frames", 7)
            with tags.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["tag_index"] = "0"
            with tags.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                build_alignment(meta, tracking, tags, root / "bad_tags", 7)

    def test_missing_columns_and_short_rows_fail_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta, tracking, tags = fixture(root)
            for index, content in enumerate((
                "tag_index,start_frame,end_frame\n0,0,1\n",
                "tag_index,team_raw\n0,Alpha FC\n",
                "tag_index,start_frame,end_frame,team_raw\n0,0,1\n",
                "tag_index,start_frame,end_frame,team_raw,team_raw\n0,0,1,Alpha,Alpha\n",
            )):
                tags.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    build_alignment(meta, tracking, tags, root / f"bad_schema_{index}", 7)


if __name__ == "__main__":
    unittest.main()
