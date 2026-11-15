"""Synthetic preservation, boundary and deliberate-corruption tests for tags."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from check_tag_table import check_tags
from possession.tagging import build_tags


class TaggingTests(unittest.TestCase):
    """Exercise public SportsCode format without opening competition files."""

    def test_source_cannot_be_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.xml"
            source.write_text("<root/>", encoding="utf-8")
            before = source.read_bytes()
            with self.assertRaises(ValueError):
                build_tags(source, source, 25)
            self.assertEqual(source.read_bytes(), before)

    def test_preservation_boundaries_and_overlaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, table = root / "tags.xml", root / "tags.csv"
            source.write_text(
                '<root><instance custom="kept"><ID>same</ID><start>0.02</start>'
                '<end>0.10</end><code>ſet pieces</code><label><group>Team</group>'
                '<text>Example FC</text></label><label><group>Type</group>'
                '<text>Corner</text></label><label><group>Type</group>'
                '<text>Corner</text></label><extra>preserved</extra></instance>'
                '<instance><ID>same</ID><start>0.04</start><end>0.08</end>'
                '<code>Future code</code></instance>'
                '<instance><start>-1</start><end>2</end><label><group>Team</group>'
                '<text>Alpha</text></label><label><group>Team</group><text>Beta</text>'
                '</label></instance><instance><start>0.001</start><end>0.002</end>'
                '</instance></root>', encoding="utf-8",
            )
            build_tags(source, table, 25)
            report = check_tags(source, table, 25)
            self.assertEqual(report, {
                "tags": 4, "unknown_teams": 3, "outside_video": 1, "empty_after_rounding": 1,
            })
            with table.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual((rows[0]["start_frame"], rows[0]["end_frame"]), ("0", "2"))
            self.assertEqual(rows[0]["set_piece_type"], "Corner")
            # End frames are excluded; at frame 2 both first intervals have ended.
            active = [
                [row["tag_index"] for row in rows
                 if int(row["start_frame"]) <= frame < int(row["end_frame"])]
                for frame in range(3)
            ]
            self.assertEqual(active, [["0", "2"], ["0", "1", "2"], ["2"]])
            original = table.read_bytes()
            build_tags(source, table, 25)
            self.assertEqual(table.read_bytes(), original)
            for field, wrong in (("end_frame", "999"), ("labels_json", "[]"),
                                 ("instance_xml", "<instance/>"), ("team_raw", "A")):
                changed = [dict(row) for row in rows]
                changed[0][field] = wrong
                with table.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(changed)
                with self.assertRaises(AssertionError):
                    check_tags(source, table, 25)

    def test_bad_times_and_wrong_format_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, table = Path(directory) / "tags.xml", Path(directory) / "tags.csv"
            for content in (
                "<root/>", "<root><instance><end>1</end></instance></root>",
                *[f"<root><instance><start>{start}</start><end>{end}</end></instance></root>"
                  for start, end in (("NaN", "1"), ("0", "inf"), ("2", "1"),
                                     ("oops", "1"), ("1e308", "1e308"))],
            ):
                source.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    build_tags(source, table, 25)


if __name__ == "__main__":
    unittest.main()
