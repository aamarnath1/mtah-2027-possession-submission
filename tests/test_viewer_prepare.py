"""Viewer data prep on tiny synthetic inputs (no match data)."""

import json
import math
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "viewer"))
import prepare_viewer_data as prep  # noqa: E402
from serve_viewer import _inside  # noqa: E402

XML = """<Root><Teams><Team id="T1"><Name>Home FC</Name></Team><Team id="T2"><Name>Away FC</Name></Team></Teams>
<Players><Player id="1" teamId="T2"/><Player id="2" teamId="T1"/></Players></Root>"""
TXT = ("1:0.1,0.2,1.0;NaN,NaN,NaN:0.5,0.5,7.0\n"
       "2:0.11,0.21,1.1;0.9,0.8,2.0:NaN,NaN,NaN\n"
       "3:0.12,0.22,1.2;0.91,0.81,2.1:0.6,0.4,8.0\n")
HOMOGRAPHY = ("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n"
              "0,100,0,10,0,50,5,0,0,1,0.2,1\n"
              "1,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,0\n"
              "2,100,0,10,0,50,5,0,0,1,0.2,1\n")
TAGS = """<file><ALL_INSTANCES>
<instance><ID>1</ID><start>0.0</start><end>0.1</end><code>BUILD UP</code>
<label><group>Team</group><text>away fc</text></label></instance>
<instance><ID>2</ID><start>0.04</start><end>9.0</end><code>SET PIECES</code>
<label><group>Team</group><text>N/A</text></label><label><group>Type</group><text>Throw-in</text></label></instance>
</ALL_INSTANCES></file>"""


def write_bundle(root, txt=TXT):
    tactical = root / "raw" / "tactical"
    tactical.mkdir(parents=True)
    (tactical / "X_tactical_atd.xml").write_text(XML)
    (tactical / "X_tactical_atd.txt").write_text(txt)
    (tactical / "X_tactical_homography.csv").write_text(HOMOGRAPHY)
    (tactical / "X_tactical_smart_tagging.xml").write_text(TAGS)
    return root / "raw"


def read_floats(path, code):
    values = array(code)
    values.frombytes(path.read_bytes())
    return values


class PrepareViewerDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, txt=TXT):
        raw = write_bundle(self.root, txt)
        out = self.root / "viewer"
        return prep.build(raw, out, prefix="X_tactical"), out

    def test_meta_and_slot_teams_follow_xml_order(self):
        meta, _ = self.build()
        self.assertEqual(meta["n_frames"], 3)
        self.assertEqual(meta["slot_teams"], ["A", "H"])  # first XML team is home
        self.assertEqual(meta["teams"], {"H": "Home FC", "A": "Away FC"})
        self.assertEqual(meta["graded_blocks"], [])  # not the real match length

    def test_positions_ball_first_then_slots_with_nan_kept(self):
        _, out = self.build()
        pos = read_floats(out / "positions.f32", "f")
        self.assertEqual(len(pos), 3 * 3 * 2)
        frame0 = pos[0:6]
        self.assertAlmostEqual(frame0[0], 0.5, places=6)  # ball x
        self.assertAlmostEqual(frame0[2], 0.1, places=6)  # slot 1 x
        self.assertTrue(math.isnan(frame0[4]) and math.isnan(frame0[5]))  # slot 2 unseen
        self.assertTrue(math.isnan(pos[6]))  # frame 1 ball unseen
        speed = read_floats(out / "ball_speed.f32", "f")
        self.assertAlmostEqual(speed[0], 7.0, places=6)
        self.assertTrue(math.isnan(speed[1]))

    def test_homography_nan_where_invalid(self):
        _, out = self.build()
        hom = read_floats(out / "homography.f64", "d")
        self.assertEqual(len(hom), 27)
        self.assertEqual(hom[0], 100.0)
        self.assertTrue(all(math.isnan(v) for v in hom[9:18]))

    def test_tags_frames_clipped_and_teams_mapped(self):
        _, out = self.build()
        tags = json.loads((out / "tags.json").read_text())
        self.assertEqual([t["team"] for t in tags], ["A", "?"])
        self.assertEqual((tags[0]["start"], tags[0]["end"]), (0, 2))  # round(0.1 * 25) = 2
        self.assertEqual(tags[1]["end"], 3)  # clipped to n_frames
        self.assertEqual(tags[1]["type"], "Throw-in")

    def test_rejects_wrong_slot_count(self):
        with self.assertRaises(ValueError):
            self.build(txt="1:0.1,0.2,1.0:0.5,0.5,7.0\n")

    def test_rejects_output_inside_raw_folder(self):
        raw = write_bundle(self.root)
        with self.assertRaises(ValueError):
            prep.build(raw, raw / "tactical" / "viewer", prefix="X_tactical")

    def test_server_refuses_paths_outside_its_folder(self):
        (self.root / "ok.json").write_text("{}")
        self.assertIsNotNone(_inside(self.root, "ok.json"))
        for name in ("../ok.json", "..", ".hidden", "sub/ok.json", ""):
            self.assertIsNone(_inside(self.root, name))


if __name__ == "__main__":
    unittest.main()
