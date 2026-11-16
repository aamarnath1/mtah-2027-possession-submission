"""Viewer data from preprocessing output matches the raw-file path (synthetic data only)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools" / "viewer"), str(Path(__file__).parent)]
import prepare_viewer_data as legacy  # noqa: E402
import prepare_viewer_from_current as from_current  # noqa: E402
from possession.alignment import build_alignment  # noqa: E402
from possession.tagging import build_tags  # noqa: E402
from possession.tracking import build as build_tracking  # noqa: E402
from test_viewer_prepare import write_bundle  # noqa: E402

N_FRAMES = 3


class ViewerFromCurrentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.raw = write_bundle(root)
        tactical = self.raw / "tactical"
        # run the real preprocessing builders to make a synthetic current/
        self.current = root / "current"
        build_tracking(tactical / "X_tactical_atd.xml", tactical / "X_tactical_atd.txt", self.current, N_FRAMES)
        build_tags(tactical / "X_tactical_smart_tagging.xml", self.current / "tags.csv", N_FRAMES)
        build_alignment(tactical / "X_tactical_atd.xml", self.current / "tracking.csv",
                        self.current / "tags.csv", self.current / "aligned", N_FRAMES)
        self.homography = tactical / "X_tactical_homography.csv"
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def test_matches_the_raw_file_path_byte_for_byte(self):
        new = from_current.build(self.current, self.homography, self.root / "viewer_new")
        old = legacy.build(self.raw, self.root / "viewer_old", prefix="X_tactical")
        for name in ("positions.f32", "ball_speed.f32", "homography.f64"):
            self.assertEqual((self.root / "viewer_new" / name).read_bytes(),
                             (self.root / "viewer_old" / name).read_bytes(), name)
        self.assertEqual(new["slot_teams"], old["slot_teams"])
        self.assertEqual(new["n_frames"], old["n_frames"])
        new_tags = json.loads((self.root / "viewer_new" / "tags.json").read_text())
        old_tags = json.loads((self.root / "viewer_old" / "tags.json").read_text())
        keys = ("index", "code", "start", "end", "team", "type")
        self.assertEqual([{k: t[k] for k in keys} for t in new_tags],
                         [{k: t[k] for k in keys} for t in old_tags])

    def test_legend_names_come_from_exact_team_matches(self):
        meta = from_current.build(self.current, self.homography, self.root / "viewer")
        self.assertEqual(meta["teams"]["A"], "away fc")  # tag text as written; "N/A" never used
        self.assertEqual(meta["teams"]["H"], "Home")     # no tag named the home team

    def test_missing_current_gives_a_helpful_error(self):
        with self.assertRaises(FileNotFoundError) as caught:
            from_current.build(self.root / "nowhere", self.homography, self.root / "viewer")
        self.assertIn("rebuild_preprocessing.py", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
