"""Publishing a verified run to data/preprocessing/current/ (synthetic files only)."""

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from possession.paths import current_file, new_run_dir, publish_current  # noqa: E402


def make_run(root, name, state="COMPLETE", marker="v1"):
    run = root / "runs" / name
    (run / "first" / "aligned").mkdir(parents=True)
    (run / "first" / "tracking.csv").write_text(f"frame\n{marker}\n")
    (run / "first" / "aligned" / "frames_with_tags.csv").write_text(marker)
    (run / "verification.json").write_text(json.dumps({"output_hashes": {"marker": marker}}))
    (run / "status.json").write_text(json.dumps({"state": state, "mapping_review_required": False}))
    return run


class PublishCurrentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "preprocessing"

    def tearDown(self):
        self.tmp.cleanup()

    def test_publish_copies_tables_verification_and_record(self):
        current = publish_current(make_run(self.root, "r1"), self.root)
        self.assertEqual((current / "tracking.csv").read_text(), "frame\nv1\n")
        self.assertEqual((current / "aligned" / "frames_with_tags.csv").read_text(), "v1")
        self.assertIn("v1", (current / "verification.json").read_text())
        record = json.loads((current / "PUBLISHED.json").read_text())
        self.assertEqual(record["run"], str(Path("runs") / "r1"))
        self.assertEqual(current_file("tracking.csv", self.root), current / "tracking.csv")

    def test_republish_replaces_and_leaves_no_temp_folders(self):
        publish_current(make_run(self.root, "r1", marker="v1"), self.root)
        current = publish_current(make_run(self.root, "r2", marker="v2"), self.root)
        self.assertEqual((current / "aligned" / "frames_with_tags.csv").read_text(), "v2")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["current", "runs"])
        # the earlier run folder is kept intact
        self.assertEqual((self.root / "runs" / "r1" / "first" / "tracking.csv").read_text(),
                         "frame\nv1\n")

    def test_failed_run_is_refused_and_current_untouched(self):
        publish_current(make_run(self.root, "r1", marker="v1"), self.root)
        with self.assertRaises(ValueError):
            publish_current(make_run(self.root, "r2", state="FAILED", marker="v2"), self.root)
        self.assertEqual((self.root / "current" / "tracking.csv").read_text(), "frame\nv1\n")

    def test_leftovers_from_an_interrupted_publish_are_cleared(self):
        (self.root / ".current-staging" / "junk").mkdir(parents=True)
        publish_current(make_run(self.root, "r1"), self.root)
        self.assertFalse((self.root / ".current-staging").exists())

    def test_missing_current_gives_a_helpful_error(self):
        with self.assertRaises(FileNotFoundError) as caught:
            current_file("tracking.csv", self.root)
        self.assertIn("rebuild_preprocessing.py", str(caught.exception))

    def test_new_run_dirs_are_unique(self):
        first = new_run_dir(self.root)
        first.mkdir(parents=True)
        self.assertNotEqual(new_run_dir(self.root), first)

    def test_publish_works_under_the_local_file_guard(self):
        make_run(self.root, "r1", marker="v1")
        publish_current(self.root / "runs" / "r1", self.root)
        make_run(self.root, "r2", marker="v2")
        script = textwrap.dedent(f"""
            import sys
            from pathlib import Path
            sys.path[:0] = [{str(REPO / 'src')!r}, {str(REPO / 'scripts')!r}]
            from local_guard import install
            from possession.paths import publish_current
            root = Path({str(self.root)!r})
            install([Path(sys.base_prefix), Path(sys.prefix), Path({str(REPO)!r})], [root])
            publish_current(root / "runs" / "r2", root)
        """)
        subprocess.run([sys.executable, "-c", script], check=True, cwd=self.tmp.name)
        self.assertEqual((self.root / "current" / "tracking.csv").read_text(), "frame\nv2\n")


if __name__ == "__main__":
    unittest.main()
