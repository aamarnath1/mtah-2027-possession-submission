"""Synthetic mutation tests for the independent checker and local restrictions."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from check_tracking_table import check, columns, fixture


class CheckerTests(unittest.TestCase):
    """The checker must reject plausible-looking but incorrect output."""

    def test_mutations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            meta, raw = fixture(folder)
            output = folder / "out"
            output.mkdir()
            players = list(ET.parse(meta).getroot().iter("Player"))
            mapping = [
                {"slot": i, "attributes": dict(p.attrib), "team": "H" if i == 1 else "A"}
                for i, p in enumerate(players, 1)
            ]
            (output / "track_slots.json").write_text(json.dumps(mapping), encoding="utf-8")
            rows = []
            for i, line in enumerate(raw.read_text().splitlines()):
                _, slots, ball = line.split(":")
                values = [str(i), str(i + 1), str(i / 25)]
                for group in [ball, *slots.split(";")]:
                    x, y, speed = map(float, group.split(","))
                    import math
                    numbers = [x, y, speed, x * 105.3, y * 68]
                    values.extend("NaN" if math.isnan(v) else str(v) for v in numbers)
                    values.append(str(int(math.isfinite(x) and math.isfinite(y))))
                rows.append(values)

            def write(changed: list[list[str]]) -> None:
                with (output / "tracking.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(columns(2))
                    writer.writerows(changed)

            write(rows)
            check(meta, raw, output, 3)
            for row_index, column, value in [
                (0, 0, "1"), (0, 2, "99"), (0, 6, "1"), (1, 3, "0"),
                (1, 14, "1"), (2, 3, "1"),
            ]:
                changed = [row[:] for row in rows]
                changed[row_index][column] = value
                write(changed)
                with self.assertRaises(AssertionError):
                    check(meta, raw, output, 3)
            write(rows[:-1])
            with self.assertRaises(AssertionError):
                check(meta, raw, output, 3)

    def test_network_subprocess_and_file_guards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code = (
                "import socket, subprocess, sys; from pathlib import Path; "
                "from local_guard import install; "
                "install([Path(sys.base_prefix)], [Path(sys.argv[1])]); "
                "checks=[lambda: socket.create_connection(('example.com',443)), "
                "lambda: subprocess.run(['whoami']), "
                "lambda: Path(sys.argv[1]).parent.joinpath('not-allowed.txt').write_text('bad')]; "
                "\nfor action in checks:\n"
                " try: action()\n except PermissionError: continue\n"
                " else: raise AssertionError('Guard failed')\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code, directory], cwd=SCRIPTS,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
