# Catalanian Compliers · More Than a Hack 2027

This repository reproduces the team's baseline possession CSV for the official tactical
recording. The pipeline aligns tracking and Smart Tagging, estimates possession from
the nearest observed team, uses player movement and pitch boundaries to detect dead
ball, and smooths brief changes before writing one label per frame.

## Reproduce the submission

Use Python 3.13.13 and Git. The organizer's validation code is pinned as a Git submodule.

```sh
git clone --recurse-submodules https://github.com/aamarnath1/mtah-2027-possession-submission.git
cd mtah-2027-possession-submission
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS or Linux, or
`.\.venv\Scripts\Activate.ps1` in PowerShell. Install the pinned dependencies.

```sh
python -m pip install -r requirements.txt
```

Place these files from the organizer's licensed bundle in `data/raw/tactical/`.

- `BAR-ATH_tactical_atd.xml`
- `BAR-ATH_tactical_atd.txt`
- `BAR-ATH_tactical_smart_tagging.xml`

```sh
python scripts/rebuild_preprocessing.py
python scripts/make_submission.py --out "submissions/Catalanian Compliers.csv"
python vendor/mtah/evaluation/validate_submission.py "submissions/Catalanian Compliers.csv"
```

To keep the bundle elsewhere, pass `--data-dir PATH` to the preprocessing command.
That path must contain the `tactical/` directory. The generated CSV has 199,319 data
rows with the header `frame,label`.

## Credits

Baseline and preprocessing by the Catalanian Compliers team. External parameter
development used the IDSSE dataset from the Deutsche Fußball Liga. See
[Bassek et al., 2025](https://doi.org/10.1038/s41597-025-04505-y) and the
[organizer repository](https://github.com/metrica-sports/more-than-a-hack-2027).
