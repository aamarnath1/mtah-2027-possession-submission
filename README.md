# Catalanian Compliers · More Than a Hack 2027

This repository reproduces the team's possession CSV with the hybrid Evidence
decoder. The organizer supplies the tactical match data to registered teams.

## Reproduce

Use Python 3.13.13. Clone the repo with `--recurse-submodules`, then activate a
fresh virtual environment and install the pinned dependencies.

```sh
python -m pip install -r requirements.txt
git submodule update --init --recursive
```

Put the organizer's three tactical preprocessing inputs under `data/raw/tactical/`:
`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`, and
`BAR-ATH_tactical_smart_tagging.xml`. The rebuild aligns inputs to the tactical
frames. The decoder uses tracking, and the tagging table is part of the
preprocessing integrity checks.

```sh
python scripts/rebuild_preprocessing.py
python scripts/make_submission.py --evidence --out "submissions/Catalanian Compliers.csv"
python vendor/mtah/evaluation/validate_submission.py "submissions/Catalanian Compliers.csv"
```

The organizer submodule pins the official validator. See [the technical
write-up](TECHNICAL.md) for methods, training data, and results.

Source snapshot: team repository commit `3a015a42aae1fe5657f9b0e98e7650f90fcaf0d8`.
