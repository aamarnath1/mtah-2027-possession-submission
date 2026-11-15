"""Build a submission CSV end to end, then validate it.

    python scripts/make_submission.py --out data/predictions/baseline.csv

Reads the published preprocessing in data/preprocessing/current/, so run
scripts/rebuild_preprocessing.py first.

This is deliberately one command. Reproducibility is one of the three graded
deliverables: a fresh clone plus the data bundle has to regenerate a byte
identical file, and that is much easier to keep true when there is only ever one
path to producing it.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "vendor" / "mtah" / "evaluation"))

from possession import baseline, decode  # noqa: E402
from possession.detectors import dead_ball  # noqa: E402
from possession.labels import (  # noqa: E402
    apply_known_dead,
    count_graded,
    fill_abstentions,
    write_submission,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "predictions" / "baseline.csv")
    parser.add_argument("--frames", type=Path, default=None,
                        help="override the frames_with_tags.csv path")
    parser.add_argument("--hold", type=int, default=30,
                        help="frames the other team must lead before possession switches. "
                             "Fitted on four IDSSE halves: 30 beats 12 by +0.021 spellF1 "
                             "and is best or near-best on three of four")
    parser.add_argument("--window", type=int, default=1,
                        help="centered smoothing window in frames; 1 disables it. "
                             "Fitted on IDSSE: the sticky baseline already resists "
                             "flicker, and smoothing on top of it over-smooths")
    parser.add_argument("--min-run", type=int, default=25,
                        help="runs shorter than this are absorbed into the previous one")
    parser.add_argument("--dead-threshold", type=float, default=None,
                        help="fraction of players slow above which play is dead; "
                             "default is fitted on IDSSE and sits on a plateau")
    parser.add_argument("--no-gap-policy", action="store_true",
                        help="keep hand-overs at the END of a ball gap (for an A/B)")
    parser.add_argument("--no-dead", action="store_true",
                        help="skip dead-ball detection (for an A/B against it)")
    args = parser.parse_args()

    slot_teams = baseline.load_slot_teams()
    print("reading preprocessing ...")
    table = baseline.load_table(args.frames, slot_teams)

    print("nearest-team evidence ...")
    nearest, _ = baseline.nearest_team_per_frame(table=table, slot_teams=slot_teams)
    labels = baseline.sticky_labels(nearest, hold_frames=args.hold)

    if not args.no_gap_policy:
        labels, moved = baseline.place_switches_in_gaps(labels, nearest)
        print(f"  relocated {moved} hand-overs to the midpoint of a ball gap")

    if not args.no_dead:
        print("dead-ball evidence ...")
        evidence = dead_ball.detect_from_table(
            table, sorted(slot_teams), slow_threshold=args.dead_threshold)
        n_dead = int((evidence >= 0.5).sum())
        print(f"  flagged dead on {n_dead} frames ({n_dead / len(table):.1%})")
        labels = ["D" if evidence[i] >= 0.5 else labels[i] for i in range(len(labels))]

    # Before decoding, so the decoder's minimum-run rule sees the real runs
    # either side of the kickoff rather than being handed a truncated one.
    labels = apply_known_dead(labels)

    print(f"decoding (centered window {args.window}, min run {args.min_run}) ...")
    labels = decode.decode(labels, window=args.window,
                           min_frames={"H": args.min_run, "A": args.min_run, "D": args.min_run})
    labels = fill_abstentions(labels)

    path = write_submission(labels, args.out)
    print(f"wrote {path}")
    print(f"possessions claimed: {decode.count_possessions(labels)}")
    print(f"graded frames filled: {count_graded(labels)}")

    import possession_metrics as pm  # the official spec, never reimplemented

    frame = pm.read_label_csv(path)
    pm.validate_submission(frame, pm.N_VIDEO_FRAMES if hasattr(pm, "N_VIDEO_FRAMES") else 199_319)
    print("official validator: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
