"""Build a submission CSV end to end, then validate it.

    python scripts/make_submission.py --out data/predictions/baseline.csv
    python scripts/make_submission.py --b6
    python scripts/make_submission.py --evidence

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

from possession import baseline, decode, sequence  # noqa: E402
from possession.detectors import dead_ball  # noqa: E402
from possession.labels import (  # noqa: E402
    apply_known_dead,
    count_graded,
    fill_abstentions,
    write_submission,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="override the output CSV path")
    parser.add_argument("--b6", action="store_true", help="enable B6 early confirmation")
    parser.add_argument("--evidence", action="store_true",
                        help="hybrid Evidence decoder: dead ball v2 mask and B6 control "
                             "spans fixed, ball-to-team and team shape voted, Viterbi "
                             "(docs/evidence-decoder.md)")
    parser.add_argument("--dead-v1", action="store_true",
                        help="with --evidence: use Hari's dead-ball mask instead of v2 (for an A/B)")
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
    dead = None
    if not args.no_dead:
        print("dead-ball evidence ...")
        evidence = dead_ball.detect_from_table(
            table, sorted(slot_teams), slow_threshold=args.dead_threshold)
        dead = evidence >= 0.5
        n_dead = int(dead.sum())
        print(f"  flagged dead on {n_dead} frames ({n_dead / len(table):.1%})")
    if args.evidence:
        if dead is None:
            parser.error("--evidence needs the dead-ball mask; drop --no-dead")
        labels = hybrid(table, slot_teams, dead, use_v2=not args.dead_v1)
    else:
        labels = rule_based(args, table, slot_teams, nearest, dead)

    # Before decoding, so the decoder's minimum-run rule sees the real runs
    # either side of the kickoff rather than being handed a truncated one.
    labels = apply_known_dead(labels)

    print(f"decoding (centered window {args.window}, min run {args.min_run}) ...")
    labels = decode.decode(labels, window=args.window,
                           min_frames={"H": args.min_run, "A": args.min_run, "D": args.min_run})
    labels = fill_abstentions(labels)

    filename = "evidence.csv" if args.evidence else "b6.csv" if args.b6 else "baseline.csv"
    path = write_submission(labels, args.out or REPO / "data" / "predictions" / filename)
    print(f"wrote {path}")
    print(f"possessions claimed: {decode.count_possessions(labels)}")
    print(f"graded frames filled: {count_graded(labels)}")

    import possession_metrics as pm  # the official spec, never reimplemented

    frame = pm.read_label_csv(path)
    problem = pm.validate_submission(
        frame, pm.N_VIDEO_FRAMES if hasattr(pm, "N_VIDEO_FRAMES") else 199_319)
    if problem is not None:      # it returns the reason rather than raising
        print(f"official validator: FAILED: {problem}")
        return 1
    print("official validator: PASSED")
    return 0


def rule_based(args, table, slot_teams, nearest, dead):
    """Hari's pipeline, with B6 anchors when --b6 is given."""
    anchors = {}
    if args.b6:
        from possession.detectors import early_confirmation

        anchors = early_confirmation.from_table(table, slot_teams, dead)
        print(f"  confirmed {len(anchors)} B6 control spans")
    labels, moved = sequence.prepare(
        nearest, dead, anchors=anchors, hold_frames=args.hold, place_gaps=not args.no_gap_policy,
    )
    if not args.no_gap_policy:
        print(f"  relocated {moved} hand-overs to the midpoint of a ball gap")
    return labels


def hybrid(table, slot_teams, dead, use_v2=True):
    """The hybrid Evidence decoder (possession.fusion.hybrid_labels).

    Settings are fixed constants fitted on IDSSE: team-shape weights in
    TeamShapeParams, decoder weights and switch cost in fusion.HYBRID_PARAMS,
    the dead-ball v2 model in detectors/dead_ball_v2_model.json.
    """
    from possession import fusion
    from possession.detectors import early_confirmation
    from possession.detectors.ball_to_team import ball_to_team
    from possession.detectors.team_shape import TeamShapeParams, team_shape
    import numpy as np

    from possession.labels import GRADED_WINDOWS, SECOND_HALF_KICKOFF

    arrays = early_confirmation.arrays_from_table(table, slot_teams)
    print("ball-to-team evidence ...")
    ball = ball_to_team(arrays).evidence
    print("team-shape evidence ...")
    halves = tuple((lo, hi + 1) for lo, hi in GRADED_WINDOWS)
    shape = team_shape(arrays, TeamShapeParams(halves=halves)).evidence
    if use_v2:
        from possession.detectors import dead_ball_v2

        print("dead-ball v2 ...")
        dead = dead_ball_v2.detect(arrays) >= dead_ball_v2.THRESHOLD
        graded = np.zeros(len(dead), dtype=bool)
        for lo, hi in GRADED_WINDOWS:
            graded[lo:hi + 1] = True
        # Graded frames only: pre-match, half-time and post-match are not scored and
        # are almost all "dead", so a whole-video share overstates it.
        print(f"  v2 flagged dead on {dead[graded].mean():.1%} of graded frames")
    # The stretch before the second-half kickoff is dead by rule: fix it as D too,
    # so no control span is anchored inside it.
    fixed_dead = dead.copy()
    fixed_dead[GRADED_WINDOWS[1][0]:SECOND_HALF_KICKOFF] = True
    runs = early_confirmation.control_runs(arrays, fixed_dead)
    print(f"  confirmed {len(runs)} B6 control spans")
    print("hybrid decoding (Viterbi per half) ...")
    return fusion.hybrid_labels({"ball": ball, "shape": shape}, fixed_dead, runs,
                                windows=GRADED_WINDOWS)


if __name__ == "__main__":
    raise SystemExit(main())
