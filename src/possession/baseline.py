"""A first signal: which team is nearest the ball, made sticky by the rules.

This is not a good detector. It exists so the pipeline is end to end and the team
has a number to improve on, and so the decoder has something real to chew on.

One thing it does that the official `baseline_raw` does not, and the difference
is the whole point. The rules say that during loose or contested phases
possession stays with the team that last had it, and changes only when the other
team gains CONTROLLED possession. So "nearest player" is not an answer on its
own. Here it is treated as evidence that only overturns the standing possession
when it is sustained and unambiguous. When the ball is untracked, the previous
label carries forward.

Proper control detection, touch detection and dead-ball detection belong in their
own modules and are somebody else's lane.
"""

import json

from possession.labels import ABSTAIN, AWAY, HOME, N_FRAMES
from possession.paths import CURRENT_DIR, current_file

NEEDED_SLOT_FIELDS = ("x_m", "y_m", "observed")


def load_slot_teams(root=None):
    """slot number -> 'H' or 'A', from the preprocessing lookup."""
    path = (root or CURRENT_DIR) / "track_slots.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw if isinstance(raw, list) else raw.get("slots", raw)
    if isinstance(entries, dict):
        entries = list(entries.values())
    teams = {}
    for entry in entries:
        slot, team = entry.get("slot"), entry.get("team")
        if slot is not None and team in (HOME, AWAY):
            teams[int(slot)] = team
    if not teams:
        raise ValueError(f"no slot->team entries found in {path}")
    return teams


def _columns(slot_teams):
    cols = ["frame", "ball_x_m", "ball_y_m", "ball_observed"]
    for slot in sorted(slot_teams):
        cols += [f"slot_{slot}_{suffix}" for suffix in NEEDED_SLOT_FIELDS]
    return cols


def load_table(frames_csv=None, slot_teams=None):
    """Read just the columns the detectors need, in float32.

    The full table is 287 columns and about 340 MB. This slice is roughly a
    third of that, and both the team and dead-ball detectors read it, so it is
    loaded once and passed around rather than read twice.
    """
    import pandas as pd

    slot_teams = slot_teams or load_slot_teams()
    path = frames_csv or current_file("aligned/frames_with_tags.csv")
    usecols = _columns(slot_teams)
    dtypes = {c: "float32" for c in usecols if c != "frame"}
    dtypes["frame"] = "int32"
    table = pd.read_csv(path, usecols=usecols, dtype=dtypes)
    if len(table) != N_FRAMES:
        raise ValueError(f"{path} has {len(table)} rows, expected {N_FRAMES}")
    return table


def nearest_team_per_frame(frames_csv=None, slot_teams=None, table=None):
    """For each frame, the team of the closest observed player to the ball.

    Returns a list of 'H', 'A' or None, where None means "no opinion": either the
    ball was not tracked, or no player was.

    Reads only the columns it needs, in float32. The full table is 287 columns and
    about 340 MB; this slice is roughly a third of that.
    """
    import numpy as np

    slot_teams = slot_teams or load_slot_teams()
    if table is None:
        table = load_table(frames_csv, slot_teams)

    slots = sorted(slot_teams)
    bx = table["ball_x_m"].to_numpy()
    by = table["ball_y_m"].to_numpy()
    ball_seen = table["ball_observed"].to_numpy() > 0

    best_d = np.full(len(table), np.inf, dtype="float32")
    best_team = np.zeros(len(table), dtype="int8")  # 0 none, 1 home, 2 away

    for slot in slots:
        seen = table[f"slot_{slot}_observed"].to_numpy() > 0
        dx = table[f"slot_{slot}_x_m"].to_numpy() - bx
        dy = table[f"slot_{slot}_y_m"].to_numpy() - by
        d = np.sqrt(dx * dx + dy * dy)
        d[~seen] = np.inf
        closer = d < best_d
        best_d = np.where(closer, d, best_d)
        best_team = np.where(closer, 1 if slot_teams[slot] == HOME else 2, best_team)

    best_team[~ball_seen] = 0
    best_team[~np.isfinite(best_d)] = 0

    lookup = {0: None, 1: HOME, 2: AWAY}
    return [lookup[int(v)] for v in best_team], best_d


def sticky_labels(nearest, hold_frames=12):
    """Carry possession forward, switching only on a sustained disagreement.

    `hold_frames` is how long the other team must be nearest before we accept that
    possession changed. It is a stand-in for the control test, not a substitute
    for one: a real control layer looks at whether the ball's motion broke and
    whether anyone kept it, which proximity cannot see.

    The switch is BACKDATED to where the disagreement started. The ground truth is
    an unsmoothed read of a per-frame owning-team field, so it has no built-in
    lag, and a system that waits for confirmation before switching is late by
    exactly the amount it waited.
    """
    labels = [ABSTAIN] * len(nearest)
    current = None
    streak_team, streak_start = None, None

    for i, team in enumerate(nearest):
        if team is not None:
            if current is None:
                # Opening possession. Require the same hold as any other switch,
                # otherwise a single noisy first frame becomes a 1-frame
                # possession that later stages faithfully preserve.
                if team != streak_team:
                    streak_team, streak_start = team, i
                elif i - streak_start + 1 >= hold_frames:
                    current = team
                    for j in range(streak_start, i + 1):
                        labels[j] = current
                    streak_team, streak_start = None, None
            elif team != current:
                if team != streak_team:
                    streak_team, streak_start = team, i
                elif i - streak_start + 1 >= hold_frames:
                    current = team
                    for j in range(streak_start, i + 1):
                        labels[j] = current
                    streak_team, streak_start = None, None
            else:
                streak_team, streak_start = None, None
        else:
            # No ball, so no opinion. A streak must be CONSECUTIVE OBSERVED
            # frames: without this reset the hold test measures an index span,
            # and one stray frame either side of a 15-second gap counts as a
            # sustained lead.
            streak_team, streak_start = None, None
        if labels[i] == ABSTAIN and current is not None:
            labels[i] = current
    return labels


def place_switches_in_gaps(labels, nearest):
    """Move a hand-over that happened inside a ball gap to the gap's midpoint.

    Where the ball is untracked we have no evidence, so `sticky_labels` holds the
    standing possession and only switches once evidence returns. That puts the
    boundary at the END of the gap. If the turnover actually happened inside the
    gap, the reported boundary is late.

    Under a uniform prior on when the switch really happened, the midpoint
    minimizes expected absolute boundary error, which is the loss temporal IoU
    charges. So when the labels either side of a gap disagree, split the gap.

    This only ever moves a boundary that already exists. It never creates one,
    which matters because a false switch costs about twice a missed one.
    """
    out = list(labels)
    n = len(out)
    i = 0
    moved = 0
    while i < n:
        if nearest[i] is not None:
            i += 1
            continue
        start = i
        while i < n and nearest[i] is None:
            i += 1
        end = i - 1
        if start == 0 or end == n - 1:
            continue                      # no evidence on one side: leave it alone
        before, after = out[start - 1], out[end + 1]
        if before == after or before == ABSTAIN or after == ABSTAIN:
            continue
        mid = start + (end - start + 1) // 2
        for j in range(start, mid):
            out[j] = before
        for j in range(mid, end + 1):
            out[j] = after
        moved += 1
    return out, moved


def build(frames_csv=None, hold_frames=12):
    """nearest-team evidence -> a sticky label array. No dead-ball handling yet."""
    nearest, _ = nearest_team_per_frame(frames_csv)
    return sticky_labels(nearest, hold_frames=hold_frames)
