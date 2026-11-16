"""Hide the ball on a public match in the same pattern it goes missing on ours.

The pattern comes from data/external/gap_profile.json, written locally by
scripts/export_gap_profile.py. It is read here and applied to IDSSE ground truth;
nothing from it is printed.

Matched to our match:
  * share of live frames with the ball hidden, and share of dead frames
  * gap lengths: drawn from our own live gaps and our own dead gaps
  * how often a live gap hides a change of team: that share of live gaps is
    placed over a real IDSSE turnover, the rest over stretches with none

Differences to keep in mind: on our match "live" and "changed team" are
estimates (dead-ball detector, sticky nearest team), while here they come from
IDSSE ground truth. And our proxy cannot see H -> A -> H inside one gap, so if
anything this under-places turnovers in gaps, which makes the test a little
easier than reality.
"""

import json
from pathlib import Path
import random

import numpy as np

from possession.paths import DATA_DIR

PROFILE = DATA_DIR / "external" / "gap_profile.json"


def load_profile(path: Path = PROFILE) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} missing: run python scripts/export_gap_profile.py")
    return json.loads(Path(path).read_text())


def hide_like_profile(labels, profile: dict, seed: int, max_tries: int = 200_000) -> np.ndarray:
    """Per-frame visibility (True = ball visible) for one IDSSE half."""
    rng = random.Random(seed)
    lab = np.asarray(labels)
    n = len(lab)
    live = np.isin(lab, ["H", "A"])
    turn = np.zeros(n, dtype=bool)
    turn[1:] = live[1:] & live[:-1] & (lab[1:] != lab[:-1])
    turn_cum = np.concatenate([[0], np.cumsum(turn)])
    live_cum = np.concatenate([[0], np.cumsum(live)])
    turnovers = np.flatnonzero(turn).tolist()
    live_starts = np.flatnonzero(live).tolist()
    dead_starts = np.flatnonzero(~live).tolist()

    gaps = profile["gaps"]
    live_lens = [g["len"] for g in gaps if g["live"]]
    dead_lens = [g["len"] for g in gaps if not g["live"]]
    known = [g["switch"] for g in gaps if g["live"] and g["switch"] is not None]
    p_switch = float(np.mean(known)) if known else 0.0

    hidden = np.zeros(n, dtype=bool)

    def free(s, e):
        lo, hi = max(0, s - 1), min(n, e + 1)          # keep a visible frame between gaps
        return not hidden[lo:hi].any()

    def turnovers_inside(s, e):
        """Turnovers strictly inside [s, e), i.e. the team differs either side of the gap."""
        return turn_cum[e] - turn_cum[s + 1] if e > s + 1 else 0

    def place(target, lengths, want_live):
        done, tries = 0, 0
        while done < target and lengths and tries < max_tries:
            tries += 1
            length = rng.choice(lengths)
            if length >= n:
                continue
            over_turnover = want_live and bool(turnovers) and rng.random() < p_switch
            if over_turnover:
                s = rng.choice(turnovers) - rng.randint(1, max(1, length - 1))
            else:
                pool = live_starts if want_live else dead_starts
                if not pool:
                    return
                s = rng.choice(pool)
            s = max(0, s)
            e = min(n, s + length)
            n_live = int(live_cum[e] - live_cum[s])
            if want_live != (n_live * 2 >= e - s):     # must be mostly the wanted kind
                continue
            if want_live and not over_turnover and turnovers_inside(s, e):
                continue                               # a "no change" gap must hide none
            if not free(s, e):
                continue
            hidden[s:e] = True
            done += n_live if want_live else (e - s) - n_live

    # Dead gaps first: they spill a little into live play, and that spill counts
    # towards the live target so the live share is not overshot.
    place(int(round(profile["dead_missing_share"] * (~live).sum())), dead_lens, False)
    live_target = int(round(profile["live_missing_share"] * live.sum()))
    place(live_target - int((hidden & live).sum()), live_lens, True)
    return ~hidden
