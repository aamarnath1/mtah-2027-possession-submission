"""Evidence decoder: weigh every detector's Evidence, then pick one label path.

This replaces the letter-by-letter steps of the current pipeline (nearest team,
30-frame hold, B6 anchors, gap placement, dead mask, short-run cleanup) with two
stages:

  1. Combine. Every detector hands over an Evidence (p_home, p_away, p_dead,
     confidence). Per frame we add up, in log-odds:
         team score  L  = sum over detectors of  weight * confidence * logit(p_home)
         dead score  Ld = dead_bias + sum of     weight * confidence * logit(p_dead)
     A detector with no opinion (NaN, or confidence 0) adds nothing, so a frame
     nobody has an opinion on scores 0: "no idea", not "probably dead".

  2. Decode. Viterbi picks the H / A / D sequence with the best total of
         per-frame scores  minus  a cost for every label change.
     The switch cost is what makes possession sticky: one stray frame cannot pay
     for a switch, a sustained run of evidence can. It also sets where a switch
     lands inside a ball gap: wherever the weaker detectors (team shape) tip the
     balance, instead of at the gap's midpoint.

Per-frame scores, given L and Ld:
    H: log sigmoid(-Ld) + log(2 sigmoid(L))
    A: log sigmoid(-Ld) + log(2 sigmoid(-L))
    D: log sigmoid(Ld)
The team term is shifted so it is 0 when L = 0; otherwise frames with no team
evidence would lean towards D just because D carries no team uncertainty.

Every weight and cost is a placeholder until fitted on other matches (IDSSE),
never on ours.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math

import numpy as np

from possession.evidence import Evidence

LABELS = ("H", "A", "D")


@dataclass(frozen=True)
class FusionParams:
    team_weights: Mapping[str, float] = field(default_factory=dict)
    """Scale per detector name for the team score. Missing names get 1.0."""

    dead_weights: Mapping[str, float] = field(default_factory=dict)
    """Scale per detector name for the dead score. Missing names get 1.0."""

    dead_bias: float = 0.0
    """Added to the dead score everywhere. Negative leans towards live play."""

    switch_cost: float = 8.0
    """Cost of an H <-> A change."""

    enter_dead_cost: float = 8.0
    """Cost of H or A -> D."""

    exit_dead_cost: float = 8.0
    """Cost of D -> H or A (a restart)."""

    max_logit: float = 6.0
    """Clip every logit to +/- this, so a detector saying 0 or 1 cannot veto all others."""


def clipped_logit(p: np.ndarray, limit: float) -> np.ndarray:
    eps = 1.0 / (1.0 + math.exp(limit))
    q = np.clip(p, eps, 1.0 - eps)
    return np.log(q) - np.log1p(-q)


def combine(evidence: Mapping[str, Evidence], params: FusionParams) -> tuple[np.ndarray, np.ndarray]:
    """Team score L (>0 favours home) and dead score Ld (>0 favours dead), per frame."""
    n = len(next(iter(evidence.values())).confidence)
    team = np.zeros(n)
    dead = np.full(n, float(params.dead_bias))
    for name, ev in evidence.items():
        conf = np.asarray(ev.confidence, dtype=float)
        p_home = np.asarray(ev.p_home, dtype=float)
        p_dead = np.asarray(ev.p_dead, dtype=float)
        w = params.team_weights.get(name, 1.0)
        if w and not np.isnan(p_home).all():
            vote = w * conf * clipped_logit(p_home, params.max_logit)
            team += np.where(np.isnan(p_home), 0.0, vote)
        w = params.dead_weights.get(name, 1.0)
        if w and not np.isnan(p_dead).all():
            vote = w * conf * clipped_logit(p_dead, params.max_logit)
            dead += np.where(np.isnan(p_dead), 0.0, vote)
    return team, dead


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def frame_scores(team: np.ndarray, dead: np.ndarray) -> np.ndarray:
    """(n, 3) scores for H, A, D."""
    alive = log_sigmoid(-dead)
    return np.column_stack([
        alive + math.log(2.0) + log_sigmoid(team),
        alive + math.log(2.0) + log_sigmoid(-team),
        log_sigmoid(dead),
    ])


def viterbi(scores: np.ndarray, params: FusionParams,
            forced: np.ndarray | None = None) -> list[str]:
    """Best H/A/D path.

    `forced`: optional per-frame constraint. 'H', 'A' or 'D' fixes the label,
    'L' means live (H or A, not D), '' leaves the frame free.
    """
    n = len(scores)
    if n == 0:
        return []
    s = scores.copy()
    if forced is not None:
        for k, label in enumerate(LABELS):
            allowed = (forced == "") | (forced == label)
            if label != "D":
                allowed |= forced == "L"
            s[~allowed, k] = -np.inf
    sw, ed, xd = params.switch_cost, params.enter_dead_cost, params.exit_dead_cost
    rows = s.tolist()
    back = bytearray(3 * n)
    h, a, d = rows[0]
    for t in range(1, n):
        sh, sa, sd = rows[t]
        # into H
        bh, ih = h, 0
        if a - sw > bh:
            bh, ih = a - sw, 1
        if d - xd > bh:
            bh, ih = d - xd, 2
        # into A
        ba, ia = a, 1
        if h - sw > ba:
            ba, ia = h - sw, 0
        if d - xd > ba:
            ba, ia = d - xd, 2
        # into D
        bd, i_d = d, 2
        if h - ed > bd:
            bd, i_d = h - ed, 0
        if a - ed > bd:
            bd, i_d = a - ed, 1
        j = 3 * t
        back[j], back[j + 1], back[j + 2] = ih, ia, i_d
        h, a, d = bh + sh, ba + sa, bd + sd
    state = max(range(3), key=(h, a, d).__getitem__)
    path = [0] * n
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[3 * t + state]
    return [LABELS[k] for k in path]


def decode(evidence: Mapping[str, Evidence], params: FusionParams,
           forced: np.ndarray | None = None) -> list[str]:
    """Combine, score and decode in one call."""
    team, dead = combine(evidence, params)
    return viterbi(frame_scores(team, dead), params, forced)


def dead_ball_evidence(score: np.ndarray, confidence: float = 1.0) -> Evidence:
    """Wrap the dead-ball detector's 0..1 score as Evidence (no team opinion)."""
    score = np.asarray(score, dtype=float)
    n = len(score)
    return Evidence(np.full(n, np.nan), np.full(n, np.nan), score, np.full(n, float(confidence)))


def hard_constraints(n: int, dead: np.ndarray | None = None,
                     anchor_runs=()) -> np.ndarray:
    """Per-frame constraints for viterbi(forced=...), the way B6 uses its inputs.

    dead         boolean mask (dead-ball evidence >= 0.5): those frames are fixed
                 to D and every other frame is live ('L'), exactly Hari's mask.
    anchor_runs  (start, end, team) spans, end exclusive, of sustained observed
                 control (B6). The whole span is fixed to that team.
    """
    forced = np.full(n, "", dtype="<U1")
    if dead is not None:
        forced[:] = "L"
        forced[np.asarray(dead, dtype=bool)] = "D"
    for start, end, team in anchor_runs:
        forced[start:end] = team
    return forced


HYBRID_PARAMS = FusionParams(team_weights={"ball": 1.0, "shape": 0.25}, switch_cost=8.0)
"""The hybrid as benchmarked: ball-to-team at full weight, team shape at a quarter,
switch cost 8. Chosen by leave-one-match-out tuning on four IDSSE matches with the
ball hidden in our match's pattern (scripts/benchmark_evidence_idsse.py): each fold
picked shape 0.25-0.5 and switch cost 4-16. Dead-ball costs do not apply, because
the hybrid fixes Hari's dead mask instead of voting on it."""


def hybrid_labels(evidence: Mapping[str, Evidence], dead: np.ndarray, control_runs=(),
                  params: FusionParams = HYBRID_PARAMS, windows=None) -> list[str]:
    """The hybrid decoder: Hari's dead mask and B6 control spans fixed, the rest voted.

    evidence      team Evidence by name, e.g. {"ball": ..., "shape": ...}
    dead          boolean dead-ball mask (fixed to D; every other frame is H or A)
    control_runs  (start, end exclusive, team) spans from early_confirmation.control_runs
    windows       (first, last) inclusive frame ranges decoded independently (one per
                  half); frames outside every window are 'X'. Default: the whole array.

    Returns raw labels: apply the usual short-run cleanup (possession.decode) after.
    """
    team, dead_score = combine(evidence, params)
    n = len(team)
    scores = frame_scores(team, dead_score)
    forced = hard_constraints(n, dead, control_runs)
    out = ["X"] * n
    for lo, hi in (windows or [(0, n - 1)]):
        out[lo:hi + 1] = viterbi(scores[lo:hi + 1], params, forced[lo:hi + 1])
    return out
