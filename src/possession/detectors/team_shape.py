"""Detector B2, Team shape: who has the ball, read from the players alone.

Works with or without the ball, so its job is the ~41% of graded frames where the
ball is untracked. It never needs a ball position.

Plain-language recipe, for every frame:
  1. Work out which goal each team attacks in each half, by code: a goalkeeper
     sits near its own goal line for the whole half (step A5 in the team guide).
  2. Put each team in its own "attacking frame": forward = toward the goal it
     attacks. Drop each team's deepest player (almost always the keeper) so one
     player does not drag the team's position back.
  3. Measure three things per team, averaged over a centred window:
        forward position   where the team's outfield block stands
        forward velocity   is the block moving toward the opponent's goal
        spread             how far players sit from their own centre
  4. Combine them into one score. The team with the ball tends to be further
     up the pitch, moving forward, and stretched out; the team without it is
     deeper, retreating and compact.
  5. Turn the score into p_home / p_away with a logistic curve. Confidence
     depends only on how many players are tracked, never on the ball.

The weights in TeamShapeParams are a logistic regression fitted on the live
frames of four public IDSSE matches (J03WOH, J03WOY, J03WQQ, J03WR9), never on
ours or the public five minutes (scripts/eval_team_shape_idsse.py). Held out
one match at a time, they name the right team on 80-85% of live frames and on
96-99% of the quarter of frames they are surest about. Spread and forward
movement carry the signal; once those are known, position of play counts
slightly against the team further up. The features are returned alongside the
evidence so a richer model can replace step 4 without touching steps 1 to 3.

Known limits, from the team guide: the signal is blurry at the moment of a
turnover, so it should steady the answer inside a ball gap rather than place the
exact switch. A deep, counter-attacking team breaks the "further up the pitch"
cue, which is why no single feature decides on its own.

It knows nothing about dead balls (p_dead stays NaN): that is detector B4's job.
"""

from dataclasses import dataclass, field
import warnings

import numpy as np

from possession.evidence import Evidence

SECOND_HALF_START = 110_808
"""First graded frame of the second half (possession.labels.GRADED_WINDOWS)."""


@dataclass(frozen=True)
class TeamShapeParams:
    """Starting guesses. Fit them on other games; do not tune on the public 5 minutes."""

    pitch_x_m: float = 105.3
    """Pitch length in metres. Our preprocessing puts x in 0..105.3 from one goal line."""

    window_s: float = 2.0
    """Half-width, in seconds, of the centred average applied to every feature."""

    velocity_span_s: float = 1.0
    """Velocity = change in forward position from this far before to this far after."""

    drop_deepest: bool = True
    """Drop each team's deepest player per frame (the keeper) before measuring."""

    min_players: int = 5
    """Each team needs at least this many tracked outfield players for an opinion."""

    full_strength: int = 10
    """Tracked outfield players per team at which confidence stops rising."""

    max_confidence: float = 1.0
    """Ceiling on confidence. Confidence is already |2 p_home - 1| times player
    coverage, and how much to trust team shape against the ball is the decoder's
    weight (possession.fusion), so by default there is no extra cap."""

    position_scale_m: float = 15.0
    velocity_scale_mps: float = 1.5
    spread_scale_m: float = 4.0
    """How big a difference in each feature counts as one unit of evidence."""

    weights: tuple = (-0.735, 0.6747, 2.3296)
    """Weight of (position, velocity, spread) in the score, each feature first divided
    by its scale above. Fitted on IDSSE (raw-unit coefficients -0.0490 per m,
    +0.4498 per m/s, +0.5824 per m, window_s 2.0)."""

    bias: float = -0.0039
    """Constant added to the score (fitted with the weights; close to 0)."""

    halves: tuple = ()
    """(start, end) frame ranges, end exclusive, each with one attacking direction.
    Empty means: split at SECOND_HALF_START if the match is long enough, else one half."""

    attack_dir: dict = field(default_factory=dict)
    """Optional override, {half_index: +1 or -1}: +1 means home attacks toward high x.
    Empty means work it out from the keepers' positions."""


@dataclass
class TeamShapeResult:
    """The evidence, plus the features, so a learned model can be trained on them."""

    evidence: Evidence
    position: np.ndarray     # (n,) metres; >0 means play sits in the away team's half
    velocity: np.ndarray     # (n,) m/s; >0 means the block is moving toward the away goal
    spread: np.ndarray       # (n,) metres; home spread minus away spread
    n_home: np.ndarray       # (n,) tracked home outfield players
    n_away: np.ndarray       # (n,) tracked away outfield players
    attack_dir: dict         # {half_index: +1 / -1}, as used


def half_ranges(n_frames: int, params: TeamShapeParams) -> list:
    if params.halves:
        return list(params.halves)
    if n_frames > SECOND_HALF_START:
        return [(0, SECOND_HALF_START), (SECOND_HALF_START, n_frames)]
    return [(0, n_frames)]


def attack_direction(x: np.ndarray, home: np.ndarray, length: float) -> int:
    """+1 if home attacks toward high x in this stretch, -1 if toward low x.

    For each team, compare how close its most extreme player sits to each goal
    line, as a median over the stretch. The keeper hugs its own line all half,
    so the side a team is consistently closest to is the side it defends.
    """
    def defends_low(team_x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            near_low = np.nanmedian(np.nanmin(team_x, axis=1))
            near_high = length - np.nanmedian(np.nanmax(team_x, axis=1))
        return near_high - near_low          # >0: hugs the low-x line more

    score = defends_low(x[:, home]) - defends_low(x[:, ~home])
    if not np.isfinite(score) or score == 0:
        raise ValueError("could not work out attacking direction; pass attack_dir")
    return 1 if score > 0 else -1


def drop_deepest_player(forward: np.ndarray) -> np.ndarray:
    """Set each row's smallest forward value to NaN, where at least two are tracked."""
    out = forward.copy()
    tracked = np.isfinite(out)
    rows = np.flatnonzero(tracked.sum(axis=1) >= 2)
    deepest = np.argmin(np.where(tracked, out, np.inf), axis=1)
    out[rows, deepest[rows]] = np.nan
    return out


def team_measures(forward: np.ndarray, y: np.ndarray):
    """Per frame: median forward position, spread, and tracked count."""
    tracked = np.isfinite(forward) & np.isfinite(y)
    f = np.where(tracked, forward, np.nan)
    yy = np.where(tracked, y, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fwd = np.nanmedian(f, axis=1)
        cy = np.nanmedian(yy, axis=1)
        spread = np.nanmean(np.hypot(f - fwd[:, None], yy - cy[:, None]), axis=1)
    return fwd, spread, tracked.sum(axis=1)


def smooth_centered(values: np.ndarray, half_width: int) -> np.ndarray:
    """NaN-aware centred moving average via cumulative sums. Never trailing."""
    if half_width <= 0:
        return values.astype(float)
    n = len(values)
    valid = np.isfinite(values)
    csum = np.concatenate([[0.0], np.cumsum(np.where(valid, values, 0.0))])
    cval = np.concatenate([[0.0], np.cumsum(valid)])
    lo = np.clip(np.arange(n) - half_width, 0, n)
    hi = np.clip(np.arange(n) + half_width + 1, 0, n)
    count = cval[hi] - cval[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (csum[hi] - csum[lo]) / np.maximum(count, 1), np.nan)


def centered_velocity(position: np.ndarray, span: int, fps: float) -> np.ndarray:
    """(position[t + span] - position[t - span]) / time, NaN at the edges."""
    out = np.full(len(position), np.nan)
    if span <= 0 or len(position) <= 2 * span:
        return out
    out[span:-span] = (position[2 * span:] - position[:-2 * span]) / (2 * span / fps)
    return out


def team_shape(arrays: dict, params: TeamShapeParams = TeamShapeParams(),
               fps: float = 25.0) -> TeamShapeResult:
    """Run the detector on the arrays from possession.loading.load_arrays()."""
    slot_xy = np.asarray(arrays["slot_xy"], dtype=float)
    home = np.asarray(arrays["slot_team"]) == "H"
    n_frames = slot_xy.shape[0]
    length = params.pitch_x_m
    x, y = slot_xy[..., 0], slot_xy[..., 1]

    # Step 1: attacking direction per half, then a per-frame direction array.
    direction = np.zeros(n_frames)
    used = {}
    for index, (lo, hi) in enumerate(half_ranges(n_frames, params)):
        d = params.attack_dir.get(index) or attack_direction(x[lo:hi], home, length)
        used[index] = d
        direction[lo:hi] = d

    # Step 2: each team in its own attacking frame, keeper dropped.
    home_fwd_x = np.where(direction[:, None] > 0, x[:, home], length - x[:, home])
    away_fwd_x = np.where(direction[:, None] > 0, length - x[:, ~home], x[:, ~home])
    if params.drop_deepest:
        home_fwd_x = drop_deepest_player(home_fwd_x)
        away_fwd_x = drop_deepest_player(away_fwd_x)

    # Step 3: measures per team, smoothed over a centred window.
    half = int(round(params.window_s * fps))
    h_fwd, h_spread, n_home = team_measures(home_fwd_x, y[:, home])
    a_fwd, a_spread, n_away = team_measures(away_fwd_x, y[:, ~home])
    enough = (n_home >= params.min_players) & (n_away >= params.min_players)
    for arr in (h_fwd, h_spread, a_fwd, a_spread):
        arr[~enough] = np.nan

    # Position of play in home's attacking direction: >0 means in the away half.
    # Away's forward position, seen from home's end, is length - a_fwd.
    position = smooth_centered((h_fwd + (length - a_fwd)) / 2.0 - length / 2.0, half)
    span = int(round(params.velocity_span_s * fps))
    velocity = smooth_centered(centered_velocity(h_fwd, span, fps)
                               - centered_velocity(a_fwd, span, fps), half) / 2.0
    spread = smooth_centered(h_spread - a_spread, half)

    # Steps 4 and 5: score, probability, confidence.
    w_pos, w_vel, w_spr = params.weights
    score = (params.bias + w_pos * position / params.position_scale_m
             + w_vel * velocity / params.velocity_scale_mps
             + w_spr * spread / params.spread_scale_m)
    has_opinion = enough & np.isfinite(score)
    p_home = np.full(n_frames, np.nan)
    p_home[has_opinion] = 1.0 / (1.0 + np.exp(-score[has_opinion]))
    p_away = 1.0 - p_home

    strength = np.clip(np.minimum(n_home, n_away) / float(params.full_strength), 0.0, 1.0)
    sureness = np.abs(2.0 * np.nan_to_num(p_home, nan=0.5) - 1.0)
    confidence = np.where(has_opinion,
                          np.minimum(params.max_confidence, sureness * strength), 0.0)

    evidence = Evidence(p_home, p_away, np.full(n_frames, np.nan), confidence)
    evidence.validate()
    return TeamShapeResult(evidence, position, velocity, spread, n_home, n_away, used)
