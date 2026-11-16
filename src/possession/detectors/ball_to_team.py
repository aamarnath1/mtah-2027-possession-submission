"""Detector B1, Ball to team: when the ball is seen, which team is it with?

Plain-language recipe, for every frame:
  1. Measure the distance (metres) from the ball to the closest tracked player of each team.
  2. The team whose player is closer is the likely owner. The bigger the gap, the surer we are.
  3. Decide whether a player really CONTROLS the ball: the ball is close to a player, moving
     slowly (speed worked out from the ball's own movement, smoothed, tiny flickers bridged), and stays that way for a few frames. A fast ball that flies past a defender is a
     pass, not a change of possession, so in that case we lower our confidence and let the
     decoder keep the previous team.
  4. If the ball was not seen, or one team has no tracked player, we give no opinion.

It reads the arrays from possession.loading.load_arrays() and returns an Evidence object.
It knows nothing about dead balls (p_dead stays NaN): that is detector B4's job.
"""

from dataclasses import dataclass
import warnings

import numpy as np

from possession.evidence import Evidence


@dataclass(frozen=True)
class BallToTeamParams:
    """Starting guesses. Tune them on other games later; do not tune on the public 5 minutes."""

    margin_scale_m: float = 1.5
    """Metres. A gap of this size between the two teams' nearest players gives about 73% for
    the closer team. Smaller values make the detector more decisive."""

    control_radius_m: float = 2.0
    """Metres. A player counts as being 'on the ball' within this distance. Tracking error is
    about 1 to 2 m, so going much below this misses real control."""

    max_control_speed_mps: float = 9.0
    """Metres per second. A ball moving faster than this is treated as a pass or shot."""

    min_control_frames: int = 5
    """The close-and-slow state must last this many frames (25 frames = 1 second)."""

    uncontrolled_weight: float = 0.25
    """Confidence is multiplied by this when nobody controls the ball (a pass in the air)."""

    speed_source: str = "positions"
    """'positions': work out the ball's speed from how far it moved (recommended). 'tracking':
    use the speed column from the tracking file. On this match that column is empty on most
    frames where the ball is seen, so do not use it unless that changes."""

    speed_span_frames: int = 4
    """For 'positions': speed = distance moved between the frames this far apart on either
    side of the current frame (an even number, 4 = about a fifth of a second)."""

    speed_smooth_frames: int = 5
    """Ball speed is noisy frame to frame. We use the median over this many frames (odd number)
    so one spike does not break a stretch of control. Use 1 to switch smoothing off."""

    max_gap_frames: int = 3
    """A break in 'close and slow' of at most this many frames is bridged, so a one-frame
    flicker does not cut a control stretch in two. Use 0 to switch bridging off."""


@dataclass
class BallToTeamResult:
    """The evidence, plus extra signals that other detectors (restart, touch) can reuse."""

    evidence: Evidence
    control: np.ndarray      # (n,) True when a player is judged to control the ball
    dist_home: np.ndarray    # (n,) metres to nearest home player, NaN if none tracked
    dist_away: np.ndarray    # (n,) metres to nearest away player, NaN if none tracked
    ball_speed: np.ndarray   # (n,) m/s actually used, NaN where it could not be worked out


def nearest_distance_per_team(ball_xy: np.ndarray, slot_xy: np.ndarray,
                              team_mask: np.ndarray) -> np.ndarray:
    """Metres from the ball to the closest tracked player of one team, per frame.

    ball_xy: (n, 2) ball position in metres. slot_xy: (n, slots, 2). team_mask: (slots,) bool.
    Returns (n,). NaN where the ball is unseen or the team has no tracked player.
    """
    if not team_mask.any():
        raise ValueError("No slots belong to this team")
    offsets = slot_xy[:, team_mask, :] - ball_xy[:, None, :]       # (n, k, 2)
    distances = np.sqrt((offsets ** 2).sum(axis=2))                # NaN if any coordinate is NaN
    distances = np.where(np.isnan(distances), np.inf, distances)   # ignore empty slots
    nearest = distances.min(axis=1)
    return np.where(np.isinf(nearest), np.nan, nearest)


def speed_from_positions(xy: np.ndarray, span: int, fps: float = 25.0) -> np.ndarray:
    """Speed in m/s from positions in metres, over `span` frames centred on each frame.

    NaN wherever either end of the span is unseen, and for the first and last span/2 frames.
    """
    if span < 2 or span % 2:
        raise ValueError("span must be an even number, at least 2")
    half, n_frames = span // 2, len(xy)
    speed = np.full(n_frames, np.nan)
    if n_frames > span:
        moved = xy[span:] - xy[:n_frames - span]
        speed[half:n_frames - half] = np.sqrt((moved ** 2).sum(axis=1)) * fps / span
    return speed


def smooth_speed(speed: np.ndarray, window: int) -> np.ndarray:
    """Median of the speed over a centred window of `window` frames, ignoring NaN."""
    if window <= 1:
        return np.asarray(speed, dtype=float).copy()
    if window % 2 == 0:
        raise ValueError("window must be odd")
    padded = np.pad(np.asarray(speed, dtype=float), window // 2, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN windows are fine
        return np.nanmedian(windows, axis=1)


def bridge_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill breaks of at most max_gap frames that sit between two stretches of True."""
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    if max_gap <= 0:
        return out
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    runs = list(zip(edges[0::2], edges[1::2]))                    # (start, stop) of each True run
    for (_, stop), (next_start, _) in zip(runs, runs[1:]):
        if next_start - stop <= max_gap:
            out[stop:next_start] = True
    return out


def keep_long_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only stretches of True that last at least min_len frames in a row."""
    mask = np.asarray(mask, dtype=bool)
    if min_len <= 1:
        return mask.copy()
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    kept = np.zeros_like(mask)
    for start, stop in zip(edges[0::2], edges[1::2]):
        if stop - start >= min_len:
            kept[start:stop] = True
    return kept


def ball_to_team(arrays: dict, params: BallToTeamParams = BallToTeamParams()) -> BallToTeamResult:
    """Run the detector on the arrays from load_arrays()."""
    ball_xy = arrays["ball_xy"]
    ball_seen, slot_xy = arrays["ball_obs"], arrays["slot_xy"]
    slot_team = np.asarray(arrays["slot_team"])
    n_frames = len(ball_xy)

    dist_home = nearest_distance_per_team(ball_xy, slot_xy, slot_team == "H")
    dist_away = nearest_distance_per_team(ball_xy, slot_xy, slot_team == "A")

    # Step 2: who is closer, and how sure are we?
    margin = dist_away - dist_home                  # positive: home player is closer
    has_opinion = ball_seen & ~np.isnan(margin)
    p_home = np.full(n_frames, np.nan)
    p_home[has_opinion] = 1.0 / (1.0 + np.exp(-margin[has_opinion] / params.margin_scale_m))
    p_away = 1.0 - p_home

    # Step 3: does somebody really control the ball?
    nearest = np.fmin(dist_home, dist_away)         # fmin ignores NaN
    close = nearest <= params.control_radius_m      # NaN compares as False
    if params.speed_source == "positions":
        ball_speed = speed_from_positions(ball_xy, params.speed_span_frames)
    elif params.speed_source == "tracking":
        ball_speed = arrays["ball_speed"]
    else:
        raise ValueError("speed_source must be 'positions' or 'tracking'")
    slow = smooth_speed(ball_speed, params.speed_smooth_frames) <= params.max_control_speed_mps
    raw_control = has_opinion & close & slow
    bridged = bridge_short_gaps(raw_control, params.max_gap_frames)
    control = keep_long_runs(bridged, params.min_control_frames) & has_opinion

    # Confidence: sure when the gap is big AND the ball is controlled; low during passes.
    gap_confidence = np.abs(2.0 * p_home - 1.0)
    weight = np.where(control, 1.0, params.uncontrolled_weight)
    confidence = np.where(has_opinion, gap_confidence * weight, 0.0)

    evidence = Evidence(p_home, p_away, np.full(n_frames, np.nan), confidence)
    evidence.validate()
    return BallToTeamResult(evidence, control, dist_home, dist_away,
                            smooth_speed(ball_speed, params.speed_smooth_frames))
