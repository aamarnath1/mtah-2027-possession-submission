"""Is the ball in play, or has the referee stopped it.

Dead is a referee state, not a physics state. A still ball during live play is
NOT dead: under the rules it belongs to the team that last had it. So ball
stillness is a weak feature and never a trigger on its own.

Two signals carry this, measured on IDSSE first-half data (46,179 alive frames,
22,951 dead):

  ball outside the pitch box     0.2% of alive frames, 53.7% of dead frames.
                                 Nearly deterministic in one direction: if the
                                 ball is visible and off the pitch, play has
                                 stopped. It only covers half the dead frames
                                 though, because the ball comes back on quickly
                                 while play is still stopped.

  the whole team slowing at once  mean player speed 2.40 m/s alive against 1.05
                                 dead; the fraction of players under 1 m/s runs
                                 0.14 alive against 0.48 dead. About 1.4 standard
                                 deviations of separation each.

Player movement supplies evidence during tracking gaps.

The published state of the art for this task is Lang et al. (Scientific Reports
2022): per-player position, speed and acceleration sampled at plus and minus
0.4, 0.8, 2, 4, 20, 40 and 80 seconds, fed to AdaBoost, reaching 92% frame
accuracy with no ball term at all. The multi-scale window is the part worth
copying even without the classifier, and their finding that the plus/minus 4
second band dominates is what sets the default here.
"""

PITCH_HALF_X = 52.5
PITCH_HALF_Y = 34.0
BALL_RADIUS_M = 0.11
"""The ball is out when it has WHOLLY crossed the line, so allow its radius."""


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def frac_slow(players, fps=25, threshold_mps=1.0):
    """Per frame: what share of tracked players are moving slower than threshold.

    Speed is derived from positions rather than trusted from the feed, so this
    works identically on any source.
    """
    n = len(players[0]) if players else 0
    out = [None] * n
    for i in range(1, n):
        slow = total = 0
        for track in players:
            a, b = track[i - 1], track[i]
            if a is None or b is None:
                continue
            speed = (((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5) * fps
            total += 1
            slow += speed < threshold_mps
        out[i] = slow / total if total else None
    if n:
        out[0] = out[1] if n > 1 else None
    return out


def smooth_centered(values, half_width):
    """Centered moving average over a signal that may contain None.

    Centered, never trailing. A trailing window delays both edges of a stoppage
    in the same direction, and a uniform shift is the expensive kind of error.
    """
    n = len(values)
    out = [None] * n
    for i in range(n):
        lo, hi = max(0, i - half_width), min(n, i + half_width + 1)
        out[i] = _mean(values[lo:hi])
    return out


def ball_out_of_box(ball_xy, half_x=PITCH_HALF_X, half_y=PITCH_HALF_Y,
                    origin_centered=True):
    """Per frame: True if the ball is wholly off the pitch, None if untracked.

    `origin_centered` is True for IDSSE, where coordinates are meters from the
    centre spot. Our own feed arrives normalized 0 to 1 and is scaled to
    105.3 by 68.0, so its origin is a corner: pass False and the box is
    0..2*half_x by 0..2*half_y.
    """
    out = []
    for ball in ball_xy:
        if ball is None:
            out.append(None)
            continue
        x, y = ball
        if origin_centered:
            is_out = abs(x) > half_x + BALL_RADIUS_M or abs(y) > half_y + BALL_RADIUS_M
        else:
            is_out = (x < -BALL_RADIUS_M or x > 2 * half_x + BALL_RADIUS_M
                      or y < -BALL_RADIUS_M or y > 2 * half_y + BALL_RADIUS_M)
        out.append(is_out)
    return out


def detect(ball_xy, players, fps=25, slow_threshold=0.34, window_s=4.0,
           origin_centered=True):
    """Per-frame dead-ball evidence in [0, 1].

    `slow_threshold` is the smoothed fraction-of-players-slow above which we call
    it dead. Fit it on external matches; the default came from a sweep on IDSSE
    and is a starting point, not a constant of nature.

    `window_s` is the half-width of the centered average, in seconds. Lang et al.
    found the plus/minus 4 second band carries most of the signal.
    """
    slow = smooth_centered(frac_slow(players, fps=fps), int(window_s * fps))
    off = ball_out_of_box(ball_xy, origin_centered=origin_centered)

    evidence = []
    for i in range(len(ball_xy)):
        if off[i]:
            evidence.append(1.0)          # visible and off the pitch: play stopped
            continue
        s = slow[i]
        if s is None:
            evidence.append(0.0)          # nothing tracked: say nothing
            continue
        # a soft ramp around the threshold rather than a cliff, so the decoder
        # sees confidence rather than a hard call it cannot argue with
        evidence.append(min(1.0, max(0.0, (s - slow_threshold) / 0.20 + 0.5)))
    return evidence


def labels_from_evidence(evidence, cut=0.5):
    """Convenience for scoring the detector on its own, before any decoding."""
    return ["D" if e >= cut else "L" for e in evidence]


# --------------------------------------------------------------------------- #
# Our own feed
# --------------------------------------------------------------------------- #
#
# The functions above take Python lists, which is fine for a 69,000-frame
# external half but wasteful on our 199,319-frame table with 46 slots. This path
# does the same arithmetic in numpy, straight off the preprocessed columns.

DEFAULT_SLOW_THRESHOLD = 0.40
"""Threshold selected on external IDSSE matches with simulated ball tracking gaps."""


def detect_from_table(table, slot_numbers, fps=25, slow_threshold=None,
                      window_s=4.0, pitch_x_m=105.3, pitch_y_m=68.0):
    """Per-frame dead evidence from the preprocessed frames table.

    `table` is a pandas frame carrying ball_x_m, ball_y_m, ball_observed and,
    for each slot, slot_<n>_x_m, slot_<n>_y_m, slot_<n>_observed.

    Our coordinates have their origin at a pitch corner, not the centre spot, so
    the out-of-play box is 0..105.3 by 0..68.0 rather than symmetric about zero.
    """
    import numpy as np

    slow_threshold = DEFAULT_SLOW_THRESHOLD if slow_threshold is None else slow_threshold
    n = len(table)

    slow_count = np.zeros(n)
    tracked = np.zeros(n)
    for slot in slot_numbers:
        x = table[f"slot_{slot}_x_m"].to_numpy(dtype="float64")
        y = table[f"slot_{slot}_y_m"].to_numpy(dtype="float64")
        seen = table[f"slot_{slot}_observed"].to_numpy() > 0
        step = np.full(n, np.nan)
        step[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2) * fps
        usable = seen & np.roll(seen, 1) & np.isfinite(step)
        usable[0] = False
        tracked += usable
        slow_count += usable & (step < 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(tracked > 0, slow_count / np.maximum(tracked, 1), np.nan)

    # centered moving average, NaN-aware, via cumulative sums
    half = int(window_s * fps)
    filled = np.nan_to_num(frac, nan=0.0)
    valid = np.isfinite(frac).astype("float64")
    csum = np.concatenate([[0.0], np.cumsum(filled)])
    cval = np.concatenate([[0.0], np.cumsum(valid)])
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    total = csum[hi] - csum[lo]
    count = cval[hi] - cval[lo]
    smoothed = np.where(count > 0, total / np.maximum(count, 1), np.nan)

    bx = table["ball_x_m"].to_numpy(dtype="float64")
    by = table["ball_y_m"].to_numpy(dtype="float64")
    ball_seen = table["ball_observed"].to_numpy() > 0
    off = ball_seen & (
        (bx < -BALL_RADIUS_M) | (bx > pitch_x_m + BALL_RADIUS_M)
        | (by < -BALL_RADIUS_M) | (by > pitch_y_m + BALL_RADIUS_M))

    ramp = np.clip((smoothed - slow_threshold) / 0.20 + 0.5, 0.0, 1.0)
    evidence = np.where(np.isfinite(ramp), ramp, 0.0)
    evidence = np.where(off, 1.0, evidence)
    return evidence
