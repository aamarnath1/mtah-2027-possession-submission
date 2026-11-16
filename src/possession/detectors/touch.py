"""When did somebody actually touch the ball.

NEGATIVE RESULT. Built, measured, and NOT WIRED INTO THE PIPELINE. Kept so that
nobody spends another day rebuilding it. Read the last section before you do.


The point of this layer is not to suppress switches. It is to make them
trustworthy enough that we can afford to be more sensitive.

Our boundary diagnostic says live possessions come out roughly 600 frames too
long on validation data: we claim about 160 possessions where the truth has 211,
so we are merging turnovers, not inventing them. The sustained-lead hold that
currently guards switches is a timer, not evidence, and the only way to shorten
it without flooding the output with flicker is to require evidence that a touch
happened at all.

The one principle every published method agrees on: **proximity alone can never
establish a touch.** We have no ball height, so a ball sailing two metres over
somebody's head and a ball at their feet are the same two numbers. What
distinguishes them is that a touched ball BREAKS, changing speed or direction,
while an untouched one carries on ballistically.

Link and Hoernig (2017) measured the trade on 6,976 hand-labelled possessions:
naive proximity scores recall .90 / precision .68, and adding the kinematic gate
moves it to recall .80 / precision .86. Ten points of recall for eighteen of
precision. Under our metric a false switch costs about twice a missed one, so
that trade is better for us than it was for them.

Published thresholds, and which to trust:

  direction change    cos 10 degrees. Fitted identically across all three
                      providers in Vidal-Codina et al., and independently the
                      default in databallpy. The one value worth copying.
  possession radius   0.5 m to 1.5 m, provider-dependent. Must be fitted.
  speed change        1 m/s to 5 m/s across providers in the same study. A
                      fivefold spread. Never copy this one.

Direction must be measured ACROSS the whole contact window, from before entry to
after exit, never frame to frame. At 3 m/s with 5 cm of position noise, a
frame-to-frame angle carries about 7 degrees of noise against a 10 degree
threshold, which is no signal at all.
"""

import math

DIRECTION_COS = math.cos(math.radians(10.0))
"""Copied directly. Fitted the same across three tracking providers."""

DEFAULT_RADIUS_M = 1.0
DEFAULT_SPEED_CHANGE_MPS = 2.0
"""Both must be fitted on external data. These are starting points only."""


def _speed(a, b, fps):
    if a is None or b is None:
        return None
    return math.hypot(b[0] - a[0], b[1] - a[1]) * fps


def _unit(a, b):
    if a is None or b is None:
        return None
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else None


def proximity_runs(ball_xy, players, teams, radius_m):
    """Maximal stretches where the same team's nearest player is within radius.

    Yields (start, end, team). A run is a candidate touch window, nothing more:
    it still has to pass the kinematic gate below.
    """
    n = len(ball_xy)
    nearest = [None] * n
    for i in range(n):
        ball = ball_xy[i]
        if ball is None:
            continue
        best_d, best_team = radius_m, None
        for team, tracks in zip(teams, players):
            for track in tracks:
                p = track[i]
                if p is None:
                    continue
                d = math.hypot(p[0] - ball[0], p[1] - ball[1])
                if d < best_d:
                    best_d, best_team = d, team
        nearest[i] = best_team

    runs, start = [], None
    for i in range(n + 1):
        here = nearest[i] if i < n else None
        if start is not None and (here != nearest[start]):
            runs.append((start, i - 1, nearest[start]))
            start = None
        if start is None and here is not None:
            start = i
    return runs


def kinematic_break(ball_xy, start, end, fps, speed_change_mps, baseline=5):
    """Did the ball's motion break across this window.

    Speed OR direction, following Vidal-Codina et al., who use the disjunction
    explicitly as the substitute for the missing z coordinate.

    The baseline for each direction vector is extended over several frames so the
    angle is measured against real displacement rather than position noise.
    """
    n = len(ball_xy)
    a0, a1 = max(0, start - baseline), start
    b0, b1 = min(n - 1, end), min(n - 1, end + baseline)
    if a0 == a1 or b0 == b1:
        return False

    v_in = _speed(ball_xy[a0], ball_xy[a1], fps / max(a1 - a0, 1))
    v_out = _speed(ball_xy[b0], ball_xy[b1], fps / max(b1 - b0, 1))
    if v_in is not None and v_out is not None:
        if abs(v_in - v_out) > speed_change_mps:
            return True

    d_in, d_out = _unit(ball_xy[a0], ball_xy[a1]), _unit(ball_xy[b0], ball_xy[b1])
    if d_in is not None and d_out is not None:
        if d_in[0] * d_out[0] + d_in[1] * d_out[1] < DIRECTION_COS:
            return True
    return False


def detect(ball_xy, players_by_team, fps=25, radius_m=DEFAULT_RADIUS_M,
           speed_change_mps=DEFAULT_SPEED_CHANGE_MPS):
    """Touch events as (frame, team), one per accepted proximity window.

    The frame reported is the START of the window, because that is when contact
    began, and the metric cares where a boundary falls rather than when we became
    sure of it.
    """
    teams = list(players_by_team)
    tracks = [players_by_team[t] for t in teams]
    events = []
    for start, end, team in proximity_runs(ball_xy, tracks, teams, radius_m):
        if kinematic_break(ball_xy, start, end, fps, speed_change_mps):
            events.append((start, team))
    return events


def gate(nearest, events, window_frames=50):
    """Mask a nearest-team signal down to frames a touch can vouch for.

    Returns a copy of `nearest` in which a frame keeps its team only if that team
    touched the ball within `window_frames` before it. Everywhere else becomes
    None, meaning no opinion, which the sticky pass reads as "carry on".

    This is what buys a shorter hold: the hold exists to filter flicker, and a
    signal that already requires evidence of contact does not need as much of it.
    """
    out = [None] * len(nearest)
    last = {}
    ev = sorted(events)
    j = 0
    for i in range(len(nearest)):
        while j < len(ev) and ev[j][0] <= i:
            last[ev[j][1]] = ev[j][0]
            j += 1
        team = nearest[i]
        if team is not None and i - last.get(team, -10 ** 9) <= window_frames:
            out[i] = team
    return out


# --------------------------------------------------------------------------- #
# Why this is not in the pipeline
# --------------------------------------------------------------------------- #
#
# Gating the nearest-team signal on touch evidence LOWERS spellF1 from 0.2212 to
# roughly 0.175 on four validation halves, and it does so across every parameter
# setting tried: radius 1.0 and 1.5 m, speed-change thresholds 1.0 and 3.0 m/s,
# gate windows of 125 and 300 frames, and holds from 8 to 30 frames. The spread
# across all of those was 0.172 to 0.177. That flatness is what makes it a real
# negative rather than a tuning failure.
#
# The cause is coverage, not correctness. The detector finds 255 to 427 touches
# per half where a real half has well over a thousand, because the ball is
# untracked on 41% of frames and a touch can only be seen when the ball is
# visible. Gating on sparse evidence blanks out more good signal than bad: the
# claimed possession count falls from 114 to about 68, against a true 211, so we
# move from under-segmenting to badly under-segmenting.
#
# This would work on a feed with a complete ball track. On ours it removes more
# signal than noise. If ball coverage is ever improved, reconsider it first: the
# published trade is ten points of recall for eighteen of precision, which is a
# good trade under a metric that punishes false switches twice as hard as missed
# ones.
