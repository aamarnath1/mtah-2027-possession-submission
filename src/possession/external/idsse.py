"""Load the IDSSE / Sportec open Bundesliga dataset as a tuning match.

Why this dataset. Every parameter we use has to be fitted somewhere, and the
rules forbid fitting anything on the provided game or its public five minutes.
IDSSE is the only open set that is simultaneously 25 fps like ours, CC BY 4.0 so
usable in a prize competition, and labelled with possession on every frame rather
than requiring us to derive labels from event data.

Attribution is a licence condition, not a courtesy. Any use must credit the
Deutsche Fussball Liga and cite Bassek et al., Scientific Data (2025),
doi 10.1038/s41597-025-04505-y. That belongs in docs/TECHNICAL.md.

Format, from the raw XML:

    <FrameSet GameSection="firstHalf" MatchId=".." TeamId=".." PersonId="..">
      <Frame N="10000" T=".." X="0.57" Y="23.23" D="0.00" S="0.00" A="0.00" M="1"/>

The ball's frames carry three extra attributes:

      Z                 height in meters, which our feed does not have
      BallStatus        1 while the ball is in play
      BallPossession    which team holds it

Coordinates are already meters with the origin at the centre spot, so unlike our
own feed there is no normalization to undo.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

HOME, AWAY, DEAD = "H", "A", "D"


@dataclass
class ExternalMatch:
    """One tuning match, in the shape our own code expects."""

    match_id: str
    frames: list            # frame numbers, 0-based and contiguous
    ball_xy: list           # (x, y) meters, or None where the ball is untracked
    ball_z: list            # meters, or None. We do not have this on our match
    players: dict           # team -> list of per-frame [(x, y), ...] per person
    labels: list            # 'H' | 'A' | 'D' per frame: the ground truth
    home_team_id: str
    away_team_id: str

    @property
    def n_frames(self):
        return len(self.frames)

    def possession_share(self):
        """Home share of alive frames. A sanity check on the team mapping."""
        alive = [lab for lab in self.labels if lab in (HOME, AWAY)]
        if not alive:
            return None
        return sum(1 for lab in alive if lab == HOME) / len(alive)


def team_roles(matchinfo_path):
    """TeamId -> 'H' or 'A', read from the match information file."""
    root = ET.parse(matchinfo_path).getroot()
    roles = {}
    for team in root.iter("Team"):
        role = (team.get("Role") or "").lower()
        if role in ("home", "guest", "away"):
            roles[team.get("TeamId")] = HOME if role == "home" else AWAY
    if len(roles) != 2:
        raise ValueError(f"expected 2 teams in {matchinfo_path}, found {len(roles)}")
    return roles


def load(positions_path, matchinfo_path, half="firstHalf"):
    """Parse one half into an ExternalMatch.

    One half at a time because the whole file is about 370 MB and iterparse keeps
    memory flat only if we drop elements as we go.

    `BallPossession` is 1 for home and 2 for away. Verified on J03WPY against the
    events file: the loader reports a 0.665 home possession share and the home
    side made 604 of 891 passes, 67.8%. Matches really are this lopsided, so
    `possession_share()` is a weak check. Compare it against the pass split in the
    events file before trusting a new match, because a flipped mapping would
    silently mirror every label we tune on.
    """
    roles = team_roles(matchinfo_path)

    ball_rows = {}
    player_rows = {}
    current = None

    for event, elem in ET.iterparse(positions_path, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            current = {
                "section": elem.get("GameSection"),
                "team": elem.get("TeamId"),
                "person": elem.get("PersonId"),
                "is_ball": False,
            }
        elif event == "end" and elem.tag == "Frame":
            if current and current["section"] == half:
                n = int(elem.get("N"))
                if elem.get("BallPossession") is not None:
                    current["is_ball"] = True
                    ball_rows[n] = (
                        float(elem.get("X")), float(elem.get("Y")),
                        float(elem.get("Z")) if elem.get("Z") else None,
                        elem.get("BallStatus"), elem.get("BallPossession"),
                    )
                else:
                    player_rows.setdefault(current["person"], {})[n] = (
                        float(elem.get("X")), float(elem.get("Y")), current["team"])
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            current = None
            elem.clear()

    if not ball_rows:
        raise ValueError(f"no ball frames found for {half} in {positions_path}")

    lo, hi = min(ball_rows), max(ball_rows)
    frames = list(range(lo, hi + 1))

    # Which BallPossession code is the home team. Both codes should appear; if
    # only one does, the half is degenerate and we bail rather than guess.
    codes = {row[4] for row in ball_rows.values() if row[4]}
    if not codes:
        raise ValueError("no BallPossession values present")
    code_to_team = {"1": HOME, "2": AWAY}

    ball_xy, ball_z, labels = [], [], []
    for n in frames:
        row = ball_rows.get(n)
        if row is None:
            ball_xy.append(None)
            ball_z.append(None)
            labels.append(DEAD)
            continue
        x, y, z, status, poss = row
        ball_xy.append((x, y))
        ball_z.append(z)
        labels.append(DEAD if status != "1" else code_to_team.get(poss, DEAD))

    players = {HOME: [], AWAY: []}
    for person, rows in player_rows.items():
        team_id = next(iter(rows.values()))[2]
        side = roles.get(team_id)
        if side is None:
            continue
        track = [(rows[n][0], rows[n][1]) if n in rows else None for n in frames]
        players[side].append(track)

    home_id = next(t for t, r in roles.items() if r == HOME)
    away_id = next(t for t, r in roles.items() if r == AWAY)

    return ExternalMatch(
        match_id=Path(positions_path).stem.split("_")[-1],
        frames=frames,
        ball_xy=ball_xy,
        ball_z=ball_z,
        players=players,
        labels=labels,
        home_team_id=home_id,
        away_team_id=away_id,
    )
