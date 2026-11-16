"""Turn observed ball control into the frozen B6 early-confirmation anchors."""

from collections.abc import Mapping
from itertools import groupby
from typing import TYPE_CHECKING

import numpy as np

from possession.detectors.ball_to_team import ball_to_team

if TYPE_CHECKING:
    from pandas import DataFrame


def from_table(
    table: "DataFrame", slot_teams: Mapping[int, str], raw_dead: np.ndarray | None,
) -> dict[int, str]:
    """Use only observed positions from the published preprocessing table."""
    return anchors(arrays_from_table(table, slot_teams),
                   raw_dead if raw_dead is not None else np.zeros(len(table), dtype=bool))


def arrays_from_table(table: "DataFrame", slot_teams: Mapping[int, str]) -> dict[str, np.ndarray]:
    """Detector arrays (ball_xy, ball_obs, slot_xy, slot_team), observed positions only."""
    observed = table["ball_observed"].to_numpy() > 0
    ball = table[["ball_x_m", "ball_y_m"]].to_numpy()
    tracks = []
    slots = sorted(slot_teams)
    for slot in slots:
        columns = [f"slot_{slot}_x_m", f"slot_{slot}_y_m"]
        seen = table[f"slot_{slot}_observed"].to_numpy() > 0
        tracks.append(np.where(seen[:, None], table[columns].to_numpy(), np.nan))
    return {
        "ball_xy": np.where(observed[:, None], ball, np.nan),
        "ball_obs": observed & np.isfinite(ball).all(axis=1),
        "slot_xy": np.stack(tracks, axis=1),
        "slot_team": np.array([slot_teams[slot] for slot in slots]),
    }


def anchors(arrays: dict[str, np.ndarray], raw_dead: np.ndarray) -> dict[int, str]:
    """Backdate the start of each 12-frame run that passes the fixed B6 gates."""
    return {start: team for start, _, team in control_runs(arrays, raw_dead)}


def control_runs(arrays: dict[str, np.ndarray],
                 raw_dead: np.ndarray) -> list[tuple[int, int, str]]:
    """(start, end exclusive, team) of each 12-frame run that passes the B6 gates."""
    result = ball_to_team(arrays)
    margin = result.dist_away - result.dist_home
    accepted = arrays["ball_obs"] & result.control & (np.abs(margin) > 0.75) & ~raw_dead
    signal = np.where(accepted, np.where(margin > 0, "H", "A"), "X")
    out = []
    start = 0
    for team, frames in groupby(signal.tolist()):
        count = sum(1 for _ in frames)
        if team != "X" and count >= 12:
            out.append((start, start + count, team))
        start += count
    return out
