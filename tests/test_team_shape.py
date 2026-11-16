"""Synthetic tests for the team-shape detector. No match data is used."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from possession.detectors.team_shape import (TeamShapeParams, attack_direction,  # noqa: E402
                                             centered_velocity, drop_deepest_player,
                                             smooth_centered, team_shape)

L = 105.3
N_PER_TEAM = 11


def team_block(keeper_x, centre_x, width, n=N_PER_TEAM):
    """One keeper plus n-1 outfield players spread around centre_x (metres)."""
    xs = [keeper_x] + list(np.linspace(centre_x - width / 2, centre_x + width / 2, n - 1))
    ys = [34.0] + list(np.linspace(34 - width / 2, 34 + width / 2, n - 1))
    return list(zip(xs, ys))


def make_arrays(frames):
    """frames: list of (home_positions, away_positions), each a list of (x, y)."""
    slot_xy = np.array([h + a for h, a in frames], dtype=float)
    teams = np.array(["H"] * len(frames[0][0]) + ["A"] * len(frames[0][1]))
    return {"slot_xy": slot_xy, "slot_team": teams}


def home_attacking_right(n, home_centre, away_centre, home_width, away_width):
    """Home keeper at x=2 (defends low x), away keeper at x=103."""
    frame = (team_block(2.0, home_centre, home_width), team_block(103.0, away_centre, away_width))
    return [frame] * n


FAST = TeamShapeParams(window_s=0.0, velocity_span_s=0.04, halves=((0, 10**9),))


class TeamShapeTests(unittest.TestCase):
    def test_home_pushed_up_and_spread_means_home(self) -> None:
        # Home block in the away half and stretched; away deep and compact.
        arrays = make_arrays(home_attacking_right(10, 70, 85, 30, 12))
        result = team_shape(arrays, FAST)
        self.assertEqual(result.attack_dir[0], 1)
        self.assertTrue((result.evidence.p_home[2:-2] > 0.7).all())

    def test_away_pushed_up_means_away(self) -> None:
        arrays = make_arrays(home_attacking_right(10, 20, 35, 12, 30))
        result = team_shape(arrays, FAST)
        self.assertTrue((result.evidence.p_away[2:-2] > 0.7).all())

    def test_mirrored_pitch_gives_same_answer(self) -> None:
        # Flip every x: home now attacks toward low x. The call must not change.
        right = make_arrays(home_attacking_right(10, 70, 85, 30, 12))
        left = dict(right, slot_xy=right["slot_xy"].copy())
        left["slot_xy"][..., 0] = L - left["slot_xy"][..., 0]
        a = team_shape(right, FAST).evidence.p_home
        b = team_shape(left, FAST).evidence.p_home
        self.assertEqual(team_shape(left, FAST).attack_dir[0], -1)
        np.testing.assert_allclose(a, b, equal_nan=True)

    def test_moving_forward_counts_for_home(self) -> None:
        # Both teams level and equally spread; home block advancing, away retreating.
        frames = []
        for t in range(30):
            frames.append((team_block(2.0, 50 + 0.2 * t, 20), team_block(103.0, 55 + 0.2 * t, 20)))
        result = team_shape(make_arrays(frames),
                            TeamShapeParams(window_s=0.0, velocity_span_s=0.2,
                                            halves=((0, 10**9),), weights=(0.0, 1.0, 0.0)))
        self.assertGreater(np.nanmean(result.velocity), 0)
        self.assertTrue((result.evidence.p_home[np.isfinite(result.evidence.p_home)] > 0.5).all())

    def test_too_few_players_gives_no_opinion(self) -> None:
        arrays = make_arrays(home_attacking_right(6, 70, 85, 30, 12))
        arrays["slot_xy"][3, N_PER_TEAM + 4:, :] = np.nan   # away down to 4 tracked
        result = team_shape(arrays, FAST)
        self.assertTrue(np.isnan(result.evidence.p_home[3]))
        self.assertEqual(result.evidence.confidence[3], 0.0)
        self.assertGreater(result.evidence.confidence[1], 0.0)

    def test_evidence_contract(self) -> None:
        result = team_shape(make_arrays(home_attacking_right(10, 70, 85, 30, 12)), FAST)
        ev = result.evidence
        ev.validate()
        self.assertTrue(np.isnan(ev.p_dead).all())
        self.assertLessEqual(ev.confidence.max(), FAST.max_confidence)

    def test_attack_dir_override(self) -> None:
        arrays = make_arrays(home_attacking_right(10, 70, 85, 30, 12))
        flipped = team_shape(arrays, TeamShapeParams(window_s=0.0, halves=((0, 10**9),),
                                                     attack_dir={0: -1}))
        self.assertEqual(flipped.attack_dir[0], -1)


class HelperTests(unittest.TestCase):
    def test_attack_direction_from_keepers(self) -> None:
        x = np.array([[2.0, 40.0, 60.0, 103.0, 50.0, 70.0]] * 5)
        home = np.array([True, True, True, False, False, False])
        self.assertEqual(attack_direction(x, home, L), 1)
        self.assertEqual(attack_direction(L - x, home, L), -1)

    def test_drop_deepest_player(self) -> None:
        f = np.array([[5.0, 30.0, np.nan], [np.nan, 40.0, np.nan]])
        out = drop_deepest_player(f)
        self.assertTrue(np.isnan(out[0, 0]))
        self.assertEqual(out[0, 1], 30.0)
        self.assertEqual(out[1, 1], 40.0)          # only one tracked: kept

    def test_smooth_centered_ignores_nan(self) -> None:
        out = smooth_centered(np.array([1.0, np.nan, 3.0]), 1)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])

    def test_centered_velocity(self) -> None:
        v = centered_velocity(np.arange(10, dtype=float), 2, fps=25.0)
        self.assertTrue(np.isnan(v[:2]).all() and np.isnan(v[-2:]).all())
        np.testing.assert_allclose(v[2:-2], 25.0)


if __name__ == "__main__":
    unittest.main()
