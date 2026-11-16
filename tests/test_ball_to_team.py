"""Synthetic tests for the ball-to-team detector. No match data is used."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from possession.detectors.ball_to_team import (BallToTeamParams, ball_to_team, bridge_short_gaps,
                                               keep_long_runs, nearest_distance_per_team,
                                               smooth_speed, speed_from_positions)
from possession.evidence import Evidence

NAN = np.nan


def make_arrays(ball, speed, slots):
    """ball: list of (x, y) or None. slots: per-frame list of 4 (x, y) positions.
    Slot teams are fixed: slots 0 and 1 are home, slots 2 and 3 are away."""
    n = len(ball)
    ball_xy = np.array([b if b else (NAN, NAN) for b in ball], dtype=float)
    return {
        "ball_xy": ball_xy,
        "ball_speed": np.array([s if b else NAN for s, b in zip(speed, ball)], dtype=float),
        "ball_obs": np.array([b is not None for b in ball]),
        "slot_xy": np.array(slots, dtype=float).reshape(n, 4, 2),
        "slot_team": np.array(["H", "H", "A", "A"]),
    }


FAR = (80.0, 60.0)


class BallToTeamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = BallToTeamParams(min_control_frames=1, speed_source="tracking", speed_smooth_frames=1, max_gap_frames=0)

    def test_home_player_on_the_ball(self) -> None:
        arrays = make_arrays([(50, 30)], [3.0], [[(50.5, 30), FAR, (58, 30), FAR]])
        result = ball_to_team(arrays, self.params)
        self.assertGreater(result.evidence.p_home[0], 0.9)
        self.assertTrue(result.control[0])
        self.assertGreater(result.evidence.confidence[0], 0.8)

    def test_away_player_on_the_ball(self) -> None:
        arrays = make_arrays([(50, 30)], [3.0], [[(58, 30), FAR, (50.5, 30), FAR]])
        result = ball_to_team(arrays, self.params)
        self.assertLess(result.evidence.p_home[0], 0.1)
        self.assertAlmostEqual(result.evidence.p_away[0], 1 - result.evidence.p_home[0])

    def test_unseen_ball_gives_no_opinion(self) -> None:
        arrays = make_arrays([None], [0.0], [[(50, 30), FAR, (58, 30), FAR]])
        evidence = ball_to_team(arrays, self.params).evidence
        self.assertTrue(np.isnan(evidence.p_home[0]))
        self.assertEqual(evidence.confidence[0], 0.0)

    def test_fast_ball_lowers_confidence(self) -> None:
        arrays = make_arrays([(50, 30), (50, 30)], [3.0, 20.0],
                             [[(50.5, 30), FAR, (58, 30), FAR]] * 2)
        result = ball_to_team(arrays, self.params)
        self.assertTrue(result.control[0])
        self.assertFalse(result.control[1])
        self.assertLess(result.evidence.confidence[1], result.evidence.confidence[0])

    def test_equal_distance_is_a_coin_flip_with_zero_confidence(self) -> None:
        arrays = make_arrays([(50, 30)], [3.0], [[(49, 30), FAR, (51, 30), FAR]])
        evidence = ball_to_team(arrays, self.params).evidence
        self.assertAlmostEqual(evidence.p_home[0], 0.5)
        self.assertAlmostEqual(evidence.confidence[0], 0.0)

    def test_team_with_no_tracked_player_gives_no_opinion(self) -> None:
        arrays = make_arrays([(50, 30)], [3.0], [[(50.5, 30), FAR, (NAN, NAN), (NAN, NAN)]])
        evidence = ball_to_team(arrays, self.params).evidence
        self.assertTrue(np.isnan(evidence.p_home[0]))

    def test_empty_slots_are_ignored(self) -> None:
        arrays = make_arrays([(50, 30)], [3.0], [[(NAN, NAN), (50.5, 30), (58, 30), FAR]])
        self.assertGreater(ball_to_team(arrays, self.params).evidence.p_home[0], 0.9)

    def test_control_needs_enough_frames_in_a_row(self) -> None:
        strict = BallToTeamParams(min_control_frames=3, speed_source="tracking", speed_smooth_frames=1, max_gap_frames=0)
        slots = [[(50.5, 30), FAR, (58, 30), FAR]] * 5
        arrays = make_arrays([(50, 30)] * 5, [3.0, 20.0, 3.0, 3.0, 3.0], slots)
        control = ball_to_team(arrays, strict).control
        self.assertEqual(control.tolist(), [False, False, True, True, True])

    def test_one_speed_spike_does_not_break_control(self) -> None:
        slots = [[(50.5, 30), FAR, (58, 30), FAR]] * 5
        arrays = make_arrays([(50, 30)] * 5, [3.0, 3.0, 30.0, 3.0, 3.0], slots)
        smooth = BallToTeamParams(min_control_frames=5, speed_source="tracking", speed_smooth_frames=5, max_gap_frames=0)
        rough = BallToTeamParams(min_control_frames=5, speed_source="tracking", speed_smooth_frames=1, max_gap_frames=0)
        self.assertTrue(ball_to_team(arrays, smooth).control.all())
        self.assertFalse(ball_to_team(arrays, rough).control.any())

    def test_short_flicker_is_bridged(self) -> None:
        slots = [[(50.5, 30), FAR, (58, 30), FAR]] * 5
        arrays = make_arrays([(50, 30)] * 5, [3.0, 3.0, 30.0, 3.0, 3.0], slots)
        bridged = BallToTeamParams(min_control_frames=5, speed_source="tracking", speed_smooth_frames=1, max_gap_frames=1)
        self.assertTrue(ball_to_team(arrays, bridged).control.all())

    def test_smoothing_and_bridging_helpers(self) -> None:
        smoothed = smooth_speed(np.array([1.0, 1.0, 50.0, 1.0, 1.0]), 3)
        self.assertEqual(smoothed.tolist(), [1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertTrue(np.isnan(smooth_speed(np.array([NAN, NAN, NAN]), 3)).all())
        mask = np.array([1, 1, 0, 1, 0, 0, 0, 1], bool)
        self.assertEqual(bridge_short_gaps(mask, 1).tolist(),
                         [True, True, True, True, False, False, False, True])
        self.assertEqual(bridge_short_gaps(mask, 0).tolist(), mask.tolist())

    def test_speed_from_positions(self) -> None:
        xy = np.column_stack([np.arange(8) * 0.2, np.zeros(8)])        # 0.2 m per frame = 5 m/s
        speed = speed_from_positions(xy, 4)
        self.assertTrue(np.isnan(speed[:2]).all() and np.isnan(speed[-2:]).all())
        self.assertTrue(np.allclose(speed[2:6], 5.0))
        xy[4] = (NAN, NAN)                                              # an unseen frame
        gapped = speed_from_positions(xy, 4)
        self.assertTrue(np.isnan(gapped[[2, 6]]).all())                 # their span ends on frame 4
        self.assertTrue(np.isfinite(gapped[[3, 4, 5]]).all())

    def test_control_works_when_the_tracking_speed_column_is_empty(self) -> None:
        """The real match has almost no ball speed values; positions must be enough."""
        n = 14
        ball = [(50 + 0.1 * i, 30) for i in range(n)]                   # 0.1 m/frame = 2.5 m/s
        slots = [[(50.5 + 0.1 * i, 30), FAR, (58, 30), FAR] for i in range(n)]
        arrays = make_arrays(ball, [NAN] * n, slots)                    # speed column all NaN
        result = ball_to_team(arrays, BallToTeamParams())
        self.assertTrue(result.control[4:10].all())
        self.assertGreater(result.evidence.p_home[6], 0.9)

    def test_a_fast_ball_is_not_controlled_when_speed_comes_from_positions(self) -> None:
        n = 14
        ball = [(30 + 1.0 * i, 30) for i in range(n)]                   # 1 m/frame = 25 m/s
        slots = [[(30.5 + 1.0 * i, 30), FAR, (80, 30), FAR] for i in range(n)]
        result = ball_to_team(make_arrays(ball, [NAN] * n, slots), BallToTeamParams())
        self.assertFalse(result.control[4:10].any())

    def test_helpers(self) -> None:
        self.assertEqual(keep_long_runs(np.array([1, 1, 0, 1, 1, 1, 0], bool), 3).tolist(),
                         [False, False, False, True, True, True, False])
        ball = np.array([[0.0, 0.0]])
        slots = np.array([[[3.0, 4.0], [1.0, 0.0], [NAN, NAN]]])
        distance = nearest_distance_per_team(ball, slots, np.array([True, True, True]))
        self.assertAlmostEqual(distance[0], 1.0)

    def test_evidence_contract_is_enforced(self) -> None:
        bad = Evidence(np.array([1.5]), np.array([NAN]), np.array([NAN]), np.array([0.5]))
        with self.assertRaises(ValueError):
            bad.validate()
        bad_conf = Evidence(np.array([NAN]), np.array([NAN]), np.array([NAN]), np.array([NAN]))
        with self.assertRaises(ValueError):
            bad_conf.validate()


if __name__ == "__main__":
    unittest.main()
