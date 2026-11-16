"""Synthetic tests for the Evidence combiner and Viterbi decoder. No match data."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from possession.evidence import Evidence                                   # noqa: E402
from possession.fusion import (FusionParams, combine, dead_ball_evidence,  # noqa: E402
                               decode, frame_scores, viterbi)


def team_ev(p_home, conf):
    p = np.asarray(p_home, dtype=float)
    return Evidence(p, 1 - p, np.full(len(p), np.nan), np.asarray(conf, dtype=float))


class FusionTests(unittest.TestCase):
    def test_no_opinion_is_neutral(self) -> None:
        team, dead = combine({"x": Evidence.empty(5)}, FusionParams())
        np.testing.assert_array_equal(team, 0)
        np.testing.assert_array_equal(dead, 0)
        s = frame_scores(team, dead)
        np.testing.assert_allclose(s[:, 0], s[:, 2])     # no lean towards D

    def test_short_blip_does_not_switch(self) -> None:
        p = np.full(100, 0.9)
        p[50:53] = 0.1
        labels = decode({"b": team_ev(p, np.ones(100))}, FusionParams(switch_cost=8))
        self.assertEqual(set(labels), {"H"})

    def test_sustained_evidence_switches(self) -> None:
        p = np.r_[np.full(50, 0.9), np.full(50, 0.1)]
        labels = decode({"b": team_ev(p, np.ones(100))}, FusionParams(switch_cost=8))
        self.assertEqual(labels[:50], ["H"] * 50)
        self.assertEqual(labels[50:], ["A"] * 50)

    def test_weak_detector_places_switch_inside_gap(self) -> None:
        # Strong detector: H until 30, silent 30..70, A after 70.
        # Weak detector flips at 40: the switch should land there, not at the midpoint 50.
        strong = np.r_[np.full(30, 0.95), np.full(40, np.nan), np.full(30, 0.05)]
        weak = np.r_[np.full(40, 0.7), np.full(60, 0.3)]
        labels = decode({"s": team_ev(strong, np.isfinite(strong).astype(float)),
                         "w": team_ev(weak, np.full(100, 0.5))}, FusionParams(switch_cost=4))
        self.assertEqual(labels.index("A"), 40)

    def test_dead_evidence(self) -> None:
        dead = np.r_[np.zeros(40), np.ones(20), np.zeros(40)]
        p = np.full(100, 0.8)
        labels = decode({"b": team_ev(p, np.ones(100)), "dead": dead_ball_evidence(dead)},
                        FusionParams())
        self.assertEqual(labels[45:55], ["D"] * 10)
        self.assertEqual(labels[:35], ["H"] * 35)

    def test_forced_labels(self) -> None:
        scores = frame_scores(np.full(10, 3.0), np.zeros(10))
        forced = np.array([""] * 10, dtype=object)
        forced[:5] = "D"
        labels = viterbi(scores, FusionParams(), forced.astype(str))
        self.assertEqual(labels[:5], ["D"] * 5)


class HardConstraintTests(unittest.TestCase):
    def test_dead_mask_and_anchor_run(self) -> None:
        from possession.fusion import hard_constraints
        dead = np.zeros(100, bool)
        dead[40:60] = True
        forced = hard_constraints(100, dead, [(70, 85, "A")])
        self.assertEqual(set(forced[40:60]), {"D"})
        self.assertEqual(set(forced[70:85]), {"A"})
        self.assertEqual(forced[0], "L")
        # Weak home evidence everywhere: the anchor run still wins, and no D outside the mask.
        scores = frame_scores(np.full(100, 0.5), np.full(100, 3.0))
        labels = viterbi(scores, FusionParams(switch_cost=2), forced)
        self.assertEqual(labels[40:60], ["D"] * 20)
        self.assertEqual(labels[70:85], ["A"] * 15)
        self.assertNotIn("D", labels[:40] + labels[60:])


class HybridTests(unittest.TestCase):
    def test_windows_and_fixed_inputs(self) -> None:
        from possession.fusion import hybrid_labels
        n = 120
        p = np.r_[np.full(60, 0.9), np.full(60, 0.1)]
        dead = np.zeros(n, bool)
        dead[20:30] = True
        labels = hybrid_labels({"ball": team_ev(p, np.ones(n))}, dead, [(40, 55, "A")],
                               windows=[(10, 100)])
        self.assertEqual(labels[:10], ["X"] * 10)        # outside every window
        self.assertEqual(labels[101:], ["X"] * 19)
        self.assertEqual(labels[20:30], ["D"] * 10)      # dead mask fixed
        self.assertEqual(labels[40:56], ["A"] * 16)      # control span fixed
        self.assertNotIn("D", labels[30:101])


if __name__ == "__main__":
    unittest.main()
