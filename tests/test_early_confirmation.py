"""Check the evidence and duration gates for early confirmation."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from possession.detectors.early_confirmation import anchors


class EarlyConfirmationTests(unittest.TestCase):
    def test_only_sustained_observed_control_anchors_a_switch(self) -> None:
        seen = np.ones(70, dtype=bool)
        dead = np.zeros_like(seen)
        control = np.zeros_like(seen)
        control[0:11] = control[15:27] = control[32:44] = control[49:61] = True
        seen[34] = False
        dead[52] = True
        evidence = SimpleNamespace(
            control=control, dist_home=np.ones(70), dist_away=np.full(70, 2.0),
        )
        with patch("possession.detectors.early_confirmation.ball_to_team", return_value=evidence):
            self.assertEqual(anchors({"ball_obs": seen}, dead), {15: "H"})
            evidence.dist_away[:] = 1.75
            self.assertEqual(anchors({"ball_obs": seen}, dead), {})
