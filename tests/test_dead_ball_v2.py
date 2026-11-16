"""Synthetic tests for dead ball v2. No match data is used."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from possession.detectors import dead_ball_v2 as D   # noqa: E402


def synthetic(n=3000, still=slice(1000, 2000), ball_seen=True):
    """22 players running in circles, all standing still inside `still`."""
    t = np.arange(n, dtype=float)
    phase = np.where(np.isin(np.arange(n), np.arange(n)[still]), 0.0, 1.0).cumsum() / 25
    k = np.arange(22)
    x = 52 + 30 * np.sin(phase[:, None] * 0.3 + k)
    y = 34 + 20 * np.cos(phase[:, None] * 0.3 + k)
    slot_xy = np.stack([x, y], axis=2)
    ball = slot_xy[:, 0] + 0.5
    obs = np.full(n, ball_seen)
    return {"slot_xy": slot_xy, "ball_xy": np.where(obs[:, None], ball, np.nan), "ball_obs": obs}


class DeadBallV2Tests(unittest.TestCase):
    def test_features_match_saved_model(self) -> None:
        X, names = D.features(synthetic())
        model = D.BinnedLogistic.load()
        self.assertEqual(names, model.names)
        self.assertEqual(X.shape, (3000, len(names)))

    def test_still_players_look_dead(self) -> None:
        p = D.detect(synthetic(ball_seen=False))
        self.assertGreater(p[1300:1700].mean(), p[200:600].mean() + 0.3)
        self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_fit_save_load_roundtrip(self) -> None:
        X, names = D.features(synthetic())
        y = np.zeros(len(X))
        y[1000:2000] = 1
        model = D.BinnedLogistic.fit(X, y, names, n_bins=4, steps=10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            model.save(path)
            again = D.BinnedLogistic.load(path)
        np.testing.assert_allclose(model.predict(X), again.predict(X))
        self.assertGreater(model.predict(X)[1200:1800].mean(), 0.8)


if __name__ == "__main__":
    unittest.main()
