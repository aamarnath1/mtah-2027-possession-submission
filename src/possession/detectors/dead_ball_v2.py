"""Dead ball v2: a learned stoppage detector, multi-scale, mostly ball-free.

Version 1 (dead_ball.py) thresholds one signal, the smoothed share of slow
players, at one time scale. On IDSSE with the ball hidden like ours it is right
on 83% of frames but finds only 65% of dead frames, misses about one stoppage in
seven outright, and breaks the rest into pieces.

Version 2, held out one IDSSE match at a time with the ball hidden like ours, is
right on 93% of frames and finds 96% of dead frames (precision 90%). Inside the
hybrid decoder it lifts possession-spell F1 from 0.363 to 0.432 (B6: 0.350),
better on all four matches.

It follows Lang et al. (Scientific Reports 2022), who reach 92% frame
accuracy from player movement alone by looking at several time scales at once:

  player movement   share of players under 1 and under 2 m/s, mean speed and
                    the 90th percentile speed, each averaged over centred
                    windows of 0.4, 1, 2, 4, 8 and 20 seconds
  change            mean speed over the next W seconds minus the last W
                    (everyone stops at a whistle; everyone moves at a restart)
  shape             how spread out all players are, and the share of players
                    inside either penalty area (corners, free kicks near goal)
  ball, when seen   off the pitch, speed, distance to the nearest line, and
                    whether it is seen at all; NaN otherwise

Model: logistic regression on quantile-binned features (one weight per bin, an
additive model, so every feature can bend however the data says). NaN gets its
own bin, so "ball not seen" is learned rather than imputed. Fitted on public
IDSSE matches only (scripts/fit_dead_ball_v2.py), never on ours. The fitted model
is stored as JSON next to this file.

Inputs: the arrays from early_confirmation.arrays_from_table() or
external.idsse_arrays (ball_xy, ball_obs, slot_xy in metres, x from 0..pitch_x_m).
"""

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np

MODEL_PATH = Path(__file__).with_name("dead_ball_v2_model.json")
THRESHOLD = 0.4
"""Dead when the probability is at least this. Every leave-one-match-out fold on
IDSSE picked 0.4 (scripts/benchmark_dead_v2.py), out of 0.3 to 0.8."""
WINDOWS_S = (0.4, 1.0, 2.0, 4.0, 8.0, 20.0)
CHANGE_S = (2.0, 4.0)
BOX_DEPTH_M = 16.5
BOX_WIDTH_M = 40.3


def centred_mean(values: np.ndarray, half: int) -> np.ndarray:
    """NaN-aware centred moving average over +/- half frames."""
    n = len(values)
    if half <= 0:
        return values.astype(float)
    ok = np.isfinite(values)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, values, 0.0))])
    cn = np.concatenate([[0.0], np.cumsum(ok)])
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    cnt = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan)


def window_mean(values: np.ndarray, start_off: int, end_off: int) -> np.ndarray:
    """NaN-aware mean over frames [t + start_off, t + end_off), clipped at the edges."""
    n = len(values)
    ok = np.isfinite(values)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, values, 0.0))])
    cn = np.concatenate([[0.0], np.cumsum(ok)])
    lo = np.clip(np.arange(n) + start_off, 0, n)
    hi = np.clip(np.arange(n) + end_off, 0, n)
    cnt = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan)


def player_speeds(slot_xy: np.ndarray, fps: float) -> np.ndarray:
    """(n, slots) speed in m/s from consecutive positions; NaN where untracked."""
    speed = np.full(slot_xy.shape[:2], np.nan)
    step = np.diff(slot_xy, axis=0)
    speed[1:] = np.sqrt((step ** 2).sum(axis=2)) * fps
    speed[0] = speed[1]
    return speed


def features(arrays: dict, pitch_x_m: float = 105.3, pitch_y_m: float = 68.0,
             fps: float = 25.0) -> tuple[np.ndarray, list[str]]:
    """(n, k) feature matrix and the feature names."""
    slot_xy = np.asarray(arrays["slot_xy"], dtype=float)
    ball_xy = np.asarray(arrays["ball_xy"], dtype=float)
    seen = np.asarray(arrays["ball_obs"], dtype=bool) & np.isfinite(ball_xy).all(axis=1)
    speed = player_speeds(slot_xy, fps)
    tracked = np.isfinite(speed)
    n_tracked = tracked.sum(axis=1)
    has = n_tracked > 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        frame = {
            "slow1": np.where(has, (tracked & (speed < 1.0)).sum(1) / np.maximum(n_tracked, 1), np.nan),
            "slow2": np.where(has, (tracked & (speed < 2.0)).sum(1) / np.maximum(n_tracked, 1), np.nan),
            "mean_speed": np.nanmean(speed, axis=1),
            "p90_speed": np.nanpercentile(speed, 90, axis=1),
        }
        x, y = slot_xy[..., 0], slot_xy[..., 1]
        cx, cy = np.nanmean(x, axis=1), np.nanmean(y, axis=1)
        spread = np.nanmean(np.hypot(x - cx[:, None], y - cy[:, None]), axis=1)
        in_y = np.abs(y - pitch_y_m / 2) <= BOX_WIDTH_M / 2
        in_low = (x <= BOX_DEPTH_M) & in_y
        in_high = (x >= pitch_x_m - BOX_DEPTH_M) & in_y
        pos_tracked = np.isfinite(x).sum(axis=1)
        in_box = np.where(pos_tracked > 0, np.maximum(in_low.sum(1), in_high.sum(1))
                          / np.maximum(pos_tracked, 1), np.nan)

    cols, names = [], []
    for name, values in frame.items():
        for w in WINDOWS_S:
            cols.append(centred_mean(values, int(round(w * fps))))
            names.append(f"{name}_{w:g}s")
    for w in CHANGE_S:
        k = int(round(w * fps))
        cols.append(window_mean(frame["mean_speed"], 0, k) - window_mean(frame["mean_speed"], -k, 0))
        names.append(f"speed_change_{w:g}s")
    for name, values in (("spread", spread), ("in_box", in_box)):
        for w in (2.0, 8.0):
            cols.append(centred_mean(values, int(round(w * fps))))
            names.append(f"{name}_{w:g}s")

    # Ball terms, NaN where the ball is not seen.
    bx = np.where(seen, ball_xy[:, 0], np.nan)
    by = np.where(seen, ball_xy[:, 1], np.nan)
    off = np.where(seen, ((bx < -0.11) | (bx > pitch_x_m + 0.11)
                          | (by < -0.11) | (by > pitch_y_m + 0.11)).astype(float), np.nan)
    line = np.minimum(np.minimum(bx, pitch_x_m - bx), np.minimum(by, pitch_y_m - by))
    bspeed = np.full(len(bx), np.nan)
    bspeed[2:-2] = np.hypot(bx[4:] - bx[:-4], by[4:] - by[:-4]) * fps / 4
    for name, values in (("ball_off", off), ("ball_line_m", line), ("ball_speed", bspeed)):
        for w in (0.0, 1.0, 4.0):
            cols.append(centred_mean(values, int(round(w * fps))))
            names.append(f"{name}_{w:g}s")
    for w in (1.0, 4.0):
        cols.append(centred_mean(seen.astype(float), int(round(w * fps))))
        names.append(f"ball_seen_{w:g}s")
    return np.column_stack(cols), names


@dataclass
class BinnedLogistic:
    """Logistic regression on quantile bins; NaN is its own bin per feature."""

    edges: list            # per feature: increasing bin edges (interior)
    weights: np.ndarray    # bias, then one weight per bin (incl. NaN bin) per feature
    names: list

    def design(self, X: np.ndarray) -> np.ndarray:
        blocks = []
        for j, e in enumerate(self.edges):
            col = X[:, j]
            idx = np.searchsorted(np.asarray(e), col, side="right")
            idx = np.where(np.isfinite(col), idx, len(e) + 1)        # NaN bin last
            block = np.zeros((len(col), len(e) + 2), dtype=np.float32)
            block[np.arange(len(col)), idx] = 1.0
            blocks.append(block)
        return np.hstack([np.ones((len(X), 1), dtype=np.float32)] + blocks)

    def predict(self, X: np.ndarray, chunk: int = 50_000) -> np.ndarray:
        out = np.empty(len(X))
        for s in range(0, len(X), chunk):
            z = self.design(X[s:s + chunk]) @ self.weights
            out[s:s + chunk] = 1.0 / (1.0 + np.exp(-z))
        return out

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, names: list, n_bins: int = 12,
            l2: float = 1e-2, steps: int = 30) -> "BinnedLogistic":
        edges = []
        for j in range(X.shape[1]):
            col = X[:, j][np.isfinite(X[:, j])]
            q = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1])) if len(col) else []
            edges.append([float(v) for v in q])
        model = cls(edges, np.zeros(1), names)
        D = model.design(X).astype(np.float64)
        w = np.zeros(D.shape[1])
        reg = np.full(D.shape[1], l2)
        reg[0] = 0.0
        for _ in range(steps):
            p = 1.0 / (1.0 + np.exp(-(D @ w)))
            g = D.T @ (p - y) / len(y) + reg * w
            H = (D * (p * (1 - p))[:, None]).T @ D / len(y) + np.diag(reg)
            w -= np.linalg.solve(H, g)
        model.weights = w
        return model

    def save(self, path: Path = MODEL_PATH, meta: dict | None = None) -> None:
        Path(path).write_text(json.dumps({"names": self.names, "edges": self.edges,
                                          "weights": self.weights.tolist(),
                                          "meta": meta or {}}, indent=0))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "BinnedLogistic":
        d = json.loads(Path(path).read_text())
        return cls(d["edges"], np.asarray(d["weights"]), d["names"])


def detect(arrays: dict, model: BinnedLogistic | None = None,
           pitch_x_m: float = 105.3, pitch_y_m: float = 68.0) -> np.ndarray:
    """Per-frame probability that play is dead."""
    model = model or BinnedLogistic.load()
    X, names = features(arrays, pitch_x_m, pitch_y_m)
    if names != model.names:
        raise ValueError("feature list does not match the fitted model; refit it")
    return model.predict(X)
