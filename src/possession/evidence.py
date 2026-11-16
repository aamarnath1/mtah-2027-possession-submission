"""The shared format every detector hands to the fusion step and the decoder.

Four arrays, one value per video frame:

    p_home      chance the home team (Barcelona) has the ball. NaN means "no opinion".
    p_away      chance the away team has the ball. NaN means "no opinion".
    p_dead      chance the ball is dead (out of play). NaN means "no opinion".
    confidence  how much to trust this detector on this frame, from 0 to 1.
                0 means "ignore me". It is never NaN.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Evidence:
    """One detector's opinion for every frame."""

    p_home: np.ndarray
    p_away: np.ndarray
    p_dead: np.ndarray
    confidence: np.ndarray

    @classmethod
    def empty(cls, n_frames: int) -> "Evidence":
        """No opinion anywhere: all NaN, confidence 0."""
        return cls(np.full(n_frames, np.nan), np.full(n_frames, np.nan),
                   np.full(n_frames, np.nan), np.zeros(n_frames))

    def validate(self) -> None:
        """Raise ValueError if the arrays break the contract."""
        n_frames = len(self.confidence)
        for name in ("p_home", "p_away", "p_dead", "confidence"):
            values = np.asarray(getattr(self, name))
            if values.shape != (n_frames,):
                raise ValueError(f"{name} must have shape ({n_frames},), got {values.shape}")
            known = values[~np.isnan(values)]
            if ((known < 0) | (known > 1)).any():
                raise ValueError(f"{name} has values outside [0, 1]")
        if np.isnan(self.confidence).any():
            raise ValueError("confidence must never be NaN (use 0 for no opinion)")
        both = ~np.isnan(self.p_home) & ~np.isnan(self.p_away)
        if not np.allclose(self.p_home[both] + self.p_away[both], 1.0):
            raise ValueError("where both are given, p_home + p_away must equal 1")

    def save(self, path: Path) -> None:
        """Write to a compressed .npz file (create the folder if needed)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, p_home=self.p_home, p_away=self.p_away,
                            p_dead=self.p_dead, confidence=self.confidence)

    @classmethod
    def load(cls, path: Path) -> "Evidence":
        data = np.load(Path(path))
        return cls(data["p_home"], data["p_away"], data["p_dead"], data["confidence"])
