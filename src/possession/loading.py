"""Load the clean per-frame table as compact numpy arrays, cached locally."""
import json
import numpy as np
import pandas as pd
from possession.paths import DATA_DIR, CURRENT_DIR, current_file

N_SLOTS = 46
CACHE = DATA_DIR / "cache" / "arrays.npz"      # under data/, so gitignored

def load_arrays(use_cache: bool = True) -> dict:
    if use_cache and CACHE.is_file():
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}

    cols = ["frame", "ball_x_m", "ball_y_m", "ball_speed_mps", "ball_observed"]
    for s in range(1, N_SLOTS + 1):
        cols += [f"slot_{s}_x_m", f"slot_{s}_y_m", f"slot_{s}_speed_mps"]
    df = pd.read_csv(current_file("aligned/frames_with_tags.csv"), usecols=cols)
    assert (df["frame"].to_numpy() == np.arange(len(df))).all(), "frames not 0..n-1"

    slots = sorted(json.loads((CURRENT_DIR / "track_slots.json").read_text("utf-8")),
                   key=lambda r: r["slot"])
    out = {
        "ball_xy": df[["ball_x_m", "ball_y_m"]].to_numpy(),            # (n, 2) metres, NaN if unseen
        "ball_speed": df["ball_speed_mps"].to_numpy(),                  # (n,)
        "ball_obs": df["ball_observed"].to_numpy().astype(bool),        # (n,)
        "slot_xy": np.stack([df[[f"slot_{s}_x_m", f"slot_{s}_y_m"]].to_numpy()
                             for s in range(1, N_SLOTS + 1)], axis=1),  # (n, 46, 2)
        "slot_speed": np.stack([df[f"slot_{s}_speed_mps"].to_numpy()
                                for s in range(1, N_SLOTS + 1)], axis=1),
        "slot_team": np.array([r["team"] for r in slots]),              # (46,) 'H' / 'A'
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out