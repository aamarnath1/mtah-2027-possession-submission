# Match viewer

A local tool for watching the match with our data on top of it, so we can see what our
predictions are doing.

- **Left:** the tactical video with the tracking drawn over it. Players are coloured by team and
  the ball is a yellow ring. A faint outline marks the pitch edge, which shows whether the
  homography fits that frame.
- **Right:** a digital pitch (105.3 × 68 m) with every tracked player and the ball. It also shows
  a short ball trail and dashed lines from the ball to the nearest player on each team.
- **Bottom:** timelines for two label files (A and B), the Smart Tagging phases and set pieces,
  with graded frames shaded. There's one for the whole match and a zoomed one around the playhead.
- **Prediction layer (P):** with a Labels A file picked, a banner on the video shows the call
  (team name and H/A, dead ball, or no answer), how long the current run has lasted, and Labels B
  with agree/differ. The predicted team's players get a white ring and the other team is dimmed
  (everyone is dimmed on a dead ball), and the video border takes the call's colour. Pick
  `data/predictions/evidence.csv` as Labels A and `b6.csv` as Labels B to watch where they differ.
- **Info cards:** the frame and time, both labels and how long each one's current run lasts,
  whether A and B agree, ball and player counts, nearest distances and active tags.

Everything runs on your machine. The server only listens on `127.0.0.1`.

## Run it

All commands run from the repo root with the venv active.

**1. Prepare the data.** The viewer reads the verified preprocessing output in
`data/preprocessing/current/`, so it shows exactly what the model reads. If you haven't
published it yet, run the rebuild first (about 6 minutes):

```bash
python scripts/rebuild_preprocessing.py
```

Then build the viewer's data. It writes `data/viewer/`, which is gitignored. Rerun it whenever
`current/` changes.

```bash
python tools/viewer/prepare_viewer_from_current.py
```

**2. Put the tactical video in place.** It isn't in `data/raw/` yet. Copy
`BAR-ATH_tactical.mp4` from the competition bundle to `data/raw/tactical/`. Without it, the
pitch view and timelines still work.

**3. Start the server:**

```bash
python tools/viewer/serve_viewer.py
```

**4. Open** http://127.0.0.1:8765/ in Chrome or Safari.

## Label files

The viewer lists every `*.csv` in `submissions/` and `data/predictions/`. Both are gitignored.
Pick one as **Labels A** and another as **Labels B** to compare them, for example a new model
against the baseline. The format is the submission format:

```
frame,label
0,X
1,X
...
```

`H` / `A` / `D` / `X`, with 0-based video frames. To point at other folders, use
`--labels <folder>` (repeatable). Press ↻ to re-scan after writing a new file.

## Controls

| Key | Action |
|---|---|
| Space | Play / pause |
| ← / → | ±1 frame |
| Shift + ← / → | ±1 second |
| ↓ / ↑ | ±10 seconds |
| `,` / `.` | Jump to the previous / next change in Labels A (good for spotting flicker) |
| `[` / `]` | Slower / faster (0.25× to 8×) |
| O · N · T · L · P | Overlay · slot numbers · ball trail · nearest-player lines · prediction |
| G | Go to frame |
| Click or drag a timeline | Seek. Scroll on the zoomed timeline to change its width. |

The viewer remembers your label choices and position between reloads (stored in the browser only).

## How it lines up

- Every frame number is the **0-based video frame**: the same as submission rows, the homography
  `frame`, and Smart Tagging `round(seconds × 25)`. The preprocessing has already applied the
  shifts, so the viewer just uses its `frame` column.
- The homography is the one input not covered by the preprocessing, so it still comes from the raw
  CSV in `data/raw/tactical/`.
- The video's current frame comes from `requestVideoFrameCallback` (`round(mediaTime × 25)`), so
  the overlay updates with each frame actually shown.
- **Sync → Video offset** adds a fixed number of frames between the video and the data. It should
  stay at 0. If the dots ever trail the players by a constant amount, this is where to check.

## Files

| File | What it does |
|---|---|
| `prepare_viewer_from_current.py` | **The data step.** `data/preprocessing/current/` tables (tracking, slot teams, tags) plus the raw homography CSV → compact binaries in `data/viewer/` |
| `prepare_viewer_data.py` | **Not used anymore.** The older version that reads the raw files directly. Kept as a fallback and an independent cross-check: it produces identical binaries. |
| `serve_viewer.py` | Local server (standard library only) with HTTP Range support so the video can seek |
| `static/` | The page: `index.html`, `viewer.css`, `viewer.js` (no external libraries, works offline) |

`data/viewer/` contains `meta.json`, `positions.f32` (frames × 47 × 2, ball first), `ball_speed.f32`,
`homography.f64` (frames × 9) and `tags.json`. It's about 90 MB in total.

## Data rules

- The prepared data, the video and anything you screenshot from the viewer are **match data**. Keep
  them local and never paste them into a cloud LLM.
- No hand-labelling from what you see. The viewer is for checking and debugging our automatic
  output, not for making labels.
