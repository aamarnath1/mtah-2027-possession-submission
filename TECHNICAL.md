# Catalanian Compliers · Possession detection

## Approach

We infer possessions with the hybrid Evidence decoder. For each half, Viterbi
selects a sequence of home, away, and dead-ball labels. The sequence pays a cost
of 8 for each direct switch between teams. A 25-frame cleanup removes brief
label runs and the output covers the complete tactical-video frame index.

## Inputs and detectors

Metrica's tactical ATD files provide time-aligned ball and player coordinates.
Missing observations stay missing. Ball-to-team evidence compares proximity
and uses speed computed from observed ball positions. Team shape uses player
position, movement, and spread, and gives a lower-weight vote (0.25) when
resolving competing signals. Its independence from ball position helps when
the ball disappears. A binned logistic dead-ball model uses player movement,
ball geometry, and tracking availability. The decoder uses a 0.40 dead-ball
threshold and fixes sustained clear ball-control stretches of 12 or more frames
to the corresponding team. The preprocessing also reads Smart Tagging to
check the frame alignment. Tags do not vote in the submitted decoder.

## Training and evaluation data

Team-shape and dead-ball settings use four public IDSSE matches from the
2022/23 German professional leagues, J03WOH, J03WOY, J03WQQ, and J03WR9.
The DFL dataset supplies ball status and team possession labels. We fitted
and evaluated with whole matches held out and used simulated ball gaps to
stress-test the approaches. No labeled frames from the competition match
were used for training or parameter tuning. The public competition excerpt
was reserved for pipeline checks.

The official possession-spell F1 averaged 0.432 on the held-out Bundesliga
matches under the competition-style gap simulation. Under that benchmark,
the B6 rule-based decoder scored 0.350 and the earlier hybrid with the
original dead-ball detector scored 0.363. These are external validation
results. The full competition score is unknown.

IDSSE source and dataset credit: Deutsche Fussball Liga and Bassek, Rein,
Weber, and Memmert (2025), [data release](https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177)
and [dataset paper](https://doi.org/10.1038/s41597-025-04505-y), CC BY 4.0.

## Possession rules

Control is inferred from proximity and sustained observed evidence. The
switch penalty favors continuity through brief challenges and loose balls.
Predicted stoppages are labeled D and play resumes with the side favored by
the detector evidence. The submitted decoder has no separate restart-taker
override. It emits X outside the competition's graded windows.

## Limitations

Ball gaps and the uncertainty of control can blur possession boundaries.
Team shape can be misleading for teams playing deep or during transitions.
Errors in the dead-ball mask can join or split otherwise valid possessions.
The IDSSE sample is small, and real tactical gaps can differ from simulated
ones. A throw-in timing and attribution prototype provided no spellF1 gain
over the combined decoder in a paired external test and was left out.

## Reproduction

Install the pinned requirements and initialize the organizer submodule, as
described in the README. With the organizer's tactical inputs in place, run:

```sh
python scripts/rebuild_preprocessing.py
python scripts/make_submission.py --evidence --out "submissions/Catalanian Compliers.csv"
python vendor/mtah/evaluation/validate_submission.py "submissions/Catalanian Compliers.csv"
```

The preprocessing checks two independent builds for matching output hashes.
The submission script uses fixed fitted settings and generates 199,319 data rows.
