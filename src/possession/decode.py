"""Turning per-frame evidence into a label sequence.

This is the only place that decides a possession changed, and it is where most of
the score is won or lost.

Why this matters more than it looks. The ranking metric scores whole possessions,
matched to the truth by temporal overlap. The official baselines differ only in
one smoothing parameter and span 0.0486 to 0.2433 on that metric while their
per-frame scores sit between 0.636 and 0.675. Same detector, same evidence, a
factor of five in score, decided entirely here.

Two rules the published work on this kind of metric is firm about, and both are
cheap:

  Smooth with a CENTERED window, never a trailing one. A trailing filter of width
  K lags every boundary by about K/2 in the same direction. That is a uniform
  shift, and a uniform shift costs roughly twice what a one-sided boundary error
  costs, because it moves both edges of a possession the same way. We decode the
  whole match offline, so there is no reason to accept any lag at all.

  Then delete runs that are too short. A spurious short run is worse than a missed
  one: it is a false positive AND it splits the possession it landed in, so one
  blip costs two ways. Published ablations put this at several points of segment
  score for no loss of frame accuracy.
"""

from possession.labels import ABSTAIN, AWAY, DEAD, HOME


def smooth(labels, window, lo, hi):
    """Majority vote over a centered window, applied to frames lo..hi inclusive.

    `window` is the full width in frames. Counts come from per-label prefix sums
    rather than an incrementally maintained counter: the incremental version is
    easy to get subtly wrong at the window edges, and a wrong count here silently
    invents possessions.
    """
    if window <= 1:
        return labels
    half = window // 2
    n = len(labels)
    classes = (HOME, AWAY, DEAD)

    prefix = {}
    for c in classes:
        acc = [0] * (n + 1)
        for i, lab in enumerate(labels):
            acc[i + 1] = acc[i] + (lab == c)
        prefix[c] = acc

    out = list(labels)
    for i in range(lo, hi + 1):
        a, b = max(0, i - half), min(n, i + half + 1)
        best, best_n = None, 0
        for c in classes:
            count = prefix[c][b] - prefix[c][a]
            # ties go to whatever this frame already says, which keeps the
            # vote from nudging a boundary for no reason
            if count > best_n or (count == best_n and c == labels[i]):
                best, best_n = c, count
        if best_n:
            out[i] = best
    return out


def runs(labels, lo, hi):
    """Contiguous same-label stretches within lo..hi, as (start, end, label)."""
    out = []
    start = lo
    for i in range(lo + 1, hi + 2):
        if i > hi or labels[i] != labels[start]:
            out.append((start, i - 1, labels[start]))
            start = i
    return out


def drop_short_runs(labels, min_frames, lo, hi):
    """Absorb any run shorter than `min_frames` into the run before it.

    `min_frames` is per class, keyed by label, because the three classes have very
    different duration distributions. Possessions run seconds; a real dead-ball
    stretch runs much longer; and the truth genuinely contains possessions under
    three seconds, so this cuts both ways and must not be set aggressively.
    """
    out = list(labels)
    changed = True
    while changed:
        changed = False
        current = runs(out, lo, hi)
        for index, (start, end, label) in enumerate(current):
            need = min_frames.get(label, 0)
            if end - start + 1 >= need:
                continue
            if start > lo:
                fill = out[start - 1]            # absorb backwards, the usual case
            elif index + 1 < len(current):
                fill = current[index + 1][2]     # head run: absorb forwards instead
            else:
                continue                         # the whole window is one short run
            for i in range(start, end + 1):
                out[i] = fill
            changed = True
            break
    return out


def decode(labels, window=51, min_frames=None, windows=None):
    """Smooth, then suppress short runs, within each graded window separately.

    Defaults are deliberately conservative placeholders. They must be fitted on
    external matches before we trust them, never on this one: tuning on the public
    five minutes is against the rules, and it also picks wrong. The organizers'
    own smooth125 beats smooth50 on that window and loses on the full match.
    """
    from possession.labels import GRADED_WINDOWS

    if min_frames is None:
        min_frames = {HOME: 25, AWAY: 25, DEAD: 25}
    out = list(labels)
    for lo, hi in (windows or GRADED_WINDOWS):
        out = smooth(out, window, lo, hi)
        out = drop_short_runs(out, min_frames, lo, hi)
    return out


def count_possessions(labels):
    """How many possessions we are claiming. A sanity number, not a score.

    A possession is a maximal run of H or of A; D splits them and is not one. The
    public five minutes hold 17 across 7,500 frames, so a full match plausibly
    holds a few hundred. Thousands means the decoder is flickering.
    """
    from possession.labels import GRADED_WINDOWS

    total = 0
    for lo, hi in GRADED_WINDOWS:
        total += sum(1 for _, _, label in runs(labels, lo, hi) if label in (HOME, AWAY))
    return total
