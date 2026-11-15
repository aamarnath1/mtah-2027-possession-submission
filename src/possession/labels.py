"""The submission format: what a label array is, and how it becomes the CSV.

Nothing here models anything. It is the contract between whatever produces labels
and the file we hand in, kept in one place so an off-by-one can only happen once.

Reference: vendor/mtah/SUBMISSION.md and vendor/mtah/CHALLENGE.md.
"""

from pathlib import Path

N_FRAMES = 199_319
"""Rows in the submission, one per frame of the TACTICAL video.

The broadcast feed has 164,488 frames and is a different timeline. A file with
that many rows means the wrong feed was indexed.
"""

FPS = 25

GRADED_WINDOWS = ((11_185, 90_651), (110_808, 190_197))
"""Inclusive frame ranges the scorer looks at. 158,857 frames in total.

Everything outside is ignored, so a label there costs nothing and gains nothing.
"""

SECOND_HALF_KICKOFF = 113_742
"""The second half is graded from 110,808 but does not kick off until here.

Those 2,934 frames are graded and every one of them is D. That is a rule, not a
prediction, so we write them rather than hoping a decoder rediscovers them.
"""

HOME, AWAY, DEAD, ABSTAIN = "H", "A", "D", "X"


def empty_labels():
    """All-X, the correct starting point. X means "no answer", not "dead"."""
    return [ABSTAIN] * N_FRAMES


def graded_frames():
    """Every frame index the scorer will look at."""
    out = []
    for lo, hi in GRADED_WINDOWS:
        out.extend(range(lo, hi + 1))
    return out


def count_graded(labels):
    """How many graded frames carry a real answer. Should be all 158,857."""
    total = 0
    for lo, hi in GRADED_WINDOWS:
        total += sum(1 for i in range(lo, hi + 1) if labels[i] != ABSTAIN)
    return total


def apply_known_dead(labels):
    """Stamp the frames that are dead by rule rather than by detection."""
    start = GRADED_WINDOWS[1][0]
    for i in range(start, SECOND_HALF_KICKOFF):
        labels[i] = DEAD
    return labels


def fill_abstentions(labels):
    """Replace X inside the graded windows by carrying the previous label.

    The scorer forward-fills X before extracting possessions anyway, so an X on a
    graded frame is never better than writing the previous label ourselves, and it
    is strictly worse for the per-frame tiebreak, which counts X as a miss.
    """
    for lo, hi in GRADED_WINDOWS:
        last = None
        for i in range(lo, hi + 1):
            if labels[i] != ABSTAIN:
                last = labels[i]
            elif last is not None:
                labels[i] = last
        # a run of X at the head of a half has nothing behind it: back-fill instead
        first = next((labels[i] for i in range(lo, hi + 1) if labels[i] != ABSTAIN), None)
        if first is not None:
            for i in range(lo, hi + 1):
                if labels[i] == ABSTAIN:
                    labels[i] = first
                else:
                    break
    return labels


def write_submission(labels, out_path):
    """Write `frame,label` and fail loudly rather than hand in a broken file.

    `frame` is redundant with the row number on purpose: the organizers use it to
    turn an off-by-one into a hard error instead of a silent 0.0 score.
    """
    if len(labels) != N_FRAMES:
        raise ValueError(f"expected {N_FRAMES} labels, got {len(labels)}")
    bad = {lab for lab in labels} - {HOME, AWAY, DEAD, ABSTAIN}
    if bad:
        raise ValueError(f"labels must be H, A, D or X; found {sorted(bad)}")
    for lo, hi in GRADED_WINDOWS:
        n_abstain = sum(1 for i in range(lo, hi + 1) if labels[i] == ABSTAIN)
        if n_abstain:
            raise ValueError(
                f"{n_abstain} abstentions inside graded window {lo}..{hi}. "
                "Run fill_abstentions() first; X is never better than the previous label.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        for i, lab in enumerate(labels):
            handle.write(f"{i},{lab}\n")
    return out_path
