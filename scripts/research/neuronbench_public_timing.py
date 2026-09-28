"""Reconstruct the published deterministic stimulus timing without simulator access."""
from __future__ import annotations

import numpy as np


SAMPLES_PER_MS = 10  # archived voltage is subsampled from 0.01 ms to 0.1 ms
BASE_MS = 20.0
TAIL_MS = 80.0


def current_and_test_start(segments) -> tuple[np.ndarray, int]:
    """Return sampled current and start of the final contiguous positive run.

    The public benchmark's deterministic protocol counts the full trace when
    no positive-current segment exists. This matters for release-only trials.
    """
    pieces = [(BASE_MS, 0.0), *((float(d), float(a)) for d, a in segments),
              (TAIL_MS, 0.0)]
    if any(not np.isfinite(d) or not np.isfinite(a) or d <= 0 for d, a in pieces):
        raise ValueError('invalid protocol segment')
    currents = []
    cursor = 0
    run_start = None
    start = 0
    for duration, amplitude in pieces:
        count = max(int(round(duration * SAMPLES_PER_MS)), 1)
        currents.append(np.full(count, amplitude, dtype=float))
        if amplitude > 0:
            if run_start is None:
                run_start = cursor
            start = run_start
        else:
            run_start = None
        cursor += count
    return np.concatenate(currents), start
