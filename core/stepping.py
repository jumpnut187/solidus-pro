"""Physics-agnostic time-stepping support shared by integration loops.

Extracted from ``core/solidification_loop.py``, where the same logic appeared
three times, once per integration path (thermal-coupled, schedule-based and
front-based).

Deliberately narrow. This module holds only what more than one caller needs
*today*; the timestep-policy and checkpointing abstractions a CFD filling solver
will want are not here, because building an interface with no second
implementation tends to fit the one caller that exists rather than the one that
arrives. Add them when the filling solver can shape them.
"""

from __future__ import annotations

from typing import Any, Optional


def should_capture_fraction_history(engine: Any) -> bool:
    """Whether the loop should record fraction-solid snapshots.

    Prefers the engine's own ``_should_capture_fraction_history`` and falls back
    to the analysis flags that imply a need for time history. The fallback
    existed verbatim in three places in the integration loop, all of them
    duplicating the engine method they were guarding against being absent.
    """
    capture_fn = getattr(engine, "_should_capture_fraction_history", None)
    if callable(capture_fn):
        return bool(capture_fn())

    options = engine.analysis_options
    return bool(
        options.fraction_liquid
        or options.hot_tear
        or options.shrinkage
        or options.feeding_path
    )


class SnapshotSchedule:
    """Decides when to record an output frame, from a progress percentage.

    Solver steps and output frames are separate concerns: the loop may take
    thousands of steps but only ~100 frames are kept for playback. The cadence
    is driven by *progress*, not step count, so a run that converges quickly and
    one that grinds both produce a comparably sampled history.

    Args:
        frames: Target number of frames across a 0-100 % run.
    """

    def __init__(self, frames: int = 100) -> None:
        if frames <= 0:
            raise ValueError(f"frames must be positive, got {frames}")
        self.frames = int(frames)
        self.interval = 100.0 / float(frames)
        # None rather than a negative sentinel: the original loop used -1.0,
        # which only makes the first call capture while the interval happens to
        # be <= 1.0. At frames=10 the interval is 10.0 and the opening frame
        # would have been dropped.
        self._last_progress: Optional[float] = None

    def should_capture(self, progress_pct: float) -> bool:
        """Return True when ``progress_pct`` has advanced a full interval.

        The first call always captures, so a run always has an opening frame.
        Records the capture point as a side effect, so call it once per step.
        """
        if self._last_progress is not None:
            if progress_pct - self._last_progress < self.interval:
                return False
        self._last_progress = float(progress_pct)
        return True

    def reset(self) -> None:
        """Forget the last capture point, so the next call captures."""
        self._last_progress = None
