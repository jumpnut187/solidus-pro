from __future__ import annotations

from typing import Any, Optional

def should_capture_fraction_history(engine: Any) -> bool:
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

    def __init__(self, frames: int = 100) -> None:
        if frames <= 0:
            raise ValueError(f"frames must be positive, got {frames}")
        self.frames = int(frames)
        self.interval = 100.0 / float(frames)
        self._last_progress: Optional[float] = None

    def should_capture(self, progress_pct: float) -> bool:
        if self._last_progress is not None:
            if progress_pct - self._last_progress < self.interval:
                return False
        self._last_progress = float(progress_pct)
        return True

    def reset(self) -> None:
        self._last_progress = None
