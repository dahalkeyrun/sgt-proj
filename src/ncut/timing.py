from __future__ import annotations

import time
from typing import Dict


class StageTimer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._initial_start = time.perf_counter() if enabled else None
        self._start = self._initial_start
        self._stages: Dict[str, float] = {}

    def record(self, stage_name: str) -> None:
        if not self.enabled:
            return

        now = time.perf_counter()
        if self._start is None:
            self._start = now
            self._initial_start = now

        elapsed_ms = (now - self._start) * 1000.0
        self._stages[stage_name] = round(elapsed_ms, 1)
        self._start = now

    def get_timings(self) -> Dict[str, float]:
        return dict(self._stages)

    def finish(self, stage_name: str = "total_ms") -> Dict[str, float]:
        if not self.enabled:
            return {}

        if self._initial_start is None:
            return {}

        now = time.perf_counter()
        elapsed_ms = (now - self._initial_start) * 1000.0
        self._stages[stage_name] = round(elapsed_ms, 1)
        return self.get_timings()
