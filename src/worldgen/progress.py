from __future__ import annotations

from dataclasses import dataclass
import re
import sys
import time
from typing import Callable, TextIO

_STAGE_RE = re.compile(r"^\[(?P<name>[^]]+)] (?P<event>starting|done(?: in (?P<seconds>[0-9.]+)s)?)$")


def expected_pipeline_stages(config, *, include_output: bool = True) -> int:
    macro = max(1, int(config.simulation.earth_system_passes))
    final = max(1, int(config.simulation.final_climate_ocean_passes))
    return 4 + 5 * macro + 2 * final + 6 + (1 if include_output else 0)


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--:--"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@dataclass(slots=True, frozen=True)
class ProgressSnapshot:
    stage: str | None
    completed: int
    total: int
    fraction: float
    elapsed_seconds: float
    eta_seconds: float | None


class StageProgress:
    """Callable stage progress sink compatible with ``WorldPipeline(progress=...)``."""

    def __init__(
        self,
        total_stages: int,
        *,
        enabled: bool = True,
        stream: TextIO | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.total = max(1, int(total_stages))
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.log = log
        self.started_at = time.perf_counter()
        self.completed = 0
        self.current_stage: str | None = None
        self._durations: list[float] = []
        self._last_width = 0

    def __call__(self, message: str) -> None:
        if self.log is not None:
            self.log(message)
        match = _STAGE_RE.match(message.strip())
        if match:
            name = match.group("name")
            event = match.group("event")
            if event == "starting":
                self.current_stage = name
            else:
                seconds = match.group("seconds")
                if seconds is not None:
                    self._durations.append(float(seconds))
                self.completed = min(self.total, self.completed + 1)
                self.current_stage = None
        if self.enabled:
            self._render(message)

    def snapshot(self) -> ProgressSnapshot:
        elapsed = time.perf_counter() - self.started_at
        fraction = min(1.0, self.completed / self.total)
        remaining = self.total - self.completed
        eta: float | None = None
        if self._durations and remaining > 0:
            recent = self._durations[-5:]
            recent_mean = sum(recent) / len(recent)
            global_mean = sum(self._durations) / len(self._durations)
            eta = remaining * (0.65 * recent_mean + 0.35 * global_mean)
        elif self.completed > 0 and remaining > 0:
            eta = elapsed / self.completed * remaining
        elif remaining == 0:
            eta = 0.0
        return ProgressSnapshot(self.current_stage, self.completed, self.total, fraction, elapsed, eta)

    def _render(self, message: str) -> None:
        s = self.snapshot()
        width = 28
        done = min(width, int(round(width * s.fraction)))
        bar = "#" * done + "-" * (width - done)
        stage = s.stage or message.strip()
        text = (
            f"[{bar}] {100*s.fraction:6.2f}% "
            f"stage {s.completed:02d}/{s.total:02d} "
            f"elapsed {format_duration(s.elapsed_seconds)} "
            f"ETA {format_duration(s.eta_seconds)} | {stage}"
        )
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        if is_tty:
            pad = " " * max(0, self._last_width - len(text))
            self.stream.write("\r" + text + pad)
            self.stream.flush()
            self._last_width = len(text)
            if self.completed >= self.total:
                self.stream.write("\n")
        else:
            self.stream.write(text + "\n")
            self.stream.flush()

    def finish(self) -> None:
        if not self.enabled:
            return
        if bool(getattr(self.stream, "isatty", lambda: False)()) and self._last_width:
            self.stream.write("\n")
            self.stream.flush()
            self._last_width = 0
