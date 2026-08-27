from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
import sys
import time
from typing import Any, Callable, TextIO

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


def _clock_after(seconds: float | None) -> str:
    if seconds is None or not (0.0 <= seconds < 10**9):
        return "--:--:--"
    return datetime.fromtimestamp(time.time() + seconds).strftime("%H:%M:%S")


@dataclass(slots=True, frozen=True)
class ProgressSnapshot:
    stage: str | None
    completed: int
    total: int
    fraction: float
    elapsed_seconds: float
    eta_seconds: float | None
    current_stage_elapsed_seconds: float = 0.0
    average_stage_seconds: float | None = None


class StageProgress:
    """Callable stage progress sink compatible with ``WorldPipeline(progress=...)``.

    In detailed mode the display includes process start time, current-stage runtime,
    observed mean stage duration, overall ETA, and estimated finish clock time. The
    public callable protocol remains unchanged for pipeline compatibility.
    """

    def __init__(
        self,
        total_stages: int,
        *,
        enabled: bool = True,
        stream: TextIO | None = None,
        log: Callable[[str], None] | None = None,
        detailed: bool = False,
    ) -> None:
        self.total = max(1, int(total_stages))
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.log = log
        self.detailed = bool(detailed)
        self.started_at = time.perf_counter()
        self.wall_started_at = time.time()
        self.completed = 0
        self.current_stage: str | None = None
        self._current_stage_started: float | None = None
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
                self._current_stage_started = time.perf_counter()
            else:
                seconds = match.group("seconds")
                if seconds is not None:
                    self._durations.append(float(seconds))
                elif self._current_stage_started is not None:
                    self._durations.append(time.perf_counter() - self._current_stage_started)
                self.completed = min(self.total, self.completed + 1)
                self.current_stage = None
                self._current_stage_started = None
        if self.enabled:
            self._render(message)

    def snapshot(self) -> ProgressSnapshot:
        now = time.perf_counter()
        elapsed = now - self.started_at
        fraction = min(1.0, self.completed / self.total)
        remaining = self.total - self.completed
        eta: float | None = None
        average: float | None = None
        if self._durations:
            recent = self._durations[-5:]
            recent_mean = sum(recent) / len(recent)
            global_mean = sum(self._durations) / len(self._durations)
            average = global_mean
            if remaining > 0:
                eta = remaining * (0.65 * recent_mean + 0.35 * global_mean)
        elif self.completed > 0 and remaining > 0:
            average = elapsed / self.completed
            eta = average * remaining
        elif remaining == 0:
            eta = 0.0
        current_elapsed = (
            now - self._current_stage_started if self._current_stage_started is not None else 0.0
        )
        return ProgressSnapshot(
            self.current_stage,
            self.completed,
            self.total,
            fraction,
            elapsed,
            eta,
            current_elapsed,
            average,
        )

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
        if self.detailed:
            text += (
                f" | current {format_duration(s.current_stage_elapsed_seconds)}"
                f" avg {format_duration(s.average_stage_seconds)}"
                f" started {datetime.fromtimestamp(self.wall_started_at).strftime('%H:%M:%S')}"
                f" finish~{_clock_after(s.eta_seconds)}"
            )
        self._write_line(text)

    def _write_line(self, text: str) -> None:
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


class RecursiveProgress:
    """Progress/ETA renderer for recursive refinement work.

    The callback consumes structured events from :class:`RefinementEngine`. ETA is
    estimated from observed per-field action durations and includes expected work in
    deeper requested levels using the configured branching factor. The current
    hierarchical node path is always preserved in the display and verbose log.
    """

    def __init__(
        self,
        *,
        planned_levels: int,
        branching_factor: int,
        enabled: bool = True,
        stream: TextIO | None = None,
        log: Callable[[str], None] | None = None,
        detailed: bool = False,
    ) -> None:
        self.planned_levels = max(1, int(planned_levels))
        self.branching_factor = max(1, int(branching_factor))
        self.enabled = bool(enabled)
        self.stream = stream or sys.stderr
        self.log = log
        self.detailed = bool(detailed)
        self.started_perf = time.perf_counter()
        self.started_wall = time.time()
        self.run_level_index = 0
        self.level_total = 0
        self.level_done = 0
        self.level_nodes = 0
        self.level_fields = 0
        self.path = "refinement"
        self.action = "starting"
        self._durations: list[float] = []
        self._last_width = 0

    def __call__(self, event: dict[str, Any]) -> None:
        kind = str(event.get("event", "work"))
        self.path = str(event.get("path", self.path))
        if kind == "level_start":
            self.run_level_index += 1
            self.level_total = max(1, int(event.get("total", 1)))
            self.level_done = 0
            self.level_nodes = int(event.get("nodes", 0))
            self.level_fields = int(event.get("fields", 0))
            res = event.get("resolution")
            self.action = f"level start resolution={res}"
        elif kind == "field_done":
            self.level_done = int(event.get("current", self.level_done + 1))
            seconds = event.get("seconds")
            if isinstance(seconds, (int, float)) and seconds >= 0:
                self._durations.append(float(seconds))
            self.action = (
                f"node {event.get('node_index','?')}/{event.get('node_total','?')} "
                f"field {event.get('field','?')}"
            )
        elif kind == "node_done":
            self.action = f"node complete {event.get('current','?')}/{event.get('total','?')}"
        elif kind == "compose_start":
            self.action = "bottom-up level composition"
        elif kind == "compose_field":
            self.action = f"compose {event.get('field','?')}"
        elif kind == "compose_done":
            self.action = "composition complete"
        elif kind == "level_done":
            self.level_done = self.level_total
            self.action = f"level complete resolution={event.get('resolution')}"
        elif kind == "base_field":
            self.action = f"materialize base field {event.get('field','?')}"
        else:
            self.action = str(event.get("message", kind))

        if self.log is not None:
            self.log(
                "refine event=%s path=%s action=%s level=%s/%s unit=%s/%s",
                kind,
                self.path,
                self.action,
                self.run_level_index,
                self.planned_levels,
                self.level_done,
                self.level_total,
            ) if hasattr(self.log, "__self__") else self.log(
                f"refine event={kind} path={self.path} action={self.action} "
                f"level={self.run_level_index}/{self.planned_levels} unit={self.level_done}/{self.level_total}"
            )
        if self.enabled:
            self._render()

    def _eta(self) -> tuple[float | None, float | None]:
        if not self._durations:
            return None, None
        recent = self._durations[-20:]
        recent_mean = sum(recent) / len(recent)
        global_mean = sum(self._durations) / len(self._durations)
        avg = 0.7 * recent_mean + 0.3 * global_mean
        remaining = max(0, self.level_total - self.level_done)
        nodes = max(1, self.level_nodes)
        fields = max(1, self.level_fields)
        future_levels = max(0, self.planned_levels - self.run_level_index)
        future_nodes = nodes
        for _ in range(future_levels):
            future_nodes *= self.branching_factor
            remaining += future_nodes * fields
        return remaining * avg, avg

    def _render(self) -> None:
        elapsed = time.perf_counter() - self.started_perf
        fraction = min(1.0, self.level_done / max(self.level_total, 1))
        eta, avg = self._eta()
        width = 24
        done = int(round(width * fraction))
        bar = "#" * done + "-" * (width - done)
        text = (
            f"[{bar}] level {self.run_level_index}/{self.planned_levels} "
            f"{100*fraction:6.2f}% unit {self.level_done}/{self.level_total} "
            f"elapsed {format_duration(elapsed)} ETA {format_duration(eta)} "
            f"| {self.path} > {self.action}"
        )
        if self.detailed:
            text += (
                f" | action-avg {format_duration(avg)}"
                f" started {datetime.fromtimestamp(self.started_wall).strftime('%H:%M:%S')}"
                f" finish~{_clock_after(eta)}"
            )
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        if is_tty:
            pad = " " * max(0, self._last_width - len(text))
            self.stream.write("\r" + text + pad)
            self.stream.flush()
            self._last_width = len(text)
        else:
            self.stream.write(text + "\n")
            self.stream.flush()

    def finish(self) -> None:
        if self.enabled and bool(getattr(self.stream, "isatty", lambda: False)()) and self._last_width:
            self.stream.write("\n")
            self.stream.flush()
            self._last_width = 0
