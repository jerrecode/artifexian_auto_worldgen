from __future__ import annotations

"""Thread-safe hierarchical timing and ETA telemetry for long-running generation.

The tracker is intentionally lightweight and JSON based so timing history can travel
with GitHub Actions checkpoint artifacts.  It records completed samples, tracks active
work, emits human-readable heartbeat lines, and writes a machine-readable snapshot.

Hierarchy used by ultra-resolution production:
    job/shard -> processing step -> substep (tile) -> subsubstep (tile phase)

ETA estimates are empirical.  They use arithmetic means from work completed in the
current processing step, with same-phase samples preferred for the active subsubstep.
Parallel processing-step ETA is expressed as wall-clock work divided by configured
parallelism.
"""

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from threading import Event, Lock, Thread
import time
from typing import Any, Mapping


@dataclass(slots=True)
class _Active:
    level: str
    name: str
    parent: str | None
    started_monotonic: float
    started_unix: float
    index: int | None
    total: int | None
    meta: dict[str, Any]


def _mean(values: list[float]) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v)) and float(v) >= 0.0]
    return (sum(finite) / len(finite)) if finite else None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


class HierarchicalProgressTracker:
    """Persist timing samples and continuously estimate nested ETAs."""

    def __init__(
        self,
        root: str | Path,
        *,
        scope: str,
        heartbeat_seconds: float = 30.0,
        parallelism: int = 1,
        subsubsteps_per_substep: int = 1,
        processing_steps_total: int = 1,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.scope = str(scope)
        self.events_path = self.root / f"{self.scope}.events.jsonl"
        self.summary_path = self.root / f"{self.scope}.summary.json"
        self.heartbeat_seconds = max(float(heartbeat_seconds), 5.0)
        self.parallelism = max(int(parallelism), 1)
        self.subsubsteps_per_substep = max(int(subsubsteps_per_substep), 1)
        self.processing_steps_total = max(int(processing_steps_total), 1)

        self._lock = Lock()
        self._stop = Event()
        self._heartbeat: Thread | None = None
        self._active: dict[str, _Active] = {}
        self._samples: dict[str, dict[str, list[float]]] = {
            "subsubstep": defaultdict(list),
            "substep": defaultdict(list),
            "processing_step": defaultdict(list),
        }
        self._all_samples: dict[str, list[float]] = {
            "subsubstep": [],
            "substep": [],
            "processing_step": [],
        }
        self._completed_substeps = 0
        self._selected_substeps_total = 0
        self._completed_subsubsteps = 0
        self._resumed_substeps = 0
        self._processing_steps_completed = 0
        self._current_processing_step: str | None = None
        self._load_history()

    def _load_history(self) -> None:
        if not self.events_path.exists():
            return
        try:
            lines = self.events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "end":
                continue
            level = str(event.get("level", ""))
            name = str(event.get("name", ""))
            duration = event.get("duration_seconds")
            if level not in self._samples or not name:
                continue
            try:
                value = float(duration)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value) or value < 0.0:
                continue
            self._samples[level][name].append(value)
            self._all_samples[level].append(value)

    def configure_substeps(self, *, completed: int, total: int, resumed: int = 0) -> None:
        with self._lock:
            self._completed_substeps = max(int(completed), 0)
            self._selected_substeps_total = max(int(total), 0)
            self._resumed_substeps = max(int(resumed), 0)
            self._completed_subsubsteps = (
                self._completed_substeps * self.subsubsteps_per_substep
            )
            self._write_snapshot_locked(reason="configure")

    def set_processing_step(self, name: str, *, completed_before: int = 0) -> None:
        with self._lock:
            self._current_processing_step = str(name)
            self._processing_steps_completed = max(int(completed_before), 0)
            self._write_snapshot_locked(reason="processing-step")

    def start_heartbeat(self) -> None:
        with self._lock:
            if self._heartbeat is not None:
                return
            self._stop.clear()
            self._heartbeat = Thread(
                target=self._heartbeat_loop,
                name=f"{self.scope}-eta-heartbeat",
                daemon=True,
            )
            self._heartbeat.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._heartbeat
        if thread is not None:
            thread.join(timeout=min(self.heartbeat_seconds + 1.0, 5.0))
        with self._lock:
            self._heartbeat = None
            self._write_snapshot_locked(reason="close")

    def begin(
        self,
        level: str,
        name: str,
        *,
        token: str,
        parent: str | None = None,
        index: int | None = None,
        total: int | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        now_mono = time.monotonic()
        now_unix = time.time()
        with self._lock:
            self._active[token] = _Active(
                level=str(level),
                name=str(name),
                parent=None if parent is None else str(parent),
                started_monotonic=now_mono,
                started_unix=now_unix,
                index=index,
                total=total,
                meta=dict(meta or {}),
            )
            self._append_event_locked(
                {
                    "event": "start",
                    "scope": self.scope,
                    "level": level,
                    "name": name,
                    "token": token,
                    "parent": parent,
                    "index": index,
                    "total": total,
                    "timestamp_unix": now_unix,
                    "meta": dict(meta or {}),
                }
            )
            self._emit_locked(reason="start", preferred_token=token)

    def end(
        self,
        token: str,
        *,
        duration_seconds: float | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> float:
        now_mono = time.monotonic()
        now_unix = time.time()
        with self._lock:
            active = self._active.pop(token, None)
            if active is None:
                raise KeyError(f"unknown progress token: {token}")
            duration = (
                max(now_mono - active.started_monotonic, 0.0)
                if duration_seconds is None
                else max(float(duration_seconds), 0.0)
            )
            if active.level in self._samples:
                self._samples[active.level][active.name].append(duration)
                self._all_samples[active.level].append(duration)
            if active.level == "subsubstep":
                self._completed_subsubsteps += 1
            elif active.level == "substep":
                self._completed_substeps += 1
            elif active.level == "processing_step":
                self._processing_steps_completed += 1

            merged_meta = dict(active.meta)
            merged_meta.update(meta or {})
            self._append_event_locked(
                {
                    "event": "end",
                    "scope": self.scope,
                    "level": active.level,
                    "name": active.name,
                    "token": token,
                    "parent": active.parent,
                    "index": active.index,
                    "total": active.total,
                    "timestamp_unix": now_unix,
                    "duration_seconds": duration,
                    "meta": merged_meta,
                }
            )
            self._emit_locked(reason="end")
            return duration

    def observe(
        self,
        level: str,
        name: str,
        duration_seconds: float,
        *,
        parent: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        token = f"observe:{level}:{name}:{time.time_ns()}"
        self.begin(level, name, token=token, parent=parent, meta=meta)
        self.end(token, duration_seconds=duration_seconds, meta=meta)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            with self._lock:
                if self._active:
                    self._emit_locked(reason="heartbeat")

    def _append_event_locked(self, payload: Mapping[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()

    def _active_subsubstep_locked(self, preferred_token: str | None = None) -> tuple[str, _Active] | None:
        if preferred_token is not None:
            item = self._active.get(preferred_token)
            if item is not None and item.level == "subsubstep":
                return preferred_token, item
        candidates = [
            (token, item)
            for token, item in self._active.items()
            if item.level == "subsubstep"
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda pair: pair[1].started_monotonic)

    def _active_substep_locked(self) -> tuple[str, _Active] | None:
        candidates = [
            (token, item)
            for token, item in self._active.items()
            if item.level == "substep"
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda pair: pair[1].started_monotonic)

    def _estimate_locked(self, *, preferred_token: str | None = None) -> dict[str, Any]:
        now = time.monotonic()
        avg_subsub = _mean(self._all_samples["subsubstep"])
        avg_substep = _mean(self._all_samples["substep"])
        avg_process = _mean(self._all_samples["processing_step"])

        active_phase = self._active_subsubstep_locked(preferred_token)
        phase_payload: dict[str, Any] | None = None
        if active_phase is not None:
            token, item = active_phase
            elapsed = max(now - item.started_monotonic, 0.0)
            same_phase = _mean(self._samples["subsubstep"].get(item.name, []))
            expected = same_phase if same_phase is not None else avg_subsub
            remaining = None if expected is None else max(expected - elapsed, 0.0)
            phase_payload = {
                "token": token,
                "name": item.name,
                "parent": item.parent,
                "elapsed_seconds": elapsed,
                "mean_same_phase_seconds": same_phase,
                "mean_all_subsubsteps_seconds": avg_subsub,
                "eta_seconds": remaining,
            }

        active_substep = self._active_substep_locked()
        substep_payload: dict[str, Any] | None = None
        if active_substep is not None:
            token, item = active_substep
            elapsed = max(now - item.started_monotonic, 0.0)
            expected = avg_substep
            if expected is None and avg_subsub is not None:
                expected = avg_subsub * self.subsubsteps_per_substep
            remaining = None if expected is None else max(expected - elapsed, 0.0)
            substep_payload = {
                "token": token,
                "name": item.name,
                "elapsed_seconds": elapsed,
                "mean_substep_seconds": avg_substep,
                "eta_seconds": remaining,
            }

        remaining_substeps = max(
            self._selected_substeps_total - self._completed_substeps,
            0,
        )
        total_subsubsteps = (
            self._selected_substeps_total * self.subsubsteps_per_substep
        )
        remaining_subsubsteps = max(
            total_subsubsteps - self._completed_subsubsteps,
            0,
        )
        # Requested processing-step estimator: arithmetic mean of all completed
        # subsubsteps multiplied by how many subsubsteps remain.  Divide by
        # effective parallelism to convert remaining work into wall-clock ETA.
        step_eta_from_subsubsteps = (
            None
            if avg_subsub is None
            else avg_subsub * remaining_subsubsteps / self.parallelism
        )
        step_eta_from_substeps = (
            None
            if avg_substep is None
            else avg_substep * remaining_substeps / self.parallelism
        )
        step_eta_candidates = [
            value
            for value in (step_eta_from_subsubsteps, step_eta_from_substeps)
            if value is not None and math.isfinite(value)
        ]
        processing_step_eta = (
            sum(step_eta_candidates) / len(step_eta_candidates)
            if step_eta_candidates
            else None
        )

        future_processing_steps = max(
            self.processing_steps_total - self._processing_steps_completed - 1,
            0,
        )
        whole_job_eta = processing_step_eta
        if avg_process is not None:
            tail = avg_process * future_processing_steps
            whole_job_eta = tail if whole_job_eta is None else whole_job_eta + tail

        return {
            "scope": self.scope,
            "current_processing_step": self._current_processing_step,
            "parallelism": self.parallelism,
            "counts": {
                "resumed_substeps": self._resumed_substeps,
                "completed_substeps": self._completed_substeps,
                "total_substeps": self._selected_substeps_total,
                "completed_subsubsteps": self._completed_subsubsteps,
                "total_subsubsteps": total_subsubsteps,
                "processing_steps_completed": self._processing_steps_completed,
                "processing_steps_total": self.processing_steps_total,
            },
            "means_seconds": {
                "all_subsubsteps": avg_subsub,
                "all_substeps": avg_substep,
                "all_processing_steps": avg_process,
            },
            "active_subsubstep": phase_payload,
            "active_substep": substep_payload,
            "eta_seconds": {
                "current_processing_step_from_subsubsteps": step_eta_from_subsubsteps,
                "current_processing_step_from_substeps": step_eta_from_substeps,
                "current_processing_step": processing_step_eta,
                "whole_job": whole_job_eta,
            },
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._estimate_locked()

    def _write_snapshot_locked(self, *, reason: str) -> dict[str, Any]:
        payload = self._estimate_locked()
        payload["reason"] = reason
        payload["timestamp_unix"] = time.time()
        _atomic_json(self.summary_path, payload)
        return payload

    @staticmethod
    def _fmt(seconds: float | None) -> str:
        if seconds is None or not math.isfinite(seconds):
            return "unknown"
        seconds = max(float(seconds), 0.0)
        if seconds < 90.0:
            return f"{seconds:.1f}s"
        if seconds < 5400.0:
            return f"{seconds / 60.0:.1f}m"
        return f"{seconds / 3600.0:.2f}h"

    def _emit_locked(self, *, reason: str, preferred_token: str | None = None) -> None:
        payload = self._write_snapshot_locked(reason=reason)
        counts = payload["counts"]
        etas = payload["eta_seconds"]
        phase = payload.get("active_subsubstep")
        substep = payload.get("active_substep")
        bits = [
            f"[progress:{self.scope}]",
            f"reason={reason}",
            f"tiles={counts['completed_substeps']}/{counts['total_substeps']}",
            f"subsubsteps={counts['completed_subsubsteps']}/{counts['total_subsubsteps']}",
        ]
        if substep:
            bits.append(
                f"substep={substep['name']} elapsed={self._fmt(substep['elapsed_seconds'])} "
                f"eta={self._fmt(substep['eta_seconds'])}"
            )
        if phase:
            bits.append(
                f"subsubstep={phase['name']} elapsed={self._fmt(phase['elapsed_seconds'])} "
                f"eta={self._fmt(phase['eta_seconds'])}"
            )
        bits.extend(
            [
                "step_eta="
                + self._fmt(etas["current_processing_step"]),
                "job_eta=" + self._fmt(etas["whole_job"]),
            ]
        )
        print(" | ".join(bits), flush=True)


__all__ = ["HierarchicalProgressTracker"]
