from __future__ import annotations

"""Interface-neutral hierarchical progress state and structured progress events."""

from dataclasses import dataclass, field
from enum import Enum
import math
import threading
import time
from typing import Any, Callable, Mapping


class ProgressState(str, Enum):
    PENDING = "pending"
    WAITING = "waiting"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    FAILED = "failed"


TERMINAL_STATES = {
    ProgressState.COMPLETED,
    ProgressState.SKIPPED,
    ProgressState.CANCELLED,
    ProgressState.FAILED,
}


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    type: str
    node_id: str
    parent_id: str | None
    timestamp_monotonic: float
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProgressNodeSnapshot:
    id: str
    parent: str | None
    name: str
    description: str
    state: str
    completed: float
    total: float | None
    units: str
    weight: float
    fraction: float | None
    elapsed_seconds: float
    eta_seconds: float | None
    throughput_per_second: float | None
    warnings: tuple[str, ...]
    children: tuple[str, ...]
    current_operation: str | None
    resource_metrics: Mapping[str, float]


@dataclass(slots=True)
class _ProgressNode:
    id: str
    parent: str | None
    name: str
    description: str = ""
    state: ProgressState = ProgressState.PENDING
    completed: float = 0.0
    total: float | None = None
    units: str = "items"
    weight: float = 1.0
    warnings: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    current_operation: str | None = None
    resource_metrics: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.perf_counter)
    started_at: float | None = None
    finished_at: float | None = None
    paused_at: float | None = None
    paused_seconds: float = 0.0
    last_update_at: float | None = None
    last_completed: float = 0.0
    smoothed_throughput: float | None = None
    last_progress_event_at: float = float("-inf")


class CancellationToken:
    """Thread-safe cooperative cancellation token suitable for worker safe-points."""

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise CancelledError("operation cancelled")


class CancelledError(RuntimeError):
    pass


class ProgressTree:
    """Authoritative nested progress state shared by all interfaces."""

    EVENT_CREATED = "TaskCreated"
    EVENT_STARTED = "TaskStarted"
    EVENT_PROGRESS = "TaskProgress"
    EVENT_PAUSED = "TaskPaused"
    EVENT_COMPLETED = "TaskCompleted"
    EVENT_FAILED = "TaskFailed"
    EVENT_CANCELLED = "TaskCancelled"
    EVENT_WARNING = "WarningRaised"
    EVENT_METRIC = "MetricUpdated"

    def __init__(
        self,
        *,
        event_sink: Callable[[ProgressEvent], None] | None = None,
        progress_event_interval_s: float = 0.05,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._nodes: dict[str, _ProgressNode] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._event_sink = event_sink
        self._progress_event_interval_s = max(0.0, float(progress_event_interval_s))
        self._clock = clock
        self._lock = threading.RLock()

    def create(
        self,
        node_id: str,
        name: str,
        *,
        parent: str | None = None,
        description: str = "",
        total: float | None = None,
        units: str = "items",
        weight: float = 1.0,
    ) -> CancellationToken:
        node_id = str(node_id)
        if not node_id:
            raise ValueError("node_id must be non-empty")
        if total is not None and (not math.isfinite(float(total)) or float(total) < 0):
            raise ValueError("total must be finite and non-negative, or None")
        if not math.isfinite(float(weight)) or float(weight) <= 0:
            raise ValueError("weight must be finite and positive")
        with self._lock:
            if node_id in self._nodes:
                raise KeyError(f"progress node already exists: {node_id}")
            if parent is not None and parent not in self._nodes:
                raise KeyError(f"parent progress node does not exist: {parent}")
            now = self._clock()
            node = _ProgressNode(
                id=node_id,
                parent=parent,
                name=str(name),
                description=str(description),
                total=None if total is None else float(total),
                units=str(units),
                weight=float(weight),
                created_at=now,
            )
            self._nodes[node_id] = node
            token = CancellationToken()
            self._tokens[node_id] = token
            if parent is not None:
                self._nodes[parent].children.append(node_id)
            self._emit(self.EVENT_CREATED, node, {"total": node.total, "units": node.units, "weight": node.weight}, now=now)
            return token

    def token(self, node_id: str) -> CancellationToken:
        with self._lock:
            return self._tokens[node_id]

    def start(self, node_id: str, *, operation: str | None = None) -> None:
        with self._lock:
            node = self._nodes[node_id]
            if node.state in TERMINAL_STATES:
                raise RuntimeError(f"cannot start terminal node {node_id}: {node.state.value}")
            now = self._clock()
            if node.started_at is None:
                node.started_at = now
            if node.paused_at is not None:
                node.paused_seconds += max(0.0, now - node.paused_at)
                node.paused_at = None
            node.state = ProgressState.RUNNING
            node.current_operation = operation
            node.last_update_at = now
            node.last_completed = node.completed
            self._emit(self.EVENT_STARTED, node, {"operation": operation}, now=now)

    def update(
        self,
        node_id: str,
        *,
        completed: float | None = None,
        advance: float | None = None,
        total: float | None = None,
        operation: str | None = None,
        force_event: bool = False,
    ) -> None:
        if completed is not None and advance is not None:
            raise ValueError("provide completed or advance, not both")
        with self._lock:
            node = self._nodes[node_id]
            if node.state in TERMINAL_STATES:
                raise RuntimeError(f"cannot update terminal node {node_id}: {node.state.value}")
            now = self._clock()
            if node.started_at is None:
                node.started_at = now
                node.state = ProgressState.RUNNING
            if total is not None:
                if not math.isfinite(float(total)) or float(total) < 0:
                    raise ValueError("total must be finite and non-negative")
                node.total = float(total)
            old_completed = node.completed
            if completed is not None:
                value = float(completed)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("completed must be finite and non-negative")
                node.completed = value
            elif advance is not None:
                delta = float(advance)
                if not math.isfinite(delta):
                    raise ValueError("advance must be finite")
                node.completed = max(0.0, node.completed + delta)
            if node.total is not None:
                node.completed = min(node.completed, node.total)
            if operation is not None:
                node.current_operation = str(operation)
            self._update_throughput(node, old_completed, now)
            if force_event or now - node.last_progress_event_at >= self._progress_event_interval_s:
                node.last_progress_event_at = now
                snap = self._snapshot_unlocked(node_id, now)
                self._emit(
                    self.EVENT_PROGRESS,
                    node,
                    {
                        "completed": node.completed,
                        "total": node.total,
                        "fraction": snap.fraction,
                        "throughput_per_second": snap.throughput_per_second,
                        "eta_seconds": snap.eta_seconds,
                        "operation": node.current_operation,
                    },
                    now=now,
                )

    def pause(self, node_id: str) -> None:
        with self._lock:
            node = self._nodes[node_id]
            if node.state != ProgressState.RUNNING:
                raise RuntimeError(f"only running nodes can be paused: {node_id}")
            now = self._clock()
            node.state = ProgressState.PAUSED
            node.paused_at = now
            self._emit(self.EVENT_PAUSED, node, {}, now=now)

    def resume(self, node_id: str, *, operation: str | None = None) -> None:
        self.start(node_id, operation=operation)

    def retry(self, node_id: str, *, operation: str | None = None) -> None:
        with self._lock:
            node = self._nodes[node_id]
            if node.state not in {ProgressState.FAILED, ProgressState.WAITING, ProgressState.RETRYING}:
                raise RuntimeError(f"node is not retryable from state {node.state.value}: {node_id}")
            node.state = ProgressState.RETRYING
            node.finished_at = None
            node.current_operation = operation

    def complete(self, node_id: str, *, skipped: bool = False) -> None:
        with self._lock:
            node = self._nodes[node_id]
            now = self._clock()
            node.state = ProgressState.SKIPPED if skipped else ProgressState.COMPLETED
            if not skipped and node.total is not None:
                node.completed = node.total
            node.finished_at = now
            node.current_operation = None
            self._emit(self.EVENT_COMPLETED, node, {"skipped": bool(skipped)}, now=now)

    def fail(self, node_id: str, error: str, *, recovery_attempted: str | None = None) -> None:
        with self._lock:
            node = self._nodes[node_id]
            now = self._clock()
            node.state = ProgressState.FAILED
            node.finished_at = now
            self._emit(
                self.EVENT_FAILED,
                node,
                {"error": str(error), "recovery_attempted": recovery_attempted},
                now=now,
            )

    def cancel(self, node_id: str, *, propagate: bool = True) -> None:
        with self._lock:
            targets = [node_id]
            if propagate:
                targets.extend(self._descendant_ids_unlocked(node_id))
            now = self._clock()
            for target in targets:
                node = self._nodes[target]
                self._tokens[target].cancel()
                if node.state not in TERMINAL_STATES:
                    node.state = ProgressState.CANCELLED
                    node.finished_at = now
                    node.current_operation = None
                    self._emit(self.EVENT_CANCELLED, node, {}, now=now)

    def warning(self, node_id: str, message: str) -> None:
        with self._lock:
            node = self._nodes[node_id]
            node.warnings.append(str(message))
            self._emit(self.EVENT_WARNING, node, {"message": str(message)})

    def metric(self, node_id: str, name: str, value: float) -> None:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("metric values must be finite")
        with self._lock:
            node = self._nodes[node_id]
            node.resource_metrics[str(name)] = numeric
            self._emit(self.EVENT_METRIC, node, {"name": str(name), "value": numeric})

    def snapshot(self, node_id: str) -> ProgressNodeSnapshot:
        with self._lock:
            return self._snapshot_unlocked(node_id, self._clock())

    def snapshot_all(self) -> dict[str, ProgressNodeSnapshot]:
        with self._lock:
            now = self._clock()
            return {node_id: self._snapshot_unlocked(node_id, now) for node_id in self._nodes}

    def roots(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(node.id for node in self._nodes.values() if node.parent is None)

    def _descendant_ids_unlocked(self, node_id: str) -> list[str]:
        result: list[str] = []
        pending = list(self._nodes[node_id].children)
        while pending:
            child = pending.pop()
            result.append(child)
            pending.extend(self._nodes[child].children)
        return result

    def _fraction_unlocked(self, node_id: str) -> float | None:
        node = self._nodes[node_id]
        if node.children:
            weighted = 0.0
            weight_sum = 0.0
            for child_id in node.children:
                child = self._nodes[child_id]
                fraction = self._fraction_unlocked(child_id)
                if fraction is None:
                    continue
                weighted += child.weight * fraction
                weight_sum += child.weight
            if weight_sum > 0:
                return min(1.0, max(0.0, weighted / weight_sum))
        if node.state in {ProgressState.COMPLETED, ProgressState.SKIPPED}:
            return 1.0
        if node.total is None:
            return None
        if node.total == 0:
            return 1.0 if node.state in TERMINAL_STATES else 0.0
        return min(1.0, max(0.0, node.completed / node.total))

    def _elapsed_unlocked(self, node: _ProgressNode, now: float) -> float:
        if node.started_at is None:
            return 0.0
        end = node.finished_at if node.finished_at is not None else now
        paused = node.paused_seconds
        if node.paused_at is not None:
            paused += max(0.0, end - node.paused_at)
        return max(0.0, end - node.started_at - paused)

    def _snapshot_unlocked(self, node_id: str, now: float) -> ProgressNodeSnapshot:
        node = self._nodes[node_id]
        fraction = self._fraction_unlocked(node_id)
        elapsed = self._elapsed_unlocked(node, now)
        throughput = node.smoothed_throughput
        eta: float | None = None
        if node.state in {ProgressState.COMPLETED, ProgressState.SKIPPED}:
            eta = 0.0
        elif not node.children and node.total is not None and throughput and throughput > 0:
            eta = max(0.0, node.total - node.completed) / throughput
        elif fraction is not None and 0.0 < fraction < 1.0 and elapsed > 0:
            eta = elapsed * (1.0 - fraction) / fraction
        return ProgressNodeSnapshot(
            id=node.id,
            parent=node.parent,
            name=node.name,
            description=node.description,
            state=node.state.value,
            completed=node.completed,
            total=node.total,
            units=node.units,
            weight=node.weight,
            fraction=fraction,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            throughput_per_second=throughput,
            warnings=tuple(node.warnings),
            children=tuple(node.children),
            current_operation=node.current_operation,
            resource_metrics=dict(node.resource_metrics),
        )

    def _update_throughput(self, node: _ProgressNode, old_completed: float, now: float) -> None:
        previous_at = node.last_update_at
        previous_completed = node.last_completed
        node.last_update_at = now
        node.last_completed = node.completed
        if previous_at is None:
            return
        dt = now - previous_at
        delta = node.completed - previous_completed
        if dt <= 0 or delta <= 0:
            return
        instant = delta / dt
        node.smoothed_throughput = instant if node.smoothed_throughput is None else 0.25 * instant + 0.75 * node.smoothed_throughput

    def _emit(self, event_type: str, node: _ProgressNode, payload: Mapping[str, Any], *, now: float | None = None) -> None:
        if self._event_sink is None:
            return
        self._event_sink(
            ProgressEvent(
                type=event_type,
                node_id=node.id,
                parent_id=node.parent,
                timestamp_monotonic=self._clock() if now is None else now,
                payload=dict(payload),
            )
        )


__all__ = [
    "CancelledError",
    "CancellationToken",
    "ProgressEvent",
    "ProgressNodeSnapshot",
    "ProgressState",
    "ProgressTree",
]
