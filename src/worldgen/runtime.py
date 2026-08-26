from __future__ import annotations

from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
import importlib.util
import math
import os
from typing import Callable, Iterable, Iterator, Literal, TypeVar

T = TypeVar("T")
R = TypeVar("R")
Backend = Literal["auto", "thread", "process", "serial"]


@dataclass(slots=True, frozen=True)
class RuntimePlan:
    backend: Backend
    workers: int
    cpu_count: int
    memory_limit_bytes: int | None
    threads_per_worker: int


def optional_backend_status() -> dict[str, bool]:
    """Report optional acceleration/storage packages without importing them."""
    names = ("numba", "numexpr", "bottleneck", "psutil", "zarr", "h5py", "PIL")
    return {name: importlib.util.find_spec(name) is not None for name in names}


def _memory_limit_bytes() -> int | None:
    """Best-effort physical/cgroup memory limit without requiring psutil."""
    candidates: list[int] = []
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(p, "rt", encoding="ascii").read().strip()
            if raw and raw != "max":
                value = int(raw)
                if 0 < value < (1 << 62):
                    candidates.append(value)
        except (OSError, ValueError):
            pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            candidates.append(int(pages) * int(page_size))
    except (AttributeError, OSError, ValueError):
        pass
    return min(candidates) if candidates else None


def resolve_runtime_plan(
    workers: int | None = None,
    *,
    worker_cap: int = 8,
    backend: Backend = "auto",
    reserve_cpus: int = 1,
    memory_per_worker_mb: int = 384,
    threads_per_worker: int = 1,
) -> RuntimePlan:
    cpu = max(1, os.cpu_count() or 1)
    cap = max(1, int(worker_cap))
    available_cpu = max(1, cpu - max(0, int(reserve_cpus)))
    mem_limit = _memory_limit_bytes()
    if mem_limit is None:
        memory_cap = cap
    else:
        bytes_per_worker = max(64, int(memory_per_worker_mb)) * 1024 * 1024
        # Leave 25% of memory outside the worker pool for the coordinator and large arrays.
        memory_cap = max(1, int((mem_limit * 0.75) // bytes_per_worker))

    if workers is None or int(workers) <= 0:
        resolved = min(cap, available_cpu, memory_cap)
    else:
        resolved = min(max(1, int(workers)), cap, available_cpu, memory_cap)

    chosen: Backend = backend
    if chosen == "auto":
        chosen = "thread" if resolved > 1 else "serial"
    if chosen == "serial":
        resolved = 1
    if chosen not in ("serial", "thread", "process"):
        raise ValueError(f"Unsupported backend: {backend!r}")

    return RuntimePlan(
        backend=chosen,
        workers=resolved,
        cpu_count=cpu,
        memory_limit_bytes=mem_limit,
        threads_per_worker=max(1, int(threads_per_worker)),
    )


def configure_numeric_threads(threads: int = 1, *, force: bool = False) -> dict[str, str]:
    """Cap BLAS/OpenMP/NumExpr thread pools before NumPy/SciPy are imported.

    Returns the environment variables that were set. Existing user values are
    respected unless ``force=True``.
    """
    n = str(max(1, int(threads)))
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    changed: dict[str, str] = {}
    for key in keys:
        if force or key not in os.environ:
            os.environ[key] = n
            changed[key] = n
    return changed


class ManagedExecutor(AbstractContextManager["ManagedExecutor"]):
    """Capped executor with deterministic ordered mapping and explicit lifecycle."""

    def __init__(self, plan: RuntimePlan):
        self.plan = plan
        self._executor: Executor | None = None

    def __enter__(self) -> "ManagedExecutor":
        if self.plan.backend == "thread":
            self._executor = ThreadPoolExecutor(max_workers=self.plan.workers, thread_name_prefix="worldgen")
        elif self.plan.backend == "process":
            self._executor = ProcessPoolExecutor(max_workers=self.plan.workers)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=exc is not None)
            self._executor = None
        return None

    def map(self, fn: Callable[[T], R], items: Iterable[T], *, chunksize: int = 1) -> Iterator[R]:
        if self.plan.backend == "serial" or self._executor is None:
            return map(fn, items)
        if isinstance(self._executor, ProcessPoolExecutor):
            return self._executor.map(fn, items, chunksize=max(1, int(chunksize)))
        return self._executor.map(fn, items)

    def submit(self, fn: Callable[..., R], /, *args, **kwargs) -> Future[R]:
        if self.plan.backend == "serial" or self._executor is None:
            future: Future[R] = Future()
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)
            return future
        return self._executor.submit(fn, *args, **kwargs)


def recommended_chunk_size(item_count: int, workers: int, *, target_chunks_per_worker: int = 8) -> int:
    if item_count <= 0:
        return 1
    denom = max(1, int(workers)) * max(1, int(target_chunks_per_worker))
    return max(1, int(math.ceil(item_count / denom)))
