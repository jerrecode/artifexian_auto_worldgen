from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable

from .drainage import install_into_hydrology
from .manifest import write_run_manifest
from .pipeline import WorldPipeline as _BaseWorldPipeline
from .workflow import StageCheckpointStore, StageRecord, stage_key

# Install the O(N) reusable drainage graph kernels once. This compatibility
# bridge keeps the long-standing hydrology public API stable while moving its
# expensive graph traversals into a dedicated optimized implementation.
install_into_hydrology()


class WorldPipeline(_BaseWorldPipeline):
    """Checkpointable pipeline façade over the deterministic core pipeline.

    The inherited generation graph still defines the numerical workflow. This
    subclass intercepts every named stage, gives it a content-addressed key,
    optionally restores/saves its result, and records provenance for the final
    run manifest. Keeping the execution layer orthogonal to the scientific
    kernels lets resumability evolve without duplicating the world model.
    """

    def __init__(
        self,
        config,
        progress: Callable[[str], None] | None = print,
        *,
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
        runtime_plan: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config, progress=progress)
        self.resume = bool(resume)
        self.checkpoints = StageCheckpointStore(checkpoint_dir) if checkpoint_dir is not None else None
        self.stage_records: list[StageRecord] = []
        self.runtime_plan = dict(runtime_plan or {})
        self._previous_stage_key: str | None = None

    def _stage(self, name: str, fn: Callable[[], Any]) -> Any:
        dep_keys = () if self._previous_stage_key is None else (self._previous_stage_key,)
        key = stage_key(
            stage_name=name,
            seed=self.cfg.seed,
            config=self.cfg,
            dependency_keys=dep_keys,
            implementation_version="2",
        )
        cacheable = self.checkpoints is not None and name != "output"
        if cacheable and self.resume and self.checkpoints.has(key):
            self.progress(f"[{name}] starting")
            t0 = time.perf_counter()
            value = self.checkpoints.load(key)
            dt = time.perf_counter() - t0
            self.timings[name] = dt
            info = self.checkpoints._index.get("stages", {}).get(key, {})
            path = str(self.checkpoints.path_for_key(key))
            self.stage_records.append(StageRecord(
                name=name,
                key=key,
                seconds=dt,
                loaded_from_checkpoint=True,
                checkpoint_path=path,
                size_bytes=info.get("size_bytes"),
            ))
            self.progress(f"[{name}] done in {dt:.3f}s")
            self._previous_stage_key = key
            return value

        self.progress(f"[{name}] starting")
        t0 = time.perf_counter()
        value = fn()
        dt = time.perf_counter() - t0
        self.timings[name] = dt
        checkpoint_path: str | None = None
        checkpoint_size: int | None = None
        if cacheable:
            path, checkpoint_size = self.checkpoints.save(name, key, value)
            checkpoint_path = str(path)
        self.stage_records.append(StageRecord(
            name=name,
            key=key,
            seconds=dt,
            loaded_from_checkpoint=False,
            checkpoint_path=checkpoint_path,
            size_bytes=checkpoint_size,
        ))
        self.progress(f"[{name}] done in {dt:.3f}s")
        self._previous_stage_key = key
        return value

    def save(self, world: dict[str, Any], out: Path) -> None:
        super().save(world, out)
        write_run_manifest(
            out / "run_manifest.json",
            self.cfg,
            stage_records=self.stage_records,
            timings=self.timings,
            runtime_plan=self.runtime_plan,
            output_root=out,
        )
