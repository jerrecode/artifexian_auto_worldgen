from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable

from .drainage import install_into_hydrology
from .field_schema import write_field_catalog
from .invariants import write_validation_report
from .manifest import write_run_manifest
from .pipeline import WorldPipeline as _BaseWorldPipeline
from .workflow import StageCheckpointStore, StageRecord, stage_key

install_into_hydrology()


class WorldPipeline(_BaseWorldPipeline):
    """Checkpointable, provenance-aware façade over the deterministic core pipeline."""

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
            implementation_version="3",
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
        arrays = self._array_export(world)
        write_field_catalog(out / "field_catalog.json", arrays)
        write_validation_report(out / "validation.json", world, strict=False)
        write_run_manifest(
            out / "run_manifest.json",
            self.cfg,
            stage_records=self.stage_records,
            timings=self.timings,
            runtime_plan=self.runtime_plan,
            output_root=out,
        )
