from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable

from .checkpoint import CheckpointStore, package_source_fingerprint, stage_cache_key
from .pipeline import WorldPipeline


class ResumableWorldPipeline(WorldPipeline):
    """WorldPipeline with transparent, content-addressed stage checkpoints.

    Existing stage code does not need to know about persistence: every call to
    ``self._stage`` is intercepted. The key includes the full resolved configuration
    and a fingerprint of all installed ``worldgen`` Python sources, so stale results
    are never reused after a source/configuration change.
    """

    def __init__(
        self,
        config,
        progress: Callable[[str], None] | None = print,
        *,
        checkpoint_dir: str | Path,
        resume: bool = True,
        checkpoint_max_bytes: int = 64 * 1024**3,
    ) -> None:
        super().__init__(config, progress=progress)
        self.checkpoint_store = CheckpointStore(checkpoint_dir, max_bytes=checkpoint_max_bytes)
        self.resume = bool(resume)
        self._source_fingerprint = package_source_fingerprint()
        self._config_dict = config.to_dict()
        self.checkpoint_hits: list[str] = []

    def _stage(self, name: str, fn: Callable[[], Any]) -> Any:
        # Output is intentionally never checkpointed: it is a side-effect stage and
        # may need to regenerate files after output flags/render settings change.
        cacheable = name != "output"
        key = stage_cache_key(name, self._config_dict, self._source_fingerprint)
        if cacheable and self.resume:
            cached = self.checkpoint_store.get(key)
            if cached is not None:
                self.progress(f"[{name}] starting")
                self.progress(f"[{name}] checkpoint hit")
                self.timings[name] = 0.0
                self.checkpoint_hits.append(name)
                self.progress(f"[{name}] done in 0.000s")
                return cached

        value = super()._stage(name, fn)
        if cacheable:
            t0 = time.perf_counter()
            self.checkpoint_store.put(name, key, value)
            self.progress(f"[{name}] checkpoint saved in {time.perf_counter() - t0:.3f}s")
        return value

    def close(self) -> None:
        self.checkpoint_store.close()

    def checkpoint_stats(self) -> dict[str, int | float]:
        s = self.checkpoint_store.stats()
        return {
            "entries": s.entries,
            "bytes_used": s.bytes_used,
            "max_bytes": s.max_bytes,
            "hits": s.hits,
            "misses": s.misses,
            "writes": s.writes,
            "evictions": s.evictions,
            "run_stage_hits": len(self.checkpoint_hits),
        }
