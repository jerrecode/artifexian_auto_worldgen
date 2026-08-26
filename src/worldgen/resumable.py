from __future__ import annotations

from pathlib import Path
import hashlib
import time
from typing import Any, Callable

from .checkpoint import CheckpointStore, package_source_fingerprint, stage_cache_key
from .pipeline import WorldPipeline


class ResumableWorldPipeline(WorldPipeline):
    """WorldPipeline with transparent, dependency-sensitive stage checkpoints.

    Every stage key includes the installed source fingerprint, only the configuration
    sections that can directly affect that stage, and a rolling digest of upstream
    stage keys. Consequently a society-only configuration change can still reuse
    astronomy/tectonics/climate checkpoints, while any upstream change invalidates
    all dependent downstream state automatically.
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
        self._dependency_digest = hashlib.sha256(b"worldgen-stage-root-v2").hexdigest()
        self.checkpoint_hits: list[str] = []
        self.stage_cache_keys: dict[str, str] = {}

    def _stage_config(self, name: str) -> dict[str, Any]:
        c = self._config_dict
        sections: set[str] = {"seed"}
        if name == "astronomy":
            sections |= {"astronomy"}
        elif name == "tectonics":
            sections |= {"resolution", "tectonics", "noise"}
        elif name == "noise_cache":
            sections |= {"resolution", "noise"}
        elif name.startswith("terrain"):
            sections |= {"terrain", "tectonics", "simulation"}
        elif name.startswith("ocean"):
            sections |= {"ocean", "terrain", "noise", "simulation"}
        elif name.startswith("climate"):
            sections |= {"climate", "terrain", "noise", "simulation"}
        elif name.startswith("geology"):
            sections |= {"noise"}
        elif name.startswith("surface") and name != "surface_appearance":
            sections |= {"hydrology", "noise", "simulation"}
        elif name == "hydrology_final":
            sections |= {"hydrology"}
        elif name == "weather":
            sections |= {"weather"}
        elif name == "surface_appearance":
            sections |= {"appearance"}
        elif name == "resources":
            sections |= {"resources"}
        elif name == "society":
            sections |= {"society"}
        elif name == "output":
            sections |= {"output"}
        else:
            # Unknown future stages default to the whole configuration for safety.
            return {"config": c, "dependency_digest": self._dependency_digest}
        return {
            "config": {key: c[key] for key in sorted(sections) if key in c},
            "dependency_digest": self._dependency_digest,
        }

    def _advance_dependency_digest(self, cache_key: str) -> None:
        h = hashlib.sha256()
        h.update(self._dependency_digest.encode("ascii"))
        h.update(cache_key.encode("ascii"))
        self._dependency_digest = h.hexdigest()

    def _stage(self, name: str, fn: Callable[[], Any]) -> Any:
        cacheable = name != "output"
        key = stage_cache_key(name, self._stage_config(name), self._source_fingerprint)
        self.stage_cache_keys[name] = key
        if cacheable and self.resume:
            cached = self.checkpoint_store.get(key)
            if cached is not None:
                self.progress(f"[{name}] starting")
                self.progress(f"[{name}] checkpoint hit")
                self.timings[name] = 0.0
                self.checkpoint_hits.append(name)
                self.progress(f"[{name}] done in 0.000s")
                self._advance_dependency_digest(key)
                return cached

        value = super()._stage(name, fn)
        if cacheable:
            t0 = time.perf_counter()
            self.checkpoint_store.put(name, key, value)
            self.progress(f"[{name}] checkpoint saved in {time.perf_counter() - t0:.3f}s")
        self._advance_dependency_digest(key)
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
