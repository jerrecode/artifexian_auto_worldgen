# Performance, runtime control and large-world operation

The generator is primarily NumPy/SciPy vectorized. Blind process-level parallelism is often slower for this workload because the largest stages move many multi-megabyte arrays between coupled ocean, atmosphere, geology and hydrology solvers. The runtime layer therefore caps concurrency using both CPU count and a memory-per-worker budget, and keeps BLAS/OpenMP/NumExpr thread counts explicit to avoid nested oversubscription.

## Optional acceleration packages

The base install remains portable. Extra backends are opt-in:

```bash
pip install 'artifexian-auto-worldgen[performance]'
pip install 'artifexian-auto-worldgen[jit]'
pip install 'artifexian-auto-worldgen[storage]'
pip install 'artifexian-auto-worldgen[render]'
```

`performance` provides NumExpr, Bottleneck, psutil, threadpoolctl and joblib. `jit` adds Numba. `storage` adds Zarr/HDF5 options for future backends while the built-in random-access store requires only NumPy and SQLite. `render` adds Pillow for post-render resampling.

## Runtime caps

```bash
worldgen --workers 0 --worker-cap 6 --parallel-backend auto \
  --threads-per-worker 1 --memory-per-worker-mb 512 --runtime-info
```

`--workers 0` enables automatic selection. The resolver reserves coordinator capacity and limits workers against the detected cgroup/physical memory limit. The current core Earth-system solver is still mostly single-coordinator vectorized computation; the managed executor is used for safely parallelizable work such as post-render image processing and is available to future tiled numerical kernels.

## Resolution control

```bash
worldgen --resolution 1536x768
worldgen --height 1024
worldgen --resolution-scale 2
worldgen --png-scale 2 --png-resample lanczos --max-png-megapixels 160
```

The simulation grid always remains 2:1 equirectangular. `--resolution-scale` changes physical simulation resolution. `--png-scale` changes only rendered PNG resolution after simulation and is therefore far cheaper than increasing the numerical grid.

## Generic configuration overrides

Every dataclass configuration field can be overridden without adding hundreds of dedicated flags:

```bash
worldgen \
  --set tectonics.plate_count=18 \
  --set climate.moisture_iterations=40 \
  --set simulation.earth_system_passes=4 \
  --set output.compress_npz=true
```

Use `--write-config resolved.yml --dry-run` to materialize and inspect the effective configuration without generating a world.

## Random-access array store

```bash
worldgen --array-store world-arrays --array-store-max-gb 64
```

The built-in `MappedArrayStore` writes each canonical exported array as a `.npy` file and tracks shape, dtype, byte size, creation time and last access in SQLite. Arrays can be reopened with NumPy memory mapping so slices are randomly accessible without loading the complete dataset. The store supports both temporary and persistent roots and LRU pruning under a hard byte cap.

## In-memory cache

`ByteBoundLRUCache` is a thread-safe LRU cache bounded by estimated resident bytes rather than entry count. It supports TTL expiration, hit/miss/eviction statistics and explicit pruning. This is intended for deterministic intermediate fields that are expensive to regenerate but unsafe to keep unbounded at high resolution.

## Tiled numerical processing

`worldgen.mathops` supplies deterministic 2-D chunk planning, halo-aware tile iteration, working-set estimation, stable reductions, optional NumExpr acceleration and an optional Numba decorator. Tiling is preferred when an operation can be expressed locally because it bounds peak temporary memory and makes parallelism explicit instead of duplicating complete global arrays.

## Progress, ETA, logging and profiling

```bash
worldgen --progress -v --log-file run.log
worldgen -vv --log-file run.jsonl --log-json
worldgen --timings-json timings.json
worldgen --profile world.prof
```

The progress reporter consumes pipeline stage events and shows stage count, percentage, elapsed time and an adaptive ETA. `--profile` writes raw cProfile data plus a human-readable cumulative-time report. `--timings-json` records stage timings and the resolved runtime plan.

## CI validation

Every push to `main` now runs compilation, CLI smoke checks, the complete pytest suite and a distributable package build. Performance work should be landed incrementally so regressions can be localized to a specific commit.
