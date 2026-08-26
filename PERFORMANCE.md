# Performance, runtime control and large-world operation

The generator is primarily NumPy/SciPy vectorized. Blind process-level parallelism is often slower for this workload because the largest stages move many multi-megabyte arrays between coupled ocean, atmosphere, geology and hydrology solvers. The runtime layer therefore caps concurrency using both CPU count and a memory-per-worker budget, and keeps BLAS/OpenMP/NumExpr thread counts explicit to avoid nested oversubscription.

Performance changes are expected to preserve numerical fidelity unless a different backend or tolerance is explicitly selected. Simulation resolution, render resolution, numerical convergence and model fidelity are separate controls.

## Optional acceleration packages

The base install remains portable. Extra backends are opt-in:

```bash
pip install 'artifexian-auto-worldgen[performance]'
pip install 'artifexian-auto-worldgen[jit]'
pip install 'artifexian-auto-worldgen[storage]'
pip install 'artifexian-auto-worldgen[render]'
```

`performance` provides NumExpr, Bottleneck, psutil, threadpoolctl and joblib. `jit` adds Numba. `storage` provides Zarr/HDF5 dependencies for optional storage work while the built-in random-access store requires only NumPy and SQLite. `render` adds Pillow for post-render resampling.

Priority-Flood and sequential drainage-graph kernels have dedicated Numba-capable implementations. The reference Priority-Flood backend remains available for semantic regression testing.

## Reproducible benchmarks

Machine-readable benchmark profiles are available through:

```bash
python -m worldgen.benchmarks --profile micro --output benchmark.json
python -m worldgen.benchmarks --profile quick --output benchmark-quick.json
python -m worldgen.benchmarks --profile high --skip-world --output kernels-high.json
```

Profiles are `micro` (128x64), `quick` (256x128), `normal` (768x384), and `high` (1536x768). The JSON records environment/backend availability, wall and CPU time, peak/current RSS when available, Priority-Flood reference/accelerated equivalence and speedup, drainage-graph construction/accumulation timings, and optional complete-world stage timings/cache statistics.

Absolute timing thresholds are intentionally not hard-coded into ordinary CI because shared runners are noisy. Correctness tests verify benchmark contracts and numerical equivalence; benchmark JSON is intended for before/after comparisons and dedicated regression analysis.

## Runtime caps

```bash
worldgen --workers 0 --worker-cap 6 --parallel-backend auto \
  --threads-per-worker 1 --memory-per-worker-mb 512 --runtime-info
```

`--workers 0` enables automatic selection. The resolver reserves coordinator capacity and limits workers against the detected cgroup/physical memory limit. The core coupled Earth-system dependency chain is not parallelized blindly; independent rendering/post-processing and future tile-local kernels use the managed execution abstraction instead.

## Resolution control

```bash
worldgen --resolution 1536x768
worldgen --height 1024
worldgen --resolution-scale 2
worldgen --png-scale 2 --png-resample lanczos --max-png-megapixels 160
```

The current simulation backend uses a 2:1 equirectangular spherical grid. `--resolution-scale` changes physical simulation resolution. `--png-scale` changes only rendered PNG resolution after simulation and is therefore far cheaper than increasing the numerical grid.

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

## Adaptive drainage refresh

Legacy worlds retain fixed receiver-graph refresh semantics by default:

```yaml
hydrology:
  flow_refresh_mode: interval
  flow_refresh_interval: 1
```

High-fidelity runs may select `adaptive` or `hybrid`. The adaptive policy compares terrain against the state at the last drainage rebuild and refreshes when a robust elevation-change threshold, coastline-change fraction, prior delta aggradation threshold, or hard maximum reuse interval is reached. This avoids rebuilding Priority-Flood/receivers/DrainageGraph after geomorphic iterations that did not materially alter drainage while guaranteeing that small changes cannot accumulate indefinitely.

## Adaptive Earth-system convergence

Fixed pass counts remain the default for backward reproducibility. Optional adaptive convergence uses physical residuals rather than a fixed number of coupled passes:

```yaml
simulation:
  adaptive_convergence: true
  min_earth_system_passes: 2
  max_earth_system_passes: 6
  convergence_temperature_c: 0.15
  convergence_precip_mm_year: 15.0
  convergence_elevation_m: 2.0
  required_consecutive_converged_passes: 2

  adaptive_final_coupling: true
  min_final_climate_ocean_passes: 1
  max_final_climate_ocean_passes: 4
```

Predictor passes cannot terminate adaptive convergence: the solver reaches full configured climate/ocean fidelity by the minimum pass count and only then counts consecutive converged passes. Coupling history stores normalized residuals and the explicit stop reason. This is a numerical stopping criterion for the reduced-order model, not a claim of first-principles Earth-system convergence.

## Random-access array store

```bash
worldgen --array-store world-arrays --array-store-max-gb 64
```

`MappedArrayStore` uses immutable content-addressed `.npy` objects and SQLite metadata. New objects are serialized and fsynced before the SQLite pointer is committed; old payloads are removed only after commit. Recovery reconciles orphan objects left by interruption. Arrays reopen through NumPy memory mapping so arbitrary slices are accessible without loading complete datasets, and LRU pruning enforces a hard byte cap.

## Checkpoint invalidation

Resumable checkpoints combine relevant configuration sections, a stage-specific source fingerprint, and a rolling upstream dependency digest. Consequently an output/render source change need not invalidate astronomy or tectonics, while a climate code change invalidates climate and downstream stages. Unknown future stages fall back conservatively to the complete package source set.

## In-memory cache

`ByteBoundLRUCache` is a thread-safe LRU cache bounded by estimated resident bytes rather than entry count. It supports TTL expiration, hit/miss/eviction statistics and explicit pruning. Deterministic great-circle distance fields use this bounded cache because they are expensive and frequently reused across physical stages.

## Spherical tiled numerical processing

`worldgen.tiling` provides genuine 2-D deterministic tiles rather than full-width row strips. Each `SphericalTile` carries a core region, expanded halo, crop region, and mapped indices. East/west halos wrap periodically; north/south halos reflect through the pole and rotate longitude by 180 degrees. Halos can be specified in cells or conservatively in physical kilometres.

`worldgen.mathops` continues to provide lower-level working-set estimation, stable reductions, NumExpr acceleration and the optional Numba decorator. Globally coupled algorithms must not be tiled as if they were local; the spherical tiler is intended for stencil/filter/noise/appearance/resource kernels whose dependencies fit inside a declared halo.

## Progress, ETA, logging and profiling

```bash
worldgen --progress -v --log-file run.log
worldgen -vv --log-file run.jsonl --log-json
worldgen --timings-json timings.json
worldgen --profile world.prof
```

The progress reporter consumes pipeline stage events and shows stage count, percentage, elapsed time and an adaptive ETA. `--profile` writes raw cProfile data plus a human-readable cumulative-time report. `--timings-json` records stage timings and the resolved runtime plan.

## CI validation

Every push to `main` runs compilation, CLI smoke checks, the complete pytest suite, Python-version compatibility jobs, optional-backend coverage and an end-to-end generation/resume path. Performance work is landed incrementally so numerical or storage regressions can be localized to a specific commit.
