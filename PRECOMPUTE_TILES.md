# Complete-prefix planetary tile precomputation

The planetary LOD system supports **both** operating modes:

1. sparse/lazy generation, where only visible/requested tiles are materialized; and
2. offline prefix precomputation, where every cube-sphere tile from z0 through a chosen depth is materialized in advance.

The modes use exactly the same tile keys, deterministic terrain functions, product caches and viewer paths. Precomputing does not create a second map format and does not disable deeper on-demand zoom.

## Semantics

`--precompute-depth Z` means every address on all six cube faces at every level `0..Z` is generated.

The exact tile count is

```text
6 * sum(4^z, z=0..Z) = 2 * (4^(Z+1) - 1)
```

Examples:

| Maximum depth | Complete prefix tiles |
|---:|---:|
| 0 | 6 |
| 1 | 30 |
| 2 | 126 |
| 3 | 510 |
| 4 | 2,046 |
| 5 | 8,190 |
| 6 | 32,766 |
| 7 | 131,070 |
| 8 | 524,286 |
| 9 | 2,097,150 |

The exponential growth is why lazy generation remains the default. A complete z7 prefix with 256-cell tiles contains more than 130,000 tiles; a z15 global prefix would be physically impractical even though individual z15 tiles remain cheap to generate lazily.

## Plan before generating

```bash
worldgen-tiles \
  --world world-out \
  --precompute-depth 6 \
  --precompute-plan-only
```

The plan reports the exact tile count and an estimated uncompressed array/mesh payload. PNG, vector and filesystem/container overhead is data-dependent and therefore not represented as fake precision in that estimate.

## Precompute base elevation

```bash
worldgen-tiles \
  --world world-out \
  --precompute-depth 6 \
  --precompute-workers 8
```

This creates every z0-z6 elevation tile. Levels z7 and deeper remain available on demand through the normal sparse API/viewer.

A successful precompute is **pinned by default**. The interactive persistent LRU will not evict generated products belonging to z0-z6 even when a later camera session has a smaller cache quota. Deeper lazily generated tiles remain normal LRU candidates.

For a disposable cache warm-up instead of an archival prefix, opt out explicitly:

```bash
worldgen-tiles \
  --world world-out \
  --precompute-depth 6 \
  --precompute-no-pin
```

## Precompute several scientific fields

```bash
worldgen-tiles \
  --world world-out \
  --precompute-depth 5 \
  --field elevation_m \
  --field plate_id \
  --field annual_temperature_c \
  --precompute-workers 8
```

Only fields that exist in the base world or are explicitly supported tile products may be requested.

## Precompute render products

The normal product flags apply to every tile in the prefix:

```bash
worldgen-tiles \
  --world world-out \
  --precompute-depth 5 \
  --mesh \
  --height-png \
  --true-color-png \
  --precompute-workers 8
```

## Precompute local physical refinement

Individual layers can be selected:

```bash
worldgen-tiles \
  --world world-out \
  --precompute-depth 4 \
  --local-temperature \
  --local-temperature-monthly \
  --precompute-orography \
  --precompute-surface \
  --precompute-hydrology \
  --precompute-geomorphology \
  --precompute-vectors
```

Or the current derived physical layers can be enabled together:

```bash
worldgen-tiles \
  --world world-out \
  --precompute-depth 4 \
  --precompute-all-derived
```

These are substantially more expensive than elevation-only precomputation because some products include 12 monthly layers and local physical solves.

## Safety limits

By default a run is rejected if either:

- the prefix exceeds 100,000 tiles; or
- the estimated uncompressed scientific/mesh payload exceeds 16 GiB.

The limits are configurable:

```bash
worldgen-tiles \
  --world world-out \
  --precompute-depth 7 \
  --precompute-max-tiles 200000 \
  --precompute-max-gib 64
```

A deliberately provisioned machine may bypass the safety limits:

```bash
worldgen-tiles \
  --world world-out \
  --precompute-depth 7 \
  --precompute-force-large
```

The override does not make the hierarchy smaller; it only confirms that the user intentionally accepts the storage/runtime cost.

## Resumability

Precomputation uses the same atomic per-tile caches as lazy generation. If a run stops after 20,000 of 32,766 tiles, rerunning the same command walks the prefix again but reuses completed products and continues generating missing products.

A job-specific status file is stored below:

```text
tiles/cubesphere_v1/precompute/
```

It records the plan, product selection, completed count, cache hits, last completed address, source-world fingerprint and state (`running`, `failed`, or `complete`).

No giant in-memory list of all tiles is required. Tile addresses are streamed breadth-first and the parallel scheduler keeps only a bounded `~2 * workers` task window in flight.

## Pinned prefix and cache eviction

On successful completion, the CLI writes:

```text
tiles/cubesphere_v1/precompute/pinned_prefix.json
```

The manifest records the deepest protected prefix and the source-world fingerprint. Pinning is monotonic: completing a shallower prefix later does not accidentally reduce an existing deeper pin.

The runtime disk LRU excludes precompute control manifests from payload accounting and will not delete any generated product with a `zNN` level less than or equal to the pinned depth. This applies consistently to scientific fields, metadata, meshes, derived local physics, imagery and vectors.

If the pinned prefix itself is larger than the configured runtime disk quota, preservation wins: cache statistics report an over-budget cache rather than deleting data the user explicitly requested to keep. Deeper unpinned tiles can still be evicted normally. Use `--precompute-no-pin` when that archival guarantee is not desired.

## Static files and dynamic RAM residency

Precomputed tiles are **not kept permanently in RAM**. Their files remain the authoritative stored representation, while the runtime maintains a separate byte-bounded in-process LRU for only the tiles currently useful to the viewer or another client.

The intended hierarchy is:

```text
static/pinned tile files on disk
        |
        v
persistent disk-cache policy
        |
        v
byte-bounded decoded RAM LRU
        |
        v
active renderer/simulation references
```

Scientific `.npy` fields are decoded into ordinary read-only NumPy arrays when requested. Repeated requests reuse the resident array until the RAM LRU evicts it. PNGs, meshes, vector files and other static products can be loaded through the same resident cache as bytes. Evicting any of these resident objects only drops the cache's process-memory reference; it does not remove or modify the static file.

`PlanetTileRuntime` defaults to a 256 MiB resident-memory budget and accepts a different `memory_cache_max_bytes` value. The runtime exposes:

- `load_field(...)` for decoded scientific arrays;
- `load_product_bytes(...)` for static binary/viewer products;
- `release_field(...)` and `release_product(...)` for explicit unloading;
- `release_tile(...)` to drop all cached products associated with one tile;
- `clear_memory_cache()` to unload the complete resident working set;
- `memory_cache_stats()` for bytes, hits, misses, evictions and oversize rejections.

The RAM cache is deliberately independent of disk pinning. A z0-z6 precomputed tile may be evicted from RAM immediately when it leaves the working set while its disk file remains permanently available. The next camera visit reloads it from disk without regeneration. Conversely, a deeper lazy tile can be evicted from RAM and later also evicted from the unpinned persistent disk cache if storage pressure requires it.

Pinned static files are not `touch`ed on reads for disk-LRU bookkeeping, so ordinary runtime access does not mutate their timestamps merely to represent cache recency.

## Viewer behavior after precomputation

Nothing changes in the viewer address space. If the camera requests a tile at or above the precomputed depth its static file already exists; it is loaded into RAM only when needed. If it requests a deeper tile, that tile is generated lazily as before and then enters the same RAM residency layer.

A common deployment pattern is therefore:

```text
z0-z5   precomputed/pinned on disk; loaded into RAM only when visible
z6-z10  optionally precomputed for important worlds/regions
z11+    generated on demand and cached/evicted normally
```

This preserves instant coarse/medium zoom while retaining effectively arbitrary deep logical resolution without requiring either the full planet or the full precomputed prefix to remain resident in process memory.
