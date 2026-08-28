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

## Interaction with cache eviction

Complete-prefix precomputation is an offline persistence request. If a runtime disk quota is configured below the size of the requested complete prefix, quota eviction and complete precomputation are contradictory policies. For archival/prebuilt maps, provision a disk quota large enough for the prefix or disable runtime eviction for that output directory.

## Viewer behavior after precomputation

Nothing changes in the viewer address space. If the camera requests a tile at or above the precomputed depth it is already a cache hit. If it requests a deeper tile, that tile is generated lazily as before.

A common deployment pattern is therefore:

```text
z0-z5   precomputed and immediately available
z6-z10  optionally precomputed for important worlds/regions
z11+    generated on demand and cached
```

This preserves instant coarse/medium zoom while retaining effectively arbitrary deep logical resolution without requiring the entire planet to be prebuilt at metre scale.
