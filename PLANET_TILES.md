# Sparse Planetary Terrain / LOD Tiles

`worldgen` now has two deliberately different refinement paths:

1. `worldgen --refine` remains the offline **globally materialized** recursive
   refinement workflow. It is useful when a complete higher-resolution raster is
   genuinely required and the requested depth still fits storage.
2. `worldgen-tiles` is the **sparse interactive** path. It maps the completed global
   simulation onto a six-face cube-sphere quadtree and generates only explicitly
   requested tiles.

The second path is the foundation intended for Google-Earth-like globe navigation,
large game worlds, and random-access planetary terrain.

## Why a sparse cube-sphere

A global raster cannot simply be doubled indefinitely. Every linear factor of two
quadruples cell count. At sufficiently deep zoom, a globally materialized raster is
not merely slow; it is physically impractical to store.

The sparse backend uses addresses

```text
(face, z, x, y)
```

where `face` is one of

```text
px nx py ny pz nz
```

and each level splits a parent into four children. A tile always contains a fixed
`tile_size x tile_size` footprint represented by `(tile_size + 1)^2` terrain
vertices. The extra row and column make same-LOD neighbours share their complete
boundary vertex sets.

A cube-sphere was chosen for the internal authoritative terrain address space rather
than Web Mercator because it covers the entire planet, has no polar singularity,
and naturally fits 3-D globe rendering. Export adapters can later produce other
standards/projections.

## Ground resolution

A useful face-centre characteristic spacing is

```text
meters_per_sample ~= pi * planet_radius_m / (2 * tile_size * 2^z)
```

For an Earth-radius body and `tile_size=256`:

| z | approximate spacing |
|---:|---:|
| 0 | 39.1 km/sample |
| 1 | 19.6 km/sample |
| 5 | 1.22 km/sample |
| 6 | 611 m/sample |
| 8 | 153 m/sample |
| 10 | 38.2 m/sample |
| 12 | 9.55 m/sample |
| 14 | 2.39 m/sample |
| 15 | 1.19 m/sample |
| 16 | 0.60 m/sample |

`worldgen-tiles --meters-per-sample ...` chooses the first level at least as fine as
the requested characteristic resolution.

A single 257 x 257 float32 elevation tile is only about 258 KiB before filesystem
metadata. By contrast, globally storing all six cube faces at z15 would imply about
`6 * (256 * 2^15)^2` height samples, around 1.7 PB for one uncompressed float32
height layer alone. Deep levels therefore **must be sparse/lazy**.

## Current command-line workflow

Generate the physically coupled base world normally. NPZ output must be enabled:

```bash
worldgen --out world-out ...
```

Initialize the tile manifest without generating any terrain tile:

```bash
worldgen-tiles --world world-out
```

Generate one explicit tile:

```bash
worldgen-tiles \
  --world world-out \
  --tile px/10/612/431 \
  --field elevation_m
```

Resolve a location and choose LOD from requested ground resolution:

```bash
worldgen-tiles \
  --world world-out \
  --at 48.2082,16.3738 \
  --meters-per-sample 20 \
  --field elevation_m
```

Generate only the tiles intersecting a small viewing cap:

```bash
worldgen-tiles \
  --world world-out \
  --visible 48.2082,16.3738 \
  --meters-per-sample 20 \
  --angular-radius-deg 0.08 \
  --maximum-visible-tiles 256
```

Generate render-ready meshes as well:

```bash
worldgen-tiles \
  --world world-out \
  --visible 48.2082,16.3738 \
  --meters-per-sample 20 \
  --angular-radius-deg 0.08 \
  --mesh
```

The CLI emits JSON containing every resolved tile key, approximate ground spacing,
cache-hit state, metadata path, field paths and optional mesh path.

## On-disk layout

The current schema is intentionally simple and inspectable:

```text
world-out/
  world_arrays.npz
  world.json
  tiles/
    cubesphere_v1/
      tileset.json
      fields/
        elevation_m/
          z10/
            px/
              x00000612/
                y00000431.npy
      metadata/
        z10/
          px/
            x00000612/
              y00000431.json
      meshes/
        z10/
          px/
            x00000612/
              y00000431.npz
```

No sibling, parent, child, other face, or complete zoom-level raster is implicitly
materialized when one tile is requested.

`tileset.json` fingerprints `world_arrays.npz` and records the complete tile
specification. Reopening an existing cache with a different source world or tile
specification is rejected rather than silently mixing incompatible terrain.

## Data semantics

### Global simulation remains authoritative

The expensive coupled planetary simulation is intentionally solved globally at a
tractable resolution. Its results provide the low-frequency boundary state:

- tectonic plates, crust type/age/stress;
- global topography and ocean state;
- climate, winds, humidity and precipitation;
- hydrology and large drainage systems;
- geology;
- ecology/appearance;
- resources and other world layers;
- atmosphere/volatile state supplied by `atmogen`.

Continuous and categorical global fields can then be sampled into a requested cube
tile. Time-dependent fields such as monthly climate arrays retain their leading time
dimension.

`flow_to` is intentionally not inherited. It contains integer receiver indices into
the original global raster and becomes meaningless after projection/refinement.
Local drainage receivers must be recomputed at the target scale.

### Current high-frequency elevation

At zoom levels above the base representation, elevation currently receives
partition-independent deterministic spherical spectral detail. Every value is a
function of absolute planet-centred coordinates, world seed and detail band. It does
not depend on which neighbouring tiles happened to be generated first.

This provides:

- stable deterministic results;
- same-LOD shared edges;
- no longitude/pole seam;
- progressively available smaller-scale relief;
- bounded memory and storage.

It does **not** yet make a scientifically solved metre-scale landscape. Procedural
sub-grid relief is a boundary condition for the local-physics work below, not a
substitute for it.

## Render meshes and mixed LOD

`worldgen.terrain_mesh` converts one elevation tile into a planet-surface triangle
mesh. Vertex positions are stored as float32 in a tile-local east/north/up-like
frame, while the planet-centred origin and basis are float64. This prevents loss of
small-scale precision from storing ~6,000 km absolute coordinates in float32.

Each mesh can include a downward perimeter skirt. Same-level tiles already share top
edge vertices; skirts conceal transient T-junction gaps when a renderer shows a
coarse parent next to finer children. A future renderer may additionally implement
geomorphing/cross-fade to suppress visual popping during LOD replacement.

## Visibility-driven use

`visible_tiles(...)` traverses from the six root faces through only quadtree branches
whose spherical bounding cap intersects the requested viewing cap. It does not
construct or enumerate all `6 * 4^z` addresses at a deep level.

A globe viewer should therefore maintain approximately this loop:

```text
camera changes
    -> estimate desired meters/sample from altitude + screen pixels
    -> choose z
    -> traverse visible quadtree branches
    -> request missing visible tiles
    -> keep parent visible until required children are ready
    -> swap/fade to children
    -> evict non-visible GPU/RAM resources
    -> optionally retain disk cache
```

The visible set, RAM cache and persistent disk cache are intentionally separate
concepts. A tile may remain on disk after leaving the camera frustum without
remaining resident in renderer memory.

## Next physical refinement stage

The sparse address space is designed so the next stages can improve *physics* rather
than merely increase sampling density. The intended local pipeline is:

```text
global physical state
    -> target tile + deterministic physical halo
    -> parent-scale boundary conditions
    -> geology-conditioned sub-grid relief
    -> high-resolution coast/shore reconstruction
    -> local depression handling / drainage rebuild
    -> inherited major river constraints
    -> local runoff + stream-power erosion
    -> sediment routing / deposition
    -> hillslope diffusion / mass wasting
    -> climate topographic downscaling
         lapse-rate temperature correction
         orographic precipitation
         wind exposure / rain shadow
    -> local soil / vegetation / snow / albedo
    -> crop halo to authoritative tile core
    -> cache terrain and derived layers
```

The halo must be wide enough for each local numerical operator. Operations that can
propagate farther than the halo must either use parent-provided boundary conditions,
a larger supertile, or a hierarchical solve. Core boundary heights should remain
anchored or be reconciled so independently requested tiles cannot create terrain
cracks.

Large river topology is a special case: a local tile cannot infer the correct
continental drainage basin from local elevation alone. The global hydrology network
must provide inflow/outflow and channel constraints; the local solver then refines
tributaries and channel morphology inside that boundary state.

## Interoperability direction

The internal cube-sphere cache is not meant to be the only delivery format.
Exporters can be built above it without changing terrain generation.

High-priority targets are:

- Cesium quantized-mesh terrain for existing globe viewers;
- OGC 3D Tiles / implicit tiling where appropriate;
- glTF/GLB tile meshes;
- raw height/normal/albedo tiles for game engines;
- PNG/WebP/AVIF imagery tiles;
- compressed scientific array chunks (for example Zarr-compatible stores) for
  analysis rather than visualization.

Cesium quantized-mesh itself uses a multi-resolution quadtree, overlapping edge
vertices and edge lists intended to support skirts. OGC 3D Tiles implicit tiling
provides random-addressable quadtree/octree subdivision plus sparse availability.
Those concepts map well to the new worldgen architecture even though the internal
cube-sphere projection differs from Cesium terrain's standard EPSG:4326/3857 tiling.

## What “completely working” still requires

The current implementation establishes the hard storage/LOD topology and a working
lazy terrain path. Remaining major work is intentionally tracked as engineering and
scientific work rather than hidden behind a high zoom number:

1. tile-local geomorphology and drainage with parent/global boundary constraints;
2. hierarchical geometric error and screen-space-error LOD selection;
3. parent/child geomorphing in addition to skirts;
4. browser/game-engine delivery encodings and a viewer/streaming service;
5. texture mip chains and derived normal/slope/albedo tiles;
6. cache quotas, request coalescing and concurrent generation locks;
7. vector feature tiling for rivers, coastlines, roads/settlements and labels;
8. optional standardized Cesium/3D Tiles export;
9. stress tests at very deep logical zoom without ever prebuilding the planet.

The key constraint is now architectural rather than numerical: deep zoom is an
address into a deterministic planetary function and cache, **not** a request to
allocate the whole planet at that resolution.
