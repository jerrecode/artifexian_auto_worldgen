# Recursive world refinement

The refinement subsystem turns one completed global world into a persistent hierarchy of increasingly fine numerical datasets without pretending that globally coupled physics can be solved independently inside arbitrary map rectangles.

## Basic workflow

Generate the global world normally and retain `world_arrays.npz`:

```bash
worldgen generate --config config/default.yaml --out world-out \
  --checkpoint-dir world-out/checkpoints --progress-detail -vv
```

Refine it one depth:

```bash
worldgen generate --out world-out --refine --progress-detail -vv
```

The default refinement splits every current parent section into `2x2` children and doubles the linear numerical resolution. A 768x384 base therefore becomes 1536x768 at depth 1, 3072x1536 at depth 2, 6144x3072 at depth 3, and so on.

Either repeat the same command later or request several new depths in one invocation:

```bash
worldgen generate --out world-out --refine --refine-levels 3
```

The hierarchy is persistent. A later `--refine` invocation starts from the deepest completely composed level rather than returning to the original coarse world.

## Refinement controls

```text
--refine
--refine-levels N
--refine-scale N
--refine-sections COLUMNSxROWS
--refine-halo-cells N
--refine-elevation-detail-strength X
--[no-]refine-keep-sections
--[no-]resume
--[no-]progress
--[no-]progress-detail
-v / -vv
--log-file PATH
--log-json
```

Example using 4x2 spatial subdivision and a 3x linear refinement at each new depth:

```bash
worldgen generate --out world-out --refine \
  --refine-sections 4x2 \
  --refine-scale 3 \
  --refine-halo-cells 16 \
  --progress-detail -vv
```

`--refine-keep-sections` retains every completed child array after full-level composition. The default deletes redundant child payloads only after a level has been safely composed, while preserving the node graph and level metadata. This considerably reduces disk multiplication for deep trees.

## Persistent layout

```text
world-out/
  world_arrays.npz
  world.json
  maps/
    02b_height_grayscale_16bit.png
    02b_height_grayscale_16bit.json
  refinement/
    manifest.json
    levels/
      level_0000/
        index.json
        arrays/*.npy
      level_0001/
        index.json
        level_state.json
        hydrology_refinement.json
        arrays/*.npy
        maps/
          height_grayscale_16bit.png
          height_grayscale_16bit.json
      level_0002/
        ...
```

Level zero is a one-time random-access materialization of `world_arrays.npz`. Subsequent levels use individual `.npy` fields so arrays can be memory-mapped rather than repeatedly decompressing a monolithic archive.

## Why section boundaries remain coherent

A child section is **not** simulated against an isolated cropped parent array. For every refined cell, source coordinates are expressed in the complete composed parent grid. Sampling therefore sees the same coarse/global forcing on both sides of a child boundary.

Each child is evaluated on an expanded halo and only its core is retained. Longitude is periodic. Latitude crossings reflect through the pole and shift longitude by 180 degrees. Tangent-vector components receive a sign reversal when their raster basis is reflected through a pole.

Deterministic elevation detail is evaluated from global spherical coordinates rather than from a random generator restarted independently in each child. Consequently changing the section layout does not introduce a different random seam pattern.

After all child cores complete, the level is composed into a full disk-backed dataset. That composed field—not a collection of independently evolving sibling edge states—is the source for the next recursive depth.

Drainage is deliberately handled **after** this composition step. No child tile is allowed to decide its own outlet. The complete refined elevation sphere is Priority-Flooded once and receives one new globally consistent receiver graph, drainage accumulation, channel hierarchy and watershed hierarchy. Thus a section boundary has no special hydrologic meaning.

## Recursive ancestry

Node IDs preserve their full ancestry. Examples:

```text
root/r0c0
root/r0c0/r1c1
root/r0c0/r1c1/r0c1
```

The index for every completed depth retains all child bounds and parent IDs even when redundant child payload files are deleted after composition.

## Crash/restart behavior

Every individual child field is written through a temporary file, flushed, fsynced, and atomically renamed. A node receives its completion marker only after all of its fields have been written.

The level is marked complete only after bottom-up composition, global refined-hydrology recomputation, and its full-level index succeed. `manifest.json` advances `deepest_complete_level` only after that point.

With the default `--resume`, restarting after interruption reuses already completed node fields. The remaining node fields and composition continue from the incomplete depth. Use `--no-resume` to deliberately recompute existing refinement payloads.

Older completed refinement trees created before refined hydrology existed are migrated lazily: before such a level becomes the parent of a deeper refinement, its drainage topology is rebuilt from its already-composed elevation and inherited forcing arrays.

The original simulation-stage checkpoint system remains separate and continues to handle interrupted base-world generation.

## Progress and ETA

Normal generation supports:

```bash
worldgen generate ... --progress-detail -vv
```

Detailed stage progress includes process start time, current-stage runtime, observed mean stage duration, total elapsed time, ETA, and estimated finish clock time.

Refinement progress is hierarchy-aware. It reports the current absolute refinement depth, recursive node path, node number, field name, current-level completion, action-duration moving average, elapsed time, estimated remaining recursive work, and estimated finish time. The global drainage rebuild emits its own `refined_hydrology_start` and `refined_hydrology_done` events. Verbose progress events can also be written to the normal text or JSONL log.

## Full-relief grayscale height map

Initial generation writes a 16-bit grayscale height map when PNG output is enabled:

```text
maps/02b_height_grayscale_16bit.png
maps/02b_height_grayscale_16bit.json
```

Every completed refinement depth writes the equivalent product under its own `maps/` directory.

The global **deepest modeled point**, normally in the ocean, maps to integer value `0`; the global highest mountain maps to `65535`. Sea level is merely an interior encoded value and is not a clipping boundary. This is therefore a true elevation/bathymetry height field rather than a land-only elevation image or water-level mask.

The JSON sidecar records the physical minimum and maximum elevation represented by the PNG.

## What is actually refined today

The engine distinguishes four operations.

**Inherited continuous forcing fields** are spherically interpolated from the complete parent level. This includes climate scalars, runoff/baseflow forcing, ocean/climate vector fields, suitability fields, and similar globally solved maps.

**Categorical/discrete inherited fields** are sampled with spherical nearest-neighbour semantics. This avoids creating nonsensical fractional plate IDs, rock classes, boolean masks, or integer class codes.

**Refinement kernels** may add genuinely new sub-grid state. The first implemented kernel synthesizes deterministic, globally continuous spherical relief on `elevation_km`; `ocean_depth_m` is then derived from that refined complete elevation/bathymetry field rather than independently interpolated.

**Topology-derived hydrology** is not interpolated at all. Once the refined sphere is fully composed, the engine recomputes:

- Priority-Flood filled elevation;
- flattened `flow_to` receiver addresses for the new raster;
- flow accumulation and drainage area;
- mean/discharge and bankfull indices;
- resolved channel classes and river masks;
- Strahler stream order and width proxy;
- depression-derived lake mask;
- unique terminal-outlet basin IDs;
- three nested sub-basin levels;
- exorheic/internal drainage classification;
- great-circle distance to outlet;
- topographic wetness index;
- height above nearest drainage (HAND);
- sub-grid drainage-density proxy;
- meander/sinuosity proxies.

Because the recomputation is performed on the complete composed level, the result is deterministic and independent of the chosen `--refine-sections` partition for a fixed refined elevation/forcing state.

## Scientific boundary of refinement

Refinement is still not equivalent to rerunning the complete atmosphere-ocean-tectonic world model at the final global resolution.

Tectonic plate history, atmospheric circulation and global ocean circulation have non-local dependencies and are inherited from their previously solved parent/global state. Solving those independently inside child rectangles would generate exactly the section artifacts this system is designed to prevent.

Drainage is different: once the complete refined topography exists, its non-local dependency can be solved globally over that complete sphere. Therefore the receiver graph and watershed hierarchy are now recomputed at every completed refined depth rather than inherited or interpolated.

The refined hydrology currently uses the refined elevation plus inherited/interpolated runoff, precipitation, baseflow and storminess forcing. It does **not** rerun the climate circulation or groundwater PDE at the fine resolution. The global land water-balance model and the multicomponent condensate partition remain parent/global forcing unless a future high-resolution climate/groundwater backend is explicitly invoked.

Passing software tests verifies implementation invariants, not physical validation of every newly synthesized sub-grid feature.

## Scaling characteristics

For constant split and scale factors, full composed cell count grows with the square of linear resolution. With scale `s` and depth `L`:

```text
width_L  = width_0  * s^L
height_L = height_0 * s^L
cells_L  = cells_0  * s^(2L)
```

The number of leaf sections grows with branching factor `B = sections_x * sections_y`:

```text
leaf_nodes_L = B^L
```

Most interpolated fields retain the section-bounded working-set behavior. The refined drainage solve is intentionally a **global O(N) topology stage**, because a correct watershed may span the entire world. It therefore adds full-level working arrays for elevation/fill, receiver addresses and graph accumulators. This is a fundamental dependency cost, not a tile-size artifact.

For very deep refinement, disk-backed field composition remains bounded by section size, while the global drainage stage becomes the principal RAM constraint. Future out-of-core Priority-Flood/receiver backends can reduce that constraint without changing the public refinement contract.
