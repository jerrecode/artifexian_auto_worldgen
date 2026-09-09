# Scientific and Software Architecture for a Deterministic Multiscale Procedural Planet

Status: architecture baseline, 2026-09-09.

This document is the implementation architecture for `artifexian_auto_worldgen`.
It builds on the repository's existing global Earth-system pipeline, 49-transcript
Worldbuilder's Log inventory, sparse cube-sphere terrain hierarchy, recursive/local
refinement, transactional checkpointing, and `atmogen` coupling. It does not create
a second unrelated generator.

Core invariant:

> One versioned project state identifies one causal planet. Coarse state constrains
> finer state; fine state refines rather than reinvents parent geography; maximum
> global resolution is never materialized merely because a local camera zooms in.

## 1. Artifexian / WorldSmith feature and adjustability matrix

`TRANSCRIPT_INVENTORY.json` contains 49 supplied transcripts covering logs 0-50
except 33 and 40. `SOURCE_COVERAGE.md` is the current transcript-to-code trace.
Artifexian is treated as the minimum benchmark for user agency and decision density,
not as an unquestionable scientific source.

| Domain | Existing foundation | Required extension |
|---|---|---|
| star/orbit/moon | `astronomy.py`, `tides.py` | parameter metadata, inverse goals, uncertainty |
| tectonic history | `tectonics.py`, `geodynamics.py` | stable feature IDs, editable constraints, event provenance |
| topography/bathymetry | `terrain.py`, `ocean.py`, `planet_tiles.py` | causal multiscale residuals |
| atmosphere/ocean/climate | `climate.py`, ocean modules, `atmogen` | conserved local downscaling and fidelity tiers |
| hydrology | priority flood, `hydrology*`, `watersheds.py` | hierarchical basin/channel boundary contracts |
| geomorphology | erosion/local geomorphology/surface evolution | fluvial, sediment, mass-wasting, glacial and coastal coupling |
| geology/resources | geology/lithology/resources | stratigraphy, deposit provenance, technology accessibility |
| ecology | appearance/local surface | soils, functional groups, dispersal/evolution |
| settlement/humans | `society.py` | component-wise suitability and migration cost surfaces |
| authoring | configuration/presets | formal intent, locks, sparse overrides, variants, inverse worldbuilding |

Every subsystem should expose Basic, Guided, Expert and Developer parameter views.
Basic controls are Earth-normalized tendencies where meaningful; Expert controls use
explicit physical units; Developer controls expose numerical methods and tolerances.

## 2. Scientific improvements over Artifexian

The engine replaces inexpensive hand-painted proxies with causal reduced-order
models where doing so has material scientific or worldbuilding value. Major
improvements are spherical calculations at source, explicit water and sediment
accounting, geology-conditioned erodibility, hydrology/topography feedback,
scale-aware terrain refinement, numerical conservation diagnostics, quantitative
Earth-calibrated validation, and provenance tags distinguishing physical inference,
empirical approximation, heuristic, stochastic procedure and artistic override.

The system does **not** claim unique deterministic prediction of an 850 Myr history.
Long-term tectonics, climate history and evolution are underconstrained and partly
chaotic; the correct target is physically admissible causal coherence, not fake
precision.

## 3. Global dependency graph

Persisted-product invalidation uses an acyclic graph:

```text
astronomy -> planet -> tectonics -> geology -> macro_terrain
macro_terrain -> atmosphere_ocean -> climate
macro_terrain -> surface_coupling
climate -> surface_coupling
tides -> surface_coupling
surface_coupling -> ecology -> habitability -> society
geology -> resources -> habitability
surface_coupling -> terrain_refinement
all visible scientific products -> viewer_products
```

Strong numerical feedback loops are collapsed into product groups when necessary.
Changing erosion therefore does not invalidate tectonics; changing plate count does.

## 4. Feedback loops

Important bounded iterative loops include climate/ocean heat transport; climate,
snow/ice and albedo; precipitation, runoff, channel flow, erosion, topography and
drainage; soil moisture, vegetation and evapotranspiration; vegetation/root cohesion
and erosion; and river sediment supply, deltas/coasts and shoreline geometry.

Loops terminate on physically scaled residual criteria and iteration ceilings.
Non-convergence is recorded explicitly and may invoke a documented fallback.

## 5. Spherical grid comparison

| Grid | Main strength | Main weakness for this engine | Role |
|---|---|---|---|
| lat/lon | simple interchange | polar singularity and severe convergence | export only |
| cubed sphere | six nonsingular regular patches, natural quadtree/GPU tiles | metric distortion and face transforms | **authoritative hierarchy** |
| icosahedral/geodesic | near-uniform triangles, strong finite volume | less convenient rectangular GPU patch hierarchy | optional solver research |
| HEALPix | exact equal-area, nested, excellent harmonic/statistical sampling | awkward walkable terrain patches | analysis/statistics adapter |
| spherical Voronoi/Delaunay | adaptive conservative meshes | complex dynamic indexing/rendering | specialized research |
| spherical clipmap | excellent camera-local rendering | not a scientific state grid | renderer technique |

The selected authoritative grid remains a metric-aware hierarchical cubed sphere.
Gradients, divergence, cell area and fluxes must use the spherical/cube-face metric,
not naive planar pixel distances. HEALPix remains useful for equal-area statistics
and spherical-harmonic diagnostics.

Scientific basis: Ronchi, Iacono & Paolucci (1996), DOI 10.1006/jcph.1996.0047;
HEALPix documentation, https://healpix.sourceforge.io/.

## 6. Selected hierarchy

The physical hierarchy is:

```text
planet -> cube face -> quadtree tile -> child tile -> ... -> local metric frame
```

The current `maximum_level <= 30` guards are operational limits, not the conceptual
architecture. Future logical addressing should be extendable until physical
resolution, numeric precision or configured resource limits make further subdivision
meaningless.

## 7. Tile-address design

Canonical geometric identity remains `(face, level, x, y)`. Product/cache identity
adds body ID, layer/product, algorithm-contract version and dependency fingerprint.
Children use `(2x+dx, 2y+dy)` for `dx,dy in {0,1}`.

Numerical kernels receive an authoritative core plus process-specific halo or
supertile. A border value may never depend on which neighbor happened to generate
first.

## 8. Master-seed derivation architecture

A canonical 256-bit master seed is derived from arbitrary-size integers, explicit
hexadecimal values, UTF-8 strings or bytes with a versioned domain-separated BLAKE2b
scheme. New randomness is derived by semantic paths, for example:

```text
master / tectonics / plate-ID / epoch-ID / event
master / hydrology / basin-ID / river-ID / process
master / terrain / tile-address / physical-scale-band / process
```

Parallel streams use Philox or explicit counter hashes. Random output therefore
depends on semantic identity and an explicit sample counter, not mutable call order,
worker count or task scheduling. Existing integer `RngPool(name)` streams remain
bit-for-bit compatible until a subsystem explicitly changes algorithm contract.

## 9. Project/state schema

A project separates editable intent from physical products and cache state:

```text
project.toml
project-state.json
intent/
checkpoints/
products/
cache/
exports/
logs/
metrics/
```

The canonical world fingerprint is:

```text
H(canonical master seed,
  canonical physical configuration,
  canonical algorithm versions,
  canonical overrides)
```

Logging, UI state, worker count, output image format and cache quota are excluded
from physical identity unless documented to change a numerical contract.

## 10. Four independent LOD domains

The engine distinguishes simulation LOD, stored-data LOD, geometric/render LOD and
material/microdetail LOD. Walking at human scale does not cause global climate or
tectonics to run at centimeter resolution. Local terrain receives coarse physical
boundary fields and resolves only processes whose scales become newly relevant.

## 11. Parent/child consistency

Continuous child fields are modeled conceptually as:

```text
child = reconstructed_parent + constrained_physical_residual
```

Residuals occupy newly resolvable spatial frequencies, honor border/flux constraints
and may not arbitrarily change parent-scale integrals. Topological features such as
plates, faults, basins, rivers, coastlines and deposits use stable lineage IDs.

High-resolution changes are classified as local-only, parent-render aggregation,
physically important upward correction, or explicit inconsistency requiring action.

## 12. Hierarchical hydrology continuity

Coarse hydrology owns major basin topology, divides, river trunks, outlets and
long-range inflow/outflow. Fine hydrology owns tributaries, meanders, bars, oxbows,
floodplains, gullies and small depressions.

A local solve receives parent basin ID, boundary water/sediment fluxes, inherited
river identities/discharge, precipitation/runoff fields and a suitable halo. A local
crop may not reroute a continental river across a parent divide because context is
missing.

## 13. High-detail refinement strategy

Macro terrain comes from tectonics, crust/isostasy, volcanism and long-wavelength
erosion. Regional terrain adds catchments, faults/scarps, lithologic boundaries,
orographic climate, fluvial incision, landslides, glaciers and coasts. Local terrain
adds tributaries, gullies, terraces, meanders, point bars, talus, fans, moraines and
dunes. Pebbles, small rocks, vegetation instances and sub-centimeter roughness
belong primarily to materials/instances rather than planetary mesh vertices.

## 14. Terrain-frequency strategy

A grid spacing `dx` cannot faithfully encode geometry with wavelength near or below
roughly `2*dx`. Newly materialized frequencies must therefore be appropriate to the
active physical scale. Wet erodible hillslopes gain drainage-linked gullies, hard
strata gain differential erosion/cliffs, arid loose sediment gains aeolian forms,
and glaciated terrain gains glacial forms. Generic white noise is not physical
detail.

Validation measures terrain spectra, structure functions, slope/curvature,
drainage density and cross-scale residual energy to detect interpolation-only detail
or grid-aligned procedural artifacts.

## 15. Storage/cache architecture

Chunked scientific arrays should prefer Zarr where random partial access is useful;
HDF5 remains suitable for compact archives/interchange; SQLite holds indexes,
manifests and job metadata rather than giant rasters.

Cache tiers are GPU -> RAM -> persistent disk -> deterministic regeneration.
Irreplaceable edits/imports are never ordinary-cache evicted. Expensive reproducible
products are preferentially persisted/compressed; cheap reproducible detail may be
regenerated. Writes are atomic and checksummed.

OGC Zarr Storage Specification: https://www.ogc.org/standards/zarr-storage-specification/.

## 16. Resource-budget architecture

CPU, RAM, VRAM, disk cache, I/O, queue depth and streaming bandwidth are explicit
budgets. Under pressure the system cancels speculative work, reduces prefetch,
coalesces/evicts cheap work, delays refinement and only then relaxes visual
screen-space error. Visible valid terrain and canonical user data have priority.

Concurrency is admitted by estimated working set as well as core count.

## 17. Asynchronous task graph

Every task carries stable ID, parent, priority, dependencies, cost/memory estimate,
backend requirement, cancellation token, progress node, retry policy and content
key. Duplicate requests share one future. Queues are bounded. Camera/preview work
uses latest-request-wins cancellation and outranks batch prerendering.

Threads serve I/O; process/native/vectorized kernels serve CPU-heavy work; GPU jobs
are VRAM-budgeted. Completion order cannot alter canonical results.

## 18. Progress architecture

The shared progress tree supports arbitrary nesting and stores ID, parent, name,
description, state, completed, total, units, workload weight, elapsed time, smoothed
throughput, ETA, warnings, children, current operation and resource metrics.

Parent progress is workload-weighted, never a naive equal average. Unknown totals
remain unknown rather than fabricating a percentage. Structured events include
`TaskCreated`, `TaskStarted`, `TaskProgress`, `TaskPaused`,
`TaskCompleted`, `TaskFailed`, `TaskCancelled`, `WarningRaised` and
`MetricUpdated`. CLI, TUI and GUIs consume this same service.

## 19. Logging, tracing and metrics

Structured logs carry component, job/task/correlation ID, world fingerprint, tile,
LOD, process/thread, operation, duration and exception context. Metrics are separate:
cache hit rate, queue depth, cells/s, tile latency, RAM/VRAM, disk bytes, upload
time, triangle count and solver residuals.

A correlation trace must connect camera request -> tile traversal -> cache lookup ->
shared generation task -> dependencies -> generation/load -> upload -> render.

## 20. Exception and recovery architecture

Domain errors distinguish configuration, validation, numerical/convergence,
storage/cache, resource-limit, generation/chunk, plugin, rendering and serialization
failures while preserving exception chaining.

Numerical guards detect NaN/infinity, impossible signs, divergence, invalid topology,
uphill/looping drainage, mass-balance failure and instability. Retries are bounded
and classified. Reproducible corrupt cache may be regenerated; irreplaceable user
state is never silently modified. Optional fine-detail failure can fall back to a
valid parent representation.

## 21. CLI / TUI / Generation GUI / Viewer architecture

All interfaces call one headless generation/service core:

```text
scientific kernels
 -> product contracts/dependency graph
 -> task/checkpoint/cache/progress/logging services
 -> world service API
 -> CLI / TUI / Generation GUI / Viewer / tests
```

The CLI remains the automation/CI authority. The TUI is keyboard-first authoring and
operations. The Generation GUI focuses on parameters, intent/locks, fast preview,
candidate comparison, jobs and diagnostics. The Viewer focuses on camera-driven
streaming and scientific inspection including first-person movement. No interface
owns a competing generation algorithm.

## 22. Preview architecture

Preview uses the same seed, stable feature identities, sphere geometry and model
family as final generation, with lower spatial/temporal resolution, fewer iterations
and expensive downstream systems disabled.

Suggested tiers: `instant` (plate partition/velocities/boundaries), `fast`
(tectonic evolution plus broad relief), `detailed` (higher tectonic resolution and
cheap derived approximations), `final` (requested full pipeline). Debounce,
cancellation and dependency-local invalidation make slider editing responsive.

## 23. Plugin architecture

Plugins register versioned contracts declaring plugin/version, input/output
products, supported scale/fidelity, determinism guarantee, CPU/GPU requirements,
working-set class, parameter schema and provenance. Physics-changing plugins
participate in world identity; viewer-only plugins do not. Plugins extend public
contracts rather than monkey-patching private implementation state.

## 24. Scientific validation methodology

Validation layers include:

- invariants: finite values, legal topology, seam continuity and conservation;
- Earth-calibrated statistics: hypsometry, bathymetry, orogen dimensions, basin
  distributions, drainage/Horton/Hack-like metrics, river sinuosity and climate;
- terrain spectra/roughness at context-appropriate scales;
- high-resolution DEM/LiDAR comparisons for slope, curvature, hillslope length and
  channel geometry where datasets permit;
- artifact detectors for cube/grid alignment, repeated phase, stripes, straight
  rivers, ridge crossings, uphill flow, loops, biome discontinuity and LOD cracks;
- determinism matrix across serial/parallel order, cache deletion, restart and
  interface entry points;
- multiple validation presets rather than tuning to one Earth-like seed.

Priority-Flood foundations: Barnes, Lehman & Mulla (2014), Computers & Geosciences
62. Stream-power incision foundations: Whipple & Tucker (1999), JGR 104(B8).

## 25. Implementation milestones

The repository is already beyond a greenfield architecture phase, so work is
reordered around missing contracts.

**M0 — current:** canonical seed identity, semantic/counter RNG, world/component
fingerprints, dependency-local invalidation, weighted nested progress, this document
and focused tests.

**M1:** project manifest/intent model, sparse hard/strong/soft/suggestion overrides,
locks, staleness planner, TOML project schema and migrations.

**M2:** bounded priority task DAG, shared futures, memory/VRAM/I/O reservations,
cancellation/backpressure/retries, tracing and scheduler metrics.

**M3:** physically constrained sparse local refinement with halos/supertiles,
hierarchical river/fault/geology contracts, local drainage/sediment/hillslope solve,
conservation and seam tests, parent+residual storage experiments.

**M4:** screen-space-error streaming service with frustum/horizon culling, minimal
visible tile traversal, bounded predictive prefetch and RAM/GPU eviction.

Projected terrain error is approximately:

```text
error_pixels ~= (geometric_error_m / distance_m)
               * viewport_height_pixels / (2*tan(vertical_fov/2))
```

In natural language, apparent error grows with unresolved terrain height error and
screen height, and shrinks with camera distance and wider field of view.

**M5:** preview service, candidate gallery, authoring CLI/TUI, then Generation GUI.

**M6:** viewer, camera-relative/local-frame rendering, batched patch draws,
virtual/material streaming, collision LOD and space-to-ground acceptance test.

**M7:** deeper climate/ocean downscaling, coasts/tides, soils, glaciers, ecology,
evolution, stratigraphy/resources, technology accessibility and migration.

**M8:** Zarr/3D Tiles/glTF interoperability, calibrated validation, fault injection,
profiling-guided CPU/GPU optimization and full documentation.

OGC 3D Tiles 1.1 is an interoperability target rather than the internal terrain
authority: https://www.ogc.org/standards/3DTiles/.

## Major deliberate deviations from the request

1. No exact long-term mantle/tectonic prediction; kinematic/stochastic history with
   geodynamic constraints is more scientifically honest and tractable.
2. No universal maximum-resolution grid; each process has its own relevant scale.
3. Existing spectral high-frequency terrain is retained only as unresolved
   deterministic boundary detail, not mislabeled as solved geomorphology.
4. Cubed sphere is authoritative, but equal-area/statistical and external geodetic
   grids remain adapters where objectively superior.
5. Global consistency comes from hierarchy, stable identities, constraints and
   flux contracts—not a physically impossible maximum-resolution global array.
6. New stochastic code may not share one mutable RNG sequence.
7. Cache identity is dependency/version/content addressed, not filename addressed.

## Definition of done

The architecture is successful only when executable acceptance tests show that:

1. the same semantic tile is identical across task order, worker count, cache
   deletion and restart;
2. a local stream can trace its causes through basin, precipitation, geology,
   mountain uplift and tectonic history;
3. space-to-ground navigation never allocates the full high-resolution planet;
4. detailed state can be evicted and later reconstructed identically;
5. unrelated parameter edits neither reroll nor invalidate independent geography;
6. CLI/TUI/GUI/viewer observe the same project/tasks/progress/generation core;
7. physical approximation and numerical error remain measurable and visible.
