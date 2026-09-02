# Artifexian Auto Worldgen Engineering Roadmap

This roadmap is the implementation plan for evolving the generated world outputs into a physically coherent, sparse, planetary-scale data set that can be streamed and viewed from orbital scale down to metre-scale logical resolution without materializing the full planet at the deepest level.

The global coupled simulation remains the low-frequency physical authority. The cube-sphere quadtree is the random-access address space. Local refinement adds physically motivated detail only where requested, with explicit parent/global boundary constraints.

## Roadmap invariants

Every phase must preserve these properties:

- deterministic output for a fixed world seed, configuration, tile key, and model version;
- bounded memory with work proportional to requested/visible regions, not total logical planet resolution;
- same-LOD watertight cube-face and tile boundaries;
- mixed-LOD renderability through parent fallback, skirts and later geomorphing;
- no reinterpretation of global raster indices such as `flow_to` after reprojection/refinement;
- global-to-local physical consistency: local solvers refine the global state rather than silently replacing continental/global constraints;
- content-addressed/provenance-aware cache invalidation when the source world or model configuration changes;
- scientific products and display products remain explicitly distinguished.

## Phase 0 — Camera-driven LOD correctness and regression gates

**Status:** in progress. `worldgen.lod` exists; comprehensive tests are the immediate gate.

Deliverables:

1. Add and pass comprehensive tests for the camera/screen-space-error (SSE) selector.
2. Verify zoom monotonicity:
   - decreasing camera altitude never produces a coarser maximum selected level for otherwise identical requests;
   - decreasing the allowed screen error never produces a coarser maximum selected level;
   - increasing viewport pixel density can only maintain or increase required detail.
3. Verify selected sets are true quadtree leaf sets: no selected tile may be an ancestor of another selected tile.
4. Verify parent/child coverage across refinement changes. A finer selection must remain covered by ancestors in a coarser compatible selection, including cube-face boundaries.
5. Verify deterministic selection ordering and repeatability.
6. Verify `maximum_level` and `maximum_tiles` safety bounds.
7. Verify resident-memory estimates are bounded by the selected tile budget.
8. Add tests for horizon/footprint behavior, poles, longitude seam and cube-face transitions.
9. Add fallback-chain tests so renderers can retain valid parents until children are resident.

Acceptance gate: all LOD tests pass on every supported Python version and the end-to-end generation/resume workflow remains green.

## Phase 1 — Runtime residency, persistent cache policy and request coalescing

**Status:** planned; sparse persistent tile files already exist.

Deliverables:

1. Add explicit independent budgets for:
   - resident CPU height/scientific arrays;
   - resident render meshes;
   - resident imagery/textures;
   - persistent disk tile cache.
2. Add byte-bounded LRU eviction with deterministic metadata and cache statistics.
3. Add request coalescing/single-flight generation: concurrent requests for the same `(product, face, z, x, y, model revision)` share one generation future/lock rather than duplicating work.
4. Make writes atomic and crash-safe and ensure readers never observe partial products.
5. Add stale-cache invalidation for source-world fingerprints, tile specification changes, local-physics model versions and exporter versions.
6. Add prefetch priority classes: visible-now, predicted-next, fallback-parent and background.
7. Add tests with concurrent callers, intentional interruptions, quota pressure and deep logical zooms.

Acceptance gate: bounded resident/disk usage under adversarial navigation and exactly one generation execution per coalesced tile request.

## Phase 2 — Hierarchical rivers and bounded local geomorphology

**Status:** partially implemented. Open-boundary local D8 hydrology exists; continental constraints and terrain evolution are next.

Deliverables:

1. Preserve continental rivers as explicit hierarchical constraints.
   - Convert the global river network/centerlines into tile-addressable channel constraints.
   - Carry inflow/outflow discharge, stream order, direction and coarse channel position into local supertiles.
   - Prevent local tributary generation from rerouting or deleting major continental rivers.
   - Allow local tributaries/channel morphology to refine around the inherited network.
2. Introduce supertile/halo hydrology with open physical boundaries rather than spherical wrapping inside local cube tiles.
3. Integrate bounded stream-power incision, for example a calibrated form of `E = K A^m S^n`, using locally refined discharge/drainage and geology-dependent erodibility.
4. Add sediment production, routing, deposition, alluvial filling and delta/channel deposition where local boundary information permits it.
5. Add hillslope diffusion/mass-wasting with a stable timestep/iteration policy and slope constraints.
6. Anchor/reconcile terrain near authoritative tile boundaries:
   - solve on a halo/supertile larger than the output core;
   - crop only the converged core;
   - apply parent-constrained boundary blending where a finite-domain operator would otherwise alter shared edges;
   - prove independently generated neighboring cores remain watertight.
7. Propagate coarse/global drainage area and discharge as boundary/source terms instead of pretending a local patch can rediscover an entire continental catchment.
8. Add conservation diagnostics for removed/deposited sediment and terrain-volume change.
9. Add seam, parent-child consistency and independent-generation tests.

Acceptance gate: neighboring independently generated tiles have identical shared terrain boundaries, major rivers remain topologically consistent with the global network, and local erosion/deposition diagnostics satisfy configured conservation tolerances.

## Phase 3 — Local topographic climate downscaling

**Status:** temperature lapse-rate downscaling exists; wind/precipitation refinement is planned.

Deliverables:

1. Retain existing terrain-aware annual/monthly lapse-rate temperature correction as the baseline.
2. Add local wind downscaling from inherited global winds using high-resolution terrain normals, slope/aspect, roughness, channel/valley orientation and exposure.
3. Add orographic precipitation enhancement on windward slopes and physically bounded rain-shadow drying on lee slopes.
4. Preserve parent/global monthly precipitation totals or area-weighted regional constraints to prevent local downscaling from creating/losing arbitrary water mass.
5. Downscale humidity, snow fraction and potential evapotranspiration consistently with the local temperature/wind/precipitation state.
6. Feed local runoff back into Phase 2 hydrology in an explicitly iterated/coupled refinement loop where enabled.
7. Define an `atmogen` coupling boundary for cases where local pressure/elevation/temperature columns materially change condensation/cloud/radiative state; use representative/batched columns rather than embedding atmospheric chemistry in the terrain package.
8. Add tests for windward/leeward asymmetry, mass-constrained precipitation redistribution, seasonal behavior and tile-boundary continuity.

Acceptance gate: local climate fields react to resolved terrain while preserving configured parent/global conservation constraints and remaining continuous across independently generated neighboring tiles.

## Phase 4 — High-resolution derived surface state

**Status:** planned; basic height/true-color display products already exist.

Deliverables:

1. Generate tile-local metric slope, aspect and render normal maps from the cube-sphere geometry rather than planar-pixel assumptions.
2. Generate soil-moisture state from downscaled precipitation/runoff, drainage position, soil/geology permeability and evapotranspiration.
3. Generate snow accumulation/persistence from monthly local temperature, precipitation, insolation/exposure and melt conditions.
4. Generate local biome/ecological suitability and vegetation fractions constrained by the global biome/ecology state.
5. Generate local surface albedo consistently from snow, water, soil, rock and vegetation fractions.
6. Add optional roughness/material/wetness layers useful to physically based renderers while documenting which are physical diagnostics versus artistic display products.
7. Produce globally consistent mip/quantization metadata so neighboring texture tiles decode identically.
8. Add numerical seam tests for every continuous derived layer and categorical parent-consistency tests for biome classes.

Acceptance gate: a complete requested terrain tile can supply geometry plus physically coupled surface layers without requiring a globally materialized high-resolution raster.

## Phase 5 — Vector feature tiling

**Status:** planned.

Deliverables:

1. Build hierarchical vector tiles for:
   - major and local rivers;
   - shorelines/coastlines and lake boundaries;
   - settlements and labels;
   - resource deposits/regions;
   - tectonic/geologic linework where useful;
   - later roads/political/society features.
2. Clip geometry to cube-sphere tile domains with stable feature IDs and parent/child lineage.
3. Preserve cross-tile line continuity and avoid duplicate/missing features at boundaries.
4. Add level-dependent simplification/generalization while retaining topology.
5. Separate world-space authoritative geometry from viewer-specific vector encodings.
6. Add spatial indexes so only intersecting features are fetched/generated.

Acceptance gate: vector features can be streamed independently at appropriate LOD without rasterizing the whole feature set.

## Phase 6 — Viewer-facing delivery and streaming service

**Status:** planned. CLI/API/disk-cache paths exist; a network/viewer layer does not yet.

Deliverables:

1. Add a local service layer exposing tileset metadata, camera/LOD resolution, tile availability and lazy product generation.
2. Provide content types for height arrays, imagery, meshes, derived scientific layers and vector tiles.
3. Use bounded worker queues, request coalescing, cancellation and backpressure so rapid camera motion cannot create an unbounded generation backlog.
4. Support HTTP caching semantics/ETags derived from immutable tile fingerprints.
5. Add a reference globe viewer demonstrating:
   - camera-driven SSE traversal;
   - parent fallback while children load;
   - mixed-LOD skirts;
   - later geomorph/cross-fade;
   - independent CPU/GPU/disk cache budgets;
   - selective scientific/display layer toggles.
6. Keep the service optional and local-first; generation APIs remain usable without a web server.
7. Add integration/load tests that simulate rapid pan/zoom across the globe.

Acceptance gate: an interactive viewer can move from orbital to local scales while network/generation work remains bounded by the visible/prefetch working set.

## Phase 7 — Standardized terrain and 3-D exporters

**Status:** planned; internal cube-sphere mesh/product representation exists.

Deliverables:

1. Add Cesium quantized-mesh export, including shared edge vertices, edge lists/skirts, geometric error and appropriate projection/tiling adaptation.
2. Add OGC 3D Tiles output where applicable, including implicit tiling/subtree availability for sparse quadtree content.
3. Add glTF/GLB mesh export for engine-neutral terrain chunks.
4. Add game-engine-oriented raw height/normal/albedo/material tile manifests.
5. Add imagery export to PNG plus optional WebP/AVIF when dependencies are available.
6. Add scientific chunk export suitable for random access, e.g. Zarr-compatible layouts where practical.
7. Version exporters separately from the authoritative internal tile schema so format changes do not invalidate physical terrain unless necessary.
8. Add conformance/round-trip tests against available validators/readers.

Acceptance gate: at least one established external globe/3-D client can consume an exported sparse tileset without custom access to the internal NumPy cache.

## Phase 8 — Deep-zoom, performance and scientific validation

**Status:** continuous/final hardening.

Deliverables:

1. Stress-test very deep logical zooms while materializing only a small working set.
2. Verify no algorithm accidentally scales with the theoretical global cell count at the requested maximum zoom.
3. Benchmark tile latency, throughput, memory, cache hit rate, request-coalescing effectiveness and storage amplification.
4. Add randomized seam tests across all six cube faces and multiple LODs.
5. Add deterministic hash regressions across worker counts and execution order.
6. Add local physical diagnostics: sediment closure, precipitation redistribution closure, runoff/discharge consistency and derived-layer bounds.
7. Add compatibility/version migration tests for tile manifests and caches.
8. Add end-to-end viewer navigation benchmarks and long-running cache-quota tests.

Acceptance gate: the system remains deterministic, bounded and navigable at metre-scale logical resolution without prebuilding the planet and with explicit diagnostics for the degree of physical fidelity at each scale.

## Cross-repository ownership

`artifexian_auto_worldgen` owns the cube-sphere address space, terrain/hydrology/climate downscaling, caching, vector tiling, viewer delivery and exporters.

`atmogen` remains the authoritative atmosphere/ocean-chemistry/cloud/radiation package. Worldgen may request representative or local atmospheric columns from it when local elevation, pressure, temperature, composition or stellar forcing require atmospheric recomputation. The terrain system must not duplicate `atmogen` chemistry or radiation logic. As of atmogen API 11, representative-column fingerprints, de-duplication, convergence/fallback counts, explicit elevation-adjusted surface-pressure boundaries and per-column provenance are supplied by atmogen and recorded by worldgen rather than reconstructed in the terrain package. Ocean bathymetry is clamped to the atmospheric datum rather than being mistaken for below-surface atmospheric elevation.

The corresponding `atmogen` roadmap tracks the API/batching/provenance work necessary to support those multiscale worldgen requests.
