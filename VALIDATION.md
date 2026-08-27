# Validation

Validation combines deterministic software tests, analytic numerical tests,
failure injection, generated-world diagnostics, and explicit model-level caveats.
Results remain machine/environment dependent unless a test specifies an exact
invariant.

## Automated tests

Run the complete suite with `python -m pytest -q`. It covers target-temperature
orbital calibration, moon/Hill-radius consistency, exact same-seed
reproducibility, different-seed divergence, populated physical/worldbuilding
layers, drainage acyclicity, adaptive convergence, checkpoint dependency hashes,
crash-transactional storage, Priority-Flood backend equivalence, runtime planning,
2-D tiling, benchmark contracts, conserved surface-liquid filling, composition
thermodynamics, barotropic circulation and the advanced exotic-planetary layers.

Spherical numerical regressions use analytic vector fields and explicit seam/pole
fixtures. Geographic morphology, connected components, tectonic boundary
classification, coarse-field resizing, noise octave interpolation, Gaussian
filtering, domain warping and liquid-volume integration all exercise longitude
wrapping plus antipodal pole crossing. These tests establish discrete
implementation correctness; they do not by themselves validate every reduced-order
physical parameterization.

## Surface-liquid conservation

The advanced volatile reservoir solver enforces explicit mass accounting per
species:

```text
total inventory = atmospheric vapor + fixed solid + mobile liquid
                + pressure/phase-incompatible non-condensed excess
```

Only mobile liquid contributes to surface volume. The resulting common liquid level
is solved over exact spherical raster wedges and independently re-integrated. Tests
cover flat spheres, irregular beds, deepest-basin-first filling, zero-liquid dry
worlds, density dependence, multiple liquids, vaporization, freeze-out and
supercritical/gas-only states.

A small non-zero floating-point volume residual is expected; it must remain tiny
relative to the requested liquid volume.

## Barotropic ocean validation

The `barotropic` backend is a reduced-order depth-integrated streamfunction model,
not a primitive-equation ocean. Validation therefore focuses on invariants that the
backend actually promises:

- wet/dry masking and finite currents;
- deterministic monthly fields;
- spherical streamfunction-derived flow;
- low discrete divergence in the wet interior;
- finite kinetic-energy/current diagnostics;
- compatibility with the existing heat-advection interface.

`config/maximal_realism_safe.yaml` explicitly enables this backend; ordinary presets
remain on the cheaper `fast` path.

## Advanced chemistry and volatile-cycle validation

The chemistry layer is intentionally split into thermodynamic and screening tiers.
Tests assert pathway and state behavior rather than laboratory-precision abundance.
Current regression cases include:

- an O2-bearing irradiated atmosphere generates trace O3 rather than bulk ozone;
- an N2/CH4 Titan-like atmosphere generates hydrocarbon, nitrile and tholin products;
- more than one atmospheric condensate can be active simultaneously;
- photochemical aerosol pseudo-species are not silently converted into surface oceans;
- the transported base precipitation field is partitioned conservatively between
  eligible condensates instead of being duplicated for every species.

The model labels photochemical outputs as production/abundance proxies because it
does not integrate a vertical kinetic network.

## Exotic-ocean and cryogeology validation

The exotic-ocean regression suite checks that mobile-liquid mass fractions control
mixture state, including ammonia-water freezing-point depression and lower-density
hydrocarbon liquids. Diagnostic arrays must be finite, correctly shaped and masked
to the modeled liquid reservoir.

Automatic geodynamic tests exercise inactive/stagnant/mobile/tidally forced regimes
and independent cryogenic activity. A high-tidal-heat icy moon must be capable of
entering active/episodic cryotectonics even if ordinary silicate plate tectonics is
weak. Cryogeology fields are checked for bounded shell, melt, fracture and venting
indices.

These are regime classifiers and reduced-order spatial priors. Passing the tests
does not validate a particular moon's shell thickness to observational precision.

## Fluid-dependent geomorphology validation

The methane-river framework no longer assumes that every working fluid has water's
density, viscosity, surface tension or Earth gravity. Tests verify that a Titan-like
CH4/C2H6 mixture resolves methane as the dominant geomorphic fluid and produces
bounded stream-power, evaporation, sediment-capacity and deposition multipliers.
Water/Earth conditions remain the reference state.

The current advanced layer exposes the fluid-scaled parameter block and diagnostic
erosion/deposition fields. It does **not** yet silently re-run the final canonical
terrain with those coefficients, because doing so would require another coupled
terrain -> liquid-level -> ocean -> climate -> hydrology convergence loop to retain
mass and coastline consistency.

## Advanced-preset/end-to-end validation

Composition-aware smoke worlds exercise the canonical `WorldPipeline` and must
return:

- `surface_liquids`;
- `volatile_cycle`;
- `exotic_ocean` when mobile liquid exists;
- `geodynamics`;
- `cryogeology`;
- `geomorphic_fluid_parameters`;
- `exotic_geomorphology`.

Output-enabled runs additionally write `surface_liquids.*`,
`advanced_planetary_physics.json` and `advanced_planetary_fields.npz`.

The full scientific presets (`titan_like.yaml`, `mars_like.yaml`,
`tidal_giant_moon.yaml`, `super_earth.yaml`, `venus_like.yaml`) are intentionally
heavier than unit fixtures. CI uses reduced-resolution/end-to-end variants to catch
pipeline/interface failures without making every Python-version job a long benchmark.

## Resolution and benchmark interpretation

The current `config/default.yaml` simulation grid is 768×384. The 512×256
`config/fast.yaml` preset and the 256×128 `--quick` override are separate fidelity
choices. Use `python -m worldgen.benchmarks --profile micro --output benchmark.json`
for machine-readable kernel and whole-pipeline measurements; generated timing JSON
and `world.json` carry run-specific stage data. Do not compare timings from unlike
resolutions, fidelity settings, optional backends, or output selections.

Benchmark numbers from GitHub-hosted runners are recorded as contextual measurements,
not hard pass/fail thresholds. Shared-runner scheduling, CPU model and thermal state
vary between runs.

## Generated-world diagnostics

A representative Earthlike validation world should include, among other checks:

- area-weighted land fraction approximately equal to the configured target where the
  legacy target-land mode is used;
- the configured home-world radiative/greenhouse target before geographic
  redistribution;
- a full 12-month temperature/precipitation/wind/pressure cube;
- a drainage graph and persistent river network;
- seeded ore/fuel/salt/gem deposit records;
- tropical-cyclone tracks where climate/configuration permits;
- optional settlement/culture/history records.

Composition-aware worlds instead validate their coastline against conserved mobile
liquid volume; they must not be forced back to an arbitrary target land fraction.

## Model-level validation caveat

Passing software tests does not make the model a full physical simulator. The
tectonic component remains kinematic rather than mantle-convection based; the
atmosphere/ocean components are procedural reduced-order models rather than GCMs;
photochemistry is pathway screening rather than vertical kinetics; exotic ocean
mixtures are not full non-ideal equations of state; and cryogeology is not a
viscoelastic shell/fracture simulation.

`SOURCE_COVERAGE.md`, `SURFACE_LIQUIDS.md` and `EXOTIC_PLANETARY_PHYSICS.md` document
where the implementation is transcript-exact, an explicit heuristic, a scientific
substitution, or a generative extension.
