# Validation

The delivered build was executed end-to-end in the sandbox before packaging.

## Automated tests

```text
test_astronomy_target ... ok
test_reproducible ... ok
test_seed_changes_world ... ok
test_world_has_expected_layers ... ok
Ran 4 tests ... OK
```

The tests verify target-temperature orbital calibration, moon/Hill-radius consistency, exact same-seed reproducibility, different-seed divergence, populated resource output, river generation, multiple climate classes, and settlement generation.

## Default 512×256 benchmark in the build environment

The latest complete default run executed all computational stages in a few seconds; PNG/NPZ/JSON output serialization and rendering was the largest single cost. Timing is machine-dependent and is included in each generated `world.json`.

The validation world produced, among other checks:

- area-weighted land fraction approximately equal to the configured 29%;
- physically calibrated configured home-world radiative/greenhouse target of 15°C before geographic redistribution;
- a full 12-month temperature/precipitation/wind/pressure cube;
- a drainage graph and persistent river network;
- hundreds of seeded ore/fuel/salt/gem deposit records;
- tropical-cyclone tracks;
- optional settlement/culture/history records.

## Model-level validation caveat

Passing software tests does not make the model a full physical simulator. The tectonic component is kinematic rather than mantle-convection based, and the atmosphere/ocean components are procedural reduced-order models rather than GCMs. `SOURCE_COVERAGE.md` documents where the implementation is transcript-exact, an explicit heuristic, a scientific substitution, or a generative extension.
