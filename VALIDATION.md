# Validation

Validation combines deterministic software tests, analytic numerical tests,
failure injection, and generated-world diagnostics. Results remain
machine/environment dependent unless a test specifies an exact invariant.

## Automated tests

Run the complete suite with `python -m pytest -q`. It covers target-temperature
orbital calibration, moon/Hill-radius consistency, exact same-seed
reproducibility, different-seed divergence, populated physical/worldbuilding
layers, drainage acyclicity, adaptive convergence, checkpoint dependency hashes,
crash-transactional storage, Priority-Flood backend equivalence, runtime planning,
2-D tiling, and benchmark contracts.

Spherical numerical regressions use analytic vector fields and explicit seam/pole
fixtures. Geographic morphology, connected components, tectonic boundary
classification, coarse-field resizing, noise octave interpolation, Gaussian
filtering, and domain warping all exercise longitude wrapping plus antipodal pole
crossing. These tests establish discrete implementation correctness; they do not
by themselves validate every reduced-order physical parameterization.

## Resolution and benchmark interpretation

The current `config/default.yaml` simulation grid is 768×384. The 512×256
`config/fast.yaml` preset and the 256×128 `--quick` override are separate fidelity
choices. Use `python -m worldgen.benchmarks --profile micro --output benchmark.json`
for machine-readable kernel and whole-pipeline measurements; generated timing JSON
and `world.json` carry run-specific stage data. Do not compare timings from unlike
resolutions, fidelity settings, optional backends, or output selections.

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
