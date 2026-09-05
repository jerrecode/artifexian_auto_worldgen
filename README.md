# Artifexian Automatic World Generator

## Standalone atmosphere/material backend

The optional `atmogen.enabled` path uses the separately versioned `atmogen` package
as the authoritative representative-column atmosphere, phase-reservoir, cloud and
spectral-radiation backend. The dependency is pinned to a compatible Git revision;
run manifests and checkpoint keys record its package/API/data versions and material
database hash. For sibling development, install `../atmogen` editable before this
package. See `config/atmogen_fast.yaml` for the compact configuration boundary.

Worldgen still owns spherical terrain, horizontal climate/ocean transport, and the
exact spherical-wedge liquid-volume fill. On the new path, the phase masses,
densities and liquid volumes passed into that geographic fill come from `atmogen`.
The legacy reduced-order composition chemistry/optics layer is not run in parallel
when `atmogen` is enabled. The current integration solves one representative FAST
column; representative-column clustering and iterative horizontal column coupling
remain future work and are not claimed as implemented.

A deterministic, configurable Python pipeline that reconstructs the *kind of work* performed across the supplied **Worldbuilder's Log** transcripts as one automated procedural system.

The optional environment-conditioned procedural erosion layer is documented in
[`PROCEDURAL_EROSION.md`](PROCEDURAL_EROSION.md), including the Runevision
phase-cell lineage, planetary forcing fields, numerical invariants, recoupling
semantics, and known limitations.

The goal is not to automate Photoshop, GPlates, Blender, Desmos, or spreadsheet clicks. The goal is to replace those manual representations with the underlying data and operations they stand for: spherical geometry, orbital physics, kinematic plate history, continuous raster fields, hydrology, climate, rule-based geology/resource generation, and a seeded constrained history generator.

## What it generates

One command produces a complete world dependency chain:

```text
seed + config
  └─ astronomy
      ├─ star / habitable zone / orbit / year
      ├─ seeded multi-planet system + mutual-Hill spacing diagnostics
      ├─ local 3-D stellar neighbourhood + approximate magnitudes
      ├─ planet mass, radius, gravity, escape velocity, temperature
      ├─ moon stability, month, tides proxy, apparent angular sizes
      └─ atmosphere partial pressures / density
          ↓
  spherical grid
          ↓
  850 Myr kinematic plate history
      ├─ current plates and boundary types
      ├─ convergence / rifting history
      ├─ orogen and rift ages
      ├─ oceanic crust ages
      └─ hotspots / LIPs
          ↓
  terrain + coastlines + shelves + bathymetry
          ↓
  ocean currents + upwelling + SST anomalies
          ↓
  monthly climate
      ├─ insolation / seasons
      ├─ temperature / elevation lapse rate / continentality
      ├─ pressure belts
      ├─ wind fields
      ├─ advected moisture / orographic precipitation
      └─ quantitative Köppen-Geiger classes
          ↓
  priority-flood hydrology
      ├─ basins / runoff / flow directions
      ├─ lakes
      └─ river network
          ↓
  weather
      ├─ fog
      ├─ thunderstorms / lightning / tornado potential
      ├─ tropical cyclone genesis + tracks
      ├─ blizzards
      ├─ sand/dust storms
      └─ aurora oval
          ↓
  geology
      ├─ paleoshallow-sea likelihood
      ├─ cratons / shields / platforms
      ├─ sedimentary basins / metallogenic belts
      └─ surface rock classes
          ↓
  fuels + ores + salt
      ├─ wood / peat / coal
      ├─ Copper Age rules
      ├─ Bronze Age rules
      ├─ Iron Age rules
      ├─ salt flats / halite / sea-salt access
      └─ technology-resource intersection maps
          ↓
  optional human settlement/history layer
      ├─ portal-location constraints
      ├─ settlement suitability
      ├─ seeded settlements / cultural branches
      ├─ contact/trade links
      └─ historical events
```

## Installation

Python 3.11+ is recommended.

```bash
cd artifexian_auto_worldgen
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Dependencies are deliberately small: NumPy, SciPy, Matplotlib, and PyYAML.

## Generate a world

```bash
worldgen --config config/default.yaml --out world-out
```

Or without installation:

```bash
PYTHONPATH=src python -m worldgen \
  --config config/default.yaml \
  --out world-out
```

Fast iteration:

```bash
worldgen --config config/default.yaml --out quick-world --quick
```

Higher-detail preset:

```bash
worldgen --config config/high_detail.yaml --out detailed-world
```

Change only the seed:

```bash
worldgen --config config/default.yaml --out another-world --seed 123456789
```

Physical/geofiction world only:

```bash
worldgen --config config/default.yaml --out physical-world --no-society
```

## Output

```text
world-out/
├── world.json                 human-readable metadata, deposits, tracks, societies
├── features.geojson           vector points/lines: deposits, settlements, portal, cyclone tracks
├── world_arrays.npz           randomly accessible numerical fields
├── world_report.md            compact generated report
└── maps/
    ├── 01_plate_ids.png
    ├── 02_elevation.png
    ├── 03_ocean_crust_age.png
    ├── 04_temperature_annual.png
    ├── 05_precipitation_annual.png
    ├── 06_koppen.png
    ├── 07_continentality.png
    ├── 08_rivers.png
    ├── 09_geology.png
    ├── 10_resource_intensity.png
    ├── 11_thunderstorms.png
    ├── 12_hurricane_genesis.png
    ├── 13_sea_ice_coral.png
    ├── 14_settlement_suitability.png
    └── legends.json
```

The `.npz` file is the main machine-readable numerical world. For example:

```python
import numpy as np

w = np.load("world-out/world_arrays.npz")
height_km = w["elevation_km"]
monthly_temperature = w["temperature_c_monthly"]
rivers = w["rivers"]
copper = w["resource_copper_rich"]
```

## Performance architecture

The implementation is designed around arrays rather than Python objects per map cell. Expensive spatial operations use NumPy/SciPy vectorization through a canonical spherical topology layer; pole crossings reflect latitude and rotate longitude by 180 degrees, while longitude is periodic. Spherical plate assignment uses chunked matrix multiplication. Hydrology remains the principal intentionally sequential operation because the drainage graph must be topologically accumulated.

Every stage receives an independent deterministic `PCG64DXSM` random stream derived from the root seed and a stage name. This is important for worldbuilding: changing ore density should not unexpectedly reroll the star, and changing the climate implementation should not intentionally reroll plate seeds.

`--quick` uses a 256×128 map, 50 Myr tectonic history sampling, and fewer moisture-advection iterations. `config/default.yaml` is 768×384; the separate `config/fast.yaml` preset is 512×256. The pipeline can be raised to 1536×768 or more in YAML, with roughly quadratic growth in raster work.

## Important mathematical substitutions

Some videos visibly use calculators/spreadsheets but do not speak every cell formula. In those places this project uses transparent standard relationships rather than trying to reverse-engineer hidden spreadsheet cells.

Examples:

### Keplerian year

In solar/Earth units,

\[
P_{\rm yr}=\sqrt{\frac{a_{\rm AU}^{3}}{M_\star/M_\odot}}.
\]

### Surface gravity

\[
\frac{g}{g_\oplus}=\frac{M/M_\oplus}{(R/R_\oplus)^2}.
\]

### Hill radius

\[
r_H \approx a(1-e)\left(\frac{m_p}{3M_\star}\right)^{1/3}.
\]

### Thermal ocean-floor subsidence

The code uses a configurable square-root age law,

\[
d(t)=d_0+k\sqrt{t},
\]

then modifies it with shelves, ridges, trenches, LIPs, and hotspots. The defaults approximately reproduce the transcript's discussed young-crust and 50 Myr depth scale.

### Monthly climate

Daily-mean insolation is computed from latitude and solar declination. Temperature then adds continental thermal amplitude, ocean-current anomalies, and an elevation lapse rate. Pressure gradients provide winds; moisture is iteratively advected along those winds and removed through convection/orographic condensation.

### Hydrology

Depressions are resolved by Priority-Flood. A D8 drainage graph then routes runoff strictly downhill. Flow accumulation gives drainage area/discharge proxies, from which a configurable upper quantile becomes the persistent river network.

## Exact heuristics retained from the transcripts

Where an explicit procedure was spoken, it is encoded rather than replaced arbitrarily. Examples include:

- the tree-line rule with 4000 m between ±30°, then -130 m per latitude degree for the next 20°, then -75 m/degree until the tree line reaches zero;
- the roughly 4.6 m/Myr global erosion control used when discussing old mountain ranges;
- target land fraction near 29%;
- crust-age-driven bathymetry beginning around 2600 m for very young oceanic crust;
- thunderstorm influence zones and the `<1`, `1–5`, `5–15`, `15+` flashes/km²/year interpretation;
- bog iron in suitable wetlands downhill of metallogenic belts;
- VMS/SEDEX/skarn/placer/laterite/oolitic and other resource rules as dependencies on tectonics, rocks, climate and hydrology;
- salt flats from arid sedimentary depressions and halite from former shallow seas;
- the distinction between strict world state, configurable assumptions, and underspecified/head-canon choices in the human layer.

The generator also emits a quantitative annual temperature-range/continentality layer and a gemstone extension. These are explicitly documented as external-guide supplements because the corresponding transcript material was missing or delegated/commissioned rather than fully demonstrated in the supplied files.

## Where automation necessarily differs from the videos

This is deliberately **not** a literal GPlates clone or a full physical Earth-system simulator.

The plate model is kinematic: spherical Voronoi plate domains rotate around seeded Euler-like axes, boundary relative motion produces convergent/divergent/transform classes, and sampled history accumulates rifting/orogeny fields. It produces the information downstream stages need, but it is not mantle convection or finite-element geodynamics.

The climate model is an efficient procedural energy/moisture model, not a 3-D general circulation model. It is more numerical than hand-painted isotherms and precipitation selections, but less physically complete than CESM/ExoPlaSim.

The source transcript for fog explicitly delegates the procedure to another guide rather than describing it. Gemstones are also commissioned/referenced rather than procedurally demonstrated. Those cannot be faithfully reconstructed *from the supplied transcript itself*, so this project uses a physical fog proxy and focuses mineral automation on the explicitly demonstrated ore/fuel/salt workflow.

The human logs contain purposeful creative judgments and deliberately underspecified values. The code therefore does not claim to mathematically derive a uniquely correct culture. It uses geographic/resource constraints and seeded generative choices, and labels synthetic population values as such.

See `SOURCE_COVERAGE.md` for episode-by-episode coverage and the two missing transcript episodes.

## Testing

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Extending the model

The easiest extension points are:

- add an ore rule to `resources.py`;
- replace climate with a heavier model while preserving the `ClimateResult` interface;
- replace the kinematic tectonic model while preserving `TectonicResult`;
- add raster/vector formats (GeoTIFF, NetCDF, GeoPackage) in `pipeline.py`;
- add a true language/culture agent model downstream of `SocietyResult`.

The pipeline intentionally separates stages by typed result objects so those upgrades do not require rewriting the whole generator.


## License

This project is released under the MIT License. See `LICENSE`.
