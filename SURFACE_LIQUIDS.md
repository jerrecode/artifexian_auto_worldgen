# Dynamic surface-liquid reservoirs

The composition-aware planetary model can derive the surface liquid level from the
amount of volatile material that is actually mobile liquid at the current climate
state instead of choosing sea level from a target land fraction.

Legacy worlds using `astronomy.greenhouse_model: legacy` keep the historical
land-fraction sea-level behavior for reproducibility.

## Configuration

For composition-aware worlds, `astronomy.surface_volatiles` supplies the volatile
inventory used by the surface-reservoir solver.  Values are currently measured in
**modern-Earth ocean masses**, where

```text
1 Earth ocean mass = 1.3321e21 kg
```

For example:

```yaml
astronomy:
  greenhouse_model: composition
  thermodynamics_backend: auto
  atmosphere_pressure_bar: 1.0
  atmosphere:
    N2: 0.78
    O2: 0.21
    Ar: 0.0096
    CO2: 0.0004
  surface_volatiles:
    H2O: 1.0
```

A Titan-like inventory can contain multiple volatile species:

```yaml
astronomy:
  greenhouse_model: composition
  atmosphere_pressure_bar: 1.47
  atmosphere:
    N2: 0.95
    CH4: 0.05
  surface_volatiles:
    CH4: 0.02
    C2H6: 0.01
```

The volatile entries are physical inventory amounts in this advanced mode, not
percentages that are automatically renormalized.

## Thermodynamic partition

For every volatile species with total inventory mass `M_total`, the solver estimates

```text
M_total = M_vapor + M_solid + M_liquid
```

at the current annual temperature field and surface pressure.

Atmospheric vapor capacity is saturation-limited.  For one raster cell of area `A`,
the hydrostatic mass represented by vapor partial pressure `p_v` is approximately

```text
m_v = p_v A / g
```

where `p_v` is bounded by the species saturation pressure and the configured model
relative humidity.  Summing the spherical cell areas gives the global vapor
capacity.  Inventory in excess of that capacity condenses.

The condensed remainder is split between fixed solid and mobile liquid from the
species phase field.  The current ice-fixation step is deliberately reduced-order;
it is not yet a glacier/ice-sheet mass-balance solver.

Liquid density is taken from the optional CoolProp real-fluid backend when available
and requested, otherwise from the built-in planetary volatile property table.  The
mobile liquid volume is

```text
V_liquid = M_liquid / rho_liquid
```

and volumes of different liquid species are currently added before the global fill.
Non-ideal methane/ethane mixture volume and activity/fugacity corrections are a
future extension.

## Exact spherical raster volume

The liquid solver does not approximate every map pixel as a rectangular vertical
prism.  Each equirectangular cell owns a spherical solid angle `Omega` obtained from
the grid's normalized spherical area weight.

For a cell whose solid bed radius is `r_b` and a global liquid surface radius `r_l`,
its filled volume is

```text
V_cell = Omega / 3 * (r_l^3 - r_b^3),  for r_l > r_b
V_cell = 0,                              otherwise
```

The implementation evaluates the equivalent numerically stable polynomial in liquid
depth `d = r_l-r_b`:

```text
V_cell = Omega * (r_b^2 d + r_b d^2 + d^3/3)
```

so the geometry remains accurate for Earth-sized, super-Earth, dwarf-planet, and
large-moon radii.

## Deepest-point-first fill algorithm

For a requested mobile-liquid volume:

1. flatten the solid-bed heightmap and sort cells by elevation;
2. activate the deepest cell or equal-elevation group;
3. raise one common equipotential liquid radius toward the next higher bed level;
4. compute exactly how much spherical-wedge volume that rise can hold;
5. if liquid remains, activate the newly reached cells and continue upward;
6. inside the final elevation interval, invert the radial shell-volume equation for
   the exact partial rise;
7. independently reintegrate the resulting depth field and report the volume
   residual.

The expensive operation is one `O(N log N)` sort for `N` raster cells.  Filling
after the sort is linear, and no repeated whole-map bisection is required.

## Coupling to climate and coastline

The advanced pipeline treats the terrain produced by tectonic/geomorphic evolution
as a **solid-bed datum**.  It then performs a short deterministic fixed-point loop:

```text
current temperature
    -> volatile vapor / solid / liquid partition
    -> mobile liquid mass and density
    -> mobile liquid volume
    -> global spherical liquid level
    -> coastline and liquid depth
    -> ocean circulation / SST
    -> climate
    -> repeat until liquid-level change is small
```

The current maximum is four iterations and the level convergence tolerance is two
metres.  After the final liquid geometry is chosen, coastline-sensitive geology,
hydrology, weather, surface appearance, resources, and society are recomputed.

The terrain array exposed to downstream stages is relative to the solved liquid
surface:

```text
relative_elevation = solid_bed_elevation - liquid_level
```

so positive values are dry land and negative values are submerged bed.  The original
physical liquid level relative to the pre-correction bed datum is retained in
metadata.

`OceanResult.depth_m` is replaced by the conserved-volume liquid-depth field so
output bathymetry is consistent with the inventory calculation rather than with an
independent target-ocean heuristic.

## Output

Composition-aware worlds with a physical volatile inventory additionally write:

```text
surface_liquids.json
surface_liquids.npz
```

The NPZ contains:

```text
liquid_depth_m
liquid_mask
relative_surface_elevation_km
```

The JSON report contains total, vapor, fixed-solid and mobile-liquid masses for each
species; densities; liquid volume; solved level; integration residual; and coupling
history.

## Conservation diagnostics

The most important numerical invariant is

```text
integral(global filled spherical-wedge volume) == sum(species liquid volume)
```

within floating-point tolerance.  Regression tests cover a flat global ocean,
irregular topography, a single deepest basin, water vaporization, complete freeze-out,
density-dependent methane versus water levels, and multi-species volume addition.

## Current scientific limitations

This is a physically stronger sea-level boundary condition, not a complete volatile
cycle solver.  In particular:

- all mobile surface liquid is currently allowed to redistribute to one global
  equipotential. Permanently isolated basins and endorheic lakes do not yet retain
  separate hydraulic levels;
- solid volatile sequestration is an area-weighted equilibrium proxy rather than an
  explicit ice-sheet/glacier thickness and flow model;
- atmospheric vapor is hydrostatic and saturation-limited but does not yet solve a
  vertically resolved moist atmosphere;
- multicomponent liquids are volume-additive pure-species reservoirs. Non-ideal
  mixture density, activity coefficients, fugacity and composition-dependent
  evaporation remain future work;
- the final exact liquid depth is currently imposed after the ocean circulation
  builder has formed its reduced-order bathymetric circulation field. A future ocean
  API should accept prescribed conserved bathymetry before computing currents;
- river erosion and sediment transport are still substantially calibrated for water.
  Exotic liquid geomorphology should eventually consume density, viscosity and other
  fluid properties explicitly;
- relative humidity, solid-fixation efficiency, fixed-point iteration count and
  tolerance are currently model constants and should become validated configuration
  parameters in a later cleanup.

These limitations are intentionally exposed so the resulting world is not described
as more physically complete than the implemented model warrants.
