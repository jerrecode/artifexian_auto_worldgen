# Exotic planetary physics

Version 0.5 adds an opt-in advanced layer for composition-aware worlds. The architecture remains hierarchical rather than replacing the fast generator:

```text
legacy deterministic world
  -> adaptive atmosphere/ocean coupling
  -> conserved volatile inventory + spherical liquid-level solve
  -> multicomponent chemistry/volatile cycle
  -> mass-conservative multicomponent condensate hydrology
  -> exotic liquid-mixture state
  -> automatic silicate + cryogenic geodynamics
  -> cryogeology
  -> fluid-property-aware landscape evolution + secondary geomorphology
  -> final composition-aware surface-liquid and atmospheric visible rendering
```

The advanced layers are enabled when `astronomy.greenhouse_model: composition` is used. Legacy configurations continue to use the established reduced-order Earthlike path.

## Chemistry registry and model tiers

`worldgen.planetary_chemistry` expands the screening chemistry registry beyond the precision thermodynamic species in `planetary_physics`. The registry includes water, CO2, methane, ethane, ammonia, nitrogen, oxygen, sulfur dioxide, hydrogen/helium/argon, and additional plausible exotic candidates including CO, H2S, HCN, acetylene, ethylene, propane, methanol, ozone, hydrogen peroxide, sulfuric acid, ammonium hydrosulfide, elemental sulfur aerosol and a `THOLIN` pseudo-species.

Two model tiers are intentional:

1. **Bulk thermodynamic fluids** have approximate phase/saturation/property data and may participate in condensation screening, precipitation or ocean state.
2. **Reaction/aerosol products** such as tholins and NH4SH are diagnostic products. They can form haze/cloud/deposition fields but are not silently treated as globally mobile oceans without defensible liquid-state data.

This is a screening chemistry model, not Gibbs free-energy minimization or a kinetic reaction network.

## Simultaneous condensates and land hydrology

The base climate solver retains one transported reference moisture tracer for performance. `volatile_cycle.py` then determines every atmospheric species that is abundant enough and sufficiently close to saturation over a non-negligible area. Several species can be active simultaneously.

For each active condensate it derives liquid/solid precipitation propensity, frost deposition, evaporation/sublimation potential and reservoir exchange. `condensate_hydrology.py` then treats the reference precipitation as mass flux, partitions that mass between all eligible condensates, and converts each species to liquid-volume depth using its own density. The partition is explicitly mass-conservative.

The resulting monthly liquid/solid condensate fields feed the actual land bucket, runoff, groundwater/baseflow and drainage solver. The bucket therefore means **mobile/stored condensate liquid**, not intrinsically water. Compatibility names such as `soil_water_storage_mm`, `groundwater_storage_mm` and `snowpack_mm` are retained for existing consumers, but new public aliases are available:

```text
soil_liquid_storage_mm
subsurface_liquid_storage_mm
solid_condensate_storage_mm
```

The final `ground_liquid_humidity_index` combines persistent soil-liquid storage with current thermodynamically liquid precipitation input. Rain on an H2O world, methane rain on Titan-like worlds, or another supported liquid condensate therefore raises ground humidity using the same chemically generic geometric-liquid semantics.

## Photochemistry and energetic-particle chemistry

`planetary_chemistry.evaluate_photochemistry` uses stellar bolometric flux, a stellar-effective-temperature UV proxy, orbital distance and an energetic-particle proxy to screen important abundance-limited pathways. Implemented pathways include:

- O2 photolysis/recombination -> O3;
- N2 + CH4 irradiation -> C2H6, C2H2, HCN and refractory tholin haze;
- CO2 photolysis -> CO and an oxygen-production proxy;
- SO2 + H2O + oxidant chemistry -> H2SO4 aerosol;
- NH3 + H2S -> NH4SH cloud material;
- reduced sulfur irradiation -> S8 aerosol/deposits;
- water radiolysis -> H2O2 oxidant deposits.

Outputs are **production and abundance proxies**, not steady-state mole fractions.

## Exotic ocean state

`exotic_ocean.py` derives its composition from the mobile-liquid masses returned by the mass-conserving surface-liquid solver. The current mixture backend computes:

- ideal volume-additive density using the actual per-species liquid density when available;
- logarithmic viscosity mixing;
- mass-weighted surface-tension screening;
- ocean class (`aqueous`, `ammonia-water cryo-ocean`, `hydrocarbon sea`, etc.);
- mixed-layer depth proxy;
- thermal density anomaly and stratification;
- sea-ice fraction;
- brine/solute concentration index;
- methane/ethane/CO2 clathrate-stability index for water-containing systems;
- hydrothermal-exchange index.

Ammonia-water and methanol-water mixtures receive bounded eutectic-like freezing-point depression. This is intentionally a reduced-order approximation, not a binary/ternary phase-diagram solver.

## Automatic geodynamic regimes

`geodynamics.py` combines internal heat, tidal heat fraction, body mass/gravity, surface temperature, configured ice-shell thickness and antifreeze content. Automatic mode can select among inactive, weakly active lid, stagnant lid, mobile lid, tidally forced, and heat-pipe/magma-dominated regimes. Ice-rich bodies independently receive inactive, conductive-shell, episodic-cryotectonic or active-cryotectonic regimes.

The low-speed tectonic initializer also respects genuinely inactive/stagnant bodies. The old terrestrial minimum speed is retained only for the historical active regime; configurations below that scale use a proportional low-speed distribution rather than failing or silently forcing Earth-like plate speeds.

## Cryogeology

`cryogeology.py` produces spatial fields for ice-shell thickness/thermal thinning, basal melt, brittle fracture, diapirism, chaos-terrain propensity, cryovolcanism, plume venting, sublimation erosion, volatile frost deposition and clathrate destabilization. It distinguishes thin, strongly fractured venting shells from thicker convecting/diapiric shells. Tidal forcing and inherited stress structure spatially focus activity.

## Methane and other non-water geomorphology

`geomorphic_fluids.py` derives a parameter block from actual mobile-liquid density, viscosity and surface tension plus local gravity. Water at Earth gravity is the reference state. The block supplies dimensionless multipliers for stream power, runoff efficiency, sediment transport, deposition, lateral bank erosion, delta retention, hillslope diffusion, evaporation loss and substrate erodibility.

These coefficients are now fed through real surface-evolution recoupling rather than remaining diagnostic only. After fluid-aware erosion/deposition changes the bed, the conserved liquid level, ocean/climate state and drainage graph are solved again. Secondary geomorphology then adds mass wasting, glacial/cryogenic erosion and deposition, subsurface-liquid/spring erosion, karst screening, floodplains, alluvial fans, wetlands, braided/avulsing channels, estuaries, submarine canyons, coastal erosion, capture susceptibility and isostatic response before the final drainage reroute.

## Composition-aware true color

`planetary_optics.py` and `appearance_planetary.py` replace the Earth-only visible assumptions after the final physical state has converged.

### Surface liquid color

The authoritative dynamic `surface_liquids.liquid_mask` and `liquid_depth_m` are rendered using the actual mobile liquid composition. The reduced-order optical model uses:

- composition-weighted broadband RGB absorption coefficients;
- two-way Beer-Lambert attenuation of bottom light;
- shallow-bottom visibility for clear liquids;
- deep-column scattering/source color;
- composition-dependent refractive index and Fresnel/sky reflection;
- suspended-sediment turbidity;
- photochemical organic deposition/suspension where present;
- exotic-ocean sea-ice fraction rather than a hard-coded 0 °C water threshold.

Consequently methane/ethane seas are no longer painted terrestrial ocean blue. Pure cryogenic hydrocarbons are treated as comparatively transparent/neutral molecular liquids; Titan-like orange/brown appearance is supplied primarily by organic haze/deposition and atmospheric transfer rather than by pretending liquid methane itself is orange.

### Atmospheric coloration

`true_color_with_clouds` is now a top-of-atmosphere composite rather than merely `surface + white cloud`. The visible screening model includes:

- molecular column scaling from surface pressure, gravity and mean molar mass;
- composition-weighted Rayleigh efficiency with wavelength^-4 behavior;
- broadband molecular visible absorption;
- cloud color from the actual precipitating condensates;
- photochemical aerosol optical depth/color, including tholins, sulfuric-acid aerosol, sulfur and other registered products.

The clear `true_color` product represents the corrected surface/liquid raster; `true_color_with_clouds` additionally passes that surface through the modeled atmosphere/cloud/haze column.

This is a **three-band reduced-order visible transfer model**, not line-by-line spectral or multiple-scattering radiative transfer. It is designed to make composition and pressure matter in the correct direction without claiming laboratory colorimetry. Venus is a particularly important limitation: H2SO4 aerosol/clouds are represented from explicit trace SO2 + H2O chemistry, but the detailed identity and spectrum of the short-wavelength absorber responsible for part of Venus's yellow/UV contrast is not hard-coded as if it were known exactly.

## Outputs

Composition-aware worlds additionally save:

```text
surface_liquids.json
surface_liquids.npz
advanced_planetary_physics.json
advanced_planetary_fields.npz
```

The normal `world_arrays.npz` now also includes the final composition-aware fields when available:

```text
liquid_condensate_input_mm_year
solid_condensate_input_mm_year
total_condensate_input_mm_year
soil_liquid_storage_mm
subsurface_liquid_storage_mm
solid_condensate_storage_mm
ground_liquid_humidity_index
solid_condensate_persistence
surface_liquid_true_color_rgb
atmospheric_haze_optical_depth
```

Additional PNGs expose surface-liquid optical color, ground-liquid humidity, atmospheric haze optical depth and generic solid-condensate persistence.

## Scientific limitations

The advanced layer deliberately records what it does not solve. Important remaining higher-fidelity backends include:

- non-ideal activity/fugacity and chemical-equilibrium minimization;
- vertical photochemical kinetics and aerosol/cloud microphysics;
- species-specific latent-heat feedback in the circulation;
- salinity/solute speciation and a true seawater/exotic-fluid equation of state;
- primitive-equation ocean circulation and dynamic sea ice;
- full non-equilibrium methane/ethane lake fractionation and basin-specific composition;
- multiple-scattering, wavelength-resolved visible radiative transfer and aerosol phase functions;
- mantle rheology/convection and plate-yield mechanics;
- viscoelastic ice-shell tidal stress and fracture propagation;
- grain-size/cohesive sediment transport and species-resolved porous-media flow.

These limitations are why advanced modules use names such as `index`, `proxy`, `screening` and `approx` rather than implying laboratory-precision predictions.
