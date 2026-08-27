# Exotic planetary physics

Version 0.5 adds an opt-in advanced layer for composition-aware worlds. The architecture remains hierarchical rather than replacing the fast generator:

```text
legacy deterministic world
  -> adaptive atmosphere/ocean coupling
  -> conserved volatile inventory + spherical liquid-level solve
  -> multicomponent chemistry/volatile cycle
  -> exotic liquid-mixture state
  -> automatic silicate + cryogenic geodynamics
  -> cryogeology
  -> fluid-property-aware geomorphic diagnostics
```

The advanced layers are enabled when `astronomy.greenhouse_model: composition` is used. Legacy configurations continue to use the established reduced-order Earthlike path.

## Chemistry registry and model tiers

`worldgen.planetary_chemistry` expands the screening chemistry registry beyond the precision thermodynamic species in `planetary_physics`. The registry includes water, CO2, methane, ethane, ammonia, nitrogen, oxygen, sulfur dioxide, hydrogen/helium/argon, and additional plausible exotic candidates including CO, H2S, HCN, acetylene, ethylene, propane, methanol, ozone, hydrogen peroxide, sulfuric acid, ammonium hydrosulfide, elemental sulfur aerosol and a `THOLIN` pseudo-species.

Two model tiers are intentional:

1. **Bulk thermodynamic fluids** have approximate phase/saturation/property data and may participate in condensation screening, precipitation or ocean state.
2. **Reaction/aerosol products** such as tholins and NH4SH are diagnostic products. They can form haze/cloud/deposition fields but are not silently treated as globally mobile oceans without defensible liquid-state data.

This is a screening chemistry model, not Gibbs free-energy minimization or a kinetic reaction network.

## Simultaneous condensates

The base climate solver retains one transported reference moisture tracer for performance. `volatile_cycle.py` then determines every atmospheric species that is abundant enough and sufficiently close to saturation over a non-negligible area. Several species can be active simultaneously.

For each active condensate it derives:

- annual liquid-equivalent precipitation;
- liquid versus solid precipitation fraction;
- frost deposition;
- evaporation potential;
- sublimation potential;
- surface-reservoir exchange.

The base transported condensate flux is partitioned conservatively between eligible species rather than independently duplicated. This preserves the reduced-order moisture budget at this layer, although each species does not yet feed its own latent heat back into the circulation.

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

`geodynamics.py` combines internal heat, tidal heat fraction, body mass/gravity, surface temperature, configured ice-shell thickness and antifreeze content. Automatic mode can select among:

- inactive;
- weakly active lid;
- stagnant lid;
- mobile lid;
- tidally forced;
- heat-pipe/magma-dominated.

Ice-rich bodies independently receive a cryogenic regime:

- inactive;
- conductive ice shell;
- episodic cryotectonics;
- active cryotectonics.

Explicit user-selected tectonic modes remain authoritative; the classifier primarily upgrades `auto`.

## Cryogeology

`cryogeology.py` produces spatial fields for:

- ice-shell thickness and thermal thinning;
- basal melt;
- brittle fracture;
- diapirism;
- chaos-terrain propensity;
- cryovolcanism;
- plume venting;
- sublimation erosion;
- volatile frost deposition;
- clathrate destabilization.

It distinguishes thin, strongly fractured venting shells from thicker convecting/diapiric shells. Tidal forcing and inherited stress structure spatially focus activity. NH3/CH3OH antifreeze fractions increase shell mobility and basal-liquid persistence.

## Methane and other non-water geomorphology

`geomorphic_fluids.py` derives a parameter block from actual mobile-liquid density, viscosity and surface tension plus local gravity. Water at Earth gravity is the reference state. The block supplies dimensionless multipliers for:

- stream power;
- runoff efficiency;
- sediment transport capacity;
- deposition efficiency;
- lateral bank erosion;
- delta retention;
- hillslope diffusion;
- evaporation loss;
- substrate erodibility.

The diagnostics therefore do not assume that a methane river on Titan has water density, water viscosity or Earth gravity. The layer also generates evaporite, organic-sediment, sublimation-landform and cryogenic mass-wasting indices.

The current 0.5 implementation uses these coefficients to describe/project the advanced geomorphic regime and to support future selectable fluid-aware landscape-evolution kernels. The legacy surface-evolution raster is not silently re-run with changed coefficients after the final conserved sea-level solve; doing that correctly requires another coupled liquid-level/climate/hydrology convergence loop rather than a one-line multiplier.

## Outputs

Composition-aware worlds additionally save:

```text
surface_liquids.json
surface_liquids.npz
advanced_planetary_physics.json
advanced_planetary_fields.npz
```

`advanced_planetary_fields.npz` contains multicomponent precipitation/frost fields, photochemical deposition, cryogeology, ocean-state and exotic-geomorphology rasters where applicable.

## Scientific limitations

The advanced layer deliberately records what it does not solve. Important remaining higher-fidelity backends include:

- non-ideal activity/fugacity and chemical-equilibrium minimization;
- vertical photochemical kinetics and aerosol/cloud microphysics;
- species-specific latent-heat feedback in the circulation;
- salinity/solute speciation and a true seawater/exotic-fluid equation of state;
- primitive-equation ocean circulation and dynamic sea ice;
- astronomical tide fields and dissipative tidal currents;
- mantle rheology/convection and plate-yield mechanics;
- viscoelastic ice-shell tidal stress and fracture propagation;
- fully fluid-aware coupled landscape evolution with grain-size/cohesive sediment physics.

These limitations are why advanced modules use names such as `index`, `proxy`, `screening` and `approx` rather than implying laboratory-precision predictions.
