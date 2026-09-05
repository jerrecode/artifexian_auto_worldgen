# Environment-conditioned procedural erosion

This document describes the procedural erosion layer implemented by
`worldgen.procedural_erosion`, `worldgen.erosion_forcing`,
`worldgen.pipeline_procedural_erosion`, and the related tile-local refinement in
`worldgen.local_geomorphology`.

The implementation is an independently written planetary adaptation of the
phase-cell / gully-filter family described by Rune Skovbo Johansen. It is not a
copy of the upstream GLSL and it is not a hydraulic-erosion simulator.

Primary conceptual references:

- https://youtu.be/r4V21_uUK8Y
- https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html
- https://www.shadertoy.com/view/wXcfWn
- https://www.shadertoy.com/view/sf23W1

The upstream final shader is released under MPL-2.0. This repository remains MIT
licensed because its implementation was written independently from the published
algorithmic description rather than copied or translated from the upstream shader.

## 1. Responsibility split

`artifexian_auto_worldgen` owns terrain geometry, lithology, hydrology, the
procedural morphology operator, sediment-aware landscape evolution, global/local
terrain mutation, diagnostics, and climate/ocean recoupling after terrain changes.

`atmogen` owns atmosphere/material thermodynamics and exposes screening-grade
liquid transport properties used as environmental inputs. The world-generator pins
a compatible atmogen revision rather than creating a bidirectional package
dependency.

The erosion boundary is therefore data-oriented:

```text
atmogen / climate / hydrology / geology / ocean / cryosphere
                       |
                       v
              erosion forcing fields
                       |
                       v
             procedural morphology core
                       |
                       v
                  terrain delta
                       |
                       v
        bounded Earth-system recoupling
```

## 2. What the Runevision-style operator represents

The source technique produces the appearance of branching erosion by evaluating
directional, blended periodic structure aligned to terrain slope. It can be sampled
point-wise and does not route explicit droplets or solve conservation laws for
water and sediment. This makes it suitable as a fast morphology primitive, not as
a replacement for the project's physical hydrology and sediment-routing solvers.

The useful conceptual ingredients retained here are:

1. directional cosine/sine phase fields aligned with the local terrain direction;
2. random spatial cells/pivots blended to remove hard cell boundaries;
3. multiple octaves, with finer octaves seeing orientation modified by coarser
   octaves;
4. sign-based orientation steering, analogous to the "straight gullies" idea;
5. stacked masking/fade-target behavior that protects larger ridges and creases;
6. separate ridge and crease shaping;
7. scale-dependent octave culling.

The planetary implementation intentionally changes the sampling domain. Instead of
a planar 2-D cell lattice, it evaluates a deterministic 3-D lattice at points on
the planet-centred unit sphere. This avoids an equirectangular seam and allows the
same primitive to be reused on cubed-sphere/local tile geometry.

## 3. Phase-cell octave

Let the unit surface position be `n`, planet radius be `R`, preferred local
wavelength be `lambda`, and the configured relative cell scale be `c`. The
world-space point is

```text
x = R n
q = x / (lambda c)
```

The integer cell containing `q` and its 26 immediate neighbours are evaluated.
Each cell receives a deterministic jittered anchor `a_j` and phase
`phi_j` from the integer-cell hash.

For displacement `d_j = q - a_j`, the compact blend weight is

```text
w_j = max(1 - ||d_j||^2 / 4.25, 0)^3
```

Let `p` be the tangent direction perpendicular to the local downhill-oriented
line field. The phase at a contributing cell is

```text
theta_j = 2 pi c (p dot d_j) + phi_j
```

The blended phase vector is

```text
C = sum_j w_j cos(theta_j)
S = sum_j w_j sin(theta_j)
W = sum_j w_j
M = sqrt(C^2 + S^2)
```

The current spherical implementation returns

```text
cos_phase = C / max(M, eps)
sin_phase = S / max(M, eps)
coherence = clamp(M / max(W, eps), 0, 1)
```

### Important difference from the final upstream filter

Runevision's final Phacelle formulation uses partial normalization: weakly
cancelled phase vectors remain weak below a configurable threshold, while
sufficiently coherent vectors are normalized. This avoids artifacts associated
with unconditional normalization.

The present worldgen implementation instead normalizes the phase direction and
carries cancellation strength separately as `coherence`, which then participates
in masking and steering. This is an intentional adaptation, not mathematical
identity with the reference shader. It should only be changed after visual and
regression comparison because the coherence coupling changes the effective
amplitude and branching statistics.

## 4. Ridge/crease profile and stacked masking

For normalized cosine value `c0`, choose a local rounding amount `r` from the
ridge or crease field according to the sign of `c0`:

```text
e = 1 + 2.5 clamp(r, 0, 1)
profile = sign(c0) * (1 - max(1 - |c0|, 0)^e)
```

Across octaves the accumulated phase mask is

```text
m <- 1 - (1 - m) (1 - clamp(coherence * detail, 0, 1))
```

and the visible profile is blended against an environment/topography-derived
ridge/valley target:

```text
visible = m * profile
        + (1 - m) * fade_target_strength * ridge_valley_target
```

This serves the same broad purpose as stacked fading in the upstream technique:
later/finer structure is prevented from uniformly destroying larger-scale
ridges and creases.

## 5. Recursive line-field steering

After each octave, the local tangent orientation is rotated. If `s` is the
blended sine phase, the turn angle is

```text
turn = steering_strength
     * sign(s)
     * coherence
     * detail
     * gain^octave
```

The south/east tangent components are rotated by the ordinary 2-D rotation matrix
and renormalized.

Using `sign(s)` rather than the raw sine magnitude is conceptually related to
Runevision's "straight gullies" construction: the internal orientation field acts
more like a constant-sided triangular gully when deciding where the next octave
branches, while the visible height profile can remain smooth.

## 6. Scale and amplitude

For octave `i`:

```text
lambda_i = preferred_scale / lacunarity^i
A_i = base_amplitude_m * gain^i * local_strength * spectral_detail
```

An octave is not evaluated where

```text
lambda_i < min_samples_per_wavelength * local_grid_spacing
```

This is an explicit anti-aliasing/LOD rule. Parameters are expressed in kilometres
or metres at the public world scale and converted to grid sampling internally.
The local tile solver performs the same check using metres per tile sample.

The implementation is deterministic for deterministic inputs and seed. The
integer 3-D hash is independent of NumPy's random state.

## 7. Environment-conditioned forcing

Environmental inputs are converted to bounded process activity fields before they
touch the morphology operator. A shared saturating transform is

```text
sat(x; scale, power) =
    1 - exp(-(max(x, 0) / scale)^power)
```

This avoids unbounded linear response to extreme climate fields.

### 7.1 Lithology

The central lithology table supplies mechanical erodibility, runoff response,
infiltration, soil capacity, chemical weatherability, frost susceptibility,
glacial abrasion susceptibility, and cohesion.

`bedrock_code` is authoritative for erosion. Legacy objects with only
`rock_code` remain supported.

### 7.2 Soil saturation

```text
soil_saturation =
    clamp(soil_water_storage / soil_capacity, 0, 1.25) * land
```

Moisture is not used as a universal multiplier. It contributes differently to
runoff-driven, rainfall-driven, chemical, and freeze-thaw terms.

### 7.3 Pluvial and fluvial activity

Representative transforms are:

```text
R = sat(surface_runoff, 650 mm/year, 0.85)
Q = sat(discharge_index, 0.22, 0.80)
P = sat(liquid_precipitation, 850 mm/year, 0.82)

fluvial =
    sqrt(R Q)
    * (0.45 + 0.55 saturation)
    * mechanical_erodibility
    * fluid_mechanical_factor
    * land

pluvial =
    P
    * (0.35 + 0.65 storminess)
    * (0.55 + 0.45 saturation)
    * mechanical_erodibility
    * fluid_mechanical_factor
    * land
```

The exact coefficients are screening/procedural parameters, not calibrated
universal geomorphic constants.

### 7.4 Planetary liquid mechanics

When condensate-species mass information is available, atmogen provides a
screening mixture density, dynamic viscosity, and surface tension. Worldgen maps
those properties and gravity to a bounded mechanical factor:

```text
F_liquid =
    (rho / 997)^0.55
    * (g / 9.80665)^0.45
    * (1e-3 / mu)^0.16
    * (0.072 / sigma)^0.10
```

The result is clipped to `[0.08, 3]`.

This is deliberately mechanical only. Acidity, alkalinity, explicit dissolution
kinetics, solvent-rock reaction networks, and temperature-dependent rheology are
not yet represented by this scalar.

### 7.5 Glacial and freeze-thaw terms

Glacial activity uses annual temperature, solid precipitation supply, lithological
abrasion susceptibility, and optional cryogeology fields such as basal melt and
brittle fracture.

Freeze-thaw activity uses crossings of the relevant liquid freezing temperature
in the monthly temperature series, climate continentality, moisture, and
lithological frost susceptibility.

These fields modulate morphology. They do not claim to solve ice dynamics,
quarrying mechanics, or subglacial sediment transport.

### 7.6 Marine term

The marine activity field depends on ocean occupancy, shelf state, local slope,
and normalized current speed. Shelf cells receive much stronger procedural marine
activity than generic deep-ocean cells.

Wave-cut platforms, longshore transport, turbidity currents, and submarine
landslides are not explicit solvers here. Those should remain separate processes
if added.

### 7.7 Chemical weathering

The current reduced term combines lithological chemical weatherability,
temperature suitability, liquid precipitation, and soil saturation. It is a
weathering propensity, not geochemical reaction kinetics.

## 8. Multi-regime composition

The final scalar morphology strength is an additive bounded mixture:

```text
strength =
    w_fluvial * fluvial
  + w_pluvial * pluvial
  + w_glacial * glacial
  + w_marine * marine
  + w_chemical * chemical
  + w_freeze_thaw * freeze_thaw

strength = clamp(strength, 0, max_local_strength)
```

An additive/process-specific construction is used intentionally instead of blindly
multiplying all environmental fields. A process that is physically irrelevant in
one regime can therefore vanish without zeroing unrelated regimes.

## 9. Preferred morphology scale and direction

Drainage density makes the preferred gully wavelength smaller; glacial and marine
regimes broaden the preferred scale. The result is clamped to configured minimum
and maximum wavelengths.

The base orientation comes from the metric spherical terrain gradient. Curvature,
topographic wetness, channel classification, and height above nearest drainage
contribute to the ridge/valley fade target.

## 10. Displacement and mass semantics

The procedural operator is morphology, not a grain-resolved sediment solver.

When `zero_mean_displacement` is enabled, the canonical global pass removes the
spherical cell-area-weighted displacement mean. If the resulting morphology would
exceed `max_displacement_m`, the complete centered field is uniformly rescaled
instead of clipping positive and negative extrema independently. Uniform scaling
preserves the zero-mean invariant and relative morphology while enforcing the cap.

The physical landscape-evolution solver still owns the sediment mass ledger.
Procedural positive/negative displacements are reported as morphological
deposition/incision for diagnostics, but are explicitly not asserted to be a
conservative sediment-transport solution.

The tile-local procedural field follows the same semantic separation. Its final
stored perturbation is area-centered, edge-tapered once, applied after the
finite-domain anchoring transform, and excluded from the physical sediment ledger.

## 11. Pipeline placement and recoupling

The public `WorldPipeline` subclasses the existing geomorphology pipeline. The
canonical world is first generated with tectonics, terrain, climate, ocean,
geology, hydrology, physical surface evolution, weather, resources and optional
society. If procedural erosion is enabled:

1. environmental forcing is assembled from the accepted world state;
2. procedural morphology produces a bounded terrain displacement;
3. terrain is rebuilt from the displaced elevation while preserving sea-level
   state;
4. the procedural diagnostics are merged into the surface-evolution result;
5. if any material displacement occurred and recoupling is enabled, ocean,
   climate, atmogen representative columns where configured, geology, hydrology,
   weather, appearance, resources, and society are recomputed using stationary
   random streams;
6. dynamic/exotic surface-liquid paths use their dedicated recoupling routines.

This bounded one-way correction avoids an uncontrolled
terrain -> climate -> erosion -> terrain feedback loop. A future iterative coupling
scheme should define explicit convergence criteria before adding repeated erosion
passes.

## 12. Diagnostics

The global operator can export:

- procedural displacement in metres;
- phase coherence;
- ridge map;
- crease map;
- effective local strength;
- effective local scale;
- fluvial, pluvial, glacial and marine activity;
- chemical weathering;
- freeze-thaw activity;
- soil saturation;
- limiter type, limiter scale, pre-constraint peak displacement, and final
  area-weighted displacement mean.

These are diagnostic/provenance fields, not separate conserved physical state.

## 13. Numerical and performance characteristics

For each octave, every active raster point examines 27 3-D neighbouring cells.
With `N` raster cells and `O` resolved octaves, CPU work is therefore
`O(27 N O)`, i.e. linear in raster size and octave count with a relatively large
constant from trigonometric operations and temporary arrays.

The implementation is vectorized NumPy. The deliberate reference backend keeps
dependencies small and behavior inspectable. The algorithm is structurally
well-suited to Numba, CuPy, JAX, compute shaders, or native kernels, but an
acceleration backend should preserve deterministic hashes, LOD decisions, boundary
semantics, diagnostics, and numerical invariants before replacing the reference
path.

The main temporary cost per cell comes from float64 position, tangent, phase,
weight, and accumulation arrays. LOD culling is therefore important on coarse
global rasters: unresolved high-frequency octaves are skipped entirely.

## 14. Validation expectations

Regression tests should preserve at least these contracts:

- deterministic seed/configuration gives deterministic morphology;
- output and diagnostics remain finite;
- zero strength is exact identity;
- configured displacement bounds are respected;
- zero-mean global morphology remains area-weighted zero-mean even when the
  displacement limit activates;
- modern `bedrock_code` and legacy `rock_code` geology contracts both work;
- geology code shapes must match the terrain raster;
- local tile morphology is deterministic and watertight;
- tile-local procedural detail has zero area-weighted mean and vanishes at the
  perimeter;
- LOD culling prevents unresolved octaves from being evaluated;
- enabled canonical erosion survives the complete post-erosion recoupling path.

## 15. Known limitations and research items

1. The current spherical phase normalization is not the same partial normalization
   used by Runevision's final shader. Comparative image/statistical tests should
   precede any change.
2. The global procedural filter creates erosion-like morphology but does not
   guarantee connected drainage paths. Real routing remains the hydrology solver's
   responsibility.
3. Mechanical liquid scaling is a reduced-order screening relationship. It is not
   derived from a calibrated Shields/stream-power/sediment-entrainment law.
4. Chemical liquid aggressiveness is not yet supplied by atmogen as an explicit
   rock-fluid reaction property.
5. Glacial, coastal and submarine processes are activity-conditioned morphology,
   not full process simulations.
6. There is no GPU backend yet. NumPy is the correctness/reference backend.
7. The canonical pass is a bounded post-process with one recoupling step, not an
   iterated climate-geomorphology equilibrium solve.
8. The procedural displacement is deliberately separate from the physical sediment
   ledger. Do not interpret its positive and negative fields as transported mass.

The design goal is therefore:

```text
fast procedural morphology
+ physically motivated spatial forcing
+ dedicated physical hydrology/sediment solvers
+ explicit diagnostics and bounded recoupling
```

rather than pretending that one fast image-space/filter-like operator is a complete
planetary erosion model.
