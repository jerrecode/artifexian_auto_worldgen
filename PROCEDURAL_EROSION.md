# Environment-conditioned advanced phase-cell erosion

This document describes the procedural erosion layer implemented by
`worldgen.procedural_erosion`, `worldgen.erosion_forcing`,
`worldgen.pipeline_procedural_erosion`, and the related tile-local refinement
in `worldgen.local_geomorphology`.

Primary conceptual references:

- Rune Skovbo Johansen, *Fast and Gorgeous Erosion Filtering*
- https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html
- https://youtu.be/r4V21_uUK8Y
- https://www.shadertoy.com/view/wXcfWn
- https://www.shadertoy.com/view/sf23W1

The code in this repository is independently written. It adapts the published
phase-cell/Phacelle ideas to a spherical planetary grid instead of translating the
reference GLSL line-for-line.

## Responsibility boundary

`artifexian_auto_worldgen` owns terrain geometry, lithology, hydrology,
landscape evolution, erosion forcing, procedural morphology, terrain mutation,
diagnostics, and post-erosion Earth-system recoupling.

`atmogen` owns atmosphere/material thermodynamics and supplies screening-grade
liquid transport properties such as density, viscosity, surface tension, and
freezing temperature. The dependency is one-way: worldgen consumes an explicit
material-property contract from atmogen.

The data flow is:

```text
climate + hydrology + geology + cryosphere + ocean + atmogen material properties
                                  |
                                  v
                       bounded erosion forcing
                                  |
                                  v
                    advanced phase-cell morphology
                                  |
                                  v
                         terrain displacement
                                  |
                                  v
                 bounded climate/ocean/hydrology recoupling
```

## Scientific interpretation

The phase-cell operator is not a hydraulic-erosion solver. It does not integrate
water momentum, sediment concentration, entrainment, deposition, or grain-scale
transport. It is a fast point-evaluable morphology generator whose output can look
like branching gullies, creases, and ridges.

In this project it is therefore used as a morphology primitive. Physically
motivated fields decide where it acts and how strongly it acts. Dedicated
hydrology and landscape-evolution code remains authoritative for connected
drainage and sediment accounting.

## Spherical phase-cell field

Let `n` be a unit surface vector, `R` the planet radius, `lambda` the local
wavelength, and `c` the cell-scale parameter:

```text
x = R n
q = x / (lambda c)
```

The integer cell containing `q` and its 26 immediate 3-D neighbours are sampled.
Each cell gets a deterministic jittered anchor `a_j` and phase `phi_j` from the
integer-cell hash.

For `d_j = q - a_j`:

```text
w_j = max(1 - ||d_j||^2 / 4.25, 0)^3
theta_j = 2 pi c (p dot d_j) + phi_j

C = sum_j w_j cos(theta_j)
S = sum_j w_j sin(theta_j)
W = sum_j w_j
M = sqrt(C^2 + S^2)
```

where `p` is the tangent vector perpendicular to the current local gully
direction.

The low-level kernel supports partial phase-vector normalization with
`n_norm in [0,1]`:

```text
D = max(M, (1 - n_norm) W, epsilon)
cos_phase = C / D
sin_phase = S / D
coherence = clamp(M / max(W, epsilon), 0, 1)
```

At `n_norm = 1`, every nonzero blended phase vector is fully normalized. At
`n_norm = 0`, cancellation amplitude is retained. The advanced high-level default
is `0.5`, matching the published principle that sufficiently coherent vectors
normalize while strongly cancelling blends remain weak. The low-level function's
default remains full normalization for backward compatibility.

The 3-D planet-centred lattice is a project-specific spherical adaptation. It
avoids the longitude seam of a planar equirectangular lattice and permits the same
primitive to operate on arbitrary spherical tile positions.

## Terrain direction and recursive internal slope

The operator computes the metric spherical terrain gradient. Its direction is used
as the local orientation, while an internal gully field can blend toward a
configured assumed slope magnitude:

```text
gully_gradient =
    (1 - assumed_blend) * measured_gradient
    + assumed_blend * assumed_slope * unit(measured_gradient)
```

This keeps the branching control field meaningful near ridge and valley extrema,
where the measured gradient approaches zero.

After each octave the internal direction is steered laterally:

```text
steer =
    steering_strength
    * sign(sin_phase)
    * coherence
    * environmental_detail
    * gain^octave
    * gully_weight
```

Using `sign(sin_phase)` instead of the sinusoidal magnitude is the spherical
analogue of the reference technique's "straight gullies" construction.

## Stacked masks, ridges, and creases

The advanced filter maintains a combined slope/fade mask across octaves rather
than summing independent stripes uniformly everywhere. Smooth onset functions and
detail-power transforms determine where finer structure survives. The previous
octave's visible result becomes part of the next fade target, preserving
large-scale ridges and creases while adding finer branching structure.

Separate ridge and crease maps are accumulated as diagnostic fields. Their values
are morphology indicators, not drainage or sediment state.

## Scale, LOD, and amplitude

For octave `i`:

```text
lambda_i = preferred_scale / lacunarity^i

amplitude_i =
    base_amplitude_m
    * gain^i
    * effective_strength
    * spectral_detail
    / gully_weight
```

An octave is skipped wherever

```text
lambda_i < min_samples_per_wavelength * local_grid_spacing
```

and iteration terminates when no active cell can resolve the octave. This makes
the filter explicitly scale-aware and prevents unresolved high-frequency work.

## Environmental forcing

The forcing layer converts source fields to bounded process-activity fields before
they reach the morphology core. A common saturating mapping is:

```text
sat(x; scale, power) =
    1 - exp(-(max(x, 0) / scale)^power)
```

This avoids unbounded linear response to extreme climate values.

### Lithology

`bedrock_code` is authoritative for erosion. Legacy objects exposing only
`rock_code` remain supported. The central lithology table exposes mechanical
erodibility, runoff response, infiltration, soil capacity, chemical
weatherability, frost susceptibility, glacial abrasion susceptibility, and
cohesion.

### Pluvial and fluvial regimes

Fluvial activity combines runoff, discharge, soil saturation, lithological
erodibility, and liquid mechanics. Pluvial activity combines liquid
precipitation, storminess, saturation, erodibility, and liquid mechanics.

Zero precipitation therefore disables rain-driven terms without disabling
unrelated marine or glacial regimes.

### Soil moisture

Soil water is normalized by lithology-dependent soil capacity. Moisture affects
different processes differently rather than serving as one universal multiplier.

### Planetary liquid mechanics

When condensate composition is available, atmogen supplies mixture density
`rho`, dynamic viscosity `mu`, and surface tension `sigma`. Worldgen maps
these plus surface gravity to a bounded screening factor:

```text
F_liquid =
    (rho / 997)^0.55
    * (g / 9.80665)^0.45
    * (1e-3 / mu)^0.16
    * (0.072 / sigma)^0.10
```

The result is clipped to a finite range. This is a reduced-order mechanical
screening relationship, not a calibrated sediment-entrainment law. Explicit
acidity/alkalinity and rock-fluid reaction kinetics are not represented by this
single factor.

### Glacial and freeze-thaw regimes

Glacial activity uses cold conditions, solid precipitation supply, lithological
abrasion susceptibility, and available cryogeology diagnostics. Freeze-thaw
activity uses monthly phase crossings, moisture, continentality, and lithological
frost susceptibility.

These fields modulate morphology; they do not solve full ice dynamics, quarrying,
or subglacial sediment transport.

### Marine regime

Marine activity is stronger on shelves than in generic deep-ocean cells and is
modulated by local slope and current speed. Wave-cut platforms, longshore
transport, turbidity currents, and submarine landslides remain separate potential
process modules.

### Chemical weathering

The reduced chemical-weathering term combines temperature suitability, liquid
precipitation, saturation, and lithological weatherability. It is a weathering
propensity, not a reaction-kinetics model.

## Multi-regime composition

The final local strength is an additive bounded mixture:

```text
strength =
    w_fluvial     * fluvial
  + w_pluvial     * pluvial
  + w_glacial     * glacial
  + w_marine      * marine
  + w_chemical    * chemical
  + w_freeze_thaw * freeze_thaw

strength = clamp(strength, 0, max_local_strength)
```

The additive process-specific construction is deliberate. An irrelevant process
can vanish without accidentally zeroing all other erosion regimes.

Drainage density narrows the preferred morphology scale; glacial and marine
regimes can broaden it. Curvature, topographic wetness, channel class, and height
above drainage contribute to ridge/valley targeting.

## Displacement and mass semantics

The procedural layer is geometric morphology and is deliberately separate from
the physical sediment ledger.

When `zero_mean_displacement` is enabled, the canonical global pass removes the
spherical cell-area-weighted displacement mean. If the centered field exceeds
`max_displacement_m`, the entire active field is uniformly rescaled. This
preserves relative morphology and the zero-mean invariant while enforcing the cap.

If zero-mean behavior is disabled, ordinary cellwise clipping is used instead.

Metadata records the limiter mode, limiter scale, pre-constraint maximum, and
post-constraint area-weighted mean.

The tile-local procedural field follows the same semantic separation. It is
area-centered, tapered to zero at the tile boundary, and applied exactly once
after the general finite-domain anchoring transform. It is excluded from the
physical sediment ledger.

## Pipeline placement and recoupling

Procedural erosion runs after the accepted terrain, climate, ocean, geology, and
hydrology fields exist. When enabled:

1. build the environmental forcing fields;
2. evaluate the advanced spherical phase-cell morphology;
3. apply the bounded terrain displacement;
4. keep physical landscape-evolution sediment accounting separate;
5. if a material displacement occurred and recoupling is enabled, recompute the
   affected ocean, climate, geology, hydrology, weather, appearance, resources,
   and optional society state;
6. use dedicated recoupling paths for dynamic/exotic surface liquids.

This is a bounded one-way correction, not an uncontrolled
terrain -> climate -> erosion -> terrain loop. Any future iterative coupling should
define explicit convergence criteria before adding repeated canonical erosion
passes.

## Diagnostics

Useful diagnostics include:

- procedural displacement in metres;
- phase coherence;
- ridge and crease maps;
- effective strength and preferred scale;
- fluvial, pluvial, glacial, marine, chemical, and freeze-thaw activity;
- soil saturation and liquid-mechanical factor;
- displacement limiter mode and scale;
- area-weighted mean displacement.

These fields are provenance/debug outputs unless explicitly documented as
conserved physical state.

## Numerical and performance characteristics

For each resolved octave every raster point evaluates 27 neighbouring 3-D cells.
With `N` raster cells and `O` resolved octaves, CPU work is therefore linear in
raster size and octave count with a relatively large trigonometric/hash constant:
approximately `O(27 N O)`.

The reference implementation is vectorized NumPy. The high-level operator defaults
to row-chunked phase-cell evaluation (`phase_chunk_rows=128`) so the 27-neighbour
temporary working set scales with chunk height rather than the full raster height.
Set `phase_chunk_rows=0` to force the unchunked reference path; chunked and
unchunked evaluation are regression-tested for bit-identical outputs. This reduces
peak memory without changing hashes, phases, normalization, or octave recurrence.

The implementation intentionally keeps the correctness path dependency-light and
inspectable. The kernel is suitable for
Numba, CuPy, JAX, native SIMD, or compute-shader acceleration, but an accelerated
backend must preserve deterministic hashing, partial-normalization behavior,
spherical coordinate conventions, LOD decisions, boundary behavior, and numerical
invariants.

## Validation contracts

The automated suite covers or should continue to cover:

- deterministic output for fixed seed/configuration;
- different seeds producing different resolved morphology;
- finite and shape validation at public boundaries;
- partial-normalization boundedness and monotonic response;
- exact identity for zero forcing;
- exact identity when all wavelengths are unresolved;
- bounded displacement;
- area-weighted zero mean even when the cap activates;
- modern `bedrock_code` and legacy `rock_code` compatibility;
- rejection of misshaped geology fields;
- dry-land zero rain/weathering forcing;
- cold/snowy glacial-regime selection;
- shelf/deep-ocean differentiation;
- deterministic watertight tile refinement;
- tile-local procedural area neutrality;
- full post-erosion recoupling.

CI executes the test suite across Python 3.11 through 3.14, exercises optional
performance/JIT/storage paths, and performs an end-to-end generation/resume job
after the matrix succeeds.

## Known limitations

1. The morphology operator is erosion-like but not a conservation-law hydraulic
   solver.
2. The 3-D spherical lattice is project-specific and is not pixel-identical to the
   2-D reference shaders.
3. Procedural branches can visually imply flow paths that are not guaranteed to be
   connected hydrological drainage; the hydrology graph remains authoritative.
4. Planetary-liquid mechanics is a screening relationship rather than a calibrated
   Shields/stream-power transport law.
5. Chemical aggressiveness is not yet a full rock-fluid reaction system.
6. Glacial, coastal, and submarine activity fields modulate morphology rather than
   solving the complete process physics.
7. Procedural positive/negative displacement is outside the physical sediment
   ledger.
8. Canonical recoupling is bounded, not a fully converged
   climate-geomorphology equilibrium.
9. NumPy is the reference backend; no GPU backend is currently required or claimed.

The intended balance is:

```text
fast procedural morphology
+ physically motivated environmental forcing
+ dedicated physical hydrology/sediment solvers
+ explicit diagnostics
+ bounded Earth-system recoupling
```
