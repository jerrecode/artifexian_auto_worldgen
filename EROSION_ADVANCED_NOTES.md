# Advanced phase-cell erosion notes

This branch continues the environment-conditioned procedural erosion integration
with a closer spherical adaptation of Rune Skovbo Johansen's final phase-cell /
Phacelle erosion filter.

Primary references:
- https://youtu.be/r4V21_uUK8Y
- https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html
- https://www.shadertoy.com/view/wXcfWn
- https://www.shadertoy.com/view/sf23W1

The upstream final shader is MPL-2.0. This repository contains an independently
written NumPy/spherical implementation of the published algorithmic ideas rather
than a line-for-line GLSL translation.

## Source-supported mechanics

The source technique is procedural morphology rather than hydraulic simulation.
A height field supplies height and gradient. Directional cosine/sine stripes are
aligned to slope. Coarser octave derivatives modify the direction of finer
octaves, producing branching ridges and gullies. Random local cells/pivots and
neighbor blending keep the stripe phase local enough to avoid the distortion of
one globally rotated pattern.

The final Runevision iteration combines:
- stacked fading between raw gullies and a recursively updated fade target;
- partial phase-vector normalization;
- straight-gully steering using the sign of the sine derivative;
- an optional assumed input slope;
- gully weighting with reciprocal strength compensation for pointier peaks;
- independent ridge and crease rounding;
- a parallel ridge-map recurrence.

Runevision explicitly notes that the ridge map is morphological rather than a
guaranteed connected drainage network.

## Phase-cell reconstruction

For weighted local phases theta_j:

    C = sum_j w_j cos(theta_j)
    S = sum_j w_j sin(theta_j)
    W = sum_j w_j
    M = sqrt(C^2 + S^2)

This branch exposes normalization n in [0, 1] and uses

    D = max(M, (1 - n) W, epsilon)
    cosine = C / D
    sine   = S / D

so n=0 retains cancellation magnitude, n=0.5 only fully normalizes sufficiently
coherent vectors, and n=1 gives the previous full unit normalization.

The stacked-mask transform is

    pow_inv(t, p) = 1 - (1 - clamp(t, 0, 1))^p
    mask_next = pow_inv(mask, detail) * new_mask

and each octave blends raw phase morphology against the previous fade target.

## Planetary adaptation

The global worldgen operator intentionally uses a seamless planet-centred 3-D cell
lattice rather than the reference 2-D Phacelle lattice. For unit surface direction
n, radius R, local wavelength lambda and cell scale c:

    x = R n
    q = x / (lambda c)

The containing 3-D cell and its 26 immediate neighbors are sampled. This removes
the longitude seam and allows the same low-level primitive to be reused on local
planet tiles. It also means this implementation is not bit-for-bit equivalent to
the ShaderToy kernel: neighbor count, weight kernel, hash and sampling domain
differ.

Straight-gully steering is likewise adapted to physical world units. The original
analytical derivative combines frequency and erosion strength directly. Here,
terrain relief is configured in metres while wavelength is in kilometres and can
vary spatially, so the internal spherical line field uses a bounded dimensionless
lateral steering contribution instead of conflating those units.

## Environmental forcing

artifexian_auto_worldgen owns terrain mutation and composes separate bounded
activity fields for fluvial, pluvial, glacial, marine, chemical-weathering and
freeze-thaw regimes. Lithology supplies erodibility, weatherability, frost/glacial
susceptibility, soil capacity and cohesion. Soil moisture affects different
processes differently instead of acting as one universal multiplier.

atmogen remains the one-way material/atmosphere dependency. It supplies liquid
density, dynamic viscosity and surface tension used by worldgen's reduced
mechanical-erosivity factor. The factor is screening-grade, not a calibrated
Shields, stream-power or sediment-entrainment equation.

The procedural delta is morphology only. Explicit hydrology and physical
landscape evolution retain ownership of drainage connectivity and the sediment
mass ledger.

## Numerical contracts

- wavelengths are world-space kilometres;
- output displacement is metres;
- unresolved octaves are skipped using local samples-per-wavelength;
- deterministic integer hashing is independent of NumPy's global RNG;
- the public low-level phase function retains full normalization by default for
  backward compatibility, while the advanced global configuration defaults to
  partial normalization;
- zero-mean global displacement is centered with spherical cell-area weights;
- if the centered field exceeds its displacement cap, one uniform scale factor is
  applied, preserving both relative morphology and zero mean;
- finite/shape checks reject malformed public operator inputs.

## Reference-backend cost

The current 3-D NumPy kernel examines 27 cells for every resolved sample and octave,
so time is O(27 N O) for N raster cells and O resolved octaves, with O(N) memory.

A local float64 microbenchmark of one phase-cell octave in the development
environment measured approximately:

| Raster | One octave |
| --- | ---: |
| 256 x 256 | 0.22 s |
| 512 x 512 | 0.89 s |
| 1024 x 1024 | 5.31 s |

These are isolated shared-CPU kernel timings, not end-to-end pipeline guarantees.
2048 x 2048 was not run because the vectorized float64 3-D temporary working set
becomes large; a compiled CPU or GPU backend is a better target at that scale.

## Known limitations

- procedural gullies are not guaranteed connected river paths;
- area-zero-mean geometry is not sediment mass conservation;
- generalized solvent-rock reaction kinetics are not implemented;
- glacial/coastal/submarine terms select morphology regimes but do not solve full
  ice, wave, turbidity-current or submarine-mass-wasting physics;
- spatially varying wavelength is most coherent when its forcing field is smooth;
- NumPy is currently the reference backend; no dedicated GPU backend is included.

## Video/transcript note

The companion video remains a primary conceptual reference, but the YouTube page
was throttled in the available retrieval environment and transcript-index searches
did not expose a complete English caption track. Missing wording was not invented.

A complete third-party copyrighted transcript is not stored in this repository.
Research notes should paraphrase the technical content and mark uncertainties
rather than reproduce the narration verbatim.
