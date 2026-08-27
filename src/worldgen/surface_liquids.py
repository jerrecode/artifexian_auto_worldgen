from __future__ import annotations

"""Mass-conserving volatile partitioning and global surface-liquid filling.

The solver deliberately separates three questions:

1. how much of each configured volatile inventory is currently atmospheric vapor,
   condensed solid, mobile liquid, or incompatible with the prescribed atmospheric
   pressure/phase state;
2. what volume the mobile liquid occupies at its thermodynamic density; and
3. what equipotential liquid level produces exactly that volume over the spherical
   solid-surface heightmap.

The geometric integration treats each raster cell as a spherical wedge. A cell with
solid angle ``omega`` and bed radius ``r_b`` contributes

    omega/3 * (r_l**3 - r_b**3)

to the reservoir when the liquid radius ``r_l`` lies above its bed. This is more
accurate than a planar area*depth approximation and remains well behaved for planets
whose radius differs substantially from Earth.

The volatile phase partition is a reduced-order equilibrium approximation. It is
not a full atmosphere-ocean-ice chemical-potential solver and multicomponent liquid
mixtures are currently volume-additive. Those limitations are recorded explicitly
in diagnostics rather than hidden behind Earth-specific constants.
"""

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np

from .grid import SphereGrid
from .planetary_physics import (
    SPECIES,
    canonical_species,
    coolprop_available,
    phase_code_grid,
    saturation_pressure_bar,
)


@dataclass(slots=True)
class VolatilePartition:
    species: str
    total_mass_kg: float
    vapor_mass_kg: float
    solid_mass_kg: float
    liquid_mass_kg: float
    noncondensed_excess_mass_kg: float
    liquid_density_kg_m3: float
    liquid_volume_m3: float
    vapor_capacity_kg: float
    solid_stable_area_fraction: float
    liquid_stable_area_fraction: float
    mean_liquid_temperature_k: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class SurfaceLiquidResult:
    liquid_level_km: float
    liquid_depth_m: np.ndarray
    liquid_mask: np.ndarray
    relative_surface_elevation_km: np.ndarray
    total_liquid_mass_kg: float
    total_liquid_volume_m3: float
    integrated_volume_m3: float
    volume_residual_m3: float
    partitions: dict[str, VolatilePartition]
    metadata: dict

    def to_dict(self) -> dict:
        out = dict(self.metadata)
        out.update(
            {
                "liquid_level_km": float(self.liquid_level_km),
                "total_liquid_mass_kg": float(self.total_liquid_mass_kg),
                "total_liquid_volume_m3": float(self.total_liquid_volume_m3),
                "integrated_volume_m3": float(self.integrated_volume_m3),
                "volume_residual_m3": float(self.volume_residual_m3),
                "partitions": {k: v.to_dict() for k, v in self.partitions.items()},
            }
        )
        return out


def _surface_area_m2(grid: SphereGrid) -> float:
    return 4.0 * math.pi * (float(grid.radius_km) * 1000.0) ** 2


def _cell_area_m2(grid: SphereGrid) -> np.ndarray:
    return np.asarray(grid.cell_area_weights, dtype=np.float64) * _surface_area_m2(grid)


def _cell_solid_angle_sr(grid: SphereGrid) -> np.ndarray:
    # cell_area_weights are normalized to sum to one over the whole sphere.
    return np.asarray(grid.cell_area_weights, dtype=np.float64) * (4.0 * math.pi)


def _shell_volume_per_sr(r0_m: float, depth_m: float) -> float:
    """Volume per steradian between radii r0 and r0+depth, evaluated stably."""
    d = max(float(depth_m), 0.0)
    r = float(r0_m)
    return r * r * d + r * d * d + d * d * d / 3.0


def _radial_increment_for_volume(r0_m: float, volume_per_sr_m3: float) -> float:
    """Invert ``r0^2*d + r0*d^2 + d^3/3`` with stable Newton iterations."""
    target = max(float(volume_per_sr_m3), 0.0)
    if target == 0.0:
        return 0.0
    r0 = max(float(r0_m), 1.0)
    d = max(target / (r0 * r0), 0.0)
    for _ in range(10):
        f = _shell_volume_per_sr(r0, d) - target
        deriv = (r0 + d) ** 2
        step = f / max(deriv, 1e-30)
        d = max(d - step, 0.0)
        if abs(step) <= max(1e-9, 1e-12 * max(d, 1.0)):
            break
    return d


def integrate_liquid_volume_m3(
    grid: SphereGrid,
    bed_elevation_km: np.ndarray,
    liquid_level_km: float,
) -> float:
    """Integrate the exact raster-wedge volume below one global liquid level."""
    bed_m = np.asarray(bed_elevation_km, dtype=np.float64) * 1000.0
    if bed_m.shape != grid.shape:
        raise ValueError("bed_elevation_km shape must match grid")
    level_m = float(liquid_level_km) * 1000.0
    depth = np.maximum(level_m - bed_m, 0.0)
    omega = _cell_solid_angle_sr(grid)
    radius_m = float(grid.radius_km) * 1000.0
    r0 = np.maximum(radius_m + bed_m, 1.0)
    per_sr = r0 * r0 * depth + r0 * depth * depth + depth**3 / 3.0
    return float(np.sum(omega * per_sr, dtype=np.float64))


def solve_global_liquid_level(
    grid: SphereGrid,
    bed_elevation_km: np.ndarray,
    liquid_volume_m3: float,
) -> tuple[float, np.ndarray, float]:
    """Fill the deepest raster elevations upward until the target volume is reached.

    The algorithm sorts cells by bed elevation once, then advances the common liquid
    radius through those breakpoints while accumulating active solid angle. The last
    partial interval is solved analytically/numerically in radius, so no iterative
    bisection over the complete map is necessary.
    """
    bed = np.asarray(bed_elevation_km, dtype=np.float64)
    if bed.shape != grid.shape:
        raise ValueError("bed_elevation_km shape must match grid")
    if not np.all(np.isfinite(bed)):
        raise ValueError("bed_elevation_km must be finite")
    target = float(liquid_volume_m3)
    if not math.isfinite(target) or target < 0.0:
        raise ValueError("liquid_volume_m3 must be finite and non-negative")

    bed_m = bed.ravel() * 1000.0
    omega = _cell_solid_angle_sr(grid).ravel()
    radius_m = float(grid.radius_km) * 1000.0
    if target == 0.0:
        level_m = float(np.min(bed_m))
        depth = np.zeros(grid.shape, dtype=np.float32)
        return level_m / 1000.0, depth, 0.0

    order = np.argsort(bed_m, kind="stable")
    z = bed_m[order]
    om = omega[order]
    remaining = target
    active_omega = 0.0
    i = 0
    n = len(z)
    current = float(z[0])

    while i < n:
        # Activate every cell at this elevation before raising the common surface.
        j = i
        while j < n and z[j] == z[i]:
            active_omega += float(om[j])
            j += 1
        current = float(z[i])
        if j >= n:
            d = _radial_increment_for_volume(
                radius_m + current, remaining / max(active_omega, 1e-30)
            )
            level_m = current + d
            remaining = 0.0
            break

        next_z = float(z[j])
        delta = next_z - current
        capacity = active_omega * _shell_volume_per_sr(radius_m + current, delta)
        if remaining <= capacity:
            d = _radial_increment_for_volume(
                radius_m + current, remaining / max(active_omega, 1e-30)
            )
            level_m = current + d
            remaining = 0.0
            break
        remaining -= capacity
        i = j
    else:  # pragma: no cover - defensive, loop always exits via final active interval
        level_m = current

    depth_m = np.maximum(
        level_m - bed.reshape(grid.shape) * 1000.0, 0.0
    ).astype(np.float32)
    integrated = integrate_liquid_volume_m3(grid, bed, level_m / 1000.0)
    return float(level_m / 1000.0), depth_m, float(integrated)


def _liquid_density(
    species: str,
    temperature_k: float,
    pressure_bar: float,
    backend: str,
) -> float:
    key = canonical_species(species)
    sp = SPECIES[key]
    if backend in {"auto", "coolprop"} and coolprop_available() and sp.coolprop_name:
        try:
            from CoolProp.CoolProp import PropsSI

            rho = float(
                PropsSI(
                    "D",
                    "T",
                    float(temperature_k),
                    "P",
                    max(float(pressure_bar), 1e-9) * 1e5,
                    sp.coolprop_name,
                )
            )
            if math.isfinite(rho) and rho > 0.0:
                return rho
        except Exception:
            if backend == "coolprop":
                raise
    return float(sp.liquid_density_kg_m3)


def partition_volatile_inventory(
    grid: SphereGrid,
    species: str,
    total_mass_kg: float,
    temperature_c: np.ndarray,
    *,
    surface_pressure_bar: float,
    gravity_m_s2: float,
    relative_humidity: float = 0.65,
    ice_fixation_efficiency: float = 0.25,
    thermodynamics_backend: str = "auto",
) -> VolatilePartition:
    """Partition one volatile inventory into vapor, fixed solid, and mobile liquid.

    Atmospheric vapor is capped by local saturation pressure (times relative
    humidity) integrated hydrostatically over the sphere. Remaining inventory is
    condensed only where a solid or liquid phase is thermodynamically available.

    If the supplied temperature/pressure field is gas-only or supercritical while
    the prescribed atmospheric column is too small to contain all configured mass,
    the remainder is reported as ``noncondensed_excess_mass_kg`` rather than being
    falsely converted to a surface liquid. Such a state means atmospheric pressure
    should ultimately be solved together with inventory in a higher-fidelity model.
    """
    key = canonical_species(species)
    total = max(float(total_mass_kg), 0.0)
    t_k = np.asarray(temperature_c, dtype=np.float64) + 273.15
    if t_k.shape != grid.shape:
        raise ValueError("temperature_c shape must match grid")
    g = max(float(gravity_m_s2), 1e-6)
    rh = float(np.clip(relative_humidity, 0.0, 1.0))
    pressure = max(float(surface_pressure_bar), 1e-12)

    psat_bar = np.asarray(
        saturation_pressure_bar(key, t_k, backend="builtin"), dtype=np.float64
    )
    vapor_pressure_pa = np.minimum(psat_bar * rh, pressure) * 1e5
    vapor_capacity = float(
        np.sum(vapor_pressure_pa * _cell_area_m2(grid) / g, dtype=np.float64)
    )
    vapor_mass = min(total, vapor_capacity)
    remainder = max(total - vapor_mass, 0.0)

    phase = phase_code_grid(key, t_k, pressure)
    weights = np.asarray(grid.cell_area_weights, dtype=np.float64)
    solid_fraction_area = float(np.sum(weights[phase == 2]))
    liquid_fraction_area = float(np.sum(weights[phase == 1]))

    solid_mass = 0.0
    liquid_mass = 0.0
    noncondensed_excess = 0.0
    if remainder > 0.0:
        if liquid_fraction_area <= 1e-12 and solid_fraction_area <= 1e-12:
            # No condensed surface phase exists anywhere at the prescribed P/T.
            noncondensed_excess = remainder
        elif liquid_fraction_area <= 1e-12:
            solid_mass = remainder
        elif solid_fraction_area <= 1e-12:
            liquid_mass = remainder
        else:
            fixed_fraction = float(
                np.clip(solid_fraction_area * ice_fixation_efficiency, 0.0, 1.0)
            )
            solid_mass = remainder * fixed_fraction
            liquid_mass = remainder - solid_mass

    liquid_cells = phase == 1
    if np.any(liquid_cells):
        w = weights[liquid_cells]
        mean_t = float(np.average(t_k[liquid_cells], weights=w))
    else:
        mean_t = float(np.average(t_k, weights=weights))
    density = _liquid_density(key, mean_t, pressure, thermodynamics_backend)
    liquid_volume = liquid_mass / max(density, 1e-30)

    # Explicit inventory closure catches future partition regressions.
    closure = vapor_mass + solid_mass + liquid_mass + noncondensed_excess
    if not math.isclose(closure, total, rel_tol=2e-12, abs_tol=1e-6):
        raise RuntimeError("volatile partition failed mass closure")

    return VolatilePartition(
        species=key,
        total_mass_kg=total,
        vapor_mass_kg=float(vapor_mass),
        solid_mass_kg=float(solid_mass),
        liquid_mass_kg=float(liquid_mass),
        noncondensed_excess_mass_kg=float(noncondensed_excess),
        liquid_density_kg_m3=float(density),
        liquid_volume_m3=float(liquid_volume),
        vapor_capacity_kg=float(vapor_capacity),
        solid_stable_area_fraction=solid_fraction_area,
        liquid_stable_area_fraction=liquid_fraction_area,
        mean_liquid_temperature_k=mean_t,
    )


def solve_surface_liquids(
    grid: SphereGrid,
    bed_elevation_km: np.ndarray,
    temperature_c: np.ndarray,
    inventories_kg: Mapping[str, float],
    *,
    surface_pressure_bar: float,
    gravity_m_s2: float,
    relative_humidity: float = 0.65,
    ice_fixation_efficiency: float = 0.25,
    thermodynamics_backend: str = "auto",
) -> SurfaceLiquidResult:
    """Partition all volatile inventories and fill their combined liquid volume."""
    if not isinstance(inventories_kg, Mapping) or not inventories_kg:
        raise ValueError("inventories_kg must be a non-empty mapping")
    bed = np.asarray(bed_elevation_km, dtype=np.float64)
    if bed.shape != grid.shape:
        raise ValueError("bed_elevation_km shape must match grid")

    partitions: dict[str, VolatilePartition] = {}
    for name, mass in inventories_kg.items():
        key = canonical_species(name)
        part = partition_volatile_inventory(
            grid,
            key,
            float(mass),
            temperature_c,
            surface_pressure_bar=surface_pressure_bar,
            gravity_m_s2=gravity_m_s2,
            relative_humidity=relative_humidity,
            ice_fixation_efficiency=ice_fixation_efficiency,
            thermodynamics_backend=thermodynamics_backend,
        )
        partitions[key] = part

    total_mass = float(sum(p.liquid_mass_kg for p in partitions.values()))
    total_volume = float(sum(p.liquid_volume_m3 for p in partitions.values()))
    level_km, depth_m, integrated = solve_global_liquid_level(grid, bed, total_volume)
    mask = depth_m > 1.0e-6
    relative = (bed - level_km).astype(np.float32)
    residual = integrated - total_volume
    total_inventory = float(sum(p.total_mass_kg for p in partitions.values()))
    vapor = float(sum(p.vapor_mass_kg for p in partitions.values()))
    solid = float(sum(p.solid_mass_kg for p in partitions.values()))
    noncondensed = float(
        sum(p.noncondensed_excess_mass_kg for p in partitions.values())
    )

    return SurfaceLiquidResult(
        liquid_level_km=float(level_km),
        liquid_depth_m=depth_m,
        liquid_mask=mask,
        relative_surface_elevation_km=relative,
        total_liquid_mass_kg=total_mass,
        total_liquid_volume_m3=total_volume,
        integrated_volume_m3=integrated,
        volume_residual_m3=residual,
        partitions=partitions,
        metadata={
            "method": "global spherical-wedge equipotential fill from deepest bed upward",
            "inventory_total_kg": total_inventory,
            "vapor_mass_kg": vapor,
            "solid_fixed_mass_kg": solid,
            "noncondensed_excess_mass_kg": noncondensed,
            "liquid_mass_kg": total_mass,
            "liquid_volume_m3": total_volume,
            "relative_humidity": float(np.clip(relative_humidity, 0.0, 1.0)),
            "ice_fixation_efficiency": float(
                np.clip(ice_fixation_efficiency, 0.0, 1.0)
            ),
            "thermodynamics_backend": thermodynamics_backend,
            "mixture_model": "volume-additive independent pure-species phase partitions; no activity/fugacity mixture correction yet",
            "pressure_coupling": "fixed supplied surface pressure; excess gas/supercritical inventory is diagnosed rather than converted to liquid",
            "geometry": "spherical raster wedges using exact radial shell volume per cell solid angle",
        },
    )


__all__ = [
    "VolatilePartition",
    "SurfaceLiquidResult",
    "integrate_liquid_volume_m3",
    "solve_global_liquid_level",
    "partition_volatile_inventory",
    "solve_surface_liquids",
]
