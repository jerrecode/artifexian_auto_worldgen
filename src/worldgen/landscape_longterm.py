from __future__ import annotations

"""Long-timescale implicit fluvial landscape relaxation.

Short explicit erosion iterations are useful for local feedback but cannot represent
hundreds of millions of years of drainage-profile adjustment.  This backend solves a
steady-state/implicit stream-power-inspired channel profile from outlets upstream,
then conservatively routes the resulting sediment through the existing drainage graph.
It is intentionally reduced-order rather than pretending to be a full Landscape
Evolution Model, but elapsed geological time and drainage hierarchy now enter the
terrain solution explicitly.
"""

from dataclasses import dataclass
import numpy as np

from . import hydrology_base as _base
from .drainage import DrainageGraph
from .grid import SphereGrid, normalize01, smooth_periodic
from .hydrology_advanced import transport_sediment_topological

_LITH_K = np.asarray([1.75, 1.20, 0.82, 0.46, 0.36, 0.55, 0.52, 0.58, 0.40], dtype=float)


@dataclass(slots=True)
class LongTermLandscapeResult:
    elevation_km: np.ndarray
    channel_incision_m: np.ndarray
    routed_deposition_m: np.ndarray
    delta_deposition_m: np.ndarray
    valley_widening_m: np.ndarray
    metadata: dict


def _receiver_edge_lengths_km(grid: SphereGrid, receiver: np.ndarray) -> np.ndarray:
    recv = np.asarray(receiver, dtype=np.int64).ravel()
    out = np.zeros(recv.size, dtype=np.float64)
    good = recv >= 0
    if np.any(good):
        xyz = np.asarray(grid.xyz, float).reshape(-1, 3)
        src = np.flatnonzero(good)
        dot = np.sum(xyz[src] * xyz[recv[src]], axis=1)
        out[src] = float(grid.radius_km) * np.arccos(np.clip(dot, -1.0, 1.0))
    return np.maximum(out, 1.0e-3)


def evolve_landscape_longterm(
    grid: SphereGrid,
    terrain,
    ocean,
    hydrology,
    geology,
    tectonics,
    hydrology_cfg,
    geomorphic_params,
    *,
    elapsed_myr: float,
) -> LongTermLandscapeResult:
    z0 = np.asarray(terrain.elevation_km, dtype=np.float64)
    land = np.asarray(terrain.land, dtype=bool)
    flow = np.asarray(hydrology.flow_to, dtype=np.int64)
    graph = DrainageGraph.from_receiver(flow, z0.shape)
    area = np.maximum(np.asarray(hydrology.drainage_area_km2, dtype=float), 1.0)
    discharge = np.asarray(hydrology.discharge_index, dtype=float)
    q = np.clip(normalize01(discharge, robust=True), 0.0, 1.0)
    channel_class = np.asarray(getattr(hydrology, "channel_class", hydrology.rivers.astype(np.uint8) * 3), dtype=np.uint8)
    channel = land & (channel_class >= 1)
    lith = _LITH_K[np.clip(np.asarray(geology.rock_code, int), 0, len(_LITH_K) - 1)]
    fluid = max(float(getattr(geomorphic_params, "stream_power_multiplier", 1.0)), 0.05)
    substrate = max(float(getattr(geomorphic_params, "substrate_erodibility_multiplier", 1.0)), 0.1)

    activity = normalize01(
        0.72 * np.asarray(tectonics.convergence_strength, float)
        + 0.18 * np.asarray(tectonics.stress_field, float)
        + 0.10 * np.asarray(tectonics.divergence_strength, float),
        robust=True,
    )
    # Stream-power steady-state scaling: stronger uplift requires a steeper channel,
    # while large drainage area / discharge / erodibility produce gentler profiles.
    a_scale = np.maximum(area / 1000.0, 1.0e-3)
    erosion_efficiency = np.maximum(fluid * substrate * lith * (0.18 + 0.82 * np.sqrt(q)), 0.04)
    equilibrium_slope = (
        0.0055
        * (0.32 + 1.75 * activity)
        / (erosion_efficiency * np.power(a_scale, 0.34))
    )
    equilibrium_slope = np.clip(equilibrium_slope, 3.0e-5, 0.075)

    # Relaxation timescale becomes short for high-discharge, weak-substrate channels
    # and long in resistant low-order headwaters.  The implicit factor remains stable
    # even when elapsed time is hundreds of Myr.
    tau_myr = 95.0 / np.maximum(erosion_efficiency * (0.18 + 1.35 * np.sqrt(q)), 0.05)
    relax = np.clip(float(max(elapsed_myr, 0.0)) / np.maximum(tau_myr, 1.0e-3), 0.0, 60.0)
    edge = _receiver_edge_lengths_km(grid, flow)

    old = z0.ravel()
    new = old.copy()
    ch = channel.ravel()
    eqs = equilibrium_slope.ravel()
    rr = relax.ravel()
    lf = land.ravel()
    # Downstream-to-upstream because each implicit node uses the already-relaxed
    # receiver elevation. Ocean/base-level receivers stay at the current shoreline.
    for kk in range(graph.order.size - 1, -1, -1):
        node = int(graph.order[kk])
        if not ch[node]:
            continue
        rec = int(flow[node])
        if rec < 0:
            continue
        rec_z = new[rec] if lf[rec] else min(new[rec], 0.0)
        target = rec_z + eqs[node] * edge[node]
        lam = rr[node]
        candidate = (old[node] + lam * target) / (1.0 + lam)
        # Erosion-only long-term pass: tectonic construction already exists in z0.
        # Preserve a tiny positive gradient and prevent pathological inversion.
        floor = rec_z + 1.0e-6
        new[node] = min(old[node], max(candidate, floor))

    channel_z = new.reshape(z0.shape)
    incision = np.maximum((z0 - channel_z) * 1000.0, 0.0) * channel

    # Valleys occupy finite width rather than one raster line. Width grows with
    # discharge/order; smoothing the channel incision approximates lateral relief
    # relaxation while retaining divides and resistant interfluves.
    width = np.clip(np.asarray(getattr(hydrology, "river_width_proxy", 0.0), float), 0.0, 1.0)
    widened = smooth_periodic(incision * (0.18 + 0.52 * width), (0.62, 0.78)) * land
    max_widen = np.maximum(incision, 0.18 * np.max(incision) if incision.size else 0.0)
    widened = np.minimum(widened, max_widen)
    z_eroded = z0 - incision / 1000.0 - 0.34 * widened / 1000.0

    cell_area = _base._cell_area_km2(grid)
    routing = np.asarray(hydrology.filled_elevation_km, float)
    slope = _base._receiver_slope(routing, flow, grid)
    dep, sediment_flux, exported = transport_sediment_topological(
        routing,
        flow,
        incision + 0.34 * widened,
        q,
        slope,
        cell_area,
        land,
        hydrology_cfg,
    )
    delta = _base._delta_deposition(
        grid,
        z_eroded,
        land,
        exported,
        cell_area,
        hydrology_cfg,
        marine_energy=getattr(ocean, "current_speed", None),
    )
    z_final = z_eroded + dep / 1000.0 + delta / 1000.0

    area_weights = np.asarray(grid.cell_area_weights, float)
    mean_incision = float(np.average(incision[land], weights=area_weights[land])) if np.any(land) else 0.0
    max_incision = float(np.max(incision)) if incision.size else 0.0
    active_fraction = float(grid.weighted_fraction(channel) / max(grid.weighted_fraction(land), 1.0e-12))
    metadata = {
        "model": "implicit downstream-controlled stream-power steady-profile relaxation + finite valley widening + conservative topological sediment routing",
        "elapsed_myr": float(elapsed_myr),
        "mean_land_channel_incision_m": mean_incision,
        "max_channel_incision_m": max_incision,
        "resolved_channel_fraction_land": active_fraction,
        "mean_relaxation_factor_channel": float(np.mean(relax[channel])) if np.any(channel) else 0.0,
        "sediment_routing": "O(N) conservative topological transport; long river paths are not hop-limited",
        "limitations": "reduced-order steady-profile backend; no transient knickpoint PDE, explicit grain-size abrasion, landslide runout or flexural isostasy in this solve",
    }
    return LongTermLandscapeResult(
        z_final.astype(np.float32),
        incision.astype(np.float32),
        dep.astype(np.float32),
        delta.astype(np.float32),
        widened.astype(np.float32),
        metadata,
    )


__all__ = ["LongTermLandscapeResult", "evolve_landscape_longterm"]
