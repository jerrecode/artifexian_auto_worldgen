from __future__ import annotations

"""Surface-evolution execution policies layered on the shared hydrology physics."""

import numpy as np

from . import hydrology_base as _base
from .drainage import DrainageGraph
from .flow_refresh import FlowRefreshState, decide_flow_refresh


def evolve_surface(
    grid,
    terrain,
    ocean,
    climate,
    geology,
    cfg,
    tectonics=None,
    rng: np.random.Generator | None = None,
    noise_cfg=None,
    static_noise=None,
):
    """Evolve terrain with fixed or physical-change-triggered drainage refresh.

    ``flow_refresh_mode='interval'`` delegates directly to the historical physical
    implementation, preserving existing worlds. Adaptive/hybrid modes run the same
    erosion, sediment, delta, uplift and diffusion equations but reuse drainage
    topology until the configured physical thresholds require a rebuild.
    """
    mode = str(getattr(cfg, "flow_refresh_mode", "interval")).strip().lower()
    if mode == "interval":
        return _base.evolve_surface(
            grid,
            terrain,
            ocean,
            climate,
            geology,
            cfg,
            tectonics,
            rng,
            noise_cfg,
            static_noise,
        )
    return _evolve_surface_adaptive(
        grid,
        terrain,
        ocean,
        climate,
        geology,
        cfg,
        tectonics,
        rng,
        noise_cfg,
        static_noise,
    )


def _evolve_surface_adaptive(
    grid,
    terrain,
    ocean,
    climate,
    geology,
    cfg,
    tectonics=None,
    rng: np.random.Generator | None = None,
    noise_cfg=None,
    static_noise=None,
):
    z = ocean.elevation_km.astype(np.float64).copy()
    erosion_total = np.zeros_like(z)
    dep_total = np.zeros_like(z)
    flux = np.zeros_like(z)
    delta_total = np.zeros_like(z)
    uplift_total = np.zeros_like(z)
    migration_total = np.zeros_like(z)
    meander = np.zeros_like(z)
    cell_area = _base._cell_area_km2(grid)
    lith = _base._LITH_ERODIBILITY[
        np.clip(geology.rock_code, 0, len(_base._LITH_ERODIBILITY) - 1)
    ]

    if rng is None:
        wiggle = np.sin(np.deg2rad(grid.lon * 3.7 + grid.lat * 1.9)) + 0.45 * np.sin(
            np.deg2rad(grid.lon * 11.3 - grid.lat * 4.1)
        )
        delta_texture = 0.5 + 0.5 * np.sin(
            np.deg2rad(grid.lon * 7.1 - grid.lat * 3.3)
        )
    elif static_noise is not None:
        wiggle = static_noise.hydro_wiggle
        delta_texture = static_noise.delta_texture
    else:
        wiggle = _base.hybrid_multifractal(
            z.shape,
            rng,
            base_scale_px=max(grid.height / 36.0, 2.5),
            **_base.noise_kwargs(
                noise_cfg,
                profile=_base.HYDRO_BLEND,
                octaves=max(5, min(8, getattr(noise_cfg, "octaves", 7))),
            ),
        )
        delta_texture = _base.hybrid_noise01(
            z.shape,
            rng,
            base_scale_px=max(grid.height / 50.0, 2.0),
            **_base.noise_kwargs(
                noise_cfg,
                profile=_base.NoiseBlend(0.36, 0.25, 0.12, 0.27),
                octaves=max(4, min(7, getattr(noise_cfg, "octaves", 6))),
            ),
        )
    wiggle = (wiggle - np.mean(wiggle)) / max(np.std(wiggle), 1e-8)
    meander_prior = (
        np.clip(lith / np.max(_base._LITH_ERODIBILITY), 0, 1)
        * terrain.lowland_strength
    )

    last_flow = None
    last_route = None
    last_graph: DrainageGraph | None = None
    refresh_state = FlowRefreshState()
    refresh_iterations: list[int] = []
    refresh_reasons: list[str] = []
    previous_delta_max_m = 0.0

    for it in range(max(0, int(cfg.surface_evolution_iterations))):
        land = z > 0
        oc = ~land
        decision = decide_flow_refresh(
            refresh_state,
            iteration=it,
            elevation_km=z,
            land=land,
            mode=str(cfg.flow_refresh_mode),
            interval=int(cfg.flow_refresh_interval),
            max_interval=int(cfg.flow_refresh_max_interval),
            elevation_threshold_m=float(cfg.flow_refresh_elevation_threshold_m),
            land_change_fraction_threshold=float(cfg.flow_refresh_land_change_fraction),
            delta_threshold_m=float(cfg.flow_refresh_delta_threshold_m),
            previous_delta_max_m=previous_delta_max_m,
            area_weights=grid.cell_area_weights,
        )
        if decision.refresh or last_flow is None or last_graph is None or last_route is None:
            micro = (
                cfg.meander_microrelief_m / 1000.0
            ) * wiggle * np.maximum(meander_prior, meander)
            route = _base._priority_flood(z + micro * land, oc, grid)
            flow = _base._flow_directions(route, oc, grid)
            graph = DrainageGraph.from_receiver(flow, z.shape)
            last_route, last_flow, last_graph = route, flow, graph
            refresh_state.mark_refreshed(it, z, land)
            refresh_iterations.append(int(it))
            refresh_reasons.append(decision.reason)
        else:
            route, flow, graph = last_route, last_flow, last_graph

        runoff = _base._runoff_mm(climate, land, geology, cfg)
        water_source = (runoff / 1000.0) * cell_area
        discharge = graph.accumulate(water_source)
        slope = _base._receiver_slope(route, flow, grid)
        vals = discharge[land & (discharge > 0)]
        qref = np.quantile(vals, 0.985) if len(vals) else 1.0
        qn = np.clip(discharge / max(qref, 1e-12), 0, 2.5)
        meander = _base._meander_field(geology, lith, slope, qn, land, cfg)

        recv = flow.astype(np.int64, copy=False)
        lf = land.ravel()
        valid_recv = recv >= 0
        safe = np.where(valid_recv, recv, 0)
        receiver_land = np.zeros(recv.size, bool)
        receiver_land[valid_recv] = lf[safe[valid_recv]]
        receiver_land = receiver_land.reshape(land.shape)
        major_mouth = land & (~receiver_land) & (qn > 0.34)
        fluvial_domain = land & (receiver_land | major_mouth)
        sref = 0.006
        stream_e = (
            cfg.max_fluvial_erosion_m_per_iteration
            * lith
            * (qn ** cfg.stream_power_m)
            * ((slope / sref) ** cfg.stream_power_n)
        )
        stream_e = (
            np.clip(stream_e, 0, cfg.max_fluvial_erosion_m_per_iteration)
            * fluvial_domain
        )
        weathering = (
            0.20
            * lith
            * np.clip(climate.annual_precipitation_mm / 1200.0, 0, 1.5)
            * land
        )
        e = np.clip(
            stream_e + weathering, 0, cfg.max_fluvial_erosion_m_per_iteration
        )

        channel = (meander > 0.06) & (qn > 0.08) & land
        banks = grid.ops.binary_dilation(channel, iterations=1) & land & ~channel
        lateral_source = grid.ops.grey_dilation(e * meander, iterations=1)
        asym = 0.35 + 0.65 * _base.normalize01(wiggle, robust=False)
        lateral = np.clip(
            cfg.lateral_erosion_fraction * lateral_source * asym * banks,
            0,
            cfg.max_fluvial_erosion_m_per_iteration * 0.75,
        )
        e_total = np.clip(
            e + lateral, 0, cfg.max_fluvial_erosion_m_per_iteration * 1.35
        )
        migration_total += lateral

        dep, load, exported = _base._transport_sediment(
            route,
            flow,
            e_total,
            np.clip(qn / 2.5, 0, 1),
            slope,
            cell_area,
            land,
            cfg,
        )
        delta = _base._delta_deposition(
            grid,
            z,
            land,
            exported,
            cell_area,
            cfg,
            marine_energy=getattr(ocean, "current_speed", None),
            distributary_texture=delta_texture,
        )
        previous_delta_max_m = float(np.max(delta)) if delta.size else 0.0

        uplift = np.zeros_like(z)
        subsidence = np.zeros_like(z)
        if tectonics is not None:
            active = (
                0.78 * tectonics.convergence_strength + 0.22 * tectonics.stress_field
            )
            uplift = cfg.tectonic_uplift_m_per_iteration * (active ** 1.15) * land
            subsidence = (
                cfg.rift_subsidence_m_per_iteration
                * tectonics.divergence_strength
                * land
            )
            uplift_total += uplift

        sm = _base.smooth_periodic(z, (0.65, 0.75))
        diffusion = (
            (sm - z)
            * cfg.hillslope_diffusion_strength
            * np.clip(lith, 0.35, 1.8)
            * land
        )
        diffusion = np.clip(diffusion, -0.010, 0.010)
        z += diffusion
        z += (uplift - subsidence) / 1000.0
        z -= e_total / 1000.0
        z += dep / 1000.0
        z += delta / 1000.0
        erosion_total += e_total
        dep_total += dep + delta
        delta_total += delta
        flux = np.maximum(flux, load)
        meander_prior = meander

    meta = {
        "iterations": int(cfg.surface_evolution_iterations),
        "max_cumulative_erosion_m": float(erosion_total.max()),
        "max_cumulative_deposition_m": float(dep_total.max()),
        "max_delta_aggradation_m": float(delta_total.max()),
        "max_tectonic_uplift_m": float(uplift_total.max()),
        "max_meander_bank_migration_m": float(migration_total.max()),
        "mean_land_erosion_m": float(
            np.average(
                erosion_total[terrain.land],
                weights=grid.cell_area_weights[terrain.land],
            )
        )
        if np.any(terrain.land)
        else 0.0,
        "model": "rainfall/snowmelt runoff + lithology-dependent stream power + lateral meander migration + hierarchical sediment routing/shelf-controlled deltas + active tectonic uplift + hillslope diffusion",
        "drainage_graph": "reusable topological order; accumulation is O(N) and Numba-capable",
        "noise_model": "shared hybrid multi-type multifractal channel microrelief and distributary texture",
        "flow_refresh_mode": str(cfg.flow_refresh_mode),
        "flow_refresh_count": len(refresh_iterations),
        "flow_refresh_iterations": refresh_iterations,
        "flow_refresh_reasons": refresh_reasons,
    }
    return _base.SurfaceEvolutionResult(
        z.astype(np.float32),
        erosion_total.astype(np.float32),
        dep_total.astype(np.float32),
        _base.normalize01(flux).astype(np.float32),
        delta_total.astype(np.float32),
        uplift_total.astype(np.float32),
        migration_total.astype(np.float32),
        meander.astype(np.float32),
        meta,
    )


__all__ = ["evolve_surface"]
