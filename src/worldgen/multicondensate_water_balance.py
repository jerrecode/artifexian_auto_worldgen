from __future__ import annotations

"""Species-aware land water balance for composition-aware condensate forcing."""

import numpy as np

from .hydrology_advanced import WaterBalanceResult, _INFILTRATION, _SOIL_CAPACITY


def build_multicondensate_water_balance(climate, land, geology, cfg) -> WaterBalanceResult:
    """Conservative soil/groundwater balance using multicomponent condensate input.

    Liquid and solid condensate depths have already been obtained from an exactly
    mass-conservative species partition.  This stage conserves their additive volume
    through a generalized seasonal solid store, soil water, groundwater, fast runoff,
    baseflow and evapotranspiration.  ``snowpack_mm`` in the returned compatibility
    object therefore means *stored solid condensate liquid-equivalent depth* for an
    exotic world, not necessarily H2O snow.
    """
    forcing = getattr(climate, "hydrologic_forcing", None)
    if forcing is None:
        raise ValueError("multicondensate water balance requires hydrologic_forcing")

    liquid_input = np.maximum(np.asarray(forcing.monthly_liquid_input_mm, dtype=np.float64), 0.0)
    solid_input = np.maximum(np.asarray(forcing.monthly_solid_input_mm, dtype=np.float64), 0.0)
    thaw_fraction = np.clip(np.asarray(forcing.monthly_thaw_fraction, dtype=np.float64), 0.0, 1.0)
    t = np.asarray(climate.temperature_c, dtype=np.float64)
    humidity = np.asarray(getattr(climate, "humidity_proxy", np.zeros_like(t)), dtype=np.float64)
    if liquid_input.ndim != 3 or solid_input.shape != liquid_input.shape or t.shape != liquid_input.shape:
        raise ValueError("multicomponent forcing and climate temperature must have shape (month,y,x)")

    lf = np.asarray(land, dtype=bool)
    shape = lf.shape
    if geology is None:
        infiltration = np.full(shape, 0.58, dtype=np.float64)
        soil_capacity = np.full(shape, 165.0, dtype=np.float64)
    else:
        rock = np.clip(np.asarray(geology.rock_code, dtype=int), 0, len(_INFILTRATION) - 1)
        infiltration = _INFILTRATION[rock]
        soil_capacity = _SOIL_CAPACITY[rock]
    soil_capacity *= max(float(getattr(cfg, "soil_storage_multiplier", 1.0)), 0.05)
    infiltration = np.clip(infiltration, 0.08, 0.92)

    soil = 0.52 * soil_capacity * lf
    groundwater = 28.0 * infiltration * lf
    solid_store = np.zeros(shape, dtype=np.float64)
    recession = float(np.clip(getattr(cfg, "groundwater_recession_fraction_month", 0.065), 0.005, 0.45))
    storm_strength = float(np.clip(getattr(cfg, "storm_runoff_strength", 1.0), 0.1, 4.0))
    spinup_years = max(1, int(getattr(cfg, "water_balance_spinup_years", 3)))

    annual_fast = np.zeros(shape, dtype=np.float64)
    annual_base = np.zeros(shape, dtype=np.float64)
    annual_recharge = np.zeros(shape, dtype=np.float64)
    annual_et = np.zeros(shape, dtype=np.float64)
    start_storage = None

    total_precip = liquid_input + solid_input
    mean_monthly_p = np.mean(total_precip, axis=0)
    p_cv = np.std(total_precip, axis=0) / np.maximum(mean_monthly_p, 1.0)
    thermal_convective = np.mean(np.clip((t - 8.0) / 27.0, 0.0, 1.0), axis=0)
    reference_species = str(getattr(forcing, "reference_species", "H2O"))

    for year in range(spinup_years):
        if year == spinup_years - 1:
            annual_fast.fill(0.0)
            annual_base.fill(0.0)
            annual_recharge.fill(0.0)
            annual_et.fill(0.0)
            start_storage = soil + groundwater + solid_store

        for month in range(total_precip.shape[0]):
            pm = total_precip[month] * lf
            tm = t[month]
            solid_store += solid_input[month] * lf

            # Generalized seasonal thaw: the forcing computes this from the actual
            # phase state of all stored/precipitating species rather than a hard 0 C
            # water threshold.  A bounded fractional release is numerically stable and
            # converges to a periodic seasonal store during spin-up.
            thaw = thaw_fraction[month] * lf
            release_fraction = np.clip(0.82 * thaw ** 1.35, 0.0, 0.92)
            melt = solid_store * release_fraction
            solid_store -= melt
            available_liquid = liquid_input[month] * lf + melt

            convective = np.clip((tm - 8.0) / 25.0, 0.0, 1.0)
            relative_eventiness = np.clip(pm / np.maximum(mean_monthly_p, 8.0) - 0.65, 0.0, 2.5)
            storm_fraction = np.clip(
                storm_strength
                * (0.08 + 0.28 * convective + 0.12 * relative_eventiness)
                * (1.10 - 0.62 * infiltration),
                0.015,
                0.72,
            )
            direct_storm = available_liquid * storm_fraction
            infiltrable = available_liquid - direct_storm

            deficit = np.clip(1.0 - soil / np.maximum(soil_capacity, 1.0), 0.0, 1.0)
            infiltration_capacity = (34.0 + 105.0 * infiltration) * (0.35 + 0.65 * deficit)
            infiltrated = np.minimum(infiltrable, infiltration_capacity)
            horton = infiltrable - infiltrated
            soil += infiltrated
            saturation = np.maximum(soil - soil_capacity, 0.0)
            soil = np.minimum(soil, soil_capacity)

            if reference_species == "H2O":
                # Preserve the established screening PET for terrestrial water.
                pet = np.maximum(0.0, 2.55 * (tm + 5.0)) * lf
            else:
                # Celsius thresholds are meaningless for methane/ammonia/etc.  Use
                # thermodynamic mobility supplied by the species phase forcing and
                # atmospheric dryness instead.  This remains a reduced-order potential
                # evaporation term, but it is phase-aware and species-agnostic.
                hum = np.clip(humidity[month], 0.0, 4.0)
                dryness = 1.0 / (1.0 + 0.85 * hum)
                mobility = np.clip(0.15 + 0.85 * thaw, 0.0, 1.0)
                pet = (12.0 + 42.0 * mobility) * (0.30 + 0.70 * dryness) * lf
            et = np.minimum(soil, pet)
            soil -= et

            recharge = np.maximum(soil - 0.70 * soil_capacity, 0.0) * (0.24 + 0.28 * infiltration)
            soil -= recharge
            groundwater += recharge
            baseflow = groundwater * recession
            groundwater -= baseflow

            fast = direct_storm + horton + saturation
            if year == spinup_years - 1:
                annual_fast += fast
                annual_base += baseflow
                annual_recharge += recharge
                annual_et += et

    if start_storage is None:
        start_storage = soil + groundwater + solid_store
    end_storage = soil + groundwater + solid_store
    annual_p = np.sum(total_precip, axis=0) * lf
    total_runoff = annual_fast + annual_base
    residual = annual_p + start_storage - (total_runoff + annual_et + end_storage)

    storminess = np.clip(
        0.48 * (annual_fast / np.maximum(total_runoff, 1.0))
        + 0.30 * np.clip(p_cv / 1.5, 0.0, 1.0)
        + 0.22 * thermal_convective,
        0.0,
        1.0,
    ) * lf
    max_abs_resid = float(np.max(np.abs(residual[lf]))) if np.any(lf) else 0.0
    meta = {
        "model": "mass-conservative multicomponent condensate forcing + generalized solid store + soil bucket + infiltration/saturation excess + groundwater/baseflow",
        "multicomponent_condensate_hydrology": True,
        "reference_species": reference_species,
        "active_hydrologic_species": list(forcing.metadata.get("active_hydrologic_species", [])),
        "condensate_mass_partition_relative_l1_residual": float(
            forcing.metadata.get("mass_conservation_relative_l1_residual", 0.0)
        ),
        "spinup_years": spinup_years,
        "max_absolute_water_balance_residual_mm_year": max_abs_resid,
        "mean_land_runoff_mm_year": float(np.mean(total_runoff[lf])) if np.any(lf) else 0.0,
        "mean_land_baseflow_fraction": float(
            np.mean(annual_base[lf] / np.maximum(total_runoff[lf], 1.0))
        ) if np.any(lf) else 0.0,
        "mean_land_solid_condensate_storage_mm": float(np.mean(solid_store[lf])) if np.any(lf) else 0.0,
        "precipitation_depth_semantics": "sum of species liquid-equivalent depths after exact condensate mass partition",
        "limitations": "global reduced-order mixed-fluid bucket; no species-resolved porous-media flow or non-ideal mixture infiltration",
    }
    return WaterBalanceResult(
        total_runoff.astype(np.float32),
        annual_fast.astype(np.float32),
        annual_base.astype(np.float32),
        annual_recharge.astype(np.float32),
        annual_et.astype(np.float32),
        soil.astype(np.float32),
        groundwater.astype(np.float32),
        solid_store.astype(np.float32),
        storminess.astype(np.float32),
        residual.astype(np.float32),
        meta,
    )


__all__ = ["build_multicondensate_water_balance"]
