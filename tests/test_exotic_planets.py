from __future__ import annotations

from pathlib import Path

import numpy as np

from worldgen.astronomy import build_astronomy
from worldgen.config import WorldConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def _astro(config_name: str):
    cfg = load_config(ROOT / "config" / config_name)
    return cfg, build_astronomy(cfg.astronomy, np.random.default_rng(1234))


def test_default_configuration_preserves_legacy_greenhouse_and_single_moon():
    cfg = WorldConfig().validate()
    astro = build_astronomy(cfg.astronomy, np.random.default_rng(1))
    assert cfg.astronomy.greenhouse_model == "legacy"
    assert len(astro.moons) == 1
    assert astro.planet["body_role"] == "planet"
    assert abs(astro.planet["greenhouse_increment_k_approx"] - cfg.astronomy.greenhouse_k) < 1e-9


def test_super_earth_preset_has_three_deterministic_moons_and_super_earth_class():
    cfg, astro = _astro("super_earth.yaml")
    assert astro.planet["bulk_class"] == "super_earth"
    assert len(astro.moons) == 3
    assert [m["name"] for m in astro.moons] == ["Selene-A", "Selene-B", "Selene-C"]
    assert all(m["orbit_km"] > 0 for m in astro.moons)
    assert astro.interior["geological_activity_regime"] in {"active", "strongly_active"}


def test_titan_like_world_is_a_moon_with_methane_cycle_and_tidal_heat():
    cfg, astro = _astro("titan_like.yaml")
    assert astro.planet["body_role"] == "moon"
    assert astro.primary["type"] == "planet"
    assert astro.primary["mass_earth"] > 90
    assert astro.volatile_chemistry["active_condensible"] == "CH4"
    assert astro.interior["tidal_heating_flux_w_m2"] > 0
    assert astro.atmosphere["surface_pressure_bar"] > 1.0
    assert astro.atmosphere["fractions"]["N2"] > astro.atmosphere["fractions"]["CH4"]


def test_venus_like_composition_greenhouse_is_far_hotter_than_equilibrium():
    _, astro = _astro("venus_like.yaml")
    assert astro.atmosphere["greenhouse_model"] == "composition"
    assert astro.planet["mean_surface_temperature_c_approx"] > 300.0
    assert astro.planet["greenhouse_increment_k_approx"] > 200.0


def test_mars_like_thin_atmosphere_has_small_greenhouse_increment():
    _, astro = _astro("mars_like.yaml")
    assert astro.atmosphere["surface_pressure_bar"] < 0.01
    assert astro.planet["greenhouse_increment_k_approx"] < 30.0


def test_all_exotic_presets_validate():
    for name in ("mars_like.yaml", "venus_like.yaml", "titan_like.yaml", "tidal_giant_moon.yaml", "super_earth.yaml"):
        cfg = load_config(ROOT / "config" / name)
        assert isinstance(cfg, WorldConfig)
