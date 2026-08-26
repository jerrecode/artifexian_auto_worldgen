import pytest

from worldgen.config import WorldConfig


def test_legacy_defaults_preserve_fixed_execution_semantics():
    cfg = WorldConfig().validate()
    assert cfg.hydrology.flow_refresh_mode == "interval"
    assert cfg.hydrology.flow_refresh_interval == 1
    assert cfg.simulation.adaptive_convergence is False
    assert cfg.simulation.adaptive_final_coupling is False
    assert cfg.simulation.earth_system_passes == 3
    assert cfg.simulation.final_climate_ocean_passes == 1


def test_adaptive_configuration_validates_when_ranges_are_consistent():
    cfg = WorldConfig()
    cfg.hydrology.flow_refresh_mode = "adaptive"
    cfg.hydrology.flow_refresh_max_interval = 4
    cfg.simulation.adaptive_convergence = True
    cfg.simulation.min_earth_system_passes = 2
    cfg.simulation.max_earth_system_passes = 7
    cfg.simulation.adaptive_final_coupling = True
    cfg.simulation.min_final_climate_ocean_passes = 1
    cfg.simulation.max_final_climate_ocean_passes = 4
    assert cfg.validate() is cfg


def test_invalid_flow_refresh_mode_is_rejected():
    cfg = WorldConfig()
    cfg.hydrology.flow_refresh_mode = "sometimes"
    with pytest.raises(ValueError, match="flow_refresh_mode"):
        cfg.validate()


def test_adaptive_pass_ranges_must_be_ordered():
    cfg = WorldConfig()
    cfg.simulation.min_earth_system_passes = 5
    cfg.simulation.max_earth_system_passes = 4
    with pytest.raises(ValueError, match="max_earth_system_passes"):
        cfg.validate()


def test_convergence_tolerances_cannot_be_negative():
    cfg = WorldConfig()
    cfg.simulation.convergence_precip_mm_year = -1.0
    with pytest.raises(ValueError, match="convergence_precip"):
        cfg.validate()
