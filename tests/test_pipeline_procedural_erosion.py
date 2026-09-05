from __future__ import annotations

from worldgen import WorldConfig, WorldPipeline


def test_public_pipeline_includes_procedural_erosion_layer_but_defaults_off():
    cfg = WorldConfig().validate()
    assert cfg.procedural_erosion.enabled is False
    assert WorldPipeline.__module__ == "worldgen.pipeline_procedural_erosion"


def test_procedural_erosion_config_validation_rejects_invalid_scale_order():
    cfg = WorldConfig()
    cfg.procedural_erosion.min_wavelength_km = 100.0
    cfg.procedural_erosion.max_wavelength_km = 50.0
    try:
        cfg.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid procedural wavelength range was accepted")
