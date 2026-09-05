from __future__ import annotations

import numpy as np

from worldgen.config import ClimateConfig, NoiseConfig, TerrainConfig, TectonicsConfig
from worldgen.geology import build_geology
from worldgen.grid import SphereGrid
from worldgen.noise import build_static_noise_fields
from worldgen.rng import RngPool
from worldgen.terrain import build_terrain
from worldgen.tectonics import generate_tectonics


def test_geology_tracks_bedrock_separately_from_offshore_surface_sediment():
    grid = SphereGrid(64, 32, 6371.0)
    rng = RngPool(20260905)
    tcfg = TectonicsConfig(plate_count=6, history_grid_height=32)
    # Keep the test on established public builders so the bedrock field is exercised
    # against real tectonic and terrain masks rather than a hand-built fixture.
    tect = generate_tectonics(
        grid,
        tcfg,
        type("Resolution", (), {"history_myr": 50, "history_step_myr": 25})(),
        rng("tectonics"),
        NoiseConfig(octaves=4),
    )
    terrain = build_terrain(
        grid, tect, tcfg, TerrainConfig(fractal_octaves=4), rng("terrain"), NoiseConfig(octaves=4)
    )
    static = build_static_noise_fields(grid.shape, NoiseConfig(octaves=4), rng)
    climate = type(
        "Climate",
        (),
        {
            "annual_temperature_c": np.full(grid.shape, 12.0, dtype=np.float32),
            "annual_precipitation_mm": np.full(grid.shape, 700.0, dtype=np.float32),
        },
    )()
    ocean = type("Ocean", (), {})()
    geology = build_geology(
        grid, tect, terrain, ocean, climate, rng("geology"), NoiseConfig(octaves=4), static
    )
    assert geology.bedrock_code.shape == grid.shape
    assert np.all(geology.rock_code[terrain.ocean] == 0)
    assert np.any(geology.bedrock_code[terrain.ocean] != 0)
    assert geology.metadata["bedrock_surface_material_separated"] is True
