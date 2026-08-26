from __future__ import annotations
import argparse
from pathlib import Path
import shutil

from .config import load_config
from .pipeline import WorldPipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Automatic procedural world generator")
    p.add_argument("generate", nargs="?", default="generate", help=argparse.SUPPRESS)
    p.add_argument("--config", type=Path, default=None, help="YAML configuration file")
    p.add_argument("--out", type=Path, default=Path("world-out"), help="Output directory")
    p.add_argument("--seed", type=int, default=None, help="Override root seed")
    p.add_argument("--quick", action="store_true", help="256x128 fast-preview mode: coarser tectonic history, fewer climate/surface iterations and reduced rendering workload")
    p.add_argument("--no-society", action="store_true", help="Stop after physical/geological generation")
    p.add_argument("--no-png", action="store_true", help="Skip PNG map rendering")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None: cfg.seed = args.seed
    if args.quick:
        cfg.resolution.width = 256
        cfg.resolution.height = 128
        cfg.resolution.history_step_myr = max(cfg.resolution.history_step_myr, 50)
        cfg.tectonics.history_grid_height = min(cfg.tectonics.history_grid_height, 56)
        cfg.tectonics.boundary_detail_octaves = min(cfg.tectonics.boundary_detail_octaves, 4)
        cfg.tectonics.boundary_deformation_iterations = min(cfg.tectonics.boundary_deformation_iterations, 1)
        cfg.tectonics.strain_boundary_warp_deg = min(cfg.tectonics.strain_boundary_warp_deg, 1.8)
        cfg.noise.octaves = min(cfg.noise.octaves, 4)
        cfg.noise.domain_warp_strength = min(cfg.noise.domain_warp_strength, 0.16)
        cfg.noise.wave_count = min(cfg.noise.wave_count, 3)
        cfg.climate.moisture_iterations = min(cfg.climate.moisture_iterations, 12)
        cfg.climate.thermal_memory_spinup_years = min(cfg.climate.thermal_memory_spinup_years, 3)
        cfg.hydrology.surface_evolution_iterations = min(cfg.hydrology.surface_evolution_iterations, 3)
        cfg.hydrology.flow_refresh_interval = max(cfg.hydrology.flow_refresh_interval, 2)
        cfg.hydrology.sediment_routing_passes = min(cfg.hydrology.sediment_routing_passes, 8)
        cfg.hydrology.max_river_centerlines = min(cfg.hydrology.max_river_centerlines, 60)
        cfg.weather.hurricane_seed_count = min(cfg.weather.hurricane_seed_count, 20)
        cfg.weather.hurricane_max_steps = min(cfg.weather.hurricane_max_steps, 70)
        cfg.society.settlement_count = min(cfg.society.settlement_count, 60)
    if args.no_society: cfg.society.enabled = False
    if args.no_png: cfg.output.save_png = False

    WorldPipeline(cfg).generate(args.out)
    print(f"World written to: {args.out.resolve()}")
    return 0
