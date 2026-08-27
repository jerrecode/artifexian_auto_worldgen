from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .grid import SphereGrid, distance_to, normalize01, smooth_periodic
from .tectonics import TectonicResult
from .terrain import TerrainResult
from .ocean import OceanResult
from .climate import ClimateResult
from .config import NoiseConfig
from .noise import hybrid_noise01, noise_kwargs, GEOLOGY_BLEND, NoiseBlend, StaticNoiseFields

ROCK_NAMES = {
    0: "unconsolidated_sediment",
    1: "sandstone_clastic",
    2: "carbonate",
    3: "granite",
    4: "metamorphic",
    5: "basalt_mafic",
    6: "andesite",
    7: "rhyolite_felsic",
    8: "ultramafic_greenstone",
}


@dataclass(slots=True)
class GeologyResult:
    rock_code: np.ndarray
    paleoshallow_sea: np.ndarray
    craton: np.ndarray
    shield: np.ndarray
    platform: np.ndarray
    metallogenic_belt: np.ndarray
    greenstone_belt: np.ndarray
    sedimentary_basin: np.ndarray
    metadata: dict

    def rock_mask(self, *names: str) -> np.ndarray:
        codes = [k for k, v in ROCK_NAMES.items() if v in names]
        return np.isin(self.rock_code, codes)


def build_geology(
    grid: SphereGrid,
    tect: TectonicResult,
    terrain: TerrainResult,
    ocean: OceanResult,
    climate: ClimateResult,
    rng: np.random.Generator,
    noise_cfg: NoiseConfig | None = None,
    static_noise: StaticNoiseFields | None = None,
) -> GeologyResult:
    land = terrain.land
    if static_noise is not None:
        lith_texture=static_noise.geology_lith; igneous_texture=static_noise.geology_igneous
    else:
        lith_texture = hybrid_noise01(
            land.shape, rng, base_scale_px=max(grid.height / 22.0, 3.0),
            **noise_kwargs(noise_cfg, profile=GEOLOGY_BLEND, octaves=max(5, min(8, getattr(noise_cfg, "octaves", 7)))),
        )
        igneous_texture = hybrid_noise01(
            land.shape, rng, base_scale_px=max(grid.height / 30.0, 2.5),
            **noise_kwargs(noise_cfg, profile=NoiseBlend(0.34,0.28,0.12,0.26), octaves=max(4, min(7, getattr(noise_cfg, "octaves", 6)))),
        )
    stable = (tect.paleo_convergence < 0.18) & (tect.paleo_divergence < 0.18) & tect.continental_crust
    # Large connected stable interiors are cratonic nuclei. This is a geographic
    # morphology operation, so it must respect longitude wrapping and antipodal
    # pole crossing instead of applying planar scipy.ndimage boundary semantics.
    craton = grid.ops.binary_opening(stable, iterations=2)
    old_low = tect.orogen_age_myr > 450
    shield = craton & old_low & (terrain.elevation_km > 0.25)
    platform = craton & ~shield

    # Paleoshallow-sea likelihood reconstructs the transcript's repeated shallow-sea masks from
    # low continental regions, rifting/sea-level-high proxies, and today's shelf remnants.
    low_cont = tect.continental_crust & (terrain.elevation_km < 0.75)
    paleo = 0.45 * normalize01(tect.paleo_divergence) + 0.35 * low_cont + 0.20 * terrain.shelf
    paleo = smooth_periodic(paleo, (3, 5))
    paleo = normalize01(paleo)

    # Sedimentary basins: lowlands, ancient shallow seas, foreland areas near mountains.
    near_orogen = np.exp(-distance_to(tect.paleo_convergence > 0.45, grid) / 450.0)
    basin_score = 0.45 * paleo + 0.30 * (terrain.lowland_strength > 0) + 0.25 * near_orogen
    basin = land & (smooth_periodic(basin_score, (2, 3)) > 0.38)

    rock = np.zeros(land.shape, dtype=np.uint8)
    # Default exposed continental substrate.
    rock[land] = 3
    rock[shield] = np.where(lith_texture[shield] < 0.56, 4, 3)
    rock[platform] = 1

    # Sedimentary facies; carbonate favored warm former shallow seas, clastic around erosion/basins.
    warm = climate.annual_temperature_c > 12
    carbonate = land & (paleo > 0.50) & warm & (near_orogen < 0.65)
    clastic = basin & ~carbonate
    sediment = land & terrain.lowland_strength.astype(bool) & (climate.annual_precipitation_mm > 700)
    rock[clastic] = 1
    rock[carbonate] = 2
    rock[sediment & ~carbonate] = 0

    # Igneous/metamorphic overprints from active and past tectonics.
    dconv = distance_to(tect.convergent, grid)
    active_orogen = land & (dconv < 300)
    andesite = active_orogen & (igneous_texture > 0.42)
    rhyolite = active_orogen & ~andesite
    rock[andesite] = 6
    rock[rhyolite] = 7
    metam = land & (tect.paleo_convergence > 0.63) & ~active_orogen
    rock[metam] = 4

    mafic = land & ((tect.lip_strength > 0.42) | (tect.hotspot_strength > 0.65) |
                    ((tect.paleo_divergence > 0.68) & (tect.rift_age_myr < 180)))
    rock[mafic] = 5

    # Greenstone/ultramafic belts: ancient mafic crust in shields, sparse and highly eroded.
    green = shield & (tect.lip_strength > 0.20) & (tect.orogen_age_myr > 250)
    green |= shield & (lith_texture > 0.64) & (igneous_texture < 0.58)
    rock[green] = 8

    metallogenic = normalize01(
        0.55 * np.exp(-dconv / 380.0) + 0.30 * tect.paleo_convergence +
        0.10 * tect.lip_strength + 0.05 * tect.hotspot_strength
    )
    meta = {"rock_codes": ROCK_NAMES, "paleoshallow_sea_model": "rift+low-continent+shelf likelihood",
            "noise_model": "shared hybrid multi-type multifractal lithologic fabrics"}
    return GeologyResult(rock, paleo.astype(np.float32), craton, shield, platform,
                         metallogenic.astype(np.float32), green, basin, meta)
