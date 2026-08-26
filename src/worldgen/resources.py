from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage

from .config import ResourcesConfig
from .grid import SphereGrid, distance_to, normalize01, smooth_periodic
from .tectonics import TectonicResult
from .terrain import TerrainResult
from .ocean import OceanResult
from .climate import ClimateResult
from .hydrology import HydrologyResult
from .geology import GeologyResult


@dataclass(slots=True)
class ResourceResult:
    suitability: dict[str, np.ndarray]
    deposits: list[dict]
    wood_potential: np.ndarray
    sea_salt_access: np.ndarray
    technology_access: dict[str, np.ndarray]
    metadata: dict


def _near(mask: np.ndarray, grid: SphereGrid, km: float) -> np.ndarray:
    return distance_to(mask, grid) <= km


def _wet_dry_zone(climate: ClimateResult) -> np.ndarray:
    # Tropical/subtropical with appreciable rain but strong seasonality.
    p = climate.precipitation_mm
    ann = p.sum(0)
    season = (p.max(0) - p.min(0)) / np.maximum(p.mean(0), 1.0)
    return (np.abs(climate.annual_temperature_c) > -999) & (climate.annual_temperature_c > 16) & (ann > 300) & (season > 0.8)


def _sample_deposits(
    name: str,
    commodity: str,
    age: str,
    field: np.ndarray,
    grid: SphereGrid,
    rng: np.random.Generator,
    density: float,
    base_count: int,
    richness: tuple[float, float, float] = (0.60, 0.32, 0.08),
) -> list[dict]:
    f = np.nan_to_num(np.clip(field.astype(float), 0, None))
    eligible = np.flatnonzero(f.ravel() > 0.08)
    if not len(eligible):
        return []
    count = int(max(0, round(base_count * density)))
    count = min(count, len(eligible))
    if count == 0:
        return []
    weights = f.ravel()[eligible]
    weights = weights ** 1.6
    weights /= weights.sum()
    picks = rng.choice(eligible, count, replace=False, p=weights)
    out = []
    h, w = f.shape
    rlabels = np.array(["poor", "moderate", "rich"])
    rprob = np.array(richness, float); rprob /= rprob.sum()
    # Compute the rich-zone threshold once per suitability field. The previous implementation
    # recomputed a whole-raster quantile for every sampled point, turning a few hundred deposits
    # into hundreds of redundant O(N) scans at high resolution.
    positive = f[f > 0]
    q85 = float(np.quantile(positive, 0.85)) if positive.size else np.inf
    for idx in picks:
        y, x = divmod(int(idx), w)
        # High suitability biases one richness tier upward while preserving transcript rarity choices.
        rp = rprob.copy()
        if f[y, x] > q85:
            rp = np.array([rp[0] * .65, rp[1] * 1.10, rp[2] * 1.85]); rp /= rp.sum()
        out.append({
            "type": name, "commodity": commodity, "technology_age": age,
            "richness": str(rng.choice(rlabels, p=rp)),
            "latitude": float(grid.lat[y, x]), "longitude": float(grid.lon[y, x]),
            "suitability": float(f[y, x]),
        })
    return out


def build_resources(
    grid: SphereGrid,
    tect: TectonicResult,
    terrain: TerrainResult,
    ocean: OceanResult,
    climate: ClimateResult,
    hydro: HydrologyResult,
    geo: GeologyResult,
    cfg: ResourcesConfig,
    rng: np.random.Generator,
) -> ResourceResult:
    land = terrain.land
    rock = geo.rock_code
    clastic = np.isin(rock, [0, 1])
    carbonate = rock == 2
    granitic = rock == 3
    metamorphic = rock == 4
    basalt = rock == 5
    andesite = rock == 6
    rhyolite = rock == 7
    greenstone = rock == 8
    felsic = rhyolite | granitic | andesite
    sedimentary = clastic | carbonate
    arid = np.char.startswith(climate.koppen, "B")
    polar = np.char.startswith(climate.koppen, "E")
    wetdry = _wet_dry_zone(climate)
    lowlat = np.abs(grid.lat) <= 45
    wet = climate.annual_precipitation_mm > 700
    highwet = climate.annual_precipitation_mm > 1400
    cold_winter = climate.temperature_c.min(0) <= 6.0
    recent_orogen = tect.orogen_age_myr < 180
    old_orogen = (tect.orogen_age_myr >= 180) & (tect.orogen_age_myr < 650)
    active_orogen = _near(tect.convergent, grid, 420) & land
    orogen = active_orogen | (geo.metallogenic_belt > 0.45)
    igneous_orogen = orogen & np.isin(rock, [5, 6, 7, 3])
    lip = tect.lip_strength > 0.38
    hotspot = tect.hotspot_strength > 0.58

    # Fuel maps.
    alat = np.abs(grid.lat)
    treeline_m = np.where(alat <= 30, 4000.0,
                  np.where(alat <= 50, 4000.0 - 130.0 * (alat - 30.0),
                           np.maximum(0.0, 1400.0 - 75.0 * (alat - 50.0))))
    wood = land & ~arid & ~polar & (ocean.elevation_km * 1000 <= treeline_m)
    subarctic = np.char.startswith(climate.koppen, "D") & (climate.annual_temperature_c < 7)
    temperate_no_dry_summer = np.char.startswith(climate.koppen, "Cf")
    continental_no_dry_summer = np.char.startswith(climate.koppen, "Df")
    glacial_legacy = (np.abs(grid.lat) > 42) & ((climate.annual_temperature_c < 8) | (terrain.mountain_strength > .65))
    peat = land & glacial_legacy & (subarctic | temperate_no_dry_summer | continental_no_dry_summer) & wet
    coal_swamp = land & lowlat & wet & (terrain.lowland_strength > 0)
    mountain_burial = np.exp(-distance_to(orogen, grid) / 500.0)
    coal = normalize01(coal_swamp.astype(float) * (0.55 + 0.45 * mountain_burial) * (1 - 0.85 * tect.lip_strength))
    coal *= (sedimentary | (terrain.mountain_strength > .58))

    # Copper/gold source belts with explicit erosion/age decay.
    age_decay = np.exp(-np.clip(tect.orogen_age_myr, 0, 1000) / 360.0)
    copper_rich = normalize01(orogen.astype(float) * (0.30 + 0.70 * age_decay) * (1 - 0.55 * lip) * (1 - 0.4 * hotspot))
    mvt_cu = normalize01((_near(basalt | hotspot, grid, 260) & carbonate).astype(float) * (0.4 + 0.6 * tect.lip_strength))
    vms_base = normalize01(geo.metallogenic_belt * (andesite | rhyolite | basalt) * (0.4 + 0.6 * tect.paleo_convergence))
    vms_secondary = normalize01(_near(vms_base > .35, grid, 260).astype(float) * clastic * wetdry * (~cold_winter))
    arsenical_cu = normalize01(copper_rich * (_near(coal > .25, grid, 350) | carbonate | (geo.paleoshallow_sea > .55)))

    gold_source = normalize01(orogen.astype(float) * (0.35 + 0.65 * age_decay) * ((andesite | rhyolite) | old_orogen))
    river_near_gold = hydro.rivers & _near(gold_source > .35, grid, 420)
    glacial_bonus = glacial_legacy & (np.abs(grid.lat) > 35)
    depositional = normalize01(hydro.cumulative_deposition_m + 0.35 * hydro.sediment_flux_index)
    gold_placer = normalize01(river_near_gold.astype(float) * (0.45 + 0.25 * normalize01(climate.annual_precipitation_mm) + 0.25 * glacial_bonus + 0.65 * depositional))
    gold_laterite = normalize01(greenstone.astype(float) * wetdry * land)
    gold_gossan = normalize01(vms_base * (rhyolite | andesite))
    silver_epithermal = normalize01(orogen.astype(float) * (rhyolite | andesite) * age_decay)
    silver_vms = normalize01(vms_secondary * 0.8)
    silver_placer = normalize01(hydro.rivers.astype(float) * _near((mvt_cu > .3) | (silver_epithermal > .4), grid, 380) * (0.45 + 0.75 * depositional))

    # Bronze-Age expansion.
    bronze_vms = vms_base
    sedex = normalize01((geo.paleoshallow_sea > .45).astype(float) * clastic * tect.paleo_divergence * (0.35 + 0.65 * tect.paleo_convergence))
    skarn = normalize01(carbonate.astype(float) * _near(rhyolite | andesite, grid, 180) * (0.45 + 0.55 * geo.metallogenic_belt))
    iocg = normalize01(geo.craton.astype(float) * _near(rhyolite | granitic, grid, 220) * (0.35 + 0.65 * copper_rich))
    tin_belt = normalize01((granitic | rhyolite).astype(float) * (0.35 + 0.65 * geo.metallogenic_belt))
    tin_placer = normalize01(hydro.rivers.astype(float) * _near(tin_belt > .35, grid, 420) * (0.45 + 0.75 * depositional))
    tin_copper = normalize01(copper_rich * _near(tin_belt > .35, grid, 300))
    porphyry_gold = normalize01(active_orogen.astype(float) * (rhyolite | andesite) * (0.4 + 0.6 * geo.metallogenic_belt))
    # Lead upgrades hydrothermal systems plus a wet MVT variant.
    mvt_lead = normalize01(sedimentary.astype(float) * wet * _near((basalt | felsic) & (terrain.mountain_strength > .35), grid, 350))
    lead = normalize01(np.maximum.reduce([0.75 * skarn, 0.70 * bronze_vms, 0.75 * sedex, 0.50 * porphyry_gold, 0.65 * mvt_lead]))
    lead_silver = normalize01(np.maximum.reduce([skarn, sedex, mvt_lead]) * lead)

    # Iron Age.
    wetlands = (hydro.lakes | (hydro.rivers & (terrain.lowland_strength > 0))) & wet
    bog_climate = subarctic | np.char.startswith(climate.koppen, "Af") | np.char.startswith(climate.koppen, "Cfa") | np.char.startswith(climate.koppen, "Cfb")
    bog_iron = normalize01(wetlands.astype(float) * bog_climate * _near(geo.metallogenic_belt > .38, grid, 500))
    skarn_iron = skarn
    iron_laterite = normalize01(lip.astype(float) * wetdry)
    warm_paleosea = (geo.paleoshallow_sea > .48) & carbonate & (climate.annual_temperature_c > 10)
    oolitic_iron = normalize01(warm_paleosea.astype(float))
    hydrothermal_iron = normalize01(np.maximum(bronze_vms, sedex) * 0.75)
    zinc_contact = (basalt & _near(metamorphic, grid, 120)) | (metamorphic & _near(basalt, grid, 120))
    zinc = normalize01(np.maximum(lead, zinc_contact.astype(float) * (0.4 + 0.6 * tect.lip_strength)) + 0.15 * carbonate)

    # Gemology extension. The supplied video transcript commissions/references the gemstone map rather
    # than deriving it on-screen; these rules follow the external Deposits & Gemology guide it cites.
    # Kimberlite districts are sparse cratonic/greenstone occurrences, with shields somewhat less exposed.
    kimberlite = normalize01(geo.craton.astype(float) * (0.35 + 0.65 * greenstone) * (1.0 - 0.35 * geo.shield))
    diamond = normalize01(kimberlite * (0.45 + 0.55 * tect.paleo_divergence))
    # Gem-bearing pegmatites are rare felsic/intermediate intrusions, especially exposed in older orogens.
    old_exposed = np.clip((tect.orogen_age_myr - 80.0) / 420.0, 0, 1)
    pegmatite = normalize01((rhyolite | andesite | granitic).astype(float) * (0.25 + 0.75 * old_exposed) * land)
    emerald_aquamarine_tourmaline = normalize01(pegmatite * (0.4 + 0.6 * geo.metallogenic_belt))
    jadeite = normalize01(metamorphic.astype(float) * _near(tect.convergent, grid, 420) * (0.35 + 0.65 * geo.metallogenic_belt))
    # Sedimentary/mineral gems from the guide.
    turquoise = normalize01(clastic.astype(float) * arid * _near(copper_rich > .30, grid, 320))
    agate = normalize01(clastic.astype(float) * _near(basalt, grid, 260))
    malachite_azurite = normalize01(clastic.astype(float) * _near((copper_rich > .32) & orogen, grid, 260))
    opal_smithsonite = normalize01(clastic.astype(float) * (geo.paleoshallow_sea > .50))
    # Resistant gems concentrate in alluvial/fossil placers downstream of primary gem districts.
    primary_gems = np.maximum.reduce([diamond, emerald_aquamarine_tourmaline, jadeite, turquoise, agate])
    gem_placer = normalize01(hydro.rivers.astype(float) * _near(primary_gems > .32, grid, 500) *
                             (0.35 + 0.30 * normalize01(climate.annual_precipitation_mm) + 0.75 * depositional))

    # Salt.
    coast_land = terrain.land & ndimage.binary_dilation(terrain.ocean, iterations=2)
    fuel = wood | peat | (coal > .22)
    sea_salt = coast_land & (arid | fuel)
    depression = hydro.lakes | ((terrain.lowland_strength > 0) & (hydro.filled_elevation_km - ocean.elevation_km > .005))
    salt_flat = normalize01((arid & land & clastic & depression).astype(float) + 0.35 * (arid & land & geo.sedimentary_basin))
    halite = normalize01(clastic.astype(float) * (geo.paleoshallow_sea > .52) * (0.35 + 0.65 * arid))

    fields = {
        "peat": peat.astype(np.float32), "coal": coal.astype(np.float32),
        "copper_rich": copper_rich.astype(np.float32), "native_copper_mvt": mvt_cu.astype(np.float32),
        "secondary_vms_copper": vms_secondary.astype(np.float32), "arsenical_copper": arsenical_cu.astype(np.float32),
        "gold_placer": gold_placer.astype(np.float32), "gold_laterite": gold_laterite.astype(np.float32),
        "gold_gossan": gold_gossan.astype(np.float32), "silver_epithermal": silver_epithermal.astype(np.float32),
        "silver_vms": silver_vms.astype(np.float32), "silver_placer": silver_placer.astype(np.float32),
        "bronze_vms": bronze_vms.astype(np.float32), "sedex": sedex.astype(np.float32), "skarn": skarn.astype(np.float32),
        "iocg": iocg.astype(np.float32), "tin_belt": tin_belt.astype(np.float32), "tin_placer": tin_placer.astype(np.float32),
        "tin_copper": tin_copper.astype(np.float32), "porphyry_gold": porphyry_gold.astype(np.float32),
        "lead": lead.astype(np.float32), "lead_silver": lead_silver.astype(np.float32),
        "bog_iron": bog_iron.astype(np.float32), "skarn_iron": skarn_iron.astype(np.float32),
        "iron_laterite": iron_laterite.astype(np.float32), "oolitic_iron": oolitic_iron.astype(np.float32),
        "hydrothermal_iron": hydrothermal_iron.astype(np.float32), "zinc": zinc.astype(np.float32),
        "salt_flat": salt_flat.astype(np.float32), "halite": halite.astype(np.float32),
        "kimberlite_diamond": diamond.astype(np.float32), "pegmatite_gems": emerald_aquamarine_tourmaline.astype(np.float32),
        "jadeite": jadeite.astype(np.float32), "turquoise": turquoise.astype(np.float32),
        "agate": agate.astype(np.float32), "malachite_azurite": malachite_azurite.astype(np.float32),
        "opal_smithsonite": opal_smithsonite.astype(np.float32), "gem_placer": gem_placer.astype(np.float32),
    }

    specs = [
        ("coal", "coal", "fuel", coal, 42, (.35,.45,.20)),
        ("native_copper_mvt", "copper", "copper", mvt_cu, 26, (.75,.23,.02)),
        ("secondary_vms_copper", "copper", "copper", vms_secondary, 30, (.55,.40,.05)),
        ("arsenical_copper", "copper+arsenic", "copper", arsenical_cu, 16, (.75,.23,.02)),
        ("gold_placer", "gold", "copper", gold_placer, 24, (.05,.72,.23)),
        ("gold_laterite", "gold", "copper", gold_laterite, 14, (.10,.70,.20)),
        ("gold_gossan", "gold", "copper", gold_gossan, 13, (.65,.31,.04)),
        ("silver_epithermal", "silver", "copper", silver_epithermal, 16, (.55,.39,.06)),
        ("silver_vms", "silver", "copper", silver_vms, 10, (.60,.36,.04)),
        ("silver_placer", "silver", "copper", silver_placer, 12, (.55,.40,.05)),
        ("vms", "copper+polymetallic", "bronze", bronze_vms, 34, (.05,.55,.40)),
        ("sedex", "lead+zinc+silver", "bronze", sedex, 22, (.03,.27,.70)),
        ("skarn", "copper+iron", "bronze", skarn, 25, (.05,.35,.60)),
        ("iocg", "iron+copper+gold", "bronze", iocg, 8, (.05,.90,.05)),
        ("tin_placer", "tin", "bronze", tin_placer, 20, (.30,.58,.12)),
        ("tin_copper", "tin+copper", "bronze", tin_copper, 15, (.25,.60,.15)),
        ("porphyry_gold", "gold", "bronze", porphyry_gold, 14, (.30,.58,.12)),
        ("lead", "lead", "bronze", lead, 25, (.20,.55,.25)),
        ("lead_silver", "lead+silver", "bronze", lead_silver, 12, (.15,.60,.25)),
        ("bog_iron", "iron", "iron", bog_iron, 40, (.25,.55,.20)),
        ("skarn_iron", "iron", "iron", skarn_iron, 30, (.05,.35,.60)),
        ("iron_laterite", "iron", "iron", iron_laterite, 22, (.05,.80,.15)),
        ("oolitic_iron", "iron", "iron", oolitic_iron, 34, (.12,.63,.25)),
        ("hydrothermal_iron", "iron", "iron", hydrothermal_iron, 12, (.02,.18,.80)),
        ("zinc", "zinc", "iron", zinc, 28, (.25,.60,.15)),
        ("kimberlite_diamond", "diamond", "gemstone", diamond, 8, (.70,.25,.05)),
        ("gem_pegmatite", "beryl+tourmaline+topaz", "gemstone", emerald_aquamarine_tourmaline, 12, (.65,.30,.05)),
        ("jadeite", "jadeite", "gemstone", jadeite, 8, (.70,.27,.03)),
        ("turquoise", "turquoise", "gemstone", turquoise, 9, (.60,.34,.06)),
        ("opal_smithsonite", "opal+smithsonite", "gemstone", opal_smithsonite, 9, (.55,.38,.07)),
        ("gem_placer", "mixed_resistant_gems", "gemstone", gem_placer, 14, (.45,.45,.10)),
        ("salt_flat", "salt", "all", salt_flat, 18, (.15,.60,.25)),
        ("halite", "salt", "all", halite, 20, (.20,.55,.25)),
    ]
    deposits: list[dict] = []
    for name, commodity, age, field, count, rich in specs:
        deposits.extend(_sample_deposits(name, commodity, age, field, grid, rng, cfg.deposit_density, count, rich))

    # Meteoric iron is intentionally ultra-rare and discoverability-biased to desert/tundra.
    meteor_field = terrain.land.astype(float) * (0.25 + 0.75 * (arid | polar))
    deposits.extend(_sample_deposits("meteoric_iron", "iron", "iron", meteor_field, grid, rng,
                                     1.0, cfg.meteorite_expected_count, (.0,.05,.95)))

    # Geological existence is distinct from preindustrial accessibility. Offshore deposits can exist,
    # but they do not unlock Copper/Bronze/Iron Age technology.
    for d in deposits:
        y = int(np.argmin(np.abs(grid.lat_1d - d["latitude"])))
        x = int(np.argmin(np.abs(((grid.lon_1d - d["longitude"] + 180.0) % 360.0) - 180.0)))
        d["submerged"] = bool(not land[y, x])
        d["accessible_preindustrial"] = bool(land[y, x])

    # Technology-resource intersections turn raw geology into cultural affordances.
    copper_access = land & (copper_rich > .25)
    bronze_access = land & copper_access & (_near((tin_belt > .30) & land, grid, 750) | _near((arsenical_cu > .25) & land, grid, 450))
    iron_sources = np.maximum.reduce([bog_iron, skarn_iron, iron_laterite, oolitic_iron, hydrothermal_iron])
    iron_access = land & (iron_sources > .24) & _near(fuel & land, grid, 650)
    brass_access = land & (copper_rich > .25) & _near((zinc > .25) & land, grid, 800)
    technology = {"copper": copper_access, "bronze": bronze_access, "iron": iron_access, "brass": brass_access}

    meta = {
        "deposit_count": len(deposits), "rule_count": len(specs) + 1,
        "submerged_deposit_count": int(sum(bool(d.get("submerged")) for d in deposits)),
        "note": "Resource maps are fuzzy geological suitability fields; point deposits are seeded samples. Geological existence is separated from preindustrial accessibility.",
    }
    return ResourceResult(fields, deposits, wood.astype(np.float32), sea_salt, technology, meta)
