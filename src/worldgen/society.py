from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy import ndimage

from .config import SocietyConfig
from .grid import SphereGrid, distance_to, normalize01
from .terrain import TerrainResult
from .climate import ClimateResult
from .hydrology import HydrologyResult
from .resources import ResourceResult
from .weather import WeatherResult
from .appearance import SurfaceAppearanceResult


@dataclass(slots=True)
class SocietyResult:
    suitability: np.ndarray
    portal: dict | None
    settlements: list[dict]
    cultures: list[dict]
    links: list[dict]
    history_events: list[dict]
    metadata: dict


def _weighted_pick_separated(
    field: np.ndarray,
    count: int,
    rng: np.random.Generator,
    min_px: int = 4,
) -> list[tuple[int, int]]:
    h, w = field.shape
    f = np.clip(field.astype(float), 0, None).copy()
    out: list[tuple[int, int]] = []
    yy, xx = np.indices(field.shape)
    for _ in range(count):
        flat = f.ravel()
        s = flat.sum()
        if s <= 0:
            break
        idx = int(rng.choice(flat.size, p=flat / s))
        y, x = divmod(idx, w)
        out.append((y, x))
        dy = np.abs(yy - y)
        dx = np.minimum(np.abs(xx - x), w - np.abs(xx - x))
        f[(dy ** 2 + dx ** 2) <= min_px ** 2] = 0
    return out




def _land_components_wrap(land: np.ndarray) -> np.ndarray:
    """8-connected land components with explicit longitude-seam merging."""
    labels, n = ndimage.label(land, structure=np.ones((3, 3), dtype=np.int8))
    if n <= 1:
        return labels.astype(np.int32)
    parent = np.arange(n + 1, dtype=np.int32)
    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = int(parent[a])
        return a
    def union(a: int, b: int) -> None:
        if a == 0 or b == 0: return
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    h, w = land.shape
    for y in range(h):
        for yy in (max(0, y-1), y, min(h-1, y+1)):
            if land[y, 0] and land[yy, w-1]: union(int(labels[y,0]), int(labels[yy,w-1]))
    roots = np.array([find(i) for i in range(n+1)], dtype=np.int32)
    rel = roots[labels]
    vals = sorted(v for v in np.unique(rel) if v != 0)
    mp = {v:i+1 for i,v in enumerate(vals)}
    out = np.zeros_like(rel, dtype=np.int32)
    for old,new in mp.items(): out[rel==old] = new
    return out


def _causal_settlement_sites(
    grid: SphereGrid, suit: np.ndarray, terrain: TerrainResult, hydro: HydrologyResult,
    portal_yx: tuple[int,int], count: int, rng: np.random.Generator, cfg: SocietyConfig,
) -> tuple[list[tuple[int,int]], np.ndarray, np.ndarray]:
    """Chronological frontier expansion using a precomputed candidate travel matrix.

    Pairwise great-circle geometry is evaluated once. During expansion we only update each
    candidate's best known land and maritime connection to the newly selected frontier node,
    reducing the old O(N_selected*N_candidates) Python routing loop to vectorized O(N²) setup
    plus O(N) updates per founded site.
    """
    count = max(1, count)
    candidate_count = max(count * 4, count + 24)
    candidates = [portal_yx] + _weighted_pick_separated(suit, candidate_count, rng, min_px=3)
    seen=set(); candidates=[p for p in candidates if not (p in seen or seen.add(p))]
    if len(candidates) <= count:
        selected = candidates
        comp = _land_components_wrap(terrain.land)
        effort=np.arange(len(selected),dtype=float)
        lm=np.array([int(comp[y,x]) for y,x in selected],dtype=np.int32)
        return selected,effort,lm

    ys=np.array([p[0] for p in candidates],dtype=np.int32)
    xs=np.array([p[1] for p in candidates],dtype=np.int32)
    n=len(candidates)
    comp_map=_land_components_wrap(terrain.land)
    comps=comp_map[ys,xs]
    river_dist=distance_to(hydro.rivers | hydro.lakes, grid)
    coast_dist=distance_to(terrain.ocean, grid)
    rugged=terrain.mountain_strength[ys,xs].astype(float)
    riverish=river_dist[ys,xs] < 100.0
    coastal=coast_dist[ys,xs] < 140.0

    vec=grid.xyz[ys,xs].astype(float)
    dots=np.clip(vec @ vec.T,-1.0,1.0)
    dist=grid.radius_km*np.arccos(dots)
    same=(comps[:,None]!=0) & (comps[:,None]==comps[None,:])
    avg_rug=0.5*(rugged[:,None]+rugged[None,:])
    land_cost=dist*(1.0+1.20*avg_rug)
    river_factor = float(np.clip(1.0 - cfg.river_navigation_bonus, 0.70, 1.0))
    land_cost*=np.where(riverish[:,None] | riverish[None,:], river_factor, 1.0)
    land_cost=np.where(same,land_cost,np.inf)
    coastal_factor = float(np.clip(1.0 - cfg.coastal_trade_bonus, 0.72, 1.0))
    cross_base=np.where((~same) & coastal[:,None] & coastal[None,:],dist * coastal_factor,np.inf)

    selected_idx=[0]
    remaining=np.ones(n,bool); remaining[0]=False
    best_land=land_cost[:,0].copy()
    best_cross=cross_base[:,0].copy()
    step_cost=[0.0]

    while remaining.any() and len(selected_idx)<count:
        progress=(len(selected_idx)-1)/max(count-1,1)
        maritime=float(np.clip(0.06+0.82*progress,0,0.90))
        reach_scale=520.0+1950.0*progress
        cross_factor=(4.35-3.25*maritime) if maritime>=0.22 else np.inf
        cost=np.minimum(best_land,best_cross*cross_factor)
        ridx=np.flatnonzero(remaining)
        c=cost[ridx]
        reach=np.exp(-np.minimum(c,12000.0)/max(reach_scale,1.0))
        ss=np.maximum(suit[ys[ridx],xs[ridx]].astype(float),1e-8)**1.35
        score=ss*reach
        if not np.any(score>0) or score.sum()<=0:
            # If the current frontier is temporarily trapped, prefer high-quality sites nearest
            # to the connected land network instead of teleporting to an arbitrary global maximum.
            fallback=np.where(np.isfinite(best_land[ridx]),best_land[ridx],np.where(np.isfinite(best_cross[ridx]),best_cross[ridx]*4.0,15000.0))
            score=ss*np.exp(-np.minimum(fallback,15000.0)/2200.0)
        score=np.power(score/max(float(score.max()),1e-12),1.55)
        score/=score.sum()
        local=int(rng.choice(len(ridx),p=score)); j=int(ridx[local])
        chosen_cost=float(cost[j]) if np.isfinite(cost[j]) else float(min(best_land[j],best_cross[j]*4.0))
        if not np.isfinite(chosen_cost): chosen_cost=5000.0
        selected_idx.append(j); remaining[j]=False; step_cost.append(chosen_cost)
        best_land=np.minimum(best_land,land_cost[:,j])
        best_cross=np.minimum(best_cross,cross_base[:,j])

    selected=[candidates[i] for i in selected_idx]
    cumulative=np.cumsum(np.sqrt(np.maximum(step_cost,0.0)))
    landmass=comps[np.asarray(selected_idx,dtype=int)].astype(np.int32)
    return selected,cumulative,landmass

def build_society(
    grid: SphereGrid,
    terrain: TerrainResult,
    climate: ClimateResult,
    hydro: HydrologyResult,
    resources: ResourceResult,
    weather: WeatherResult,
    cfg: SocietyConfig,
    rng: np.random.Generator,
    appearance: SurfaceAppearanceResult | None = None,
) -> SocietyResult:
    if not cfg.enabled:
        empty = np.zeros_like(terrain.elevation_km, dtype=np.float32)
        return SocietyResult(empty, None, [], [], [], [], {"enabled": False})

    land = terrain.land
    # Settlement desirability: freshwater and productive moderate climates dominate;
    # coasts and metal/fuel access matter, extreme hazards/elevation penalize.
    river_dist = distance_to(hydro.rivers | hydro.lakes, grid)
    coast_dist = distance_to(terrain.ocean, grid)
    fresh = np.exp(-river_dist / 160.0)
    coast = np.exp(-coast_dist / 500.0)
    t = climate.annual_temperature_c
    thermal = np.exp(-0.5 * ((t - 15.0) / 13.0) ** 2)
    precip = climate.annual_precipitation_mm
    water = np.exp(-0.5 * ((np.log1p(precip) - np.log1p(950.0)) / 1.1) ** 2)
    elevation_penalty = np.exp(-np.maximum(terrain.elevation_km, 0) / 3.0)
    metal = normalize01(
        resources.suitability["copper_rich"] + resources.suitability["bog_iron"] +
        resources.suitability["oolitic_iron"] + resources.suitability["tin_belt"]
    )
    fuel = np.maximum(resources.wood_potential, resources.suitability["coal"])
    hazard = normalize01(weather.hurricane_genesis + weather.tornado_potential + weather.blizzard)
    if appearance is not None:
        productivity = normalize01(0.52 * appearance.vegetation_fraction + 0.30 * appearance.soil_moisture_index +
                                   0.18 * (1.0 - appearance.bare_ground_fraction)) * land
    else:
        productivity = normalize01(thermal * water) * land
    suit = land * thermal * water * elevation_penalty
    river_bonus = float(np.clip(cfg.river_navigation_bonus, 0.0, 0.35))
    coast_bonus = float(np.clip(cfg.coastal_trade_bonus, 0.0, 0.25))
    suit *= (0.48 + (0.20 + river_bonus) * fresh + (0.06 + coast_bonus) * coast + 0.08 * metal + 0.08 * fuel)
    suit *= (1.0 - 0.45 * hazard)
    pw=float(np.clip(cfg.agricultural_productivity_weight,0.0,0.55))
    suit *= (1.0 - pw) + pw * (0.30 + 0.70 * productivity)
    suit = normalize01(suit) * land

    # Portal: source narrative favors an isolated mountain/temperate region and intentionally
    # leaves exact human population unspecified. We therefore choose location, not a magic fixed count.
    portal_score = suit.copy()
    if cfg.portal_prefer_mountains:
        portal_score *= (0.25 + 0.75 * terrain.mountain_strength)
    portal_score *= (np.abs(grid.lat) <= cfg.portal_latitude_max_deg)
    if portal_score.max() <= 0:
        portal_score = suit.copy()
    pidx = int(np.argmax(portal_score))
    py, px = divmod(pidx, grid.width)
    portal = {
        "latitude": float(grid.lat[py, px]), "longitude": float(grid.lon[py, px]),
        "elevation_km": float(terrain.elevation_km[py, px]),
        "canon": "portal exists; arrival mechanism intentionally configurable/head-canon",
    }

    # Settlements expand from an existing frontier. New sites are chosen by suitability *and* travel
    # reachability from previously founded sites; maritime crossings unlock gradually.
    pts, expansion_effort, landmass_ids = _causal_settlement_sites(
        grid, suit, terrain, hydro, (py, px), cfg.settlement_count, rng, cfg)
    maxeff = max(float(expansion_effort[-1]) if len(expansion_effort) else 0.0, 1.0)
    founding = -cfg.history_years + (expansion_effort / maxeff) * (cfg.history_years * 0.90)
    # Small noise is constrained so chronology remains causal/monotonic.
    founding += rng.normal(0, cfg.history_years * 0.008, len(pts))
    founding = np.maximum.accumulate(founding)
    founding[0] = -cfg.history_years
    founding = np.minimum(founding, -20)
    effective_d = expansion_effort

    # Three broad cultural waves/ages emerge from chronological expansion centers.
    centers_idx = [0]
    if len(pts) > 8:
        centers_idx.append(len(pts) // 3)
        centers_idx.append(2 * len(pts) // 3)
    centers = [pts[i] for i in centers_idx]
    cultures = []
    for cid, (y, x) in enumerate(centers):
        cultures.append({
            "culture_id": cid,
            "name": f"Culture-{cid+1}",
            "wave": cid + 1,
            "origin_latitude": float(grid.lat[y, x]),
            "origin_longitude": float(grid.lon[y, x]),
            "traits": {
                "portal_tradition": float(np.clip(1.0 - cid * 0.28 + rng.normal(0, .08), 0, 1)),
                "maritime_orientation": float(np.clip(rng.beta(2, 2), 0, 1)),
                "foodcraft_specialization": float(np.clip(0.65 + rng.normal(0, .12), 0, 1)),
                "centralization": float(rng.beta(2, 2)),
            },
        })

    settlements = []
    for sid, ((y, x), fy, d) in enumerate(zip(pts, founding, effective_d)):
        cd = []
        for cy, cx in centers:
            cd.append(grid.great_circle_km(float(grid.lat[y, x]), float(grid.lon[y, x]),
                                           float(grid.lat[cy, cx]), float(grid.lon[cy, cx])))
        cid = int(np.argmin(cd))
        local_s = float(suit[y, x])
        age = max(0.0, -float(fy))
        # Relative population, not asserted literal canon population count: intentionally marked synthetic.
        prod = float(productivity[y, x])
        carrying = (800 + 24000 * local_s ** 1.8) * (0.72 + 0.62 * prod)
        pop = carrying / (1 + 8 * math.exp(-age / 450.0))
        pop *= float(rng.lognormal(0, .28))
        tech = "copper"
        if resources.technology_access["bronze"][y, x]: tech = "bronze"
        if resources.technology_access["iron"][y, x]: tech = "iron"
        settlements.append({
            "settlement_id": sid, "culture_id": cid,
            "latitude": float(grid.lat[y, x]), "longitude": float(grid.lon[y, x]),
            "founded_year_relative_present": int(round(fy)),
            "synthetic_population_index": int(max(10, round(pop))),
            "suitability": local_s, "agricultural_productivity_index": prod, "technology_resource_ceiling": tech,
            "river_access": bool(river_dist[y, x] < 120), "coastal": bool(coast_dist[y, x] < 100),
            "landmass_id": int(landmass_ids[sid]),
        })

    # Trade/contact graph by great-circle proximity, technology complementarity and culture mixing.
    links: list[dict] = []
    if len(settlements) > 1:
        for i in range(len(settlements)):
            ds = []
            for j in range(len(settlements)):
                if i == j: continue
                d = grid.great_circle_km(settlements[i]["latitude"], settlements[i]["longitude"],
                                         settlements[j]["latitude"], settlements[j]["longitude"])
                ds.append((d, j))
            for d, j in sorted(ds)[:3]:
                same_land = settlements[i]["landmass_id"] == settlements[j]["landmass_id"]
                maritime_ok = settlements[i]["coastal"] and settlements[j]["coastal"] and d < 950
                if i < j and d < 1800 and (same_land or maritime_ok):
                    if same_land:
                        river_link = settlements[i]["river_access"] and settlements[j]["river_access"]
                        penalty = 1.0 + (cfg.river_navigation_bonus if river_link else 0.0)
                    else:
                        penalty = 0.55 + cfg.coastal_trade_bonus
                    links.append({"a": i, "b": j, "distance_km": round(d, 1),
                                  "type": "trade/contact", "strength": round(float(np.clip(penalty*np.exp(-d / 750), 0, 1)), 3),
                                  "maritime": bool(not same_land)})

    events: list[dict] = [
        {"year": -cfg.history_years, "type": "arrival", "description": "First-age humans arrive at the portal; initial population intentionally underspecified."},
        {"year": int(-cfg.history_years * .82), "type": "adaptation", "description": "Foodcraft knowledge becomes a stable subsistence dependency."},
    ]
    if len(cultures) > 1:
        events.append({"year": int(-cfg.history_years * .58), "type": "cultural_branch", "description": "Second-age regional traditions diverge from the portal community."})
    if len(cultures) > 2:
        events.append({"year": int(-cfg.history_years * .30), "type": "cultural_branch", "description": "Third-age expansion produces wider trade, pilgrimage and mixed traditions."})
    # Seeded optional social shocks make the process generative but reproducible.
    for year in range(-cfg.history_years + cfg.history_step_years, 0, cfg.history_step_years):
        if rng.random() < 0.018:
            events.append({"year": year, "type": str(rng.choice(["migration", "trade_shift", "religious_revival", "regional_conflict"])),
                           "description": "Seeded historical perturbation generated from settlement/contact structure."})
    events.sort(key=lambda x: x["year"])

    meta = {
        "enabled": True, "settlement_count": len(settlements), "culture_count": len(cultures),
        "population_note": "Population values are synthetic indices because the source intentionally leaves founding numbers underspecified.",
        "expansion_model": "sequential reachability-weighted frontier expansion with terrain/river/landmass costs and gradual maritime unlock",
        "settled_landmass_count": int(len(set(int(x) for x in landmass_ids if x != 0))),
        "agricultural_productivity_coupled": bool(appearance is not None),
        "river_navigation_bonus": float(cfg.river_navigation_bonus),
        "coastal_trade_bonus": float(cfg.coastal_trade_bonus),
    }
    return SocietyResult(suit.astype(np.float32), portal, settlements, cultures, links, events, meta)
