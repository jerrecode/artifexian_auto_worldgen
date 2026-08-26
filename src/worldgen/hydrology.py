from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import numpy as np
from scipy import ndimage

from .config import HydrologyConfig, NoiseConfig
from .grid import SphereGrid, normalize01, smooth_periodic
from .terrain import TerrainResult
from .ocean import OceanResult
from .climate import ClimateResult
from .geology import GeologyResult
from .tectonics import TectonicResult
from .noise import hybrid_multifractal, hybrid_noise01, noise_kwargs, HYDRO_BLEND, NoiseBlend, StaticNoiseFields


@dataclass(slots=True)
class SurfaceEvolutionResult:
    elevation_km: np.ndarray
    cumulative_erosion_m: np.ndarray
    cumulative_deposition_m: np.ndarray
    sediment_flux_index: np.ndarray
    delta_deposition_m: np.ndarray
    tectonic_uplift_m: np.ndarray
    meander_migration_m: np.ndarray
    meander_potential: np.ndarray
    metadata: dict


@dataclass(slots=True)
class HydrologyResult:
    filled_elevation_km: np.ndarray
    flow_to: np.ndarray
    accumulation: np.ndarray
    drainage_area_km2: np.ndarray
    discharge_index: np.ndarray
    rivers: np.ndarray
    stream_order: np.ndarray
    river_width_proxy: np.ndarray
    lakes: np.ndarray
    basin_id: np.ndarray
    runoff: np.ndarray
    cumulative_erosion_m: np.ndarray
    cumulative_deposition_m: np.ndarray
    sediment_flux_index: np.ndarray
    delta_deposition_m: np.ndarray
    tectonic_uplift_m: np.ndarray
    meander_migration_m: np.ndarray
    meander_potential: np.ndarray
    sinuosity_proxy: np.ndarray
    river_centerlines: list[dict]
    metadata: dict


_FLOOD_NEIGHBORS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
# Extended D16/D24-style stencil reduces raster-angle artifacts in long channels.
_FLOW_NEIGHBORS = _FLOOD_NEIGHBORS + [
    (-2,-1),(-2,1),(2,-1),(2,1),(-1,-2),(-1,2),(1,-2),(1,2),
    (-2,0),(2,0),(0,-2),(0,2),
]
# Relative erodibility; calibrated as morphology factors, not literal SI K values.
_LITH_ERODIBILITY = np.array([1.75, 1.20, 0.82, 0.46, 0.36, 0.55, 0.52, 0.58, 0.40], dtype=float)
_LITH_RUNOFF = np.array([0.90, 0.88, 0.68, 1.08, 1.12, 0.96, 1.00, 0.96, 1.05], dtype=float)


def _cell_area_km2(grid: SphereGrid) -> np.ndarray:
    return grid.cell_area_weights * (4.0 * math.pi * grid.radius_km ** 2)


def _priority_flood(elev: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    """Priority-Flood only across land, seeded from coastal land cells.

    This avoids putting every ocean pixel into a Python heap and is much faster than the original
    implementation while producing the same required monotonic drainage surface over continents.
    """
    h,w=elev.shape
    z=elev.astype(np.float64).copy()
    visited=ocean.astype(bool).copy()
    heap:list[tuple[float,int,int]]=[]
    land=~ocean
    coastal=land & ndimage.binary_dilation(ocean, iterations=1)
    ys,xs=np.where(coastal)
    for y,x in zip(ys.tolist(), xs.tolist()):
        visited[y,x]=True; heapq.heappush(heap,(float(z[y,x]),y,x))
    if not heap:
        for x in range(w):
            for y in (0,h-1):
                if land[y,x] and not visited[y,x]:
                    visited[y,x]=True; heapq.heappush(heap,(float(z[y,x]),y,x))
    eps=1e-7
    while heap:
        cur,y,x=heapq.heappop(heap)
        for dy,dx in _FLOOD_NEIGHBORS:
            ny=y+dy; nx=(x+dx)%w
            if ny<0:
                ny=0; nx=(nx+w//2)%w
            elif ny>=h:
                ny=h-1; nx=(nx+w//2)%w
            if visited[ny,nx]: continue
            visited[ny,nx]=True
            nz=float(z[ny,nx])
            if nz<=cur:
                nz=cur+eps; z[ny,nx]=nz
            heapq.heappush(heap,(nz,ny,nx))
    return z


def _neighbor_geometry(shape: tuple[int,int], dy: int, dx: int) -> tuple[np.ndarray,np.ndarray]:
    h,w=shape
    yy,xx=np.indices(shape)
    ny=yy+dy; nx=xx+dx
    # Reflect across a pole and rotate longitude by 180°. Offsets are <=2 so one reflection suffices.
    north=ny<0; south=ny>=h
    ny=np.where(north,-ny-1,ny)
    ny=np.where(south,2*h-ny-1,ny)
    nx=np.where(north|south,nx+w//2,nx)%w
    ny=np.clip(ny,0,h-1)
    return ny.astype(np.int32),nx.astype(np.int32)


def _neighbor_values(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    ny,nx=_neighbor_geometry(a.shape,dy,dx)
    return a[ny,nx]


def _neighbor_indices(shape:tuple[int,int],dy:int,dx:int)->np.ndarray:
    h,w=shape; ny,nx=_neighbor_geometry(shape,dy,dx)
    return (ny*w+nx).astype(np.int32)


def _flow_directions(z: np.ndarray, ocean: np.ndarray, grid: SphereGrid) -> np.ndarray:
    """Vectorized spherical-ish D8 steepest-descent routing with pole reflection."""
    h,w=z.shape
    best=np.zeros((h,w),float)
    receiver=np.full((h,w),-1,np.int32)
    for dy,dx in _FLOW_NEIGHBORS:
        nb=_neighbor_values(z,dy,dx)
        if dx and dy:
            dist=np.hypot(abs(dy)*grid.dy_km, abs(dx)*grid.dx_km)
        elif dx:
            dist=abs(dx)*grid.dx_km
        else:
            dist=np.full_like(grid.dx_km,abs(dy)*grid.dy_km)
        slope=(z-nb)/np.maximum(dist,1e-6)
        better=slope>best
        receiver[better]=_neighbor_indices((h,w),dy,dx)[better]
        best[better]=slope[better]
    receiver[ocean]=-1
    return receiver.ravel()


def _receiver_slope(z: np.ndarray, flow: np.ndarray, grid: SphereGrid) -> np.ndarray:
    h,w=z.shape
    flat=z.ravel(); idx=np.arange(flat.size); j=flow
    good=j>=0
    out=np.zeros(flat.size,float)
    yi,xi=np.divmod(idx[good],w); yj,xj=np.divmod(j[good],w)
    dlat=np.abs(yj-yi)
    rawdx=np.abs(xj-xi); dlon=np.minimum(rawdx,w-rawdx)
    # Pole reflection can produce a half-world x jump but is physically one north/south step.
    pole=(dlat==0)&(dlon>w//4)&((yi==0)|(yi==h-1))
    dlon=np.where(pole,0,dlon)
    dist=np.hypot(dlat*grid.dy_km, dlon*grid.dx_km[yi,xi])
    out[good]=np.maximum(flat[good]-flat[j[good]],0)/np.maximum(dist,1e-6)
    return out.reshape(h,w)


def _accumulate(z: np.ndarray, flow: np.ndarray, source: np.ndarray) -> np.ndarray:
    acc=source.astype(np.float64).ravel().copy()
    order=np.argsort(z.ravel())[::-1]
    for idx in order:
        j=int(flow[idx])
        if j>=0: acc[j]+=acc[idx]
    return acc.reshape(z.shape)


def _basins(flow: np.ndarray, ocean: np.ndarray, shape: tuple[int,int]) -> np.ndarray:
    h,w=shape; n=h*w
    root=np.full(n,-2,np.int32)
    # Ocean connected components need longitude seam support. Label triplicated ocean and use middle IDs.
    trip=np.concatenate([ocean,ocean,ocean],axis=1)
    labs,_=ndimage.label(trip)
    ol=labs[:,w:2*w].ravel().astype(np.int32)
    of=ocean.ravel(); root[of]=ol[of]
    for i in range(n):
        if root[i]!=-2: continue
        path=[]; cur=i; seen=set()
        while cur>=0 and root[cur]==-2 and cur not in seen:
            seen.add(cur); path.append(cur); cur=int(flow[cur])
        rid=0 if cur<0 else (int(root[cur]) if root[cur]!=-2 else 0)
        for p in path: root[p]=rid
    return root.reshape(shape)


def _runoff_mm(climate: ClimateResult, land: np.ndarray, geology: GeologyResult | None, cfg: HydrologyConfig) -> np.ndarray:
    p=climate.annual_precipitation_mm.astype(float)
    frac=cfg.runoff_base_fraction+0.46*(1.0-np.exp(-p/1050.0))
    if geology is not None:
        frac*= _LITH_RUNOFF[np.clip(geology.rock_code,0,len(_LITH_RUNOFF)-1)]
    # Very warm/dry surfaces lose more to evapotranspiration; snow supplies delayed runoff.
    evap_penalty=np.clip((climate.annual_temperature_c-8.0)/38.0,0,0.28)
    frac=np.clip(frac-evap_penalty+0.16*climate.snow_fraction,0.05,0.92)
    return (p*frac*land).astype(np.float64)


def _transport_sediment(
    routing_z: np.ndarray, flow: np.ndarray, erosion_m: np.ndarray,
    discharge_norm: np.ndarray, slope: np.ndarray, cell_area: np.ndarray,
    land: np.ndarray, cfg: HydrologyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Route sediment, deposit on low-gradient reaches, and retain export at river mouths."""
    n=routing_z.size
    area=cell_area.ravel(); lf=land.ravel(); recv=flow.astype(np.int64,copy=False)
    q=discharge_norm.ravel(); s=slope.ravel()
    flatness=np.exp(-s/0.0028)
    depfrac=np.clip(cfg.deposition_strength*flatness*(0.70+0.30*(1.0-q)),0.0,0.80)
    depfrac*=lf
    load=(erosion_m*cell_area).ravel().astype(float)
    deposited=np.zeros(n,float)
    transit=np.zeros(n,float)
    exported=np.zeros(n,float)
    valid_source=(recv>=0)&lf
    safe_recv=np.where(recv>=0,recv,0)
    target_land=np.zeros(n,bool)
    target_land[valid_source]=lf[safe_recv[valid_source]]
    inland=valid_source & target_land
    outlet=valid_source & ~target_land
    # Only concentrated, channelized mouth discharge constructs a delta. Low-discharge coastal
    # denudation remains diffuse marine sediment and must not paint every coastline as a delta.
    delta_outlet=outlet & (q>=float(np.clip(cfg.delta_min_outlet_discharge_norm, 0.0, 1.0)))
    passes=max(1,int(cfg.sediment_routing_passes))
    for _ in range(passes):
        if float(load.sum()) < 1e-9: break
        transit += load
        dvol=load*depfrac
        deposited+=dvol
        remain=load-dvol
        # Material that reaches an ocean receiver becomes delta/coastal sediment instead of vanishing.
        if np.any(delta_outlet):
            exported += np.bincount(safe_recv[delta_outlet],weights=remain[delta_outlet]*np.clip(q[delta_outlet],0.0,1.0),minlength=n)
        if np.any(inland):
            load=np.bincount(safe_recv[inland],weights=remain[inland],minlength=n).astype(float,copy=False)
        else:
            load=np.zeros(n,float)
    dep_depth=deposited/np.maximum(area,1e-9)
    dep_depth=np.minimum(dep_depth,16.0)*lf
    return (dep_depth.reshape(routing_z.shape),
            normalize01(np.log1p(transit.reshape(routing_z.shape))),
            exported.reshape(routing_z.shape))


def _delta_deposition(
    grid: SphereGrid, z: np.ndarray, land: np.ndarray, exported_volume: np.ndarray,
    cell_area: np.ndarray, cfg: HydrologyConfig, marine_energy: np.ndarray | None = None,
    distributary_texture: np.ndarray | None = None,
) -> np.ndarray:
    """Spread concentrated river-mouth sediment across shallow shelves and permit progradation.

    Deposition is favored on low-gradient shelves and reduced where marine energy is high. A coherent
    distributary texture breaks circular blobs into lobes while preserving mass approximately.
    """
    total=float(np.sum(exported_volume))*float(np.clip(cfg.delta_retention_fraction,0.0,1.0))
    if total<=1e-12:
        return np.zeros_like(z)
    ocean=~land
    shallow=ocean & (z<0.0) & (z>=-cfg.delta_max_depth_m/1000.0)
    if not np.any(shallow):
        return np.zeros_like(z)
    sigma=max(0.55,float(cfg.delta_spread_cells))
    spread=smooth_periodic(exported_volume.astype(float),(sigma,sigma*1.45))
    # Along-coast redistribution approximates tidal/wave reworking without pretending to solve tides.
    if cfg.delta_tide_reworking_strength > 0:
        along=smooth_periodic(exported_volume.astype(float),(max(0.5,sigma*0.65),max(0.8,sigma*2.2)))
        spread=(1.0-float(cfg.delta_tide_reworking_strength))*spread+float(cfg.delta_tide_reworking_strength)*along
    shallow_pref=np.exp(np.clip(z, -cfg.delta_max_depth_m/1000.0, 0.0)/(max(cfg.delta_max_depth_m,1.0)/1000.0))
    coast_pref=0.45+0.55*ndimage.binary_dilation(land,iterations=max(1,int(round(sigma)))).astype(float)
    gy,gx=np.gradient(z.astype(float))
    shelf_slope=np.hypot(gx,gy)
    slope_pref=np.exp(-shelf_slope/max(float(cfg.delta_shelf_slope_scale),1e-5))
    if marine_energy is None:
        marine_pref=1.0
    else:
        me=np.clip(np.asarray(marine_energy,float),0.0,1.5)
        marine_pref=np.clip(1.0-float(cfg.delta_wave_reworking_strength)*me,0.18,1.0)
    texture_pref=1.0
    if distributary_texture is not None:
        tex=np.clip(np.asarray(distributary_texture,float),0.0,1.0)
        st=float(np.clip(cfg.delta_distributary_texture_strength,0.0,0.75))
        texture_pref=(1.0-st)+2.0*st*tex
    weights=spread*shallow*shallow_pref*coast_pref*slope_pref*marine_pref*texture_pref
    ws=float(weights.sum())
    if ws<=1e-12:
        return np.zeros_like(z)
    volume=weights*(total/ws)
    depth=volume/np.maximum(cell_area,1e-9)
    return np.clip(depth,0.0,float(cfg.delta_max_aggradation_m_per_iteration))*shallow

def _meander_field(
    geology: GeologyResult, lith: np.ndarray, slope: np.ndarray, qn: np.ndarray,
    land: np.ndarray, cfg: HydrologyConfig,
) -> np.ndarray:
    # Soft alluvium/sandstone/carbonate encourages lateral migration; resistant crystalline rock confines it.
    soft=np.clip((lith-0.32)/1.45,0.0,1.0)
    alluvial=np.isin(geology.rock_code,[0,1,2]).astype(float)
    lowgrad=np.exp(-slope/max(float(cfg.meander_slope_scale),1e-5))
    q=np.clip(qn/1.8,0.0,1.0)
    m=(q**0.62)*lowgrad*(0.35+0.65*soft)*(0.72+0.28*alluvial)*land
    return np.clip(cfg.river_meander_strength*m,0.0,1.0)


def evolve_surface(
    grid: SphereGrid,
    terrain: TerrainResult,
    ocean: OceanResult,
    climate: ClimateResult,
    geology: GeologyResult,
    cfg: HydrologyConfig,
    tectonics: TectonicResult | None = None,
    rng: np.random.Generator | None = None,
    noise_cfg: NoiseConfig | None = None,
    static_noise: StaticNoiseFields | None = None,
) -> SurfaceEvolutionResult:
    z=ocean.elevation_km.astype(np.float64).copy()
    erosion_total=np.zeros_like(z); dep_total=np.zeros_like(z); flux=np.zeros_like(z)
    delta_total=np.zeros_like(z); uplift_total=np.zeros_like(z); migration_total=np.zeros_like(z)
    meander=np.zeros_like(z)
    cell_area=_cell_area_km2(grid)
    lith=_LITH_ERODIBILITY[np.clip(geology.rock_code,0,len(_LITH_ERODIBILITY)-1)]
    if rng is None:
        # Deterministic geometry-only fallback used by external callers/tests.
        wiggle=np.sin(np.deg2rad(grid.lon*3.7+grid.lat*1.9))+0.45*np.sin(np.deg2rad(grid.lon*11.3-grid.lat*4.1))
        delta_texture=0.5+0.5*np.sin(np.deg2rad(grid.lon*7.1-grid.lat*3.3))
    else:
        if static_noise is not None:
            wiggle=static_noise.hydro_wiggle; delta_texture=static_noise.delta_texture
        else:
            wiggle=hybrid_multifractal(
                z.shape,rng,base_scale_px=max(grid.height/36.0,2.5),
                **noise_kwargs(noise_cfg,profile=HYDRO_BLEND,octaves=max(5,min(8,getattr(noise_cfg,"octaves",7)))),
            )
            delta_texture=hybrid_noise01(
                z.shape,rng,base_scale_px=max(grid.height/50.0,2.0),
                **noise_kwargs(noise_cfg,profile=NoiseBlend(0.36,0.25,0.12,0.27),octaves=max(4,min(7,getattr(noise_cfg,"octaves",6)))),
            )
    wiggle=(wiggle-np.mean(wiggle))/max(np.std(wiggle),1e-8)
    meander_prior=np.clip(lith/np.max(_LITH_ERODIBILITY),0,1)*terrain.lowland_strength
    last_flow=None; last_route=None

    for it in range(max(0,cfg.surface_evolution_iterations)):
        land=z>0; oc=~land
        if it%max(1,cfg.flow_refresh_interval)==0 or last_flow is None:
            # Metre-scale floodplain microrelief matters only where rock/gradient permit channel migration;
            # mountains remain controlled by topographic slope rather than noise.
            micro=(cfg.meander_microrelief_m/1000.0)*wiggle*np.maximum(meander_prior,meander)
            route=_priority_flood(z+micro*land,oc)
            flow=_flow_directions(route,oc,grid)
            last_route,last_flow=route,flow
        else:
            route,flow=last_route,last_flow
        runoff=_runoff_mm(climate,land,geology,cfg)
        water_source=(runoff/1000.0)*cell_area
        discharge=_accumulate(route,flow,water_source)
        slope=_receiver_slope(route,flow,grid)
        vals=discharge[land & (discharge>0)]
        qref=np.quantile(vals,0.985) if len(vals) else 1.0
        qn=np.clip(discharge/max(qref,1e-12),0,2.5)
        meander=_meander_field(geology,lith,slope,qn,land,cfg)

        # Vertical stream-power incision. Coastal cells should not all behave like giant river
        # knickpoints merely because their receiver is below sea level: only substantial mouths keep
        # the fluvial term at the coastline; other coasts receive only weak background weathering.
        recv=flow.astype(np.int64,copy=False)
        lf=land.ravel(); valid_recv=recv>=0; safe=np.where(valid_recv,recv,0)
        receiver_land=np.zeros(recv.size,bool)
        receiver_land[valid_recv]=lf[safe[valid_recv]]
        receiver_land=receiver_land.reshape(land.shape)
        major_mouth=land & (~receiver_land) & (qn>0.34)
        fluvial_domain=land & (receiver_land | major_mouth)
        sref=0.006
        stream_e=cfg.max_fluvial_erosion_m_per_iteration*lith*(qn**cfg.stream_power_m)*((slope/sref)**cfg.stream_power_n)
        stream_e=np.clip(stream_e,0,cfg.max_fluvial_erosion_m_per_iteration)*fluvial_domain
        weathering=0.20*lith*np.clip(climate.annual_precipitation_mm/1200.0,0,1.5)*land
        e=np.clip(stream_e+weathering,0,cfg.max_fluvial_erosion_m_per_iteration)

        # Lateral bank migration/widening is strongest on low-gradient, erodible floodplains. An asymmetric
        # correlated field prevents both banks from retreating identically and seeds evolving bends.
        channel=(meander>0.06)&(qn>0.08)&land
        banks=ndimage.binary_dilation(channel,iterations=1)&land&~channel
        lateral_source=ndimage.maximum_filter(e*meander,size=3)
        asym=0.35+0.65*normalize01(wiggle,robust=False)
        lateral=np.clip(cfg.lateral_erosion_fraction*lateral_source*asym*banks,0,cfg.max_fluvial_erosion_m_per_iteration*0.75)
        e_total=np.clip(e+lateral,0,cfg.max_fluvial_erosion_m_per_iteration*1.35)
        migration_total+=lateral

        dep,load,exported=_transport_sediment(route,flow,e_total,np.clip(qn/2.5,0,1),slope,cell_area,land,cfg)
        delta=_delta_deposition(
            grid,z,land,exported,cell_area,cfg,
            marine_energy=getattr(ocean,"current_speed",None),distributary_texture=delta_texture,
        )

        # Active tectonics continues to build ranges while rivers/hillslopes denude them. This lets relief
        # emerge from the competition between uplift and erosion instead of being a static initial texture.
        uplift=np.zeros_like(z)
        subsidence=np.zeros_like(z)
        if tectonics is not None:
            active=(0.78*tectonics.convergence_strength + 0.22*tectonics.stress_field)
            uplift=cfg.tectonic_uplift_m_per_iteration*(active**1.15)*land
            # Divergent continental crust experiences subsidence at rift axes while shoulders remain in terrain.
            subsidence=cfg.rift_subsidence_m_per_iteration*tectonics.divergence_strength*land
            uplift_total+=uplift

        sm=smooth_periodic(z,(0.65,0.75))
        diffusion=(sm-z)*cfg.hillslope_diffusion_strength*np.clip(lith,0.35,1.8)*land
        diffusion=np.clip(diffusion,-0.010,0.010)
        z += diffusion
        z += (uplift-subsidence)/1000.0
        z -= e_total/1000.0
        z += dep/1000.0
        z += delta/1000.0
        erosion_total+=e_total; dep_total+=dep+delta; delta_total+=delta; flux=np.maximum(flux,load)
        meander_prior=meander

    meta={
        'iterations': int(cfg.surface_evolution_iterations),
        'max_cumulative_erosion_m': float(erosion_total.max()),
        'max_cumulative_deposition_m': float(dep_total.max()),
        'max_delta_aggradation_m': float(delta_total.max()),
        'max_tectonic_uplift_m': float(uplift_total.max()),
        'max_meander_bank_migration_m': float(migration_total.max()),
        'mean_land_erosion_m': float(np.average(erosion_total[terrain.land],weights=grid.cell_area_weights[terrain.land])) if np.any(terrain.land) else 0.0,
        'model': 'rainfall/snowmelt runoff + lithology-dependent stream power + lateral meander migration + hierarchical sediment routing/shelf-controlled deltas + active tectonic uplift + hillslope diffusion',
        'noise_model': 'shared hybrid multi-type multifractal channel microrelief and distributary texture',
    }
    return SurfaceEvolutionResult(z.astype(np.float32),erosion_total.astype(np.float32),dep_total.astype(np.float32),
                                  normalize01(flux).astype(np.float32),delta_total.astype(np.float32),
                                  uplift_total.astype(np.float32),migration_total.astype(np.float32),
                                  meander.astype(np.float32),meta)

def _lake_mask(
    grid: SphereGrid, z: np.ndarray, filled: np.ndarray, land: np.ndarray,
    drainage_area: np.ndarray, runoff_acc: np.ndarray, climate: ClimateResult, cfg: HydrologyConfig,
) -> np.ndarray:
    depth_m=(filled-z)*1000.0
    # A depression must be deep enough, have a nontrivial catchment, and receive enough integrated
    # runoff to offset evaporation. This replaces the old "every filled depression is a lake" rule.
    pet=np.maximum(0.0,26.0*(climate.annual_temperature_c+5.0))
    moisture=climate.annual_precipitation_mm/np.maximum(pet,250.0)
    rv=runoff_acc[land & (runoff_acc>0)]
    rref=np.quantile(rv,0.60) if len(rv) else 1.0
    candidate=(land & (depth_m>=cfg.lake_min_depth_m) &
               (drainage_area>=cfg.lake_min_catchment_km2) &
               ((runoff_acc>=rref)|(moisture>0.85)))
    labs,n=ndimage.label(candidate)
    out=np.zeros_like(candidate)
    cell_area=_cell_area_km2(grid)
    components=[]
    for lab in range(1,n+1):
        m=labs==lab
        area=float(cell_area[m].sum())
        if area<max(20.0,0.08*cfg.lake_min_catchment_km2): continue
        if area>850000 and float(np.mean(moisture[m]))<1.15: continue
        score=float(np.max(depth_m[m]))*math.log1p(float(np.max(drainage_area[m])))*float(np.clip(np.mean(moisture[m]),0.15,3.0))
        components.append((score,lab,area))
    # Rank hydrologically convincing basins and softly constrain pathological global lake coverage.
    # This is a calibration safeguard, not a fixed number of lakes: individual basin size/count remain emergent.
    max_area=float(cell_area[land].sum())*float(np.clip(cfg.lake_area_soft_cap_fraction_land,0.001,0.20))
    used=0.0
    for _,lab,area in sorted(components,reverse=True):
        if used+area>max_area and used>0: continue
        out|=(labs==lab); used+=area
    return out



def _strahler_order(flow: np.ndarray, channel: np.ndarray) -> np.ndarray:
    """Compute Strahler stream order on a sparse channel graph in O(number of channel cells)."""
    cf=np.asarray(channel,bool).ravel(); recv=np.asarray(flow,np.int64).ravel(); n=cf.size
    inds=np.flatnonzero(cf)
    order=np.zeros(n,np.uint8)
    if not inds.size:
        return order.reshape(channel.shape)
    indeg=np.zeros(n,np.int16)
    valid=(recv[inds]>=0)
    src=inds[valid]; tgt=recv[src]
    good=cf[tgt]
    if np.any(good):
        np.add.at(indeg,tgt[good],1)
    maxin=np.zeros(n,np.uint8); countmax=np.zeros(n,np.uint8)
    queue=list(map(int,inds[indeg[inds]==0].tolist()))
    for q in queue:
        order[q]=1
    head=0
    while head<len(queue):
        cur=queue[head]; head+=1
        r=int(recv[cur])
        if r<0 or not cf[r]:
            continue
        oc=int(order[cur])
        if oc>int(maxin[r]):
            maxin[r]=oc; countmax[r]=1
        elif oc==int(maxin[r]):
            countmax[r]=min(255,int(countmax[r])+1)
        indeg[r]-=1
        if indeg[r]==0:
            base=max(1,int(maxin[r]))
            order[r]=min(255,base+(1 if int(countmax[r])>=2 else 0))
            queue.append(r)
    # Defensive fallback for any pathological unresolved loop (should not occur on filled terrain).
    order[cf & (order==0)]=1
    return order.reshape(channel.shape)


def _build_river_centerlines(
    grid: SphereGrid, flow: np.ndarray, rivers: np.ndarray, accumulation: np.ndarray,
    meander: np.ndarray, geology: GeologyResult | None, max_paths: int = 120,
) -> list[dict]:
    """Create sub-cell vector centerlines for major rivers.

    The global raster solves catchments/discharge. These vectors add a finer planform scale: low-gradient
    rivers in erodible rock receive larger lateral oscillations, while resistant/steep reaches remain close
    to the drainage thalweg. Terrain feedback still comes from the lateral-erosion field in evolve_surface.
    """
    h,w=rivers.shape; n=h*w
    rf=rivers.ravel(); recv=flow.astype(np.int64,copy=False)
    up=np.zeros(n,np.int16)
    src=np.flatnonzero(rf & (recv>=0))
    if src.size:
        tgt=recv[src]
        good=rf[tgt]
        np.add.at(up,tgt[good],1)
    heads=np.flatnonzero(rf & (up==0))
    if heads.size==0:
        return []
    af=accumulation.ravel(); mf=meander.ravel()
    # Prefer headwaters feeding larger trunks, but cap tracing work.
    heads=heads[np.argsort(af[heads])[::-1]]
    heads=heads[:min(len(heads),900)]
    visited=np.zeros(n,bool)
    raw=[]
    for head in heads:
        path=[]; cur=int(head); seen=set()
        while cur>=0 and cur not in seen and len(path)<2400:
            seen.add(cur); path.append(cur)
            nxt=int(recv[cur])
            if nxt<0: break
            cur=nxt
            if not rf[cur] and len(path)>2:
                path.append(cur); break
        if len(path)<6: continue
        overlap=float(np.mean(visited[np.asarray(path[:-1],dtype=int)]))
        if overlap>0.68: continue
        score=len(path)*(0.55+0.45*float(np.mean(mf[np.asarray(path[:-1],dtype=int)])))
        raw.append((score,path))
        visited[np.asarray(path[:-1],dtype=int)]=True
        if len(raw)>=max_paths*2: break
    raw.sort(key=lambda x:x[0],reverse=True)
    out=[]
    for rid,(_,path) in enumerate(raw[:max_paths]):
        idx=np.asarray(path,dtype=int)
        yy,xx=np.divmod(idx,w)
        lat=grid.lat_1d[yy].astype(float)
        lon=np.rad2deg(np.unwrap(np.deg2rad(grid.lon_1d[xx].astype(float))))
        mvals=mf[idx]
        # Densify the raster thalweg, then impose a smooth lateral oscillation whose amplitude is
        # controlled by the local meander field (itself lithology/slope/discharge dependent).
        k=max(2,int(round(4.0*512/max(w,1))))
        k=max(2,min(k,4))
        t=np.arange(len(path),dtype=float)
        td=np.linspace(0,len(path)-1,(len(path)-1)*k+1)
        lati=np.interp(td,t,lat); loni=np.interp(td,t,lon); mi=np.interp(td,t,mvals)
        if len(td)>=3:
            dlat=np.gradient(lati); dlon=np.gradient(loni)*np.cos(np.deg2rad(lati))
            norm=np.hypot(dlat,dlon)+1e-9
            # tangent in east/north coordinates; perpendicular is (-north,+east)
            east=dlon/norm; north=dlat/norm
            pe=-north; pn=east
            step_km=grid.dy_km/max(k,1)
            ss=np.arange(len(td))*step_km
            mean_m=float(np.mean(mi))
            wavelength=max(grid.dy_km*3.0, grid.dy_km*(5.4-1.6*mean_m))
            phase=(int(head)*0.61803398875)%(2*np.pi)
            amp=(0.020+0.19*mi)*grid.dy_km
            # Several incommensurate bend wavelengths avoid the visibly periodic sine-wave rivers of
            # the earlier version while remaining deterministic and smooth.
            wave=(0.58*np.sin(2*np.pi*ss/max(wavelength,1.0)+phase)
                  +0.27*np.sin(2*np.pi*ss/max(wavelength*1.73,1.0)+1.91*phase+0.7)
                  +0.15*np.sin(2*np.pi*ss/max(wavelength*0.63,1.0)+2.47*phase+1.8))
            taper=np.minimum(1.0,np.minimum(np.arange(len(td))/max(3,k*2),(len(td)-1-np.arange(len(td)))/max(3,k*2)))
            offset=amp*wave*np.clip(taper,0,1)
            lati=lati+pn*offset/111.2
            loni=loni+pe*offset/(111.2*np.maximum(np.cos(np.deg2rad(lati)),0.18))
        loni=((loni+180.0)%360.0)-180.0
        rock_code=None
        if geology is not None:
            vals=geology.rock_code.ravel()[idx[:-1] if len(idx)>1 else idx]
            if vals.size:
                rock_code=int(np.bincount(vals.astype(int)).argmax())
        points=[[float(a),float(b)] for a,b in zip(lati,loni)]
        out.append({
            'river_id':rid,'points_lat_lon':points,'source_cell':int(head),
            'mean_meander_potential':float(np.mean(mvals)),
            'sinuosity_proxy':float(1.0+2.35*np.mean(mvals)),
            'dominant_rock_code':rock_code,
        })
    return out

def build_hydrology(
    grid: SphereGrid,
    terrain: TerrainResult,
    ocean: OceanResult,
    climate: ClimateResult,
    cfg: HydrologyConfig,
    geology: GeologyResult | None = None,
    surface: SurfaceEvolutionResult | None = None,
) -> HydrologyResult:
    z0=ocean.elevation_km.astype(float)
    filled=_priority_flood(z0,terrain.ocean)
    flow=_flow_directions(filled,terrain.ocean,grid)
    runoff=_runoff_mm(climate,terrain.land,geology,cfg)
    cell_area=_cell_area_km2(grid)
    water_source=(runoff/1000.0)*cell_area
    acc=_accumulate(filled,flow,water_source)
    drainage=_accumulate(filled,flow,cell_area*terrain.land)
    slope=_receiver_slope(filled,flow,grid)

    source_valid=terrain.land & (climate.annual_precipitation_mm>=cfg.min_river_precip_mm_year) & (drainage>=cfg.min_drainage_area_km2)
    vals=acc[source_valid]
    qthr=np.quantile(vals,cfg.river_accumulation_quantile) if len(vals) else np.inf
    # Established trunks can cross locally dry reaches. A lower discharge threshold exposes their
    # tributary tree; Strahler order then captures actual network hierarchy instead of flat binary rivers.
    valid=terrain.land & (drainage>=cfg.min_drainage_area_km2)
    tributary_thr=float(np.clip(cfg.tributary_discharge_fraction,0.05,0.95))*qthr
    candidate=valid & (acc>=tributary_thr)
    stream_order=_strahler_order(flow,candidate)
    # Keep all higher-order channels plus stronger first-order tributaries; this prunes isolated ephemeral lines.
    rivers=candidate & ((stream_order>=2)|(acc>=0.52*qthr))
    if np.any(rivers):
        # Preserve connectivity immediately around higher-order trunk junctions.
        near=ndimage.binary_dilation(rivers & (stream_order>=2),iterations=1)
        rivers |= candidate & near
    stream_order=np.where(rivers,stream_order,0).astype(np.uint8)
    logq=normalize01(np.log1p(acc)).astype(np.float32)
    order_norm=stream_order.astype(float)/max(float(stream_order.max()),1.0)
    river_width=(logq*(0.62+0.38*order_norm)*rivers).astype(np.float32)

    lakes=_lake_mask(grid,z0,filled,terrain.land,drainage,acc,climate,cfg)
    basins=_basins(flow,terrain.ocean,z0.shape)
    discharge_norm=normalize01(np.log1p(acc)).astype(np.float32)
    lith = _LITH_ERODIBILITY[np.clip(geology.rock_code,0,len(_LITH_ERODIBILITY)-1)] if geology is not None else np.ones_like(z0)
    meander=_meander_field(geology,lith,slope,np.clip(acc/max(float(np.quantile(acc[terrain.land & (acc>0)],0.985)) if np.any(terrain.land & (acc>0)) else 1.0,1e-9),0,2.5),terrain.land,cfg) if geology is not None else np.zeros_like(z0)
    meander*=rivers
    sinuosity=(1.0+2.35*meander).astype(np.float32)
    if surface is None:
        zero=np.zeros_like(z0,dtype=np.float32); er=dep=sf=dd=up=mm=zero
    else:
        er=surface.cumulative_erosion_m; dep=surface.cumulative_deposition_m; sf=surface.sediment_flux_index
        dd=surface.delta_deposition_m; up=surface.tectonic_uplift_m; mm=surface.meander_migration_m
    centerlines=_build_river_centerlines(grid,flow,rivers,acc,meander,geology,max_paths=max(20,int(cfg.max_river_centerlines)))
    meta={
        'river_threshold_discharge_index': float(qthr) if np.isfinite(qthr) else None,
        'river_area_fraction_of_land': grid.weighted_fraction(rivers)/max(grid.weighted_fraction(terrain.land),1e-12),
        'lake_area_fraction_of_land': grid.weighted_fraction(lakes)/max(grid.weighted_fraction(terrain.land),1e-12),
        'mean_runoff_mm_year_land': float(np.average(runoff[terrain.land],weights=grid.cell_area_weights[terrain.land])) if np.any(terrain.land) else 0.0,
        'routing': 'vectorized extended-direction steepest descent (20 headings) with longitude wrap and pole reflection',
        'mean_major_river_sinuosity_proxy': float(np.mean(sinuosity[rivers])) if np.any(rivers) else 1.0,
        'delta_area_fraction': grid.weighted_fraction(dd>0.15),
        'river_centerline_count': len(centerlines),
        'max_strahler_order': int(stream_order.max()) if np.any(rivers) else 0,
        'tributary_discharge_fraction': float(cfg.tributary_discharge_fraction),
    }
    return HydrologyResult(filled.astype(np.float32),flow,acc.astype(np.float32),drainage.astype(np.float32),discharge_norm,
                           rivers,stream_order,river_width,lakes,basins,runoff.astype(np.float32),er,dep,sf,dd,up,mm,meander.astype(np.float32),sinuosity,centerlines,meta)
