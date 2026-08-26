from __future__ import annotations
import json, math, time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import ndimage

from worldgen.config import load_config
from worldgen.pipeline import WorldPipeline
from worldgen.render import _save_field

SEED = 731_928_461
ROOT = Path('/mnt/data')
OUT = ROOT / 'worldgen_v02_validation_seed_731928461'
BASE_NPZ = ROOT / 'worldgen_run' / 'world_arrays.npz'
BASE_JSON = ROOT / 'worldgen_run' / 'world.json'


def area_weights(lat1d: np.ndarray, width: int) -> np.ndarray:
    return np.cos(np.deg2rad(lat1d))[:, None] * np.ones((1, width))


def weighted_mean(x, w, mask=None):
    x=np.asarray(x,float); w=np.asarray(w,float)
    if mask is not None:
        m=np.asarray(mask,bool); x=x[m]; w=np.broadcast_to(w, mask.shape)[m]
    return float(np.sum(x*w)/max(np.sum(w),1e-30))


def weighted_quantile(x,w,q,mask=None):
    x=np.asarray(x,float); w=np.broadcast_to(np.asarray(w,float),x.shape)
    if mask is not None:
        m=np.asarray(mask,bool); x=x[m]; w=w[m]
    else:
        x=x.ravel(); w=w.ravel()
    o=np.argsort(x); x=x[o]; w=w[o]; c=np.cumsum(w); c/=max(c[-1],1e-30)
    return float(np.interp(q,c,x))


def neighbor_boundary(ids):
    ids=np.asarray(ids)
    b=(ids != np.roll(ids,1,1)) | (ids != np.roll(ids,-1,1))
    b |= ids != np.vstack((ids[:1],ids[:-1]))
    b |= ids != np.vstack((ids[1:],ids[-1:]))
    return b


def flow_heading_count(flow_to, river_mask):
    h,w=river_mask.shape; idx=np.arange(h*w,dtype=np.int64).reshape(h,w)
    src=idx[river_mask].ravel(); dst=np.asarray(flow_to).ravel()[src]
    valid=(dst>=0)&(dst!=src); src=src[valid]; dst=dst[valid]
    if len(src)==0:return 0,{}
    si,sj=np.divmod(src,w); di,dj=np.divmod(dst,w)
    dr=di-si; dc=dj-sj
    dc=np.where(dc>w//2,dc-w,np.where(dc<-w//2,dc+w,dc))
    c=Counter(zip(dr.tolist(),dc.tolist()))
    return len(c), {f'{a},{b}':n for (a,b),n in sorted(c.items())}


def land_components(land):
    # 8-connected then merge edge labels that touch across longitude.
    lab,n=ndimage.label(land,structure=np.ones((3,3),int))
    if n==0:return lab,0
    parent=np.arange(n+1)
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    for i in range(land.shape[0]):
        if land[i,0] and land[i,-1]: union(lab[i,0],lab[i,-1])
        if i>0:
            if land[i,0] and land[i-1,-1]: union(lab[i,0],lab[i-1,-1])
            if land[i,-1] and land[i-1,0]: union(lab[i,-1],lab[i-1,0])
    roots={}; out=np.zeros_like(lab); nxt=1
    for x in np.unique(lab[lab>0]):
        r=find(int(x))
        if r not in roots: roots[r]=nxt;nxt+=1
        out[lab==x]=roots[r]
    return out,nxt-1


def baseline_metrics():
    z=np.load(BASE_NPZ,allow_pickle=False); lat=z['lat']; w=area_weights(lat,len(z['lon']))
    land=z['elevation_km']>0
    cc=np.asarray(z['continentality_class']).astype(str)
    classes={k:weighted_mean((cc==k).astype(float),w,land) for k in np.unique(cc[land])}
    heading_n, headings=flow_heading_count(z['flow_to'],z['rivers'].astype(bool))
    b=neighbor_boundary(z['plate_id'])
    return {
      'compute_seconds': float(sum(json.load(open(BASE_JSON))['timings_seconds'].values())),
      'plate_boundary_area_fraction': weighted_mean(b.astype(float),w),
      'river_heading_count': heading_n,
      'river_headings': headings,
      'lake_fraction_land': weighted_mean(z['lakes'].astype(float),w,land),
      'precip_land_mean_mm': weighted_mean(z['annual_precipitation_mm'],w,land),
      'precip_land_p99_mm': weighted_quantile(z['annual_precipitation_mm'],w,.99,land),
      'precip_land_max_mm': float(np.max(z['annual_precipitation_mm'][land])),
      'precip_at_9600_fraction_land': weighted_mean((z['annual_precipitation_mm']>=9599.9).astype(float),w,land),
      'continentality_classes': classes,
      'land_fraction': weighted_mean(land.astype(float),w),
    }


def main():
    cfg=load_config('config/default.yaml'); cfg.seed=SEED
    cfg.output.save_png=False
    pipe=WorldPipeline(cfg,progress=None)
    t=time.perf_counter(); world=pipe.generate(); total=time.perf_counter()-t
    g=world['grid']; tr=world['terrain']; te=world['tectonics']; cl=world['climate']; hy=world['hydrology']; ge=world['geology']; re=world['resources']; so=world['society']
    w=g.cell_area_weights
    land=tr.land
    b=te.boundary
    heading_n, headings=flow_heading_count(hy.flow_to,hy.rivers)
    cc=np.asarray(cl.continentality_class).astype(str)
    classes={k:weighted_mean((cc==k).astype(float),w,land) for k in np.unique(cc[land])}
    # Local dimensionless slope proxy for deposition-vs-incision validation.
    z=tr.elevation_km
    gx=(np.roll(z,-1,1)-np.roll(z,1,1))*0.5
    gy=np.empty_like(z); gy[1:-1]=(z[2:]-z[:-2])*0.5; gy[0]=z[1]-z[0]; gy[-1]=z[-1]-z[-2]
    slope=np.hypot(gx,gy)
    e=hy.cumulative_erosion_m; d=hy.cumulative_deposition_m
    er_slope=float(np.sum(slope*e*w)/max(np.sum(e*w),1e-30))
    dp_slope=float(np.sum(slope*d*w)/max(np.sum(d*w),1e-30))
    # Area-weighted precip/runoff correlation on land.
    m=land & np.isfinite(hy.runoff)
    x=cl.annual_precipitation_mm[m]; y=hy.runoff[m]; ww=np.broadcast_to(w,land.shape)[m]; ww/=ww.sum()
    mx=np.sum(ww*x); my=np.sum(ww*y); corr=float(np.sum(ww*(x-mx)*(y-my))/math.sqrt(max(np.sum(ww*(x-mx)**2)*np.sum(ww*(y-my)**2),1e-30)))
    labels,ncomp=land_components(land)
    areas=[]
    for k in range(1,ncomp+1): areas.append(float(np.sum(np.broadcast_to(w,land.shape)[labels==k])))
    areas=sorted(areas,reverse=True)
    latest={
      'compute_seconds_wall':total,
      'stage_timings_seconds':pipe.timings,
      'land_fraction':weighted_mean(land.astype(float),w),
      'landmass_count':ncomp,
      'largest_landmass_share': areas[0]/sum(areas) if areas else 0,
      'plate_count_final':te.metadata['plate_count_final'],
      'subplate_count':te.metadata['subplate_count'],
      'mean_subplates_per_plate_final':te.metadata['mean_subplates_per_plate_final'],
      'plate_split_events':te.metadata['split_events'],
      'plate_fusion_events':te.metadata['fusion_events'],
      'plate_boundary_area_fraction':weighted_mean(b.astype(float),w),
      'subplate_boundary_area_fraction':weighted_mean(te.subplate_boundary.astype(float),w),
      'intraplate_fault_area_fraction':weighted_mean(te.intraplate_fault.astype(float),w),
      'tectonic_stress_p95':weighted_quantile(te.stress_field,w,.95),
      'river_heading_count':heading_n,
      'river_headings':headings,
      'river_area_fraction_land':weighted_mean(hy.rivers.astype(float),w,land),
      'lake_fraction_land':weighted_mean(hy.lakes.astype(float),w,land),
      'precip_land_mean_mm':weighted_mean(cl.annual_precipitation_mm,w,land),
      'precip_land_p99_mm':weighted_quantile(cl.annual_precipitation_mm,w,.99,land),
      'precip_land_max_mm':float(np.max(cl.annual_precipitation_mm[land])),
      'precip_at_9600_fraction_land':weighted_mean((cl.annual_precipitation_mm>=9599.9).astype(float),w,land),
      'continentality_land_mean_c':weighted_mean(cl.continentality_index_c,w,land),
      'continentality_land_max_c':float(np.max(cl.continentality_index_c[land])),
      'continentality_classes':classes,
      'precip_runoff_weighted_corr':corr,
      'erosion_weighted_slope_proxy':er_slope,
      'deposition_weighted_slope_proxy':dp_slope,
      'erosion_total_weighted_index':float(np.sum(e*w)),
      'deposition_total_weighted_index':float(np.sum(d*w)),
      'resource_deposit_count':len(re.deposits),
      'submerged_resource_count':sum(bool(x.get('submerged')) for x in re.deposits),
      'submerged_accessible_preindustrial_count':sum(bool(x.get('submerged')) and bool(x.get('accessible_preindustrial')) for x in re.deposits),
      'settlement_count':len(so.settlements),
      'settled_landmass_count':so.metadata.get('settled_landmass_count'),
      'society_expansion_model':so.metadata.get('expansion_model'),
    }
    base=baseline_metrics()
    comparison={'seed':SEED,'baseline_v0_1':base,'optimized_v0_2':latest}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'validation_metrics.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    # Save machine-readable world without the expensive complete map suite.
    save_cfg=cfg.output; save_cfg.save_png=False
    pipe.save(world,OUT)
    # Selected diagnostic maps from the exact same world.
    maps=OUT/'diagnostic_maps'; maps.mkdir(exist_ok=True)
    _save_field(maps/'01_parent_plates.png',te.plate_id,'v0.2 parent plates','tab20')
    _save_field(maps/'02_subplates.png',te.subplate_id,'v0.2 subplates','nipy_spectral')
    _save_field(maps/'03_tectonic_stress.png',te.stress_field,'tectonic stress','inferno',vmin=0,vmax=1)
    _save_field(maps/'04_elevation.png',world['ocean'].elevation_km,'post-fluvial elevation / bathymetry (km)','terrain')
    _save_field(maps/'05_precipitation.png',cl.annual_precipitation_mm,'annual precipitation (mm)','Blues')
    _save_field(maps/'06_rivers_lakes.png',hy.rivers.astype(float)+0.5*hy.lakes.astype(float),'rainfall-fed rivers + lakes','Blues',vmin=0,vmax=1.5)
    _save_field(maps/'07_erosion.png',hy.cumulative_erosion_m,'cumulative modeled erosion (m)','inferno')
    _save_field(maps/'08_deposition.png',hy.cumulative_deposition_m,'cumulative modeled deposition (m)','copper')
    _save_field(maps/'09_continentality.png',cl.continentality_index_c,'annual temperature range / continentality (°C)','magma')
    print(json.dumps(comparison,indent=2))

if __name__=='__main__': main()
