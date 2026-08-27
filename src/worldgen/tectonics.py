from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .config import ResolutionConfig, TectonicsConfig, NoiseConfig
from .grid import SphereGrid, normalize01, smooth_periodic, distance_to
from .noise import hybrid_multifractal, noise_kwargs, TECTONIC_BLEND, NoiseBlend
from .topology import (
    _map_spherical_lattice_indices,
    apply_bilinear_sampler,
    prepare_spherical_bilinear_sampler,
)


@dataclass(slots=True)
class TectonicResult:
    plate_id: np.ndarray
    subplate_id: np.ndarray
    continental_crust: np.ndarray
    boundary: np.ndarray
    subplate_boundary: np.ndarray
    intraplate_fault: np.ndarray
    convergent: np.ndarray
    divergent: np.ndarray
    transform: np.ndarray
    convergence_strength: np.ndarray
    divergence_strength: np.ndarray
    transform_strength: np.ndarray
    stress_field: np.ndarray
    strain_field: np.ndarray
    crust_age_myr: np.ndarray
    orogen_age_myr: np.ndarray
    rift_age_myr: np.ndarray
    paleo_convergence: np.ndarray
    paleo_divergence: np.ndarray
    hotspot_strength: np.ndarray
    lip_strength: np.ndarray
    plate_centers_xyz: np.ndarray
    subplate_centers_xyz: np.ndarray
    plate_is_continental: np.ndarray
    subplate_parent: np.ndarray
    subplate_omega_xyz: np.ndarray
    plate_omega_xyz: np.ndarray
    subplate_stress: np.ndarray
    metadata: dict


def random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.normal(size=(n, 3))
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def rodrigues_rotate(v: np.ndarray, axis: np.ndarray, angle: np.ndarray | float) -> np.ndarray:
    v = np.asarray(v, float)
    axis = np.asarray(axis, float)
    if axis.ndim == 1:
        axis = np.broadcast_to(axis, v.shape)
    ang = np.asarray(angle, float)
    if ang.ndim == 0:
        ang = np.full((len(v),), ang)
    c = np.cos(ang)[:, None]
    s = np.sin(ang)[:, None]
    return v * c + np.cross(axis, v) * s + axis * np.sum(axis * v, axis=1, keepdims=True) * (1.0 - c)


def _nearest_ids(points_xyz: np.ndarray, seeds_xyz: np.ndarray, chunk: int = 65536) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    dtype = np.int16 if len(seeds_xyz) < 32767 else np.int32
    out = np.empty(len(pts), dtype=dtype)
    for i in range(0, len(pts), chunk):
        out[i:i + chunk] = np.argmax(pts[i:i + chunk] @ seeds_xyz.T, axis=1)
    return out


def _rough_nearest_ids(points_xyz: np.ndarray, seeds_xyz: np.ndarray, rough_field: np.ndarray, amp_score: float, chunk: int = 65536) -> np.ndarray:
    """Nearest-seed ownership with coherent perturbation only near Voronoi competition zones.

    Large interiors remain stable; cells close to the two-seed bisector can switch to the runner-up
    according to a multi-scale roughness field and a pair-stable sign. This produces crenulated,
    fault-like edges without salt-and-pepper fragmentation.
    """
    pts=np.asarray(points_xyz,float).reshape(-1,3); rough=np.asarray(rough_field,float).ravel()
    dtype=np.int16 if len(seeds_xyz)<32767 else np.int32
    out=np.empty(len(pts),dtype=dtype)
    for k in range(0,len(pts),chunk):
        dot=pts[k:k+chunk]@seeds_xyz.T
        pair=np.argpartition(dot,-2,axis=1)[:,-2:]
        a,b=pair[:,0],pair[:,1]
        va=dot[np.arange(len(pair)),a]; vb=dot[np.arange(len(pair)),b]
        win=np.where(va>=vb,a,b); runner=np.where(va>=vb,b,a)
        vw=np.maximum(va,vb); vr=np.minimum(va,vb); margin=vw-vr
        lo=np.minimum(win,runner).astype(np.int64); hi=np.maximum(win,runner).astype(np.int64)
        sign=np.where(((lo*73856093 + hi*19349663) & 1)==0,1.0,-1.0)
        force=rough[k:k+len(pair)]*sign
        flip=(force>0)&(margin < amp_score*np.minimum(force,2.5))
        win=win.astype(dtype,copy=False); win[flip]=runner[flip]
        out[k:k+len(pair)]=win
    return out


def _boundary_roughness(shape: tuple[int,int], rng: np.random.Generator, noise_cfg: NoiseConfig | None = None, octaves: int = 6) -> np.ndarray:
    field = hybrid_multifractal(
        shape, rng, base_scale_px=max(shape[0] / 13.0, 3.0),
        **noise_kwargs(noise_cfg, profile=TECTONIC_BLEND, octaves=octaves),
    )
    return np.clip(field, -2.7, 2.7).astype(np.float32)

def _tangent_basis(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(c, ref))) > 0.92:
        ref = np.array([0.0, 1.0, 0.0])
    a = np.cross(ref, c)
    a /= max(np.linalg.norm(a), 1e-12)
    b = np.cross(c, a)
    b /= max(np.linalg.norm(b), 1e-12)
    return a, b


def _offset_on_sphere(c: np.ndarray, angle_rad: float, bearing: float) -> np.ndarray:
    a, b = _tangent_basis(c)
    tangent = math.cos(bearing) * a + math.sin(bearing) * b
    p = math.cos(angle_rad) * c + math.sin(angle_rad) * tangent
    return p / max(np.linalg.norm(p), 1e-12)


def _subplate_counts(nplates: int, cfg: TectonicsConfig, rng: np.random.Generator) -> np.ndarray:
    target_total = max(nplates, int(round(nplates * cfg.mean_subplates_per_plate)))
    base = rng.poisson(max(cfg.mean_subplates_per_plate - 1.0, 0.1), nplates) + 1
    base = np.clip(base, cfg.min_subplates_per_plate, cfg.max_subplates_per_plate).astype(int)
    # Adjust toward the exact requested average without producing singleton plates.
    while int(base.sum()) < target_total:
        cand = np.flatnonzero(base < cfg.max_subplates_per_plate)
        if not len(cand): break
        base[int(rng.choice(cand))] += 1
    while int(base.sum()) > target_total:
        cand = np.flatnonzero(base > cfg.min_subplates_per_plate)
        if not len(cand): break
        base[int(rng.choice(cand))] -= 1
    return base


def _initial_subplates(
    macro_centers: np.ndarray,
    cfg: TectonicsConfig,
    grid: SphereGrid,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = _subplate_counts(len(macro_centers), cfg, rng)
    parent: list[int] = []
    centers: list[np.ndarray] = []
    omega_des: list[np.ndarray] = []
    sub_cont: list[bool] = []

    macro_axes = random_unit_vectors(rng, len(macro_centers))
    speeds = rng.uniform(0.8, cfg.max_plate_speed_cm_yr, len(macro_centers))
    macro_rates = speeds * 10.0 / grid.radius_km  # rad/Myr proxy
    macro_sign = rng.choice([-1.0, 1.0], len(macro_centers))
    macro_omega = macro_axes * (macro_rates * macro_sign)[:, None]
    macro_cont = rng.random(len(macro_centers)) < cfg.continental_plate_fraction
    if not np.any(macro_cont):
        macro_cont[int(rng.integers(0, len(macro_cont)))] = True

    for p, (c, k) in enumerate(zip(macro_centers, counts)):
        # Larger macro domains receive subplate seeds spread over a broad cap. The union of
        # their irregular subdomains creates non-circular, crenulated parent boundaries.
        spread = np.deg2rad(rng.uniform(8.0, 23.0))
        for j in range(int(k)):
            angle = 0.0 if j == 0 else abs(float(rng.normal(spread * 0.62, spread * 0.26)))
            angle = min(angle, spread * 1.45)
            sc = _offset_on_sphere(c, angle, float(rng.uniform(0, 2 * np.pi)))
            centers.append(sc)
            parent.append(p)
            sub_cont.append(bool(macro_cont[p]))
            perturb = rng.normal(size=3)
            perturb -= np.dot(perturb, macro_omega[p]) * macro_omega[p] / max(np.dot(macro_omega[p], macro_omega[p]), 1e-12)
            pnorm = np.linalg.norm(perturb)
            if pnorm > 0:
                perturb /= pnorm
            mag = max(np.linalg.norm(macro_omega[p]), 1e-8)
            om = macro_omega[p] + perturb * mag * rng.normal(0, cfg.subplate_motion_dispersion)
            om *= rng.uniform(0.82, 1.18)
            omega_des.append(om)

    return (np.asarray(centers, float), np.asarray(parent, np.int32),
            np.asarray(omega_des, float), np.asarray(sub_cont, bool))


def _parent_means(parent: np.ndarray, values: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for p in np.unique(parent):
        m = parent == p
        out[int(p)] = values[m].mean(axis=0)
    return out


def _actual_omega(parent: np.ndarray, desired: np.ndarray, coupling: float) -> np.ndarray:
    means = _parent_means(parent, desired)
    out = desired.copy()
    for i, p in enumerate(parent):
        out[i] = (1.0 - coupling) * desired[i] + coupling * means[int(p)]
    return out


def _neighbor_pairs(centers: np.ndarray, k: int = 6) -> list[tuple[int, int]]:
    dots = centers @ centers.T
    np.fill_diagonal(dots, -2.0)
    kk = min(k, max(1, len(centers) - 1))
    idx = np.argpartition(-dots, kth=kk - 1, axis=1)[:, :kk]
    pairs: set[tuple[int, int]] = set()
    for i in range(len(centers)):
        for j in idx[i]:
            if i != int(j):
                pairs.add((min(i, int(j)), max(i, int(j))))
    return sorted(pairs)


def _closing_score(ci: np.ndarray, cj: np.ndarray, oi: np.ndarray, oj: np.ndarray) -> tuple[float, float]:
    # Scalar 3-vector math is substantially faster here than allocating np.cross temporaries for
    # thousands of tiny neighbour interactions during subplate dynamics.
    mx=float(ci[0]+cj[0]); my=float(ci[1]+cj[1]); mz=float(ci[2]+cj[2])
    nm=math.sqrt(mx*mx+my*my+mz*mz)
    if nm < 1e-10:
        mx,my,mz=map(float,ci); nm=1.0
    else:
        mx/=nm; my/=nm; mz/=nm
    dotj=float(cj[0])*mx+float(cj[1])*my+float(cj[2])*mz
    tx=float(cj[0])-dotj*mx; ty=float(cj[1])-dotj*my; tz=float(cj[2])-dotj*mz
    nt=math.sqrt(tx*tx+ty*ty+tz*tz)
    if nt < 1e-10: return 0.0,0.0
    tx/=nt; ty/=nt; tz/=nt
    vix=float(oi[1])*mz-float(oi[2])*my; viy=float(oi[2])*mx-float(oi[0])*mz; viz=float(oi[0])*my-float(oi[1])*mx
    vjx=float(oj[1])*mz-float(oj[2])*my; vjy=float(oj[2])*mx-float(oj[0])*mz; vjz=float(oj[0])*my-float(oj[1])*mx
    rx=vix-vjx; ry=viy-vjy; rz=viz-vjz
    closing=rx*tx+ry*ty+rz*tz
    sx=rx-closing*tx; sy=ry-closing*ty; sz=rz-closing*tz
    shear=math.sqrt(sx*sx+sy*sy+sz*sz)
    return closing,shear

def _split_parent(parent: np.ndarray, desired: np.ndarray, p: int, min_sub: int) -> bool:
    members = np.flatnonzero(parent == p)
    if len(members) < max(2 * min_sub, 4):
        return False
    dirs = desired[members] / np.maximum(np.linalg.norm(desired[members], axis=1, keepdims=True), 1e-12)
    dots = dirs @ dirs.T
    a, b = np.unravel_index(np.argmin(dots), dots.shape)
    da, db = dirs[a], dirs[b]
    ga = dirs @ da >= dirs @ db
    if ga.sum() < min_sub or (~ga).sum() < min_sub:
        # Fallback deterministic bisection along principal axis in motion-vector space.
        x = desired[members] - desired[members].mean(axis=0)
        _, _, vh = np.linalg.svd(x, full_matrices=False)
        score = x @ vh[0]
        order = np.argsort(score)
        ga = np.zeros(len(members), bool)
        ga[order[:len(order)//2]] = True
    newp = int(parent.max()) + 1
    parent[members[~ga]] = newp
    return True


def _relabel_parent(parent: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    vals = sorted(map(int, np.unique(parent)))
    mp = {old: i for i, old in enumerate(vals)}
    return np.array([mp[int(p)] for p in parent], dtype=np.int32), mp


def _simulate_subplates(
    centers: np.ndarray,
    parent: np.ndarray,
    desired: np.ndarray,
    cfg: TectonicsConfig,
    res: ResolutionConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]]:
    centers = centers.copy(); parent = parent.copy(); desired = desired.copy()
    dt = float(res.history_step_myr)
    steps = max(1, int(round(res.history_myr / dt)))
    target_plates = cfg.plate_count
    fuse_memory: dict[tuple[int, int], int] = {}
    events: list[dict] = []
    snapshots: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    sub_stress = np.zeros(len(centers), float)

    for step in range(steps + 1):
        elapsed = min(step * dt, float(res.history_myr))
        actual = _actual_omega(parent, desired, cfg.parent_coupling)
        snapshots.append((elapsed, centers.copy(), parent.copy(), actual.copy()))
        if step == steps:
            break

        pairs = _neighbor_pairs(centers, k=6)
        local_stress = np.zeros(len(centers), float)
        parent_pair_alignment: dict[tuple[int, int], list[float]] = {}
        adjusted = actual.copy()
        rate_ref = max(float(np.median(np.linalg.norm(actual, axis=1))), 1e-8)
        pmeans_for_pairs = _parent_means(parent, desired)

        for i, j in pairs:
            pi, pj = int(parent[i]), int(parent[j])
            closing, shear = _closing_score(centers[i], centers[j], actual[i], actual[j])
            if pi == pj:
                ni = max(np.linalg.norm(desired[i]), 1e-12); nj = max(np.linalg.norm(desired[j]), 1e-12)
                angle = math.acos(float(np.clip(np.dot(desired[i], desired[j]) / (ni * nj), -1, 1)))
                mismatch = angle / np.pi + abs(ni - nj) / rate_ref
                local_stress[i] += mismatch; local_stress[j] += mismatch
            else:
                # Convergent neighbours push/drag each other's Euler motion rather than passing through
                # unchanged. This is a reduced-order torque/coupling approximation, not a mantle solver.
                if closing > 0:
                    nudge = np.clip((closing / rate_ref) * cfg.collision_nudge, 0.0, 0.22)
                    oi, oj = adjusted[i].copy(), adjusted[j].copy()
                    adjusted[i] = (1.0 - nudge) * oi + nudge * oj
                    adjusted[j] = (1.0 - nudge) * oj + nudge * oi
                    local_stress[i] += closing / rate_ref + 0.25 * shear / rate_ref
                    local_stress[j] += closing / rate_ref + 0.25 * shear / rate_ref
                mi = pmeans_for_pairs[pi]
                mj = pmeans_for_pairs[pj]
                ai = mi / max(np.linalg.norm(mi), 1e-12); aj = mj / max(np.linalg.norm(mj), 1e-12)
                angle_deg = math.degrees(math.acos(float(np.clip(np.dot(ai, aj), -1, 1))))
                key = (min(pi, pj), max(pi, pj))
                parent_pair_alignment.setdefault(key, []).append(angle_deg)

        # Stress has memory, but relaxes when motion incompatibility disappears.
        sub_stress = 0.72 * sub_stress + 0.28 * local_stress
        desired = 0.94 * desired + 0.06 * adjusted

        # Split the single most stressed eligible parent per step to avoid cascade explosions.
        p_stress = []
        for p in np.unique(parent):
            m = parent == p
            p_stress.append((float(np.mean(sub_stress[m]) + 0.35 * np.max(sub_stress[m])), int(p)))
        p_stress.sort(reverse=True)
        if p_stress and p_stress[0][0] > cfg.split_stress_threshold:
            ps, p = p_stress[0]
            if _split_parent(parent, desired, p, cfg.min_subplates_per_plate):
                events.append({"time_myr_from_start": elapsed, "type": "plate_split", "parent": p, "stress": round(ps, 4)})
                sub_stress[parent == parent.max()] *= 0.55

        # Persistent similar motion across a shared neighbourhood permits plate fusion.
        fused = False
        for key, angles in parent_pair_alignment.items():
            mean_ang = float(np.mean(angles))
            if mean_ang <= cfg.fuse_direction_deg:
                fuse_memory[key] = fuse_memory.get(key, 0) + 1
            else:
                fuse_memory[key] = 0
            if fuse_memory[key] >= cfg.fuse_persistence_steps:
                a, b = key
                if np.any(parent == a) and np.any(parent == b):
                    # Fuse only if the resulting plate is not absurdly overfull.
                    count = int(np.sum(parent == a) + np.sum(parent == b))
                    if count <= int(1.8 * cfg.max_subplates_per_plate):
                        parent[parent == b] = a
                        events.append({"time_myr_from_start": elapsed, "type": "plate_fusion", "plates": [a, b], "direction_difference_deg": round(mean_ang, 3)})
                        fused = True
                        break
        if fused:
            fuse_memory.clear()

        # Homeostasis: keep the hierarchy near its requested mean subplate/plate ratio.
        parent, _ = _relabel_parent(parent)
        npar = len(np.unique(parent))
        if npar < target_plates - 1:
            # Split the largest sufficiently heterogeneous plate.
            candidates = sorted(((int(np.sum(parent == p)), int(p)) for p in np.unique(parent)), reverse=True)
            for _, p in candidates:
                if _split_parent(parent, desired, p, cfg.min_subplates_per_plate):
                    events.append({"time_myr_from_start": elapsed, "type": "homeostatic_split", "parent": p})
                    break
        elif npar > target_plates + 1:
            # Merge the most directionally aligned neighbouring parents.
            pmean = _parent_means(parent, desired)
            best = None
            for i, j in pairs:
                a, b = int(parent[i]), int(parent[j])
                if a == b: continue
                va, vb = pmean[a], pmean[b]
                ang = math.acos(float(np.clip(np.dot(va, vb) / max(np.linalg.norm(va)*np.linalg.norm(vb), 1e-12), -1, 1)))
                if best is None or ang < best[0]: best = (ang, a, b)
            if best is not None:
                _, a, b = best
                parent[parent == b] = a
                events.append({"time_myr_from_start": elapsed, "type": "homeostatic_fusion", "plates": [a, b]})

        parent, _ = _relabel_parent(parent)
        actual = _actual_omega(parent, desired, cfg.parent_coupling)
        # Advance centers by their Euler rotations. A tiny stochastic torque prevents exact periodicity.
        torque = rng.normal(0.0, 0.012, desired.shape)
        desired *= (1.0 + torque)
        rates = np.linalg.norm(actual, axis=1)
        axes = actual / np.maximum(rates[:, None], 1e-12)
        centers = rodrigues_rotate(centers, axes, rates * dt)
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)

    parent, _ = _relabel_parent(parent)
    actual = _actual_omega(parent, desired, cfg.parent_coupling)
    return centers, parent, desired, sub_stress, events, snapshots


def _warp_fields(shape: tuple[int, int], amp_deg: float, octaves: int, rng: np.random.Generator, noise_cfg: NoiseConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    # Two independently generated hybrid multifractal displacement fields.  Every displacement map itself
    # contains decreasing-amplitude octaves and multiple noise families; a mild extra domain warp bends
    # long boundaries without destroying coherent plate interiors.
    out_y = hybrid_multifractal(
        shape, rng, base_scale_px=max(h / 8.5, 3.0),
        **noise_kwargs(noise_cfg, profile=NoiseBlend(0.37,0.18,0.08,0.37), octaves=octaves),
    )
    out_x = hybrid_multifractal(
        shape, rng, base_scale_px=max(h / 9.3, 3.0),
        **noise_kwargs(noise_cfg, profile=NoiseBlend(0.33,0.20,0.10,0.37), octaves=octaves),
    )
    # Robustly bound very rare displacement spikes.
    out_y = np.tanh(out_y / 2.0) * float(amp_deg)
    out_x = np.tanh(out_x / 2.0) * float(amp_deg)
    lat = np.linspace(90 - 90/h, -90 + 90/h, h)[:, None]
    out_x *= np.clip(np.cos(np.deg2rad(lat)), 0.15, 1.0)
    return out_y.astype(np.float32), out_x.astype(np.float32)

def _warped_xyz(grid: SphereGrid, warp_lat_deg: np.ndarray, warp_lon_deg: np.ndarray) -> np.ndarray:
    lat = np.clip(grid.lat + warp_lat_deg, -89.999, 89.999)
    lon = (grid.lon + warp_lon_deg + 180.0) % 360.0 - 180.0
    lr = np.deg2rad(lat); orr = np.deg2rad(lon)
    c = np.cos(lr)
    return np.stack((c*np.cos(orr), c*np.sin(orr), np.sin(lr)), axis=-1)


def _deform_plate_ownership(
    grid: SphereGrid, subcenters: np.ndarray, parent: np.ndarray, pair_type: np.ndarray, pair_strength: np.ndarray,
    sub_id: np.ndarray, warp_lat: np.ndarray, warp_lon: np.ndarray, cfg: TectonicsConfig,
    rng: np.random.Generator, noise_cfg: NoiseConfig | None, rough_amp: float, base_rough: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively deform Voronoi-like ownership using the active strain/boundary geometry.

    This is still a reduced-order block model, but it avoids treating the initial nearest-seed cells as
    immutable polygons. Convergence/divergence pushes boundaries normal to themselves, transform motion
    drags them tangentially, and coherent hybrid fields wrinkle the deformation zone.
    """
    sid = np.asarray(sub_id, dtype=np.int16).copy()
    wy = np.asarray(warp_lat, dtype=np.float32).copy()
    wx = np.asarray(warp_lon, dtype=np.float32).copy()
    h, w = sid.shape
    coslat = np.clip(np.cos(np.deg2rad(grid.lat)), 0.16, 1.0)
    # Structural heterogeneity is stationary through the few numerical relaxation iterations; generate
    # it once rather than rerolling expensive multifractals each iteration.
    wrinkle_y_base = hybrid_multifractal(
        sid.shape, rng, base_scale_px=max(h / 23.0, 3.0),
        **noise_kwargs(noise_cfg, profile=TECTONIC_BLEND, octaves=max(4, cfg.boundary_detail_octaves - 1)),
    )
    wrinkle_x_base = hybrid_multifractal(
        sid.shape, rng, base_scale_px=max(h / 27.0, 3.0),
        **noise_kwargs(noise_cfg, profile=NoiseBlend(0.31,0.27,0.08,0.34), octaves=max(4, cfg.boundary_detail_octaves - 1)),
    )
    rough_static = np.asarray(base_rough if base_rough is not None else _boundary_roughness(sid.shape,rng,noise_cfg,cfg.boundary_detail_octaves),np.float32)
    for it in range(max(0, int(cfg.boundary_deformation_iterations))):
        sb, intra, cv, dv, tr, cs, ds, ts = _classify_subplate_boundaries(grid, sid, parent, pair_type, pair_strength)
        activity = smooth_periodic(cs + ds + 0.85 * ts + 0.22 * intra.astype(float), (1.15, 1.55))
        # Influence fades into plate interiors over a few hundred km.
        dist = distance_to(sb, grid)
        envelope = np.exp(-dist / (240.0 + 80.0 * it))
        gy, gx = grid.ops.metric_gradient(activity)
        gn = np.hypot(gx, gy) + 1e-8
        ny, nx = gy / gn, gx / gn
        ty, tx = -nx, ny
        # Translate the stationary structural texture through the same spherical
        # topology used by every other geographic neighborhood operation.
        wrinkle_y = grid.ops.shift(wrinkle_y_base, -2 * it, -3 * it)
        wrinkle_x = grid.ops.shift(wrinkle_x_base, 3 * it, -2 * it)
        normal_drive = np.clip(ds - cs, -1.0, 1.0)
        shear_sign = np.tanh(wrinkle_x)
        dy = envelope * (0.42 * wrinkle_y + 0.70 * normal_drive * ny + 0.60 * ts * shear_sign * ty)
        dx = envelope * (0.42 * wrinkle_x + 0.70 * normal_drive * nx + 0.60 * ts * shear_sign * tx)
        amp = float(cfg.strain_boundary_warp_deg) / max(1, int(cfg.boundary_deformation_iterations))
        wy += (amp * np.tanh(dy)).astype(np.float32)
        wx += (amp * np.tanh(dx) * coslat).astype(np.float32)
        wxyz = _warped_xyz(grid, wy, wx)
        sid = _rough_nearest_ids(wxyz, subcenters, rough_static, rough_amp).reshape(h, w).astype(np.int16)
    return sid, wy, wx



def _shape_control_seeds(
    subcenters: np.ndarray, cfg: TectonicsConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return multiple nearby control anchors per subplate.

    Ownership of the union of these anchors maps back to one subplate.  This cheaply introduces
    lobate/non-convex block geometry without increasing the *physical* subplate count or changing
    its Euler motion.  The main center is always retained, so anchors only perturb shape.
    """
    k = max(1, int(cfg.shape_control_points_per_subplate))
    if k == 1:
        return np.asarray(subcenters, float), np.arange(len(subcenters), dtype=np.int32)
    controls: list[np.ndarray] = []
    owners: list[int] = []
    max_spread = math.radians(max(0.1, float(cfg.shape_control_spread_deg)))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i, c in enumerate(np.asarray(subcenters, float)):
        controls.append(c); owners.append(i)
        base_bearing = float(rng.uniform(0.0, 2.0 * math.pi))
        for j in range(1, k):
            # Irregular but bounded satellite anchors; later satellites are slightly farther out.
            frac = (j / max(k - 1, 1)) ** 0.72
            angle = max_spread * frac * float(rng.uniform(0.58, 1.0))
            bearing = base_bearing + j * golden + float(rng.normal(0.0, 0.22))
            controls.append(_offset_on_sphere(c, angle, bearing)); owners.append(i)
    return np.asarray(controls, float), np.asarray(owners, np.int32)

def _pair_motion_matrices(centers: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized relative-motion classification for all subplate pairs."""
    c = np.asarray(centers, float); o = np.asarray(omega, float)
    n = len(c)
    ci = c[:, None, :]
    cj = c[None, :, :]
    mid = ci + cj
    mn = np.linalg.norm(mid, axis=2, keepdims=True)
    mid = np.divide(mid, np.maximum(mn, 1e-12))
    anti = mn[..., 0] < 1e-10
    if np.any(anti):
        mid[anti] = np.broadcast_to(ci, (n, n, 3))[anti]
    dotjm = np.sum(cj * mid, axis=2, keepdims=True)
    tangent = cj - dotjm * mid
    tn = np.linalg.norm(tangent, axis=2, keepdims=True)
    tangent = np.divide(tangent, np.maximum(tn, 1e-12))
    vi = np.cross(o[:, None, :], mid)
    vj = np.cross(o[None, :, :], mid)
    rel = vi - vj
    closing = np.sum(rel * tangent, axis=2)
    shear_vec = rel - closing[..., None] * tangent
    shear = np.linalg.norm(shear_vec, axis=2)
    ref = max(float(np.percentile(np.linalg.norm(o, axis=1), 75)), 1e-8)
    th = 0.11 * ref
    typ = np.where(closing > th, -1, np.where(closing < -th, 1, 0)).astype(np.int8)
    strength = np.clip((np.abs(closing) + 0.35 * shear) / (2.3 * ref), 0, 1).astype(np.float32)
    np.fill_diagonal(typ, 0); np.fill_diagonal(strength, 0.0)
    return typ, strength

def _classify_subplate_boundaries(
    grid: SphereGrid,
    sub: np.ndarray,
    parent: np.ndarray,
    pair_type: np.ndarray,
    pair_strength: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classify 8-neighbor subplate boundaries on the canonical sphere topology."""
    h, w = sub.shape
    if (h, w) != (grid.height, grid.width):
        raise ValueError("subplate raster shape must match grid")
    conv = np.zeros((h,w), bool); div = np.zeros_like(conv); trans = np.zeros_like(conv)
    intra = np.zeros_like(conv); allb = np.zeros_like(conv)
    cs = np.zeros((h,w), np.float32); ds = np.zeros_like(cs); ts = np.zeros_like(cs)
    nbs = [grid.ops.shift(sub, dy, dx) for dy, dx in grid.ops.neighbors8()]
    for nb in nbs:
        diff = sub != nb
        allb |= diff
        same_parent = parent[sub] == parent[nb]
        intra |= diff & same_parent
        macro = diff & ~same_parent
        t = pair_type[sub, nb]
        s = pair_strength[sub, nb]
        m = macro & (t < 0); conv |= m; cs = np.maximum(cs, np.where(m, s, 0))
        m = macro & (t > 0); div |= m; ds = np.maximum(ds, np.where(m, s, 0))
        m = macro & (t == 0); trans |= m; ts = np.maximum(ts, np.where(m, s, 0))
    return allb, intra, conv, div, trans, cs, ds, ts


def _resize(a: np.ndarray, shape: tuple[int,int], order: int = 1) -> np.ndarray:
    """Resize a cell-centered global raster with spherical seam/pole semantics.

    ``order=1`` uses bilinear interpolation whose corner indices are mapped through
    longitude periodicity and antipodal pole reflection. ``order=0`` uses the same
    spherical lattice mapping for nearest-neighbor categorical fields.
    """
    src = np.asarray(a)
    if src.ndim != 2:
        raise ValueError("tectonic raster resize expects a two-dimensional field")
    target = (int(shape[0]), int(shape[1]))
    if src.shape == target:
        return src
    if target[0] <= 0 or target[1] <= 0:
        raise ValueError("target shape dimensions must be positive")
    if int(order) not in (0, 1):
        raise ValueError("spherical tectonic resize supports only order 0 or 1")

    sh, sw = src.shape
    th, tw = target
    sy1 = (np.arange(th, dtype=np.float64) + 0.5) * (sh / th) - 0.5
    sx1 = (np.arange(tw, dtype=np.float64) + 0.5) * (sw / tw) - 0.5
    sy, sx = np.broadcast_arrays(sy1[:, None], sx1[None, :])
    if int(order) == 0:
        iy = np.floor(sy + 0.5).astype(np.int64)
        ix = np.floor(sx + 0.5).astype(np.int64)
        my, mx = _map_spherical_lattice_indices(iy, ix, src.shape)
        return src[my, mx]
    sampler = prepare_spherical_bilinear_sampler(sy, sx, src.shape)
    return apply_bilinear_sampler(src, sampler)


def _blob_field(grid: SphereGrid, centers_xyz: np.ndarray, sigmas_deg: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pts = grid.xyz.reshape(-1, 3)
    out = np.zeros(len(pts), dtype=np.float64)
    chunk = 65536
    for k, c in enumerate(centers_xyz):
        sigma = np.deg2rad(sigmas_deg[k])
        for i in range(0, len(pts), chunk):
            dots = np.clip(pts[i:i+chunk] @ c, -1, 1)
            ang = np.arccos(dots)
            out[i:i+chunk] += weights[k] * np.exp(-0.5 * (ang/max(sigma,1e-6))**2)
    return out.reshape(grid.height, grid.width)


def generate_tectonics(grid: SphereGrid, cfg: TectonicsConfig, res: ResolutionConfig, rng: np.random.Generator, noise_cfg: NoiseConfig | None = None) -> TectonicResult:
    macro0 = random_unit_vectors(rng, cfg.plate_count)
    sub0, parent0, desired0, sub_cont = _initial_subplates(macro0, cfg, grid, rng)
    subcenters, parent, desired, sub_stress, events, snapshots = _simulate_subplates(sub0, parent0, desired0, cfg, res, rng)
    actual = _actual_omega(parent, desired, cfg.parent_coupling)

    warp_lat, warp_lon = _warp_fields((grid.height, grid.width), cfg.boundary_warp_deg, cfg.boundary_detail_octaves, rng, noise_cfg)
    wxyz = _warped_xyz(grid, warp_lat, warp_lon)
    boundary_rough = _boundary_roughness((grid.height,grid.width), rng, noise_cfg, cfg.boundary_detail_octaves)
    rough_amp = 0.0045 + 0.0022 * cfg.boundary_warp_deg
    sub_id = _rough_nearest_ids(wxyz, subcenters, boundary_rough, rough_amp).reshape(grid.height, grid.width)
    pair_type, pair_strength = _pair_motion_matrices(subcenters, actual)
    sub_id, warp_lat, warp_lon = _deform_plate_ownership(
        grid, subcenters, parent, pair_type, pair_strength, sub_id, warp_lat, warp_lon,
        cfg, rng, noise_cfg, rough_amp, boundary_rough,
    )
    # A final multi-anchor ownership solve removes the remaining convex-cell visual signature while
    # preserving one physical motion/stress state per subplate. This is only paid once per world.
    if int(cfg.shape_control_points_per_subplate) > 1:
        control_xyz, control_owner = _shape_control_seeds(subcenters, cfg, rng)
        final_xyz = _warped_xyz(grid, warp_lat, warp_lon)
        control_id = _rough_nearest_ids(final_xyz, control_xyz, boundary_rough, rough_amp * 0.82).reshape(grid.height, grid.width)
        sub_id = control_owner[control_id].astype(np.int16)
    plate_id = parent[sub_id]
    sub_bnd, intra, conv, div, trans, conv_s, div_s, trans_s = _classify_subplate_boundaries(grid, sub_id, parent, pair_type, pair_strength)
    macro_bnd = conv | div | trans

    # Historical deformation is accumulated on a smaller grid then smoothly upsampled. This retains
    # fine current geometry without paying full-resolution Voronoi cost for every 25 Myr snapshot.
    hh = min(grid.height, max(48, cfg.history_grid_height)); hw = 2 * hh
    hgrid = SphereGrid(hw, hh, grid.radius_km)
    hwy = _resize(warp_lat, (hh,hw), 1); hwx = _resize(warp_lon, (hh,hw), 1)
    hrough = _resize(boundary_rough,(hh,hw),1)
    hwxyz = _warped_xyz(hgrid, hwy, hwx)
    pconv = np.zeros((hh,hw), np.float32); pdiv = np.zeros_like(pconv); strain = np.zeros_like(pconv)
    oage = np.full((hh,hw), np.inf, np.float32); rage = np.full_like(oage, np.inf)
    # Record roughly every history step but cap at 48 rasterized snapshots for performance.
    stride = max(1, int(math.ceil(len(snapshots)/48)))
    used = snapshots[::stride]
    if snapshots[-1] is not used[-1]: used.append(snapshots[-1])
    for elapsed, c, par, om in used:
        sid = _rough_nearest_ids(hwxyz, c, hrough, rough_amp).reshape(hh,hw)
        typ, strength = _pair_motion_matrices(c, om)
        _, intr, cv, dv, tr, cs, ds, ts = _classify_subplate_boundaries(hgrid, sid, par, typ, strength)
        age = float(res.history_myr - elapsed)
        cvz = hgrid.ops.binary_dilation(cv, iterations=2)
        dvz = hgrid.ops.binary_dilation(dv, iterations=1)
        wt = math.exp(-age/420.0)
        pconv += smooth_periodic(cs + 0.18*cvz, (1.0,1.25)).astype(np.float32) * wt
        pdiv += smooth_periodic(ds + 0.18*dvz, (1.0,1.25)).astype(np.float32) * wt
        strain += smooth_periodic(cs + ds + 0.65*ts + 0.35*intr, (1.1,1.4)).astype(np.float32) * wt
        oage[cvz] = np.minimum(oage[cvz], age)
        rage[dvz] = np.minimum(rage[dvz], age)

    pconv = normalize01(_resize(pconv, plate_id.shape, 1)).astype(np.float32)
    pdiv = normalize01(_resize(pdiv, plate_id.shape, 1)).astype(np.float32)
    strain = normalize01(_resize(strain, plate_id.shape, 1)).astype(np.float32)
    oage = _resize(np.where(np.isfinite(oage), oage, res.history_myr+500), plate_id.shape, 0).astype(np.float32)
    rage = _resize(np.where(np.isfinite(rage), rage, res.history_myr+500), plate_id.shape, 0).astype(np.float32)

    # Current stress includes internal subplate incompatibility and active plate boundaries.
    stress_seed = sub_stress[sub_id]
    stress = normalize01(0.45*stress_seed + 0.55*smooth_periodic(conv_s + div_s + 0.55*trans_s + 0.35*intra, (1.2,1.8)))

    # Continental affinity lives on subplates so fused/split parent plates can contain inherited fragments.
    aff = np.where(sub_cont[sub_id], 1.0, -0.42)
    noise = hybrid_multifractal(
        plate_id.shape, rng, base_scale_px=max(grid.height / 12.0, 4.0),
        **noise_kwargs(noise_cfg, profile=NoiseBlend(0.50,0.15,0.20,0.15), octaves=max(5, cfg.boundary_detail_octaves)),
    )
    cont_score = 0.80*aff + 0.62*noise + 0.12*pconv - 0.16*pdiv
    cont_target = float(np.clip(cfg.continental_fraction_target + 0.13, 0.30, 0.56))
    cthr = grid.weighted_quantile(cont_score, 1.0-cont_target)
    continental = cont_score >= cthr

    # Ocean crust is youngest at spreading zones; fracture/transform strain adds local age offsets.
    dridge = distance_to(div, grid)
    crust_age = np.clip(dridge / 48.0 + 18.0*stress*trans.astype(float), 0.0, 230.0)
    crust_age[continental] = np.maximum(crust_age[continental], 250.0)

    hs_xyz = random_unit_vectors(rng, cfg.hotspot_count)
    hotspot = normalize01(_blob_field(grid, hs_xyz, rng.uniform(0.7,2.8,cfg.hotspot_count), rng.uniform(0.55,1.0,cfg.hotspot_count)))
    lip_count = max(1, int(round(res.history_myr/max(cfg.lip_interval_myr,1.0))))
    lip_xyz = random_unit_vectors(rng, lip_count); lip_age = rng.uniform(0,res.history_myr,lip_count)
    lip = normalize01(_blob_field(grid, lip_xyz, rng.uniform(1.5,6.5,lip_count), np.exp(-lip_age/250.0)*rng.uniform(.45,1.2,lip_count)))

    # Final parent centers and continental majority flag.
    pvals = np.unique(parent)
    pcenters=[]; pcont=[]; pomega=[]
    for p in pvals:
        m=parent==p; c=subcenters[m].mean(axis=0); c/=max(np.linalg.norm(c),1e-12); pcenters.append(c)
        pcont.append(bool(np.mean(sub_cont[m])>=0.5)); pomega.append(actual[m].mean(axis=0))

    split_count = sum(e['type'] in {'plate_split','homeostatic_split'} for e in events)
    fuse_count = sum(e['type'] in {'plate_fusion','homeostatic_fusion'} for e in events)
    counts = [int(np.sum(parent==p)) for p in pvals]
    meta = {
        'plate_count_target': cfg.plate_count,
        'plate_count_final': int(len(pvals)),
        'subplate_count': int(len(subcenters)),
        'mean_subplates_per_plate_final': float(np.mean(counts)),
        'subplates_per_plate_final': counts,
        'history_myr': res.history_myr,
        'history_step_myr': res.history_step_myr,
        'history_rasterized_snapshots': len(used),
        'split_events': int(split_count), 'fusion_events': int(fuse_count),
        'events': events,
        'boundary_warp_deg': cfg.boundary_warp_deg,
        'shape_control_points_per_subplate': int(cfg.shape_control_points_per_subplate),
        'shape_control_spread_deg': float(cfg.shape_control_spread_deg),
        'boundary_deformation_iterations': int(cfg.boundary_deformation_iterations),
        'strain_boundary_warp_deg': float(cfg.strain_boundary_warp_deg),
        'noise_model': 'shared hybrid multi-type multifractal with decreasing octave amplitudes and domain warping',
        'subplate_speeds_cm_yr_current': (np.linalg.norm(actual,axis=1)*grid.radius_km/10.0).round(4).tolist(),
        'model_note': 'Hierarchical deforming subplate kinematics with coupled Euler motions, stress memory, split/fuse events, collision nudging and multi-scale warped boundaries. Reduced-order model; not a mantle-convection PDE solver.'
    }
    return TectonicResult(
        plate_id.astype(np.int16), sub_id.astype(np.int16), continental, macro_bnd, sub_bnd, intra,
        conv, div, trans, conv_s, div_s, trans_s, stress.astype(np.float32), strain,
        crust_age.astype(np.float32), oage, rage, pconv, pdiv,
        hotspot.astype(np.float32), lip.astype(np.float32), np.asarray(pcenters,np.float32),
        subcenters.astype(np.float32), np.asarray(pcont,bool), parent.astype(np.int16),
        actual.astype(np.float32), np.asarray(pomega,np.float32), sub_stress.astype(np.float32), meta
    )
