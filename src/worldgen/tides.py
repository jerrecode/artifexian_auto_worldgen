from __future__ import annotations

"""Reduced-order equilibrium tides for arbitrary moon systems plus the host star."""

from dataclasses import dataclass
import math
import numpy as np

from .grid import SphereGrid, normalize01, smooth_periodic

EARTH_MASS_KG = 5.9722e24
SOLAR_MASS_KG = 1.98847e30
AU_M = 1.495978707e11


@dataclass(slots=True)
class TideResult:
    equilibrium_tide_amplitude_m: np.ndarray
    tidal_range_m: np.ndarray
    tidal_current_index: np.ndarray
    intertidal_potential: np.ndarray
    constituent_count: int
    constituents: list[dict]
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "constituent_count": self.constituent_count,
            "constituents": self.constituents,
            "metadata": self.metadata,
        }


def _equilibrium_height_m(body_mass_earth: float, radius_km: float, perturber_mass_kg: float, distance_m: float) -> float:
    body_mass = max(float(body_mass_earth), 1.0e-12) * EARTH_MASS_KG
    r = max(float(radius_km), 1.0) * 1000.0
    d = max(float(distance_m), r * 1.01)
    return float((perturber_mass_kg / body_mass) * r * (r / d) ** 3)


def _moon_constituents(astronomy) -> list[dict]:
    out: list[dict] = []
    planet = getattr(astronomy, "planet", {}) or {}
    body_mass = float(planet.get("mass_earth", 1.0))
    radius_km = float(planet.get("radius_earth", 1.0)) * 6371.0
    for i, moon in enumerate(getattr(astronomy, "moons", []) or []):
        mass_earth = float(moon.get("mass_earth", 0.0) or 0.0)
        orbit_km = float(moon.get("orbit_km", moon.get("semimajor_axis_km", 0.0)) or 0.0)
        if mass_earth <= 0.0 or orbit_km <= 0.0:
            continue
        amp = _equilibrium_height_m(body_mass, radius_km, mass_earth * EARTH_MASS_KG, orbit_km * 1000.0)
        period = float(moon.get("sidereal_period_days", moon.get("orbital_period_days", 0.0)) or 0.0)
        out.append({
            "name": str(moon.get("name", f"moon_{i+1}")),
            "kind": "moon",
            "mass_earth": mass_earth,
            "distance_km": orbit_km,
            "equilibrium_amplitude_m": amp,
            "period_days": period,
            "phase_rad": float((i + 1) * 2.399963229728653),
        })
    return out


def _stellar_constituent(astronomy) -> dict | None:
    planet = getattr(astronomy, "planet", {}) or {}
    star = getattr(astronomy, "star", {}) or {}
    body_mass = float(planet.get("mass_earth", 1.0))
    radius_km = float(planet.get("radius_earth", 1.0)) * 6371.0
    a_au = float(planet.get("semimajor_axis_au", 0.0) or 0.0)
    star_mass = float(star.get("mass_solar", 0.0) or 0.0)
    if a_au <= 0.0 or star_mass <= 0.0:
        return None
    amp = _equilibrium_height_m(body_mass, radius_km, star_mass * SOLAR_MASS_KG, a_au * AU_M)
    year_days = float((getattr(astronomy, "calendar", {}) or {}).get("local_year_days", 0.0) or 0.0)
    return {
        "name": "stellar_semidiurnal",
        "kind": "star",
        "mass_solar": star_mass,
        "distance_au": a_au,
        "equilibrium_amplitude_m": amp,
        "period_days": year_days,
        "phase_rad": 0.0,
    }


def build_tides(grid: SphereGrid, astronomy, terrain, ocean) -> TideResult:
    constituents = _moon_constituents(astronomy)
    stellar = _stellar_constituent(astronomy)
    if stellar is not None:
        constituents.append(stellar)

    lat = np.deg2rad(np.asarray(grid.lat, dtype=float))
    lon = np.deg2rad(np.asarray(grid.lon, dtype=float))
    water = np.asarray(terrain.ocean, dtype=bool)
    depth = np.maximum(np.asarray(ocean.depth_m, dtype=float), 0.0)
    if not constituents:
        zero = np.zeros(grid.shape, dtype=np.float32)
        return TideResult(zero, zero, zero, zero, 0, [], {"model": "no external tidal constituents"})

    # Sum RMS constituent envelopes rather than fixing an arbitrary epoch.  Each
    # constituent receives a deterministic orbital-plane orientation/phase so the
    # spatial field remains reproducible and multi-moon systems exhibit interference.
    rms_sq = np.zeros(grid.shape, dtype=np.float64)
    coherent = np.zeros(grid.shape, dtype=np.float64)
    for i, c in enumerate(constituents):
        phase = float(c["phase_rad"])
        obliquity = 0.18 * math.sin((i + 1) * 1.7)
        mu = np.cos(lat - obliquity) * np.cos(lon - phase)
        p2 = 0.5 * (3.0 * mu * mu - 1.0)
        amp = float(c["equilibrium_amplitude_m"])
        field = amp * p2
        rms_sq += 0.5 * field * field
        coherent += field

    equilibrium_rms = np.sqrt(np.maximum(rms_sq, 0.0))
    # Continental shelves and semi-enclosed shallow seas amplify equilibrium forcing;
    # abyssal oceans retain the small open-ocean signal. This is a screening model,
    # not a shallow-water harmonic tide solver.
    shallow = np.exp(-depth / 850.0)
    coast = water & grid.ops.binary_dilation(terrain.land, iterations=2)
    shelf_amp = 1.0 + 2.1 * shallow + 1.35 * coast.astype(float)
    tide_amp = equilibrium_rms * shelf_amp * water
    tidal_range = 2.0 * tide_amp

    gy, gx = grid.ops.metric_gradient(coherent)
    forcing_grad = np.hypot(gx, gy)
    constriction = np.exp(-depth / 420.0) * water
    current_raw = forcing_grad * (0.25 + 0.75 * constriction) + 0.22 * tide_amp / max(grid.dy_km, 1.0)
    current = normalize01(smooth_periodic(current_raw, (0.7, 0.9)), robust=True) * water

    intertidal = normalize01(tidal_range * np.exp(-depth / 18.0), robust=True) * water
    amplitudes = [float(c["equilibrium_amplitude_m"]) for c in constituents]
    periods = [float(c.get("period_days", 0.0)) for c in constituents if float(c.get("period_days", 0.0)) > 0]
    beat_days = None
    if len(periods) >= 2:
        freqs = sorted(1.0 / p for p in periods)
        differences = [abs(freqs[j] - freqs[i]) for i in range(len(freqs)) for j in range(i + 1, len(freqs)) if abs(freqs[j] - freqs[i]) > 1.0e-12]
        if differences:
            beat_days = 1.0 / min(differences)

    metadata = {
        "model": "multi-constituent equilibrium tide envelope + shelf/constriction amplification",
        "constituent_count": len(constituents),
        "sum_equilibrium_amplitudes_m": float(sum(amplitudes)),
        "max_equilibrium_constituent_amplitude_m": float(max(amplitudes)),
        "max_screened_tidal_range_m": float(np.max(tidal_range)) if tidal_range.size else 0.0,
        "longest_pairwise_beat_period_days": None if beat_days is None else float(beat_days),
        "limitations": "no global shallow-water PDE, amphidromic points, resonance eigenmodes or wetting/drying solver yet",
    }
    return TideResult(
        tide_amp.astype(np.float32),
        tidal_range.astype(np.float32),
        current.astype(np.float32),
        intertidal.astype(np.float32),
        len(constituents),
        constituents,
        metadata,
    )


__all__ = ["TideResult", "build_tides"]
