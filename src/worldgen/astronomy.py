from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import numpy as np
from .config import AstronomyConfig

G = 6.67430e-11
SIGMA = 5.670374419e-8
M_SUN = 1.98847e30
R_SUN = 6.957e8
L_SUN = 3.828e26
M_EARTH = 5.9722e24
R_EARTH = 6.371e6
AU = 1.495978707e11
G0 = 9.80665


@dataclass(slots=True)
class AstronomyResult:
    star: dict
    planet: dict
    moon: dict
    calendar: dict
    atmosphere: dict
    planetary_system: list[dict]
    stellar_neighborhood: list[dict]
    sky: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _mass_luminosity(m: float) -> float:
    if m < 0.43:
        return 0.23 * m ** 2.3
    if m < 2.0:
        return m ** 4.0
    return 1.5 * m ** 3.5


def _mass_radius_star(m: float) -> float:
    return m ** 0.8 if m <= 1.5 else m ** 0.57


def _kepler_years(a_au: float, m_solar: float) -> float:
    return math.sqrt(a_au ** 3 / m_solar)


def _rocky_radius_earth(m_earth: float, density_g_cm3: float) -> float:
    # Density-specified radius preserves the spreadsheet-like user control.
    earth_density = 5.514
    return (m_earth * earth_density / density_g_cm3) ** (1 / 3)


def _hill_radius_km(a_au: float, e: float, m_planet_earth: float, m_star_solar: float) -> float:
    return a_au * AU / 1000 * (1 - e) * ((m_planet_earth * M_EARTH) / (3 * m_star_solar * M_SUN)) ** (1 / 3)


def _roche_limit_km(radius_earth: float, rho_planet: float, rho_moon: float = 3.34) -> float:
    return 2.44 * radius_earth * R_EARTH / 1000 * (rho_planet / rho_moon) ** (1 / 3)



def _spectral_class_from_temp(t: float) -> str:
    if t >= 30000: return "O"
    if t >= 10000: return "B"
    if t >= 7500: return "A"
    if t >= 6000: return "F"
    if t >= 5200: return "G"
    if t >= 3700: return "K"
    return "M"


def _generate_planetary_system(cfg: AstronomyConfig, home_a: float, star_mass: float, lum: float,
                               rng: np.random.Generator) -> list[dict]:
    """Generate a compact stable-ish orbital architecture around the configured home world.

    This is a fast procedural spacing/Hill-separation generator, not an N-body integrator.
    """
    n = max(1, int(cfg.system_planet_count))
    frost = 4.85 * math.sqrt(max(lum, 1e-8))
    # Geometrically distributed slots around home; home is always included exactly.
    inner = max(0.06, 0.12 * math.sqrt(max(lum, 0.05)))
    outer = max(frost * 5.0, home_a * 3.0)
    raw = np.geomspace(inner, outer, max(n * 3, 12))
    # Score slots by geometric separation from selected slots; initialize with home.
    selected = [home_a]
    while len(selected) < n:
        candidates = [float(x) for x in raw if all(abs(math.log(x / y)) > 0.12 for y in selected)]
        if not candidates: break
        scores = [min(abs(math.log(x / y)) for y in selected) for x in candidates]
        # Bias toward filling both inner and outer system, while adding seeded variation.
        scores = np.asarray(scores) * rng.uniform(.85, 1.15, len(scores))
        selected.append(candidates[int(np.argmax(scores))])
    selected = sorted(selected)[:n]
    # If truncation accidentally removed home, replace nearest slot.
    if all(abs(a - home_a) > 1e-9 for a in selected):
        selected[np.argmin(np.abs(np.asarray(selected) - home_a))] = home_a
        selected.sort()

    masses = []
    for a in selected:
        if abs(a - home_a) < 1e-9:
            mass = cfg.planet_mass_earth
            kind = "home_rocky"
        elif a < frost:
            mass = float(np.exp(rng.uniform(math.log(.08), math.log(4.0))))
            kind = "rocky"
        else:
            mass = float(np.exp(rng.uniform(math.log(5.0), math.log(180.0))))
            kind = "ice_or_gas_giant"
        masses.append((mass, kind))

    result = []
    for i, (a, (mass, kind)) in enumerate(zip(selected, masses)):
        period = _kepler_years(a, star_mass)
        if kind in {"rocky", "home_rocky"}:
            radius = mass ** 0.27
        else:
            radius = min(11.5, 3.0 * mass ** 0.18)
        flux = lum / (a * a)
        teq = 278.5 * flux ** 0.25 * ((1 - cfg.albedo) / 0.7) ** 0.25
        result.append({
            "index": i, "kind": kind, "semimajor_axis_au": float(a),
            "mass_earth": float(mass), "radius_earth_approx": float(radius),
            "orbital_period_earth_years": float(period), "stellar_flux_earth": float(flux),
            "equilibrium_temperature_k_approx": float(teq), "inside_frost_line": bool(a < frost),
            "is_home_world": bool(abs(a - home_a) < 1e-9),
        })
    # Report adjacent mutual-Hill spacing as a stability diagnostic.
    for i in range(len(result) - 1):
        p1, p2 = result[i], result[i + 1]
        a1, a2 = p1["semimajor_axis_au"], p2["semimajor_axis_au"]
        mt = (p1["mass_earth"] + p2["mass_earth"]) * M_EARTH
        rh = ((mt / (3 * star_mass * M_SUN)) ** (1 / 3)) * ((a1 + a2) / 2)
        delta = (a2 - a1) / max(rh, 1e-12)
        p1["mutual_hill_spacing_to_next"] = float(delta)
    if result:
        result[-1]["mutual_hill_spacing_to_next"] = None
    return result


def _generate_stellar_neighborhood(cfg: AstronomyConfig, rng: np.random.Generator) -> list[dict]:
    rmax = max(float(cfg.stellar_neighborhood_radius_ly), 0.1)
    expected = max(0.0, cfg.stellar_density_per_ly3 * 4 * math.pi * rmax ** 3 / 3)
    n = int(rng.poisson(expected))
    if n == 0:
        return []
    # Uniform volume positions.
    dirs = rng.normal(size=(n, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rad = rmax * rng.random(n) ** (1 / 3)
    xyz = dirs * rad[:, None]
    # Fast low-mass-biased present-day stellar mass distribution.
    u = rng.random(n)
    masses = np.where(u < .78, np.exp(rng.uniform(np.log(.09), np.log(.65), n)),
                      np.exp(rng.uniform(np.log(.65), np.log(1.6), n)))
    out = []
    for i in range(n):
        m = float(masses[i]); l = _mass_luminosity(m); rr = _mass_radius_star(m)
        t = 5772 * (l / max(rr ** 2, 1e-9)) ** .25
        dly = float(rad[i]); dpc = max(dly / 3.26156, 1e-6)
        abs_mag = 4.74 - 2.5 * math.log10(max(l, 1e-12))
        app_mag = abs_mag + 5 * math.log10(dpc / 10.0)
        out.append({
            "id": i, "x_ly": float(xyz[i,0]), "y_ly": float(xyz[i,1]), "z_ly": float(xyz[i,2]),
            "distance_ly": dly, "mass_solar": m, "luminosity_solar": float(l),
            "temperature_k_approx": float(t), "spectral_class_approx": _spectral_class_from_temp(t),
            "absolute_bolometric_magnitude_approx": float(abs_mag),
            "apparent_magnitude_approx": float(app_mag), "naked_eye_visible_approx": bool(app_mag <= 6.0),
        })
    out.sort(key=lambda x: x["distance_ly"])
    return out

def build_astronomy(cfg: AstronomyConfig, rng: np.random.Generator) -> AstronomyResult:
    m = cfg.star_mass_solar
    lum = _mass_luminosity(m)
    rstar = _mass_radius_star(m)
    temp = 5772.0 * (lum / (rstar ** 2)) ** 0.25
    lifespan_gyr = 10.0 * m / max(lum, 1e-9)

    hz_inner = math.sqrt(lum / 1.10)
    hz_outer = math.sqrt(lum / 0.53)
    if cfg.semimajor_axis_au is None:
        # Set orbit from target mean T using equilibrium radiation + configurable greenhouse.
        target_k = cfg.target_mean_surface_c + 273.15 - cfg.greenhouse_k
        target_k = max(target_k, 100.0)
        a = math.sqrt(lum * (1 - cfg.albedo) / 0.7) * (278.5 / target_k) ** 2
        a = float(np.clip(a, hz_inner * 1.01, hz_outer * 0.99))
    else:
        a = cfg.semimajor_axis_au

    e = cfg.eccentricity
    year_earth = _kepler_years(a, m)
    peri = a * (1 - e)
    apo = a * (1 + e)
    teq = 278.5 * (lum ** 0.25) / math.sqrt(a) * ((1 - cfg.albedo) / 0.7) ** 0.25
    mean_t_c = teq + cfg.greenhouse_k - 273.15

    rp = _rocky_radius_earth(cfg.planet_mass_earth, cfg.planet_density_g_cm3)
    gravity_g = cfg.planet_mass_earth / rp ** 2
    escape_kms = math.sqrt(2 * G * cfg.planet_mass_earth * M_EARTH / (rp * R_EARTH)) / 1000
    hill = _hill_radius_km(a, e, cfg.planet_mass_earth, m)
    roche = _roche_limit_km(rp, cfg.planet_density_g_cm3)

    moon_a = float(np.clip(cfg.moon_orbit_km, roche * 1.5, hill * 0.45))
    mu = G * (cfg.planet_mass_earth + cfg.moon_mass_earth) * M_EARTH
    moon_sidereal_s = 2 * math.pi * math.sqrt((moon_a * 1000) ** 3 / mu)
    moon_sidereal_days = moon_sidereal_s / (cfg.rotation_hours * 3600)
    year_local_days = year_earth * 365.256 * 24 / cfg.rotation_hours
    # 1/Psyn = 1/Psid - 1/Pyear for prograde moon.
    moon_synodic_days = 1.0 / max(1e-12, 1.0 / moon_sidereal_days - 1.0 / year_local_days)

    # Relative tidal forcing M/r^3, Earth-Moon baseline.
    tide_rel = (cfg.moon_mass_earth / 0.012300) * (384400.0 / moon_a) ** 3
    moon_radius_earth = (cfg.moon_mass_earth * 5.514 / 3.34) ** (1 / 3)
    moon_radius_km = moon_radius_earth * R_EARTH / 1000
    moon_ang_deg = math.degrees(2 * math.atan2(moon_radius_km, moon_a))
    star_ang_deg = math.degrees(2 * math.atan2(rstar * R_SUN / 1000, a * AU / 1000))

    atm = {k: float(v) for k, v in cfg.atmosphere.items()}
    total = sum(atm.values())
    if total <= 0:
        raise ValueError("Atmospheric fractions must sum to a positive number")
    atm = {k: v / total for k, v in atm.items()}
    partial = {k: v * cfg.atmosphere_pressure_bar for k, v in atm.items()}

    # Mean molecular mass in g/mol. Unknown species default to N2-like value rather than crash.
    mw = {"N2": 28.0134, "O2": 31.9988, "Ar": 39.948, "CO2": 44.0095, "H2O": 18.01528,
          "CH4": 16.043, "He": 4.0026, "H2": 2.01588}
    mean_mw = sum(atm[k] * mw.get(k, 28.0) for k in atm)
    rho0 = cfg.atmosphere_pressure_bar * 1e5 * (mean_mw / 1000) / (8.314462618 * max(mean_t_c + 273.15, 150))

    star = {
        "mass_solar": m, "luminosity_solar": lum, "radius_solar": rstar,
        "effective_temperature_k": temp, "main_sequence_lifetime_gyr_approx": lifespan_gyr,
        "habitable_zone_au": [hz_inner, hz_outer],
    }
    planet = {
        "mass_earth": cfg.planet_mass_earth, "radius_earth": rp, "density_g_cm3": cfg.planet_density_g_cm3,
        "surface_gravity_g": gravity_g, "surface_gravity_m_s2": gravity_g * G0,
        "escape_velocity_km_s": escape_kms, "semimajor_axis_au": a, "eccentricity": e,
        "periapsis_au": peri, "apoapsis_au": apo, "orbital_period_earth_years": year_earth,
        "rotation_hours": cfg.rotation_hours, "axial_tilt_deg": cfg.axial_tilt_deg,
        "longitude_periapsis_deg": cfg.longitude_periapsis_deg,
        "equilibrium_temperature_k": teq, "mean_surface_temperature_c_approx": mean_t_c,
        "hill_radius_km": hill, "roche_limit_fluid_km": roche,
        "star_angular_diameter_deg": star_ang_deg,
    }
    moon = {
        "mass_earth": cfg.moon_mass_earth, "orbit_km": moon_a,
        "sidereal_period_local_days": moon_sidereal_days, "synodic_period_local_days": moon_synodic_days,
        "tidal_forcing_relative_earth_moon": tide_rel, "radius_km_approx": moon_radius_km,
        "angular_diameter_deg": moon_ang_deg,
        "can_total_eclipse_star_geometrically": moon_ang_deg >= star_ang_deg,
    }
    calendar = {
        "local_day_hours": cfg.rotation_hours, "local_year_days": year_local_days,
        "synodic_month_days": moon_synodic_days,
        "synodic_months_per_year": year_local_days / moon_synodic_days,
    }
    atmosphere = {
        "surface_pressure_bar": cfg.atmosphere_pressure_bar, "fractions": atm,
        "partial_pressures_bar": partial, "mean_molar_mass_g_mol": mean_mw,
        "surface_density_kg_m3_approx": rho0,
    }
    system = _generate_planetary_system(cfg, a, m, lum, rng)
    neighbors = _generate_stellar_neighborhood(cfg, rng)
    sunlike_mag = -26.74 - 2.5 * math.log10(max(lum / (a * a), 1e-12))
    # Simple full-moon photometric scaling from Earth's full moon; enough for relative worldbuilding.
    full_moon_mag = -12.74 - 2.5 * math.log10(max((cfg.moon_mass_earth / 0.0123) ** (2/3) * (384400 / moon_a) ** 2, 1e-12))
    sky = {
        "home_star_apparent_magnitude_approx": float(sunlike_mag),
        "full_moon_apparent_magnitude_approx": float(full_moon_mag),
        "naked_eye_neighbor_star_count_approx": int(sum(x["naked_eye_visible_approx"] for x in neighbors)),
        "neighbor_catalog_radius_ly": cfg.stellar_neighborhood_radius_ly,
    }
    return AstronomyResult(star, planet, moon, calendar, atmosphere, system, neighbors, sky)
