from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np

from .config import AstronomyConfig
from .planetary_physics import (
    SPECIES,
    atmosphere_diagnostics,
    canonical_species,
    composition_greenhouse_temperature_k,
    geological_activity_regime,
    phase_at,
    select_active_condensible,
    species_metadata,
    tidal_heating_flux_w_m2,
)

G = 6.67430e-11
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
    moons: list[dict]
    primary: dict
    interior: dict
    volatile_chemistry: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _mass_luminosity(m: float) -> float:
    if m < 0.43: return 0.23 * m**2.3
    if m < 2.0: return m**4.0
    return 1.5 * m**3.5


def _mass_radius_star(m: float) -> float:
    return m**0.8 if m <= 1.5 else m**0.57


def _kepler_years(a_au: float, m_solar: float) -> float:
    return math.sqrt(a_au**3 / m_solar)


def _rocky_radius_earth(m_earth: float, density_g_cm3: float) -> float:
    return (m_earth * 5.514 / density_g_cm3) ** (1 / 3)


def _hill_radius_km(a_au: float, e: float, m_planet_earth: float, m_star_solar: float) -> float:
    return a_au * AU / 1000 * (1 - e) * ((m_planet_earth * M_EARTH) / (3 * m_star_solar * M_SUN)) ** (1 / 3)


def _satellite_hill_radius_km(orbit_km: float, e: float, satellite_mass_earth: float, primary_mass_earth: float) -> float:
    return orbit_km * (1 - e) * (satellite_mass_earth / (3.0 * primary_mass_earth)) ** (1 / 3)


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


def _bulk_planet_class(mass_earth: float, density_g_cm3: float) -> str:
    if mass_earth < 0.15: return "dwarf_or_subterrestrial"
    if mass_earth < 0.8: return "sub_earth"
    if mass_earth <= 1.5: return "earth_mass"
    if mass_earth <= 10.0 and density_g_cm3 >= 3.5: return "super_earth"
    if mass_earth <= 20.0 and density_g_cm3 < 3.5: return "volatile_rich_super_earth_or_mini_neptune"
    if density_g_cm3 < 2.5: return "ice_or_gas_rich"
    return "massive_rocky_candidate"


def _generate_planetary_system(cfg: AstronomyConfig, home_a: float, star_mass: float, lum: float,
                               rng: np.random.Generator) -> list[dict]:
    n = max(1, int(cfg.system_planet_count)); frost = 4.85 * math.sqrt(max(lum, 1e-8))
    inner = max(0.06, 0.12 * math.sqrt(max(lum, 0.05))); outer = max(frost * 5.0, home_a * 3.0)
    raw = np.geomspace(inner, outer, max(n * 3, 12)); selected = [home_a]
    while len(selected) < n:
        candidates = [float(x) for x in raw if all(abs(math.log(x / y)) > 0.12 for y in selected)]
        if not candidates: break
        scores = np.asarray([min(abs(math.log(x / y)) for y in selected) for x in candidates]) * rng.uniform(.85, 1.15, len(candidates))
        selected.append(candidates[int(np.argmax(scores))])
    selected = sorted(selected)[:n]
    if all(abs(a - home_a) > 1e-9 for a in selected):
        selected[np.argmin(np.abs(np.asarray(selected) - home_a))] = home_a; selected.sort()
    masses = []
    for a in selected:
        if abs(a - home_a) < 1e-9:
            mass = cfg.planet_mass_earth if cfg.body_role == "planet" else float(cfg.parent_body_mass_earth or 317.8)
            kind = "home_world" if cfg.body_role == "planet" else "home_parent_planet"
        elif a < frost:
            mass = float(np.exp(rng.uniform(math.log(.08), math.log(4.0)))); kind = "rocky"
        else:
            mass = float(np.exp(rng.uniform(math.log(5.0), math.log(180.0)))); kind = "ice_or_gas_giant"
        masses.append((mass, kind))
    result = []
    for i, (a, (mass, kind)) in enumerate(zip(selected, masses)):
        radius = mass**0.27 if kind in {"rocky", "home_world"} else min(11.5, 3.0 * mass**0.18)
        flux = lum / a**2
        result.append({"index": i, "kind": kind, "semimajor_axis_au": float(a), "mass_earth": float(mass),
                       "radius_earth_approx": float(radius), "orbital_period_earth_years": float(_kepler_years(a, star_mass)),
                       "stellar_flux_earth": float(flux), "equilibrium_temperature_k_approx": float(278.5 * flux**0.25 * ((1-cfg.albedo)/0.7)**0.25),
                       "inside_frost_line": bool(a < frost), "is_home_or_parent_orbit": bool(abs(a-home_a)<1e-9)})
    for i in range(len(result)-1):
        p1,p2=result[i],result[i+1]; a1,a2=p1["semimajor_axis_au"],p2["semimajor_axis_au"]
        mt=(p1["mass_earth"]+p2["mass_earth"])*M_EARTH; rh=((mt/(3*star_mass*M_SUN))**(1/3))*((a1+a2)/2)
        p1["mutual_hill_spacing_to_next"] = float((a2-a1)/max(rh,1e-12))
    if result: result[-1]["mutual_hill_spacing_to_next"] = None
    return result


def _generate_stellar_neighborhood(cfg: AstronomyConfig, rng: np.random.Generator) -> list[dict]:
    rmax=max(float(cfg.stellar_neighborhood_radius_ly),0.1); expected=max(0.0,cfg.stellar_density_per_ly3*4*math.pi*rmax**3/3); n=int(rng.poisson(expected))
    if n == 0: return []
    dirs=rng.normal(size=(n,3)); dirs/=np.linalg.norm(dirs,axis=1,keepdims=True); rad=rmax*rng.random(n)**(1/3); xyz=dirs*rad[:,None]
    u=rng.random(n); masses=np.where(u<.78,np.exp(rng.uniform(np.log(.09),np.log(.65),n)),np.exp(rng.uniform(np.log(.65),np.log(1.6),n)))
    out=[]
    for i in range(n):
        m=float(masses[i]); l=_mass_luminosity(m); rr=_mass_radius_star(m); t=5772*(l/max(rr**2,1e-9))**.25
        dly=float(rad[i]); dpc=max(dly/3.26156,1e-6); abs_mag=4.74-2.5*math.log10(max(l,1e-12)); app_mag=abs_mag+5*math.log10(dpc/10.0)
        out.append({"id":i,"x_ly":float(xyz[i,0]),"y_ly":float(xyz[i,1]),"z_ly":float(xyz[i,2]),"distance_ly":dly,
                    "mass_solar":m,"luminosity_solar":float(l),"temperature_k_approx":float(t),"spectral_class_approx":_spectral_class_from_temp(t),
                    "absolute_bolometric_magnitude_approx":float(abs_mag),"apparent_magnitude_approx":float(app_mag),"naked_eye_visible_approx":bool(app_mag<=6.0)})
    out.sort(key=lambda x:x["distance_ly"]); return out


def _moon_records(cfg: AstronomyConfig, *, home_radius_earth: float, home_density: float, home_hill_km: float,
                  star_angular_diameter_deg: float, year_local_days: float) -> list[dict]:
    source = cfg.moons if cfg.moons else [{"name":"Moon","mass_earth":cfg.moon_mass_earth,"orbit_km":cfg.moon_orbit_km}]
    roche = _roche_limit_km(home_radius_earth, home_density)
    result=[]
    for i, raw in enumerate(source):
        mass=float(raw["mass_earth"]); density=float(raw.get("density_g_cm3",3.34)); requested=float(raw["orbit_km"])
        orbit=float(np.clip(requested, roche*1.05, max(roche*1.06,home_hill_km*0.45)))
        mu=G*(cfg.planet_mass_earth+mass)*M_EARTH; sidereal_s=2*math.pi*math.sqrt((orbit*1000)**3/mu)
        sidereal_local=sidereal_s/(cfg.rotation_hours*3600); syn=1.0/max(1e-12,1.0/sidereal_local-1.0/year_local_days)
        radius_earth=(mass*5.514/density)**(1/3); radius_km=radius_earth*R_EARTH/1000
        angular=math.degrees(2*math.atan2(radius_km,orbit)); ecc=float(raw.get("eccentricity",0.0))
        k2=float(raw.get("love_number_k2",0.10)); q=float(raw.get("quality_factor_q",100.0))
        heat=tidal_heating_flux_w_m2(satellite_radius_earth=radius_earth,primary_mass_earth=cfg.planet_mass_earth,
                                     orbit_km=orbit,eccentricity=ecc,love_number_k2=k2,quality_factor_q=q)
        result.append({"index":i,"name":str(raw.get("name",f"Moon {i+1}")),"mass_earth":mass,"density_g_cm3":density,
                       "radius_earth_approx":radius_earth,"radius_km_approx":radius_km,"requested_orbit_km":requested,"orbit_km":orbit,
                       "eccentricity":ecc,"sidereal_period_local_days":sidereal_local,"synodic_period_local_days":syn,
                       "tidal_forcing_relative_earth_moon":(mass/0.0123)*(384400.0/orbit)**3,
                       "tidal_heating_flux_w_m2_approx":heat,"love_number_k2":k2,"quality_factor_q":q,
                       "angular_diameter_deg":angular,"can_total_eclipse_star_geometrically":angular>=star_angular_diameter_deg})
    return result


def build_astronomy(cfg: AstronomyConfig, rng: np.random.Generator) -> AstronomyResult:
    m=cfg.star_mass_solar; lum=_mass_luminosity(m); rstar=_mass_radius_star(m); temp=5772.0*(lum/rstar**2)**0.25; lifespan_gyr=10.0*m/max(lum,1e-9)
    hz_inner=math.sqrt(lum/1.10); hz_outer=math.sqrt(lum/0.53)

    # For composition greenhouse worlds solve the target-orbit relation through the
    # grey optical-depth factor rather than subtracting a fixed Kelvin offset.
    if cfg.semimajor_axis_au is None:
        target_k=max(cfg.target_mean_surface_c+273.15,100.0)
        if cfg.greenhouse_model == "composition":
            surface_unit,_=composition_greenhouse_temperature_k(1.0,cfg.atmosphere,cfg.atmosphere_pressure_bar)
            target_teq=target_k/max(surface_unit,1e-6)
        else:
            target_teq=max(target_k-cfg.greenhouse_k,100.0)
        a=math.sqrt(lum*(1-cfg.albedo)/0.7)*(278.5/target_teq)**2
        a=float(np.clip(a,hz_inner*1.01,hz_outer*0.99))
    else:
        a=float(cfg.semimajor_axis_au)

    e=float(cfg.eccentricity); year_earth=_kepler_years(a,m); peri=a*(1-e); apo=a*(1+e)
    teq=278.5*(lum**0.25)/math.sqrt(a)*((1-cfg.albedo)/0.7)**0.25
    rp=_rocky_radius_earth(cfg.planet_mass_earth,cfg.planet_density_g_cm3); gravity_g=cfg.planet_mass_earth/rp**2; gravity=gravity_g*G0
    escape_kms=math.sqrt(2*G*cfg.planet_mass_earth*M_EARTH/(rp*R_EARTH))/1000

    if cfg.greenhouse_model == "composition":
        surface_k, greenhouse_terms = composition_greenhouse_temperature_k(teq,cfg.atmosphere,cfg.atmosphere_pressure_bar)
        greenhouse_k=surface_k-teq
    else:
        surface_k=teq+cfg.greenhouse_k; greenhouse_k=cfg.greenhouse_k; greenhouse_terms={"model":"legacy_fixed_offset","total":None}
    mean_t_c=surface_k-273.15

    if cfg.body_role == "moon":
        parent_mass=float(cfg.parent_body_mass_earth); parent_orbit=float(cfg.parent_orbit_km)
        home_hill=_satellite_hill_radius_km(parent_orbit,cfg.parent_orbit_eccentricity,cfg.planet_mass_earth,parent_mass)
        primary={"type":"planet","mass_earth":parent_mass,"radius_earth":float(cfg.parent_body_radius_earth),"home_orbit_km":parent_orbit,
                 "home_orbit_eccentricity":float(cfg.parent_orbit_eccentricity),"stellar_semimajor_axis_au":a}
        home_tidal=tidal_heating_flux_w_m2(satellite_radius_earth=rp,primary_mass_earth=parent_mass,orbit_km=parent_orbit,
                                           eccentricity=cfg.parent_orbit_eccentricity,love_number_k2=cfg.tidal_love_number_k2,quality_factor_q=cfg.tidal_quality_factor_q)
        hill=home_hill; roche=_roche_limit_km(rp,cfg.planet_density_g_cm3,float(cfg.parent_body_mass_earth)/max(float(cfg.parent_body_radius_earth)**3,1e-9)*5.514)
    else:
        hill=_hill_radius_km(a,e,cfg.planet_mass_earth,m); roche=_roche_limit_km(rp,cfg.planet_density_g_cm3); home_tidal=0.0
        primary={"type":"star","mass_solar":m,"stellar_semimajor_axis_au":a}

    year_local_days=year_earth*365.256*24/cfg.rotation_hours
    star_ang_deg=math.degrees(2*math.atan2(rstar*R_SUN/1000,a*AU/1000))
    moons=_moon_records(cfg,home_radius_earth=rp,home_density=cfg.planet_density_g_cm3,home_hill_km=hill,
                        star_angular_diameter_deg=star_ang_deg,year_local_days=year_local_days)
    moon=moons[0] if moons else {}

    atmosphere=atmosphere_diagnostics(composition=cfg.atmosphere,pressure_bar=cfg.atmosphere_pressure_bar,temperature_k=surface_k,gravity_m_s2=gravity)
    scale=float(atmosphere["scale_height_km_approx"]); thickness=cfg.atmosphere_thickness_km
    if thickness is None:
        thickness=scale*math.log(cfg.atmosphere_pressure_bar/cfg.atmosphere_top_pressure_bar)
        thickness_source="hydrostatic_scale_height"
    else:
        thickness_source="configured_override"
    atmosphere.update({"effective_thickness_km_approx":float(thickness),"thickness_source":thickness_source,
                       "top_pressure_bar":float(cfg.atmosphere_top_pressure_bar),"greenhouse_model":cfg.greenhouse_model,
                       "greenhouse_temperature_increment_k_approx":float(greenhouse_k),"greenhouse_optical_depth_terms":greenhouse_terms,
                       "thermodynamics_backend":cfg.thermodynamics_backend})

    active=select_active_condensible(atmosphere["fractions"],cfg.surface_volatiles,surface_k,cfg.atmosphere_pressure_bar,requested=cfg.surface_condensible)
    phases={}
    for name,inventory in cfg.surface_volatiles.items():
        key=canonical_species(name); phases[key]={"inventory_weight":float(inventory),"surface_phase_at_global_mean":phase_at(key,surface_k,cfg.atmosphere_pressure_bar,backend=cfg.thermodynamics_backend)}
    for key,partial in atmosphere["partial_pressures_bar"].items():
        if key in SPECIES:
            phases.setdefault(key,{})["atmospheric_phase_at_global_mean_partial_pressure"] = phase_at(key,surface_k,float(partial),backend=cfg.thermodynamics_backend)

    total_heat=float(cfg.radiogenic_heat_flux_w_m2)+home_tidal
    interior={"radiogenic_heat_flux_w_m2":float(cfg.radiogenic_heat_flux_w_m2),"tidal_heating_flux_w_m2":float(home_tidal),
              "total_internal_heat_flux_w_m2_approx":total_heat,"geological_activity_regime":geological_activity_regime(total_heat),
              "tidal_model":"synchronous small-eccentricity equilibrium tide; excludes obliquity/libration/resonance evolution"}
    volatile={"active_condensible":active,"surface_and_atmospheric_phases":phases,"supported_species":species_metadata()}

    star={"mass_solar":m,"luminosity_solar":lum,"radius_solar":rstar,"effective_temperature_k":temp,
          "main_sequence_lifetime_gyr_approx":lifespan_gyr,"habitable_zone_au":[hz_inner,hz_outer]}
    planet={"body_role":cfg.body_role,"bulk_class":_bulk_planet_class(cfg.planet_mass_earth,cfg.planet_density_g_cm3),
            "mass_earth":cfg.planet_mass_earth,"radius_earth":rp,"density_g_cm3":cfg.planet_density_g_cm3,
            "surface_gravity_g":gravity_g,"surface_gravity_m_s2":gravity,"escape_velocity_km_s":escape_kms,
            "semimajor_axis_au":a,"eccentricity":e,"periapsis_au":peri,"apoapsis_au":apo,"orbital_period_earth_years":year_earth,
            "rotation_hours":cfg.rotation_hours,"axial_tilt_deg":cfg.axial_tilt_deg,"longitude_periapsis_deg":cfg.longitude_periapsis_deg,
            "equilibrium_temperature_k":teq,"mean_surface_temperature_c_approx":mean_t_c,"greenhouse_increment_k_approx":greenhouse_k,
            "hill_radius_km":hill,"roche_limit_fluid_km":roche,"star_angular_diameter_deg":star_ang_deg}
    calendar={"local_day_hours":cfg.rotation_hours,"local_year_days":year_local_days,
              "moon_synodic_periods_local_days":[float(x["synodic_period_local_days"]) for x in moons],
              "synodic_month_days":float(moon.get("synodic_period_local_days",year_local_days)),
              "synodic_months_per_year":year_local_days/max(float(moon.get("synodic_period_local_days",year_local_days)),1e-12)}
    system=_generate_planetary_system(cfg,a,m,lum,rng); neighbors=_generate_stellar_neighborhood(cfg,rng)
    sunlike_mag=-26.74-2.5*math.log10(max(lum/a**2,1e-12)); moon_mags=[]
    for x in moons:
        moon_mags.append(-12.74-2.5*math.log10(max((x["mass_earth"]/0.0123)**(2/3)*(384400/x["orbit_km"])**2,1e-12)))
    sky={"home_star_apparent_magnitude_approx":float(sunlike_mag),"moon_full_apparent_magnitudes_approx":[float(x) for x in moon_mags],
         "full_moon_apparent_magnitude_approx":float(moon_mags[0]) if moon_mags else None,
         "naked_eye_neighbor_star_count_approx":int(sum(x["naked_eye_visible_approx"] for x in neighbors)),
         "neighbor_catalog_radius_ly":cfg.stellar_neighborhood_radius_ly}
    return AstronomyResult(star,planet,moon,calendar,atmosphere,system,neighbors,sky,moons,primary,interior,volatile)
