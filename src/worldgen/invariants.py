from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np

from .drainage import graph_for


@dataclass(slots=True)
class InvariantCheck:
    name: str
    status: str
    value: float | int | str | None
    tolerance: float | str | None
    message: str


@dataclass(slots=True)
class ValidationReport:
    passed: bool
    checks: list[InvariantCheck]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks]}


def _finite_check(name: str, a: np.ndarray) -> InvariantCheck:
    finite = np.isfinite(np.asarray(a, dtype=float))
    bad = int(finite.size - finite.sum())
    return InvariantCheck(
        name=f"finite:{name}",
        status="pass" if bad == 0 else "fail",
        value=bad,
        tolerance=0,
        message="number of non-finite cells",
    )


def validate_world(world: dict[str, Any], *, strict: bool = False) -> ValidationReport:
    """Evaluate numerical/scientific invariants without changing generated state."""
    grid = world["grid"]
    terrain = world["terrain"]
    climate = world["climate"]
    hydro = world["hydrology"]
    weather = world["weather"]
    resources = world["resources"]
    checks: list[InvariantCheck] = []

    for name, a in (
        ("elevation_km", world["ocean"].elevation_km),
        ("annual_temperature_c", climate.annual_temperature_c),
        ("annual_precipitation_mm", climate.annual_precipitation_mm),
        ("drainage_area_km2", hydro.drainage_area_km2),
        ("runoff", hydro.runoff),
    ):
        checks.append(_finite_check(name, a))

    target_land = float(world["config"].tectonics.continental_fraction_target)
    actual_land = float(grid.weighted_fraction(terrain.land))
    land_error = abs(actual_land - target_land)
    checks.append(InvariantCheck(
        "land_fraction_target",
        "pass" if land_error <= 0.035 else "warn",
        actual_land,
        0.035,
        f"absolute error from configured target={target_land:.5f}",
    ))

    if np.any(terrain.land):
        pmean = float(np.average(
            climate.annual_precipitation_mm[terrain.land],
            weights=grid.cell_area_weights[terrain.land],
        ))
    else:
        pmean = 0.0
    ptarget = float(world["config"].climate.precip_scale_mm_year)
    prel = abs(pmean - ptarget) / max(ptarget, 1.0)
    checks.append(InvariantCheck(
        "land_precipitation_calibration",
        "pass" if prel <= 0.03 else "warn",
        pmean,
        "3% relative",
        f"configured area-weighted target={ptarget:.3f} mm/year",
    ))

    graph = graph_for(hydro.flow_to)
    checks.append(InvariantCheck(
        "drainage_graph_acyclic",
        "pass" if graph.unresolved_count == 0 else "fail",
        graph.unresolved_count,
        0,
        "receiver nodes unresolved by topological ordering",
    ))

    if np.any(terrain.ocean):
        ocean_w = grid.cell_area_weights[terrain.ocean]
        max_ice = float(np.average(weather.sea_ice_max[terrain.ocean].astype(float), weights=ocean_w))
        min_ice = float(np.average(weather.sea_ice_min[terrain.ocean].astype(float), weights=ocean_w))
    else:
        max_ice = min_ice = 0.0
    checks.append(InvariantCheck(
        "sea_ice_seasonal_order",
        "pass" if min_ice <= max_ice + 1e-12 else "fail",
        f"min={min_ice:.6f}, max={max_ice:.6f}",
        "minimum <= maximum",
        "area-weighted ocean fractions",
    ))

    submerged_mismatch = 0
    for d in resources.deposits:
        if "submerged" not in d or "accessible_preindustrial" not in d:
            submerged_mismatch += 1
        elif bool(d["submerged"]) == bool(d["accessible_preindustrial"]):
            submerged_mismatch += 1
    checks.append(InvariantCheck(
        "resource_accessibility_flags",
        "pass" if submerged_mismatch == 0 else "fail",
        submerged_mismatch,
        0,
        "deposits with missing/inconsistent submerged vs preindustrial-accessible flags",
    ))

    passed = not any(c.status == "fail" for c in checks)
    if strict and not passed:
        failed = ", ".join(c.name for c in checks if c.status == "fail")
        raise RuntimeError(f"world invariant validation failed: {failed}")
    return ValidationReport(passed, checks)


def write_validation_report(path: str | Path, world: dict[str, Any], *, strict: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    report = validate_world(world, strict=strict)
    target.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return target
