from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any, Mapping

import numpy as np


@dataclass(slots=True, frozen=True)
class FieldMetadata:
    name: str
    dimensions: tuple[str, ...]
    dtype: str
    units: str | None
    role: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dimensions"] = list(self.dimensions)
        return payload


_UNITS: dict[str, str] = {
    "lat": "degrees_north",
    "lon": "degrees_east",
    "elevation_km": "km",
    "ocean_depth_m": "m",
    "temperature_c_monthly": "degC",
    "annual_temperature_c": "degC",
    "precipitation_mm_monthly": "mm/month",
    "annual_precipitation_mm": "mm/year",
    "continentality_index_c": "degC",
    "drainage_area_km2": "km2",
    "runoff_mm_year": "mm/year",
    "cumulative_erosion_m": "m",
    "cumulative_deposition_m": "m",
    "delta_deposition_m": "m",
    "tectonic_uplift_m": "m",
    "meander_migration_m": "m",
    "lightning_flashes_km2_year": "flashes/km2/year",
    "crust_age_myr": "Myr",
}

_CATEGORICAL = {
    "plate_id", "subplate_id", "subplate_parent", "continental_crust", "plate_boundary",
    "subplate_boundary", "intraplate_fault", "convergent", "divergent", "transform",
    "koppen", "continentality_class", "flow_to", "rivers", "stream_order", "lakes",
    "rock_code", "thunderstorm_level", "sandstorm", "duststorm", "sea_ice_max",
    "sea_ice_min", "coral_reef",
}


def infer_field_metadata(name: str, value: np.ndarray, grid_shape: tuple[int, int]) -> FieldMetadata:
    a = np.asarray(value)
    h, w = grid_shape
    if name == "lat":
        dims = ("lat",)
    elif name == "lon":
        dims = ("lon",)
    elif a.ndim == 3 and a.shape[:1] == (12,) and a.shape[1:] == (h, w):
        dims = ("month", "lat", "lon")
    elif a.ndim == 3 and a.shape == (h, w, 3):
        dims = ("lat", "lon", "channel")
    elif a.ndim == 2 and a.shape == (h, w):
        dims = ("lat", "lon")
    elif a.ndim == 2 and a.shape[1:] == (3,):
        entity = "subplate" if "subplate" in name else "plate" if "plate" in name else f"{name}_item"
        dims = (entity, "xyz")
    elif a.ndim == 1 and "subplate" in name:
        dims = ("subplate",)
    elif a.ndim == 1 and "plate" in name:
        dims = ("plate",)
    else:
        dims = tuple(f"{name}_dim{i}" for i in range(a.ndim))
    role = "coordinate" if name in {"lat", "lon"} else "categorical" if name in _CATEGORICAL else "data"
    description = name.replace("_", " ")
    return FieldMetadata(name, dims, str(a.dtype), _UNITS.get(name), role, description)


def build_field_catalog(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    if "lat" not in arrays or "lon" not in arrays:
        raise ValueError("canonical array export must contain lat and lon coordinates")
    grid_shape = (len(arrays["lat"]), len(arrays["lon"]))
    return {
        name: infer_field_metadata(name, np.asarray(value), grid_shape).to_dict()
        for name, value in arrays.items()
    }


def write_field_catalog(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_field_catalog(arrays), indent=2, ensure_ascii=False), encoding="utf-8")
    return target
