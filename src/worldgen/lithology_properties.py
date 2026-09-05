from __future__ import annotations

"""Central material-response table shared by hydrology and procedural geomorphology."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LithologyProperties:
    code: int
    name: str
    mechanical_erodibility: float
    runoff_multiplier: float
    infiltration_fraction: float
    soil_capacity_mm: float
    chemical_weatherability: float
    frost_susceptibility: float
    glacial_abrasion_susceptibility: float
    cohesion: float


LITHOLOGY_PROPERTIES: tuple[LithologyProperties, ...] = (
    LithologyProperties(0, "unconsolidated_sediment", 1.75, 0.90, 0.62, 190.0, 1.35, 1.20, 1.10, 0.30),
    LithologyProperties(1, "sandstone_clastic",       1.20, 0.88, 0.54, 145.0, 1.05, 1.05, 0.95, 0.55),
    LithologyProperties(2, "carbonate",               0.82, 0.68, 0.80, 245.0, 1.55, 0.85, 0.75, 0.60),
    LithologyProperties(3, "granite",                 0.46, 1.08, 0.47, 115.0, 0.62, 0.72, 0.80, 0.88),
    LithologyProperties(4, "metamorphic",             0.36, 1.12, 0.40, 105.0, 0.52, 0.62, 0.72, 0.92),
    LithologyProperties(5, "basalt_mafic",            0.55, 0.96, 0.52, 135.0, 0.78, 0.82, 0.92, 0.84),
    LithologyProperties(6, "andesite",                0.52, 1.00, 0.49, 125.0, 0.73, 0.78, 0.88, 0.86),
    LithologyProperties(7, "rhyolite_felsic",         0.58, 0.96, 0.58, 155.0, 0.60, 0.92, 0.76, 0.80),
    LithologyProperties(8, "ultramafic_greenstone",   0.40, 1.05, 0.44, 110.0, 0.45, 0.70, 0.84, 0.90),
)


def _array(attribute: str) -> np.ndarray:
    return np.asarray([float(getattr(item, attribute)) for item in LITHOLOGY_PROPERTIES], dtype=float)


MECHANICAL_ERODIBILITY = _array("mechanical_erodibility")
RUNOFF_MULTIPLIER = _array("runoff_multiplier")
INFILTRATION_FRACTION = _array("infiltration_fraction")
SOIL_CAPACITY_MM = _array("soil_capacity_mm")
CHEMICAL_WEATHERABILITY = _array("chemical_weatherability")
FROST_SUSCEPTIBILITY = _array("frost_susceptibility")
GLACIAL_ABRASION_SUSCEPTIBILITY = _array("glacial_abrasion_susceptibility")
COHESION = _array("cohesion")


def properties_for_codes(codes: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.clip(np.asarray(codes, dtype=np.int64), 0, len(LITHOLOGY_PROPERTIES) - 1)
    return {
        "mechanical_erodibility": MECHANICAL_ERODIBILITY[idx],
        "runoff_multiplier": RUNOFF_MULTIPLIER[idx],
        "infiltration_fraction": INFILTRATION_FRACTION[idx],
        "soil_capacity_mm": SOIL_CAPACITY_MM[idx],
        "chemical_weatherability": CHEMICAL_WEATHERABILITY[idx],
        "frost_susceptibility": FROST_SUSCEPTIBILITY[idx],
        "glacial_abrasion_susceptibility": GLACIAL_ABRASION_SUSCEPTIBILITY[idx],
        "cohesion": COHESION[idx],
    }


__all__ = [
    "CHEMICAL_WEATHERABILITY",
    "COHESION",
    "FROST_SUSCEPTIBILITY",
    "GLACIAL_ABRASION_SUSCEPTIBILITY",
    "INFILTRATION_FRACTION",
    "LITHOLOGY_PROPERTIES",
    "LithologyProperties",
    "MECHANICAL_ERODIBILITY",
    "RUNOFF_MULTIPLIER",
    "SOIL_CAPACITY_MM",
    "properties_for_codes",
]
