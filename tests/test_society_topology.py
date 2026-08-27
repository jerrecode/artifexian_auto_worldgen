import numpy as np

from worldgen.grid import SphereGrid
from worldgen.society import _land_components_wrap


def test_society_land_components_use_full_spherical_topology():
    grid = SphereGrid(64, 32)
    land = np.zeros((32, 64), dtype=bool)

    # One landmass crosses the north pole antipodally. The previous society-only
    # helper merged the east/west seam but treated these two cells as different
    # continents, which could incorrectly force a maritime crossing.
    land[0, 5] = True
    land[0, 37] = True

    labels = _land_components_wrap(grid, land)
    assert labels[0, 5] > 0
    assert labels[0, 5] == labels[0, 37]


def test_society_land_components_still_merge_longitude_seam():
    grid = SphereGrid(64, 32)
    land = np.zeros((32, 64), dtype=bool)
    land[15, 0] = True
    land[15, -1] = True

    labels = _land_components_wrap(grid, land)
    assert labels[15, 0] > 0
    assert labels[15, 0] == labels[15, -1]
