from __future__ import annotations

"""Canonical spherical raster topology and vector calculus.

Low-level interpolation/filtering primitives live in :mod:`worldgen.topology_base`.
This module owns the public ``SphericalRasterOps`` contract and adds the metric and
physical operators that must use the project's explicit east/south vector
convention. Keeping the public class here lets numerical semantics evolve without
silently changing the low-level sampling primitives used by older checkpoints.
"""

from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from .topology_base import (
    _map_spherical_lattice_indices,
    apply_bilinear_sampler,
    prepare_spherical_bilinear_sampler,
    spherical_gaussian_filter,
    SphericalRasterOps as _BaseSphericalRasterOps,
)


class SphericalRasterOps(_BaseSphericalRasterOps):
    """Canonical topology/metric operations for an equirectangular sphere.

    Coordinate conventions:

    * ``u`` / x are eastward-positive;
    * ``v`` / raster-y are southward-positive;
    * latitude ``phi`` is northward-positive;
    * :meth:`curl` returns outward-radial physical vorticity.

    The scalar topology remains periodic in longitude and antipodally reflected
    through the poles. Vector divergence/curl use flux-form spherical formulas,
    including the curvature terms omitted by Cartesian raster derivatives.
    """

    def metric_gradient(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return scalar ``(d/dsouth, d/deast)`` in units per kilometre."""
        a = np.asarray(values, dtype=np.float64)
        if a.shape[-2:] != self.shape:
            raise ValueError(f"last two dimensions must be {self.shape}, got {a.shape}")
        north = self.shift(a, -1, 0)
        south = self.shift(a, 1, 0)
        west = self.shift(a, 0, -1)
        east = self.shift(a, 0, 1)
        gy = (south - north) / max(2.0 * float(self.grid.dy_km), 1e-12)
        cosphi = np.cos(np.deg2rad(self.grid.lat))
        dx = float(self.grid.radius_km) * float(self.grid.dlon_rad) * np.maximum(cosphi, 1e-12)
        gx = (east - west) / np.maximum(2.0 * dx, 1e-12)
        return gy, gx

    def _meridional_derivative(self, values: np.ndarray) -> np.ndarray:
        """d/dsouth with second-order one-sided derivatives in polar rows.

        The latitude-longitude basis is singular at the mathematical poles. Pixel
        centres never sit exactly on a pole, but reflecting a metric-weighted vector
        flux through the pole and then dividing by ``cos(phi)`` causes severe
        cancellation error. A second-order one-sided flux derivative in the first
        and last latitude bands is substantially more accurate and corresponds to a
        finite-volume treatment of the polar cap.
        """
        a = np.asarray(values, dtype=np.float64)
        if a.shape[-2:] != self.shape:
            raise ValueError(f"last two dimensions must be {self.shape}, got {a.shape}")
        h = int(self.grid.height)
        dy = max(float(self.grid.dy_km), 1e-12)
        out = np.empty_like(a, dtype=np.float64)
        if h == 1:
            out[...] = 0.0
        elif h == 2:
            delta = (a[..., 1, :] - a[..., 0, :]) / dy
            out[..., 0, :] = delta
            out[..., 1, :] = delta
        else:
            out[..., 1:-1, :] = (a[..., 2:, :] - a[..., :-2, :]) / (2.0 * dy)
            out[..., 0, :] = (
                -3.0 * a[..., 0, :] + 4.0 * a[..., 1, :] - a[..., 2, :]
            ) / (2.0 * dy)
            out[..., -1, :] = (
                3.0 * a[..., -1, :] - 4.0 * a[..., -2, :] + a[..., -3, :]
            ) / (2.0 * dy)
        return out

    def _zonal_derivative(self, values: np.ndarray) -> np.ndarray:
        a = np.asarray(values, dtype=np.float64)
        if a.shape[-2:] != self.shape:
            raise ValueError(f"last two dimensions must be {self.shape}, got {a.shape}")
        cosphi = np.cos(np.deg2rad(self.grid.lat))
        dx = float(self.grid.radius_km) * float(self.grid.dlon_rad) * np.maximum(cosphi, 1e-12)
        return (
            np.roll(a, -1, axis=-1) - np.roll(a, 1, axis=-1)
        ) / np.maximum(2.0 * dx, 1e-12)

    def divergence(self, u_east: np.ndarray, v_south: np.ndarray) -> np.ndarray:
        r"""Physical horizontal divergence on the sphere, in inverse kilometres.

        For eastward ``u``, southward ``v``, latitude :math:`\phi`, longitude
        :math:`\lambda`, and colatitude :math:`\theta=\pi/2-\phi`::

            div = 1/(R cos(phi)) * d(u)/d(lambda)
                + 1/(R cos(phi)) * d(v cos(phi))/d(theta)

        The second term contains the meridional curvature contribution that is
        missing from ``du/dx + dv/dy`` on a Cartesian raster.
        """
        u = np.asarray(u_east, dtype=np.float64)
        v = np.asarray(v_south, dtype=np.float64)
        if u.shape != v.shape or u.shape[-2:] != self.shape:
            raise ValueError(
                "u_east and v_south must have identical grid-shaped trailing dimensions"
            )
        cosphi = np.cos(np.deg2rad(self.grid.lat))
        return self._zonal_derivative(u) + self._meridional_derivative(
            v * cosphi
        ) / np.maximum(cosphi, 1e-12)

    def curl(self, u_east: np.ndarray, v_south: np.ndarray) -> np.ndarray:
        r"""Outward-radial curl/vorticity on the sphere, in inverse kilometres.

        With southward-positive meridional velocity::

            curl_r = 1/(R cos(phi)) * d(u cos(phi))/d(theta)
                   - 1/(R cos(phi)) * d(v)/d(lambda)

        This returns positive vorticity for solid-body eastward rotation in the
        northern hemisphere, matching the conventional outward-radial orientation.
        """
        u = np.asarray(u_east, dtype=np.float64)
        v = np.asarray(v_south, dtype=np.float64)
        if u.shape != v.shape or u.shape[-2:] != self.shape:
            raise ValueError(
                "u_east and v_south must have identical grid-shaped trailing dimensions"
            )
        cosphi = np.cos(np.deg2rad(self.grid.lat))
        return self._meridional_derivative(
            u * cosphi
        ) / np.maximum(cosphi, 1e-12) - self._zonal_derivative(v)

    def binary_opening(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        return self.binary_dilation(self.binary_erosion(mask, iterations), iterations)

    def binary_closing(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        return self.binary_erosion(self.binary_dilation(mask, iterations), iterations)

    def grey_erosion(self, values: np.ndarray, iterations: int = 1) -> np.ndarray:
        out = np.asarray(values).copy()
        if out.shape[-2:] != self.shape:
            raise ValueError(f"last two dimensions must be {self.shape}, got {out.shape}")
        for _ in range(max(0, int(iterations))):
            src = out
            eroded = src.copy()
            for dy, dx in self.neighbors8():
                eroded = np.minimum(eroded, self.shift(src, dy, dx))
            out = eroded
        return out

    def local_reduction(
        self,
        values: np.ndarray,
        *,
        radius_cells: int = 1,
        reduction: Literal["min", "max", "mean"] = "mean",
    ) -> np.ndarray:
        """Apply a square cell-neighborhood reduction with spherical boundaries."""
        a = np.asarray(values)
        if a.shape[-2:] != self.shape:
            raise ValueError(f"last two dimensions must be {self.shape}, got {a.shape}")
        radius = max(0, int(radius_cells))
        if radius == 0:
            return a.copy()
        fields = [
            self.shift(a, dy, dx)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]
        if reduction == "min":
            return np.minimum.reduce(fields)
        if reduction == "max":
            return np.maximum.reduce(fields)
        if reduction == "mean":
            acc = np.zeros_like(a, dtype=np.float64)
            for field in fields:
                acc += field
            return acc / float(len(fields))
        raise ValueError(f"unsupported reduction: {reduction}")

    def binary_dilation_km(self, mask: np.ndarray, radius_km: float) -> np.ndarray:
        """Dilate a region by a great-circle distance rather than raster cells."""
        m = np.asarray(mask, dtype=bool)
        radius = max(0.0, float(radius_km))
        if radius == 0.0 or not np.any(m) or np.all(m):
            return m.copy()
        return geodesic_distance_to(m, self.grid) <= radius

    def binary_erosion_km(self, mask: np.ndarray, radius_km: float) -> np.ndarray:
        """Erode a region by a great-circle distance rather than raster cells."""
        m = np.asarray(mask, dtype=bool)
        radius = max(0.0, float(radius_km))
        if radius == 0.0 or not np.any(m) or np.all(m):
            return m.copy()
        return m & (geodesic_distance_to(~m, self.grid) > radius)

    def binary_opening_km(self, mask: np.ndarray, radius_km: float) -> np.ndarray:
        return self.binary_dilation_km(
            self.binary_erosion_km(mask, radius_km), radius_km
        )

    def binary_closing_km(self, mask: np.ndarray, radius_km: float) -> np.ndarray:
        return self.binary_erosion_km(
            self.binary_dilation_km(mask, radius_km), radius_km
        )


def geodesic_distance_to(
    mask: np.ndarray, grid: "object", *, chunk: int = 262_144
) -> np.ndarray:
    """Great-circle distance in km to the nearest True-region boundary."""
    m = np.asarray(mask, dtype=bool)
    if m.shape != (grid.height, grid.width):
        raise ValueError("mask shape must match grid")
    if not np.any(m):
        return np.full(m.shape, np.inf, dtype=np.float64)
    if np.all(m):
        return np.zeros(m.shape, dtype=np.float64)

    ops = SphericalRasterOps(grid)
    boundary = ops.boundary(m)
    points = np.asarray(grid.xyz[boundary], dtype=np.float64)
    if points.size == 0:
        points = np.asarray(grid.xyz[m], dtype=np.float64)
    tree = cKDTree(points)
    flat_xyz = np.asarray(grid.xyz, dtype=np.float64).reshape(-1, 3)
    out = np.empty(len(flat_xyz), dtype=np.float64)
    step = max(1, int(chunk))
    for start in range(0, len(flat_xyz), step):
        stop = min(len(flat_xyz), start + step)
        chord, _ = tree.query(flat_xyz[start:stop], k=1, workers=1)
        chord = np.clip(chord, 0.0, 2.0)
        out[start:stop] = 2.0 * float(grid.radius_km) * np.arcsin(0.5 * chord)
    out = out.reshape(m.shape)
    out[m] = 0.0
    return out


__all__ = [
    "SphericalRasterOps",
    "prepare_spherical_bilinear_sampler",
    "apply_bilinear_sampler",
    "spherical_gaussian_filter",
    "geodesic_distance_to",
    "_map_spherical_lattice_indices",
]
