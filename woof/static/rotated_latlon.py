"""Rotated latitude/longitude lattices (GRIB2 GDT 3.1) for declared sources.

ECCC publishes HRDPS (2.5 km) and RDPS (10 km) on a regular lattice in a
ROTATED spherical frame.  The mapped route treats such a source like a
declared Lambert source: its canonical axes are distances along the
lattice in ``axis_unit_m`` units (index times the rotated-equator spacing),
and the one thing a consumer needs is the map from a target's geographic
``(lat, lon)`` to the lattice's one-based ``(x, y)`` grid coordinates --
:meth:`RotatedLatLonGrid.latlon_to_ij`, the same interface
:class:`woof.static.lambert.LambertGrid` offers, so
:func:`woof.ingest.horiz.source_coordinate_transform` pairs either family
without knowing which it holds.

The transform is the one the Rust GRIB2 decoder places GDT 3.1 points with
(``grib-core`` ``grib2::grid::rotated_to_geographic``) and the Rust mapped
engine rotates winds with (``mapped-engine/src/rotated.rs``): the rotated
frame's north pole sits at geographic latitude ``-south_pole_lat`` on
meridian ``south_pole_lon + 180``; the rotated origin sits at
``(90 + south_pole_lat, south_pole_lon)``; the angle of rotation is
subtracted from the rotated longitude.  Setup-time NumPy, vectorized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RotatedLatLonGrid:
    """A declared rotated lattice: first point and increments in rotated
    degrees, the southern pole of the rotated frame in geographic degrees."""

    lat1: float
    lon1: float
    dlat_deg: float
    dlon_deg: float
    nx: int
    ny: int
    south_pole_lat: float
    south_pole_lon: float
    rotation_angle: float = 0.0

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object]) -> "RotatedLatLonGrid":
        return cls(
            lat1=float(parameters["lat1"]),
            lon1=float(parameters["lon1"]),
            dlat_deg=float(parameters["dlat_deg"]),
            dlon_deg=float(parameters["dlon_deg"]),
            nx=int(parameters["nx"]),
            ny=int(parameters["ny"]),
            south_pole_lat=float(parameters["south_pole_lat"]),
            south_pole_lon=float(parameters["south_pole_lon"]),
            rotation_angle=float(parameters.get("rotation_angle", 0.0)),
        )

    def _alpha(self):
        alpha = np.radians(-self.south_pole_lat)
        return np.sin(alpha), np.cos(alpha)

    def rotated_to_geographic(self, rlat, rlon):
        """Geographic ``(lat, lon)`` degrees of rotated points (lon in [-180, 180))."""

        sin_a, cos_a = self._alpha()
        rlat = np.radians(np.asarray(rlat, dtype=np.float64))
        rlon = np.radians(np.asarray(rlon, dtype=np.float64)) \
            - np.radians(self.rotation_angle)
        rx = np.cos(rlat) * np.cos(rlon)
        ry = np.cos(rlat) * np.sin(rlon)
        rz = np.sin(rlat)
        qx = sin_a * rx - cos_a * rz
        qz = cos_a * rx + sin_a * rz
        lat = np.degrees(np.arcsin(np.clip(qz, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(ry, qx)) + self.south_pole_lon
        lon = (lon + 180.0) % 360.0 - 180.0
        return lat, lon

    def geographic_to_rotated(self, lat, lon):
        """Rotated ``(rlat, rlon)`` degrees of geographic points: the
        transpose of :meth:`rotated_to_geographic`'s rotation."""

        sin_a, cos_a = self._alpha()
        lat = np.radians(np.asarray(lat, dtype=np.float64))
        lon = np.radians(np.asarray(lon, dtype=np.float64) - self.south_pole_lon)
        qx = np.cos(lat) * np.cos(lon)
        qy = np.cos(lat) * np.sin(lon)
        qz = np.sin(lat)
        rx = sin_a * qx + cos_a * qz
        rz = -cos_a * qx + sin_a * qz
        rlat = np.degrees(np.arcsin(np.clip(rz, -1.0, 1.0)))
        rlon = np.degrees(np.arctan2(qy, rx)) + self.rotation_angle
        return rlat, rlon

    def latlon_to_ij(self, lat, lon):
        """One-based ``(x, y)`` grid coordinates of geographic points.

        The rotated longitude is unwrapped to the half revolution centred
        on the lattice's first meridian, so a point west of the lattice
        lands at a negative coordinate (outside, refused by the coverage
        check) rather than a revolution away.
        """

        rlat, rlon = self.geographic_to_rotated(lat, lon)
        offset = (np.asarray(rlon) - self.lon1 + 180.0) % 360.0 - 180.0
        x = offset / self.dlon_deg + 1.0
        y = (np.asarray(rlat) - self.lat1) / self.dlat_deg + 1.0
        return x, y

    def ij_to_latlon(self, x, y):
        """Geographic ``(lat, lon)`` of one-based grid coordinates."""

        rlon = self.lon1 + (np.asarray(x, dtype=np.float64) - 1.0) * self.dlon_deg
        rlat = self.lat1 + (np.asarray(y, dtype=np.float64) - 1.0) * self.dlat_deg
        return self.rotated_to_geographic(rlat, rlon)


__all__ = ["RotatedLatLonGrid"]
