# routing/geodesy.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Tuple, List
import math
from pyproj import Geod, Transformer

GEOD = Geod(ellps="WGS84")

# lon/lat <-> WebMercator meters
_transform_ll_to_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_transform_m_to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

def to_m(lon: float, lat: float) -> Tuple[float, float]:
    return _transform_ll_to_m.transform(lon, lat)

def to_ll(x_m: float, y_m: float) -> Tuple[float, float]:
    return _transform_m_to_ll.transform(x_m, y_m)

def normalize_lon_to_pacific_view(lon: float) -> float:
    return lon if lon >= 0.0 else lon + 360.0

def angle_diff(a_deg: float, b_deg: float) -> float:
    return (a_deg - b_deg + 540.0) % 360.0 - 180.0

def geodesic_sample(a: Tuple[float,float], b: Tuple[float,float], step_km: float=3.0) -> List[Tuple[float,float]]:
    lon1, lat1 = a; lon2, lat2 = b
    _, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
    n = max(1, int(dist_m / (step_km * 1000.0)))
    pts = GEOD.npts(lon1, lat1, lon2, lat2, n)
    return [(lon1, lat1)] + pts + [(lon2, lat2)]

def gc_distance_km(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    _, _, d_m = GEOD.inv(a[0], a[1], b[0], b[1])
    return d_m / 1000.0

def geodesic_midpoint(a: Tuple[float,float], b: Tuple[float,float]) -> Tuple[float,float]:
    pts = geodesic_sample(a, b, step_km=500.0)
    return pts[len(pts)//2]

# --- compatibility aliases ---
def great_circle_midpoint(a, b):
    """Alias for backward compatibility."""
    return geodesic_midpoint(a, b)

def bearing_xy(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    ax, ay = to_m(a[0], a[1]); bx, by = to_m(b[0], b[1])
    return math.degrees(math.atan2(by - ay, bx - ax)) % 360.0

def geodesic_azimuth(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    fwd_az, _, _ = GEOD.inv(a[0], a[1], b[0], b[1])
    return (fwd_az + 360.0) % 360.0
