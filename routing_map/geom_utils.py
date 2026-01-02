from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, List

from pyproj import CRS, Transformer
from shapely.geometry import LineString
from shapely.ops import transform

from .types import LonLat, BBoxLL

@dataclass
class AOIProjector:
    """Local metric projection centered at AOI centroid (Azimuthal Equidistant)."""
    crs_ll: CRS
    crs_m: CRS
    to_m: Transformer
    to_ll: Transformer

def make_aoi_bbox(origin_ll: LonLat, dest_ll: LonLat, pad_deg: float) -> BBoxLL:
    (lon1, lat1) = origin_ll
    (lon2, lat2) = dest_ll
    min_lon = min(lon1, lon2) - pad_deg
    max_lon = max(lon1, lon2) + pad_deg
    min_lat = min(lat1, lat2) - pad_deg
    max_lat = max(lat1, lat2) + pad_deg
    return (float(min_lon), float(min_lat), float(max_lon), float(max_lat))

def build_projector_from_bbox(bbox_ll: BBoxLL) -> AOIProjector:
    (min_lon, min_lat, max_lon, max_lat) = bbox_ll
    lon0 = (min_lon + max_lon) / 2.0
    lat0 = (min_lat + max_lat) / 2.0
    crs_ll = CRS.from_epsg(4326)
    crs_m = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    to_m = Transformer.from_crs(crs_ll, crs_m, always_xy=True)
    to_ll = Transformer.from_crs(crs_m, crs_ll, always_xy=True)
    return AOIProjector(crs_ll=crs_ll, crs_m=crs_m, to_m=to_m, to_ll=to_ll)

def geom_to_m(geom, proj: AOIProjector):
    return transform(lambda x, y, z=None: proj.to_m.transform(x, y), geom)

def geom_to_ll(geom, proj: AOIProjector):
    return transform(lambda x, y, z=None: proj.to_ll.transform(x, y), geom)

# routing_map/geom_utils.py

from shapely.geometry import Point

def linestring_sample_points(line, ds_m: float):
    """
    Sample points along a LineString / LinearRing every ds_m meters.
    - Works for open LineString and closed rings (LinearRing or closed LineString).
    - Always includes a final point at distance L (end), for rings it's the same as start.
    """
    if line is None or line.is_empty:
        return []

    L = float(line.length)
    if L <= 0:
        # degenerate
        try:
            return [Point(line.coords[0])]
        except Exception:
            return []

    ds_m = float(ds_m)
    if ds_m <= 0:
        return [line.interpolate(0.0), line.interpolate(L)]

    # number of points (include 0, exclude L first; we'll force L at the end)
    n = int(L // ds_m) + 1
    pts = [line.interpolate(i * ds_m) for i in range(n)]

    # --- Force last point to be at L, but do NOT use boundary for rings (boundary empty) ---
    try:
        end_pt = line.interpolate(L)
        if not pts:
            pts = [end_pt]
        else:
            # if last isn't close to end_pt, replace last
            if pts[-1].distance(end_pt) > 1e-6:
                pts[-1] = end_pt
    except Exception:
        # fallback: just return what we have
        pass

    return pts

# ---- bbox helpers ----

def expand_bbox_ll(bbox_ll: Tuple[float, float, float, float], pad_deg: float) -> Tuple[float, float, float, float]:
    """
    Expand lon/lat bbox by pad_deg on each side.
    NOTE: This is a simple implementation (no dateline split). Works well for your E/SE Asia AOIs.
    """
    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox_ll]
    p = float(pad_deg)

    min_lon2 = min_lon - p
    max_lon2 = max_lon + p
    min_lat2 = min_lat - p
    max_lat2 = max_lat + p

    # clamp latitude to valid range
    min_lat2 = max(-89.9999, min_lat2)
    max_lat2 = min( 89.9999, max_lat2)

    # keep lon within [-180, 180] (simple clamp)
    min_lon2 = max(-180.0, min_lon2)
    max_lon2 = min( 180.0, max_lon2)

    return (min_lon2, min_lat2, max_lon2, max_lat2)
