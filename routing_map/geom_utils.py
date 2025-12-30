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

def linestring_sample_points(line: LineString, ds_m: float) -> List:
    """Sample along a LineString by distance (meters). Includes endpoints."""
    if line.is_empty:
        return []
    L = float(line.length)
    if L <= 0:
        return []
    n = max(2, int(math.ceil(L / ds_m)) + 1)
    pts = [line.interpolate(i * ds_m) for i in range(n)]
    # ensure last is end
    if pts and pts[-1].distance(line.boundary.geoms[-1]) > 1e-6:
        pts[-1] = line.interpolate(L)
    return pts
