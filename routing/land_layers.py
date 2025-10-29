# routing/land_layers.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import fiona
import shapely
from shapely.geometry import shape, Polygon, Point, LineString, GeometryCollection
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree

from .geodesy import to_m, to_ll
from .config import BUFFER_KM, COLLISION_SAFETY_KM

def dynamic_bboxes_idl(origin, dest, pad_deg: float) -> List[Polygon]:
    def pac(lon: float) -> float: return lon if lon >= 0 else lon + 360
    o_lon = pac(origin[0]); d_lon = pac(dest[0])
    min_lon = min(o_lon, d_lon) - pad_deg
    max_lon = max(o_lon, d_lon) + pad_deg
    min_lat = min(origin[1], dest[1]) - pad_deg
    max_lat = max(origin[1], dest[1]) + pad_deg
    bboxes=[]
    if min_lon < 0:  # wrap left
        bboxes.append(Polygon([(min_lon,min_lat),(0,min_lat),(0,max_lat),(min_lon,max_lat)]))
        min_lon = 0
    if max_lon > 360:  # wrap right
        bboxes.append(Polygon([(0,min_lat),(max_lon-360,min_lat),(max_lon-360,max_lat),(0,max_lat)]))
        max_lon = 360
    lon_min_std = min_lon if min_lon <= 180 else min_lon-360
    lon_max_std = max_lon if max_lon <= 180 else max_lon-360
    if lon_min_std <= lon_max_std:
        bboxes.append(Polygon([(lon_min_std,min_lat),(lon_max_std,min_lat),(lon_max_std,max_lat),(lon_min_std,max_lat)]))
    else:
        bboxes.append(Polygon([(lon_min_std,min_lat),(180,min_lat),(180,max_lat),(lon_min_std,max_lat)]))
        bboxes.append(Polygon([(-180,min_lat),(lon_max_std,min_lat),(lon_max_std,max_lat),(-180,max_lat)]))
    return bboxes

def union_lonlat_bboxes(bboxes: List[Polygon]) -> Polygon:
    u = unary_union(bboxes)
    if isinstance(u, GeometryCollection):
        return u.envelope
    return u

def load_polys_in_bboxes(shp_path: Path, bbox_polys: List[Polygon]) -> List[Polygon]:
    polys=[]
    with fiona.open(shp_path) as src:
        for feat in src:
            g = shape(feat["geometry"])
            for box in bbox_polys:
                if g.intersects(box):
                    gi = g.intersection(box)
                    try:
                        parts = list(shapely.get_parts(gi))
                    except Exception:
                        parts = list(gi.geoms) if gi.geom_type=="MultiPolygon" else [gi]
                    for p in parts:
                        if not p.is_empty:
                            polys.append(p)
                    break
    return polys

def build_land_layers(polys: List[Polygon]) -> Dict[str, object]:
    parts_m = [to_metric(p).buffer(0) for p in polys]
    union_m = unary_union(parts_m)
    collision_m = union_m.buffer(COLLISION_SAFETY_KM * 1000.0)
    ring_m      = union_m.buffer(BUFFER_KM * 1000.0)
    return {
        "UNION_M":          union_m,
        "COLLISION_PREP_M": prep(collision_m),
        "COLLISION_WGS":    to_wgs(collision_m),
        "RING_M":           ring_m,
        "RING_WGS":         to_wgs(ring_m),
        "LAND_RAW_WGS":     to_wgs(union_m),
        "LAND_PARTS_M":     parts_m,
        "COLLISION_M":      collision_m,
    }

def build_land_strtree(parts_m: List[shapely.geometry.base.BaseGeometry]) -> STRtree:
    return STRtree(parts_m)

# local helpers for metric<->wgs
def to_metric(g): return shapely.ops.transform(lambda x,y: to_m(x,y), g)
def to_wgs(g):    return shapely.ops.transform(lambda x,y: to_ll(x,y), g)

def nudge_to_ring_if_inside_fast(pt_ll, inner_ring_m, target_boundary_m):
    px, py = to_m(pt_ll[0], pt_ll[1])
    p_m = Point(px, py)
    if inner_ring_m.contains(p_m):
        q_m = shapely.ops.nearest_points(p_m, target_boundary_m)[1]
        q_llx, q_lly = to_ll(q_m.x, q_m.y)
        return (q_llx, q_lly), True
    return pt_ll, False
