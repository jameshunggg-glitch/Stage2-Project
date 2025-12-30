from __future__ import annotations
from typing import List, Tuple
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, LineString, GeometryCollection
from shapely.ops import unary_union

def _iter_polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        polys = []
        for g in geom.geoms:
            polys.extend(_iter_polygons(g))
        return polys
    return []

def build_coast_rings_smooth_v2(
    union_smooth_m,
    *,
    avoid_km: float,
    island_area_min_km2: float,
) -> tuple[object, List[LineString], pd.DataFrame]:
    """Build coast rings from A2 smooth union.

    - Expand by avoid_km (meters) to create ring polygons.
    - Drop small islands (<island_area_min_km2).
    Returns:
      ring_base_m: union of ring polygons (for viz)
      rings_m: list of LineString exteriors
      rings_df: per-ring stats
    """
    avoid_m = avoid_km * 1000.0
    polys = _iter_polygons(union_smooth_m)
    if not polys:
        empty = Polygon()
        return empty, [], pd.DataFrame(columns=["ring_id","area_km2","length_km","is_mainland","original_area_km2"])

    all_rings = []
    rows = []
    ring_id = 0

    for p in polys:
        area_km2 = float(p.area) / 1e6
        # buffer outward then use exterior
        rp = p.buffer(avoid_m).buffer(0)
        if rp.is_empty:
            continue
        rp_polys = _iter_polygons(rp)
        if not rp_polys:
            continue

        max_area = max(float(pp.area) for pp in rp_polys)
        for pp in rp_polys:
            ring_area_km2 = float(pp.area) / 1e6
            if ring_area_km2 < island_area_min_km2:
                continue
            ring_length_km = float(pp.exterior.length) / 1000.0
            is_mainland = (float(pp.area) == max_area)
            all_rings.append(pp.exterior)
            rows.append({
                "ring_id": ring_id,
                "area_km2": ring_area_km2,
                "length_km": ring_length_km,
                "is_mainland": bool(is_mainland),
                "original_area_km2": area_km2,
            })
            ring_id += 1

    if all_rings:
        ring_base_m = unary_union([Polygon(r) for r in all_rings]).buffer(0)
    else:
        ring_base_m = Polygon()

    rings_df = pd.DataFrame(rows)
    return ring_base_m, all_rings, rings_df
