# routing/visibility.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, Tuple

from shapely.geometry import LineString
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree

from .geodesy import geodesic_sample, to_m

def _to_metric_geom(geom):
    # shapely.ops.transform 需要 (x, y[, z]) -> (X, Y) 的 callable
    return shp_transform(lambda x, y, z=None: to_m(x, y), geom)

def visible(
    a: Tuple[float, float],
    b: Tuple[float, float],
    COLLISION_PREP_M,                      # prepared geometry (land buffer for collision)
    land_tree: Optional[STRtree] = None,   # STRtree of land parts in meters
    step_km: float = 3.0,
) -> bool:
    """
    以大圓折線測試 a→b 是否不與擴張陸地碰撞。
    - 先用 geodesic_sample 取樣 -> LineString (WGS84)
    - 轉 WebMercator (meters)
    - 若 land_tree 存在：先 query 預篩（0 代表無衝突可能，直接 True）
    - 最後用 COLLISION_PREP_M.intersects 檢查
    """
    ls_ll = LineString(geodesic_sample(a, b, step_km=step_km))
    ls_m  = _to_metric_geom(ls_ll)

    if land_tree is not None:
        if len(land_tree.query(ls_m)) == 0:
            return True

    return not COLLISION_PREP_M.intersects(ls_m)

__all__ = ["visible"]
