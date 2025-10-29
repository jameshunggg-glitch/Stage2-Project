# routing/visibility.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, Tuple, List

from shapely.geometry import LineString
from shapely.strtree import STRtree
from shapely.ops import transform as shp_transform

from .geodesy import geodesic_sample, to_m

def _to_metric_geom(geom):
    # shapely.ops.transform(fn, geom): fn 要接受 (x, y[, z]) 並回傳同樣長度的 tuple
    return shp_transform(lambda x, y, z=None: to_m(x, y), geom)

def visible(
    a: Tuple[float, float],
    b: Tuple[float, float],
    COLLISION_PREP_M,                      # 由 land_layers.build_land_layers 回傳的 prepared geometry
    land_tree: Optional[STRtree] = None,   # 由 land_layers.build_land_strtree(parts_m) 建的 STRtree
    step_km: float = 3.0,                  # 大圓取樣解析度（公里）
) -> bool:
    """
    以「大圓折線」進行可視性檢查：
      1) 先把 (lon,lat) 兩點用大圓路徑離散成 LineString（WGS84 經緯度）
      2) 轉成 WebMercator 公尺座標
      3) 若有 land_tree：先做快速預篩（無候選就視為可通過）
      4) 用 COLLISION_PREP_M（帶安全擴張）做 intersects 檢查
    回傳：True 代表可直連（不穿越安全膨脹的陸地）
    """
    ls_ll = LineString(geodesic_sample(a, b, step_km=step_km))
    ls_m  = _to_metric_geom(ls_ll)

    if land_tree is not None:
        # 若空集合，代表附近沒有陸地相交的可能性 → 可視
        cand = land_tree.query(ls_m)
        if len(cand) == 0:
            return True

    # prepared geometry 的 intersects 很快
    return not COLLISION_PREP_M.intersects(ls_m)

__all__ = ["visible"]
