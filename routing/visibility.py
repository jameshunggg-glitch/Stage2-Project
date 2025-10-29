# routing/visibility.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, Tuple, Callable
from shapely.geometry import LineString
from .geodesy import geodesic_sample
from .config import STEP_KM_GEODESIC

class Visibility:
    """可視策略介面"""
    def is_visible(self, a: Tuple[float,float], b: Tuple[float,float]) -> bool:
        raise NotImplementedError

class PreparedCollisionVisibility(Visibility):
    """基於 COLLISION_PREP_M（prepared geometry）與可選 STRtree 的可視檢查"""
    def __init__(self, collision_prep_m, land_tree=None, step_km: float=STEP_KM_GEODESIC, to_metric: Optional[Callable]=None):
        self.collision_prep_m = collision_prep_m
        self.land_tree = land_tree
        self.step_km = step_km
        self.to_metric = to_metric  # 函式：WGS -> metric

    def is_visible(self, a, b) -> bool:
        ls_ll = LineString(geodesic_sample(a, b, step_km=self.step_km))
        ls_m  = self.to_metric(ls_ll) if self.to_metric else ls_ll
        if self.land_tree is not None:
            candidates = self.land_tree.query(ls_m)
            if len(candidates) == 0:
                return True
        return not self.collision_prep_m.intersects(ls_m)
