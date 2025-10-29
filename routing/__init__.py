# routing/__init__.py
# -*- coding: utf-8 -*-

from .planner import plan_route
from .geodesy import (
    geodesic_sample, gc_distance_km, geodesic_midpoint,
    bearing_xy, geodesic_azimuth, angle_diff, normalize_lon_to_pacific_view
)
from .simplify import simplify_path_gc
from .lvs import make_inject_gateways_fn
from .land_layers import (
    dynamic_bboxes_idl, union_lonlat_bboxes, load_polys_in_bboxes,
    build_land_layers, build_land_strtree, nudge_to_ring_if_inside_fast
)
from .features import extract_feature_points_bbox
