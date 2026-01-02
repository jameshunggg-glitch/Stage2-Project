from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .types import LonLat, BBoxLL

@dataclass
class LandConfig:
    shp_path: Path
    buffer_km: float = 20.0
    avoid_km: float = 5.0
    collision_safety_km: float = 2.0
    precision_grid_m: float = 5.0  # set_precision grid size

@dataclass
class AoiConfig:
    origin_ll: Optional[LonLat] = None
    dest_ll: Optional[LonLat] = None
    bbox_ll: Optional[BBoxLL] = None
    pad_deg: float = 2.0

@dataclass
class SmoothConfig:
    a2_smooth_km: float = 5.0
    a2_tol_km: float = 8.0

@dataclass
class CChainConfig:
    c_step_km: float = 20.0
    round_decimals: int = 5
    island_area_min_km2: float = 20.0

@dataclass
class FeatureConfig:
    f_sample_step_km: float = 15.0
    f_angle_deg_min: float = 35.0
    f_nms_radius_km: float = 25.0
    f_max_keep: int = 2500

@dataclass
class GateAConfig:
    min_ring_length_km: float = 500.0
    short_ring_no_gate_km: float = 150.0
    short_ring_one_gate_km: float = 400.0
    snap_to_f_km: float = 15.0

@dataclass
class GateFConfig:
    min_spacing_km: float = 250.0
    max_per_ring: int = 3
    global_max: int = 3000

@dataclass
class SeaConfig:
    # Gate-B v1 parameters
    r_max_km: float = 250.0
    candidate_top_n: int = 15
    k_connect: int = 3
    deg_min: int = 1
    aoi_pad_deg: float = 2.0
    use_largest_component_only: bool = True

@dataclass
class RoutingMapConfig:
    aoi: AoiConfig
    land: LandConfig
    smooth: SmoothConfig = SmoothConfig()
    cchain: CChainConfig = CChainConfig()
    features: FeatureConfig = FeatureConfig()
    gate_a: GateAConfig = GateAConfig()
    gate_f: GateFConfig = GateFConfig()
    sea: SeaConfig = SeaConfig()
