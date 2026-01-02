# routing_map/build_aoi.py
from __future__ import annotations

from typing import Dict, Any

import numpy as np
from shapely.prepared import prep

from .config import RoutingMapConfig
from .geom_utils import make_aoi_bbox, build_projector_from_bbox
from .io_land import load_polys_in_bbox
from .land_layers import build_land_layers
from .smooth import smooth_union_for_features_from_union
from .rings import build_coast_rings_smooth_v2
from .cchain import build_C_chain_from_rings
from .features import extract_F_nodes_from_union_smooth
from .gates_a import build_gate_A_from_C_and_F_v1
from .gates_merge import merge_gates
from .inject import attach_F_to_nearest_C
from .gates_f import build_gate_F_primary
from .sea_nodes import build_sea_nodes_from_bundle, filter_sea_nodes
from .gates_b import build_gateB_connectors
from . import scgraph_bridge

from .gates_coverage import attach_gates_to_nearest_C, coverage_sample_gates_on_rings


def _pad_bbox_ll(bbox_ll, pad_deg: float):
    x0, y0, x1, y1 = [float(v) for v in bbox_ll]
    return (x0 - pad_deg, y0 - pad_deg, x1 + pad_deg, y1 + pad_deg)


def _filter_df_in_bbox(df, bbox_ll, lon_col="lon", lat_col="lat"):
    if df is None or len(df) == 0:
        return df
    x0, y0, x1, y1 = [float(v) for v in bbox_ll]
    m = (df[lon_col].between(x0, x1)) & (df[lat_col].between(y0, y1))
    return df.loc[m].reset_index(drop=True)


def build_aoi(cfg: RoutingMapConfig) -> Dict[str, Any]:
    """
    AOI pipeline (Phase 1-ready): build land->rings->C/F/Gates + Sea subnet + Gate-B connectors.
    """
    # --- AOI bbox ---
    bbox_ll = cfg.aoi.bbox_ll
    if bbox_ll is None:
        if cfg.aoi.origin_ll is None or cfg.aoi.dest_ll is None:
            raise ValueError("Provide either aoi.bbox_ll or (aoi.origin_ll & aoi.dest_ll)")
        bbox_ll = make_aoi_bbox(cfg.aoi.origin_ll, cfg.aoi.dest_ll, cfg.aoi.pad_deg)

    proj = build_projector_from_bbox(bbox_ll)

    # --- Land ---
    polys_ll = load_polys_in_bbox(cfg.land.shp_path, bbox_ll)
    layers = build_land_layers(
        polys_ll, proj,
        buffer_km=cfg.land.buffer_km,
        avoid_km=cfg.land.avoid_km,
        collision_safety_km=cfg.land.collision_safety_km,
        grid_size_m=cfg.land.precision_grid_m,
    )
    union_m = layers["UNION_M"]

    # --- Smooth union ---
    union_smooth_m = smooth_union_for_features_from_union(
        union_m,
        a2_smooth_km=cfg.smooth.a2_smooth_km,
        a2_tol_km=cfg.smooth.a2_tol_km,
    )

    # --- Rings ---
    ring_base_m, rings_m, rings_df = build_coast_rings_smooth_v2(
        union_smooth_m,
        avoid_km=cfg.land.avoid_km,
        island_area_min_km2=cfg.cchain.island_area_min_km2,
    )

    # --- C chain ---
    C_nodes, C_edges = build_C_chain_from_rings(
        rings_m, proj,
        c_step_km=cfg.cchain.c_step_km,
        round_decimals=cfg.cchain.round_decimals,
    )

    # --- Feature nodes ---
    F_nodes = extract_F_nodes_from_union_smooth(
        union_smooth_m, proj,
        sample_step_km=cfg.features.f_sample_step_km,
        angle_deg_min=cfg.features.f_angle_deg_min,
        nms_radius_km=cfg.features.f_nms_radius_km,
        max_keep=cfg.features.f_max_keep,
    )

    # Attach F -> nearest C (for Gate-F primary selection)
    F_att = attach_F_to_nearest_C(F_nodes, C_nodes)

    # --- Gates A/F ---
    Gate_A = build_gate_A_from_C_and_F_v1(
        C_nodes, rings_df, F_nodes,
        min_ring_length_km=cfg.gate_a.min_ring_length_km,
        short_ring_no_gate_km=cfg.gate_a.short_ring_no_gate_km,
        short_ring_one_gate_km=cfg.gate_a.short_ring_one_gate_km,
        snap_to_f_km=cfg.gate_a.snap_to_f_km,
    )

    Gate_F = build_gate_F_primary(
        F_att, rings_df,
        min_spacing_km=cfg.gate_f.min_spacing_km,
        max_per_ring=cfg.gate_f.max_per_ring,
        global_max=cfg.gate_f.global_max,
        round_decimals=cfg.cchain.round_decimals,
    )

    Gate_all = merge_gates(Gate_A, Gate_F, round_decimals=cfg.cchain.round_decimals)

    # 僅用 AOI bbox 內的 gates 做 coverage/connectability（避免算到外面）
    Gate_all_aoi = _filter_df_in_bbox(Gate_all, bbox_ll)

    # --- Coverage (Gate-A + Gate-F) ---
    # 1) 補 s_km（靠近哪個 C 就用它的 s_km）
    Gate_all_att = attach_gates_to_nearest_C(
        Gate_all_aoi, C_nodes, rings_df,
        debug=getattr(cfg.coverage, "debug", True),
    )

    # 2) spacing 抽樣（閉合 ring）
    Gate_all_cov = coverage_sample_gates_on_rings(
        Gate_all_att,
        gate_spacing_km=float(cfg.coverage.gate_spacing_km),
        min_per_ring=int(cfg.coverage.min_per_ring),
        prefer_source_order=list(cfg.coverage.prefer_source_order),
        debug=getattr(cfg.coverage, "debug", True),
    )

    # 給 Gate-B 用的 gate_uid（確保 connectors 可以回指 gate）
    Gate_all_cov = Gate_all_cov.copy()
    Gate_all_cov["gate_uid"] = np.arange(len(Gate_all_cov), dtype=int)

    # --- Sea subnet (scgraph) ---
    bbox_ll_sea = _pad_bbox_ll(bbox_ll, pad_deg=float(cfg.sea.aoi_pad_deg))
    bundle = scgraph_bridge.sc_edges_in_bbox(bbox_ll=bbox_ll_sea)  # 你現在的 adapter 會回 nodes/edges/stats
    S_nodes, S_edges, G, kdt = build_sea_nodes_from_bundle(proj, bundle)

    sea_ok_set = filter_sea_nodes(
        S_nodes, G,
        deg_min=int(cfg.sea.deg_min),
        use_largest_component_only=bool(cfg.sea.use_largest_component_only),
    )

    # --- Gate-B connectors (Gate -> Sea) ---
    collision_prep = prep(layers["COLLISION_M"])
    gateB_connectors = build_gateB_connectors(
        Gate_all_cov, S_nodes,
        sea_ok_set=sea_ok_set,
        kdt=kdt,
        collision_prep=collision_prep,
        top_n=cfg.sea.candidate_top_n,
        r_max_km=cfg.sea.r_max_km,
        k_connect=cfg.sea.k_connect,
    )

    # Gate-B kept gates（能連上才算 Gate-B）
    Gate_B_kept_gates = Gate_all_cov.iloc[0:0].copy()
    if gateB_connectors is not None and len(gateB_connectors) > 0:
        if "gate_uid" in gateB_connectors.columns:
            keep_uid = set(gateB_connectors["gate_uid"].astype(int).tolist())
            Gate_B_kept_gates = Gate_all_cov[Gate_all_cov["gate_uid"].isin(keep_uid)].reset_index(drop=True)
        elif "gate_row" in gateB_connectors.columns:
            keep_row = sorted(set(gateB_connectors["gate_row"].astype(int).tolist()))
            Gate_B_kept_gates = Gate_all_cov.iloc[keep_row].reset_index(drop=True)

    return {
        "cfg": cfg,
        "bbox_ll": bbox_ll,
        "bbox_ll_sea": bbox_ll_sea,
        "proj": proj,

        "polys_ll": polys_ll,
        "layers": layers,
        "union_smooth_m": union_smooth_m,

        "ring_base_m": ring_base_m,
        "rings_m": rings_m,
        "rings_df": rings_df,

        "C_nodes": C_nodes,
        "C_edges": C_edges,

        "F_nodes": F_nodes,
        "F_att": F_att,

        "Gate_A": Gate_A,
        "Gate_F": Gate_F,
        "Gate_all": Gate_all,

        # Coverage outputs
        "Gate_all_aoi": Gate_all_aoi,
        "Gate_all_cov": Gate_all_cov,

        # Sea outputs
        "sc_bundle_stats": bundle.get("stats", {}) if isinstance(bundle, dict) else {},
        "S_nodes": S_nodes,
        "S_edges": S_edges,
        "sea_graph": G,
        "sea_kdt": kdt,
        "sea_ok_set": sea_ok_set,

        # Gate-B outputs
        "gateB_connectors": gateB_connectors,
        "Gate_B_kept_gates": Gate_B_kept_gates,
    }
