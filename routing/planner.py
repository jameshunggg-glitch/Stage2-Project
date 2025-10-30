# routing/planner.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import folium

from .config import (
    BUFFER_KM, LVS_MAX_NODES, PAD_DEG, AVOID_KM,
    DRAW_STEP_KM, SIMPLIFY_MAX_PASSES,
    # 允許你在 config 補上以下三個參數；沒有的話這裡給預設
)
try:
    from .config import SCNET_EPS_KM, SCNET_MAX_NODES, SC_WHITELIST_ROUNDING
except Exception:
    SCNET_EPS_KM = 6.0
    SCNET_MAX_NODES = 3000
    SC_WHITELIST_ROUNDING = 6

try:
    from .config import FALLBACK_USE_SC_OD_PATH, FALLBACK_SIMPLIFY_PASSES
except Exception:
    FALLBACK_USE_SC_OD_PATH = True
    FALLBACK_SIMPLIFY_PASSES = 2

from .geodesy import great_circle_midpoint, normalize_lon_to_pacific_view, gc_distance_km
from .land_layers import (
    dynamic_bboxes_idl, load_polys_in_bboxes, build_land_layers,
    build_land_strtree, nudge_to_ring_if_inside_fast, union_lonlat_bboxes
)
from .features import extract_feature_points_bbox
from .visibility import visible
from .lvs import lazy_visibility_search, make_inject_gateways_fn
from .draw import draw_gc_polyline_continuous, convert_geom_to_pacific, add_scgraph_layer, add_scgraph_network_layer
from .simplify import simplify_path_gc
from .scgraph_bridge import sc_shortest_path_lonlat, sc_edges_in_bbox, sc_keypoints_in_bbox

# === Lightweight debug prints (no logging) ===
DEBUG = True  # 想關掉訊息就改成 False

def dbg(msg: str):
    if DEBUG:
        print(msg)


def plan_route(
    origin: Tuple[float, float],
    dest: Tuple[float, float],
    land_path: str | Path,
    out_html: str | Path,
    add_feature_layer: bool = True,
    # === scgraph 混合 ===
    use_scgraph: bool = True,
    sc_kwargs: Optional[Dict] = None,  # 轉交 sc_shortest_path_lonlat 的參數
):
    """
    流程：
      (A) 準備陸地與可視檢查
      (B) 取 bbox → 取得 scgraph 子網 (nodes+edges) 與 O→D 路徑 (od_path)
      (C) 併入候選節點 + 白名單邊（兩向）
      (D) 雙向 LVS → 簡化 → 比較距離；若皆失敗且允許 fallback，使用 sc od_path
      (E) folium 地圖輸出；meta 紀錄所有資訊
    回傳: (waypoints_ll, total_km_simplified, html_path, meta)
    """
    dbg(f"[PLAN] start | origin={origin} dest={dest} use_scgraph={use_scgraph}")
    land_path = Path(land_path)
    out_html = Path(out_html)

    # ---------------- (A) Land & layers ----------------
    bboxes = dynamic_bboxes_idl(origin, dest, pad_deg=PAD_DEG)
    polys = load_polys_in_bboxes(land_path, bboxes)
    if not polys:
        raise RuntimeError("No land polygons found in bbox")
    layers = build_land_layers(polys)
    UNION_M          = layers["UNION_M"]
    COLLISION_PREP_M = layers["COLLISION_PREP_M"]
    RING_WGS         = layers["RING_WGS"]
    land_raw_wgs     = layers["LAND_RAW_WGS"]
    LAND_PARTS_M     = layers["LAND_PARTS_M"]
    INNER_RING_M     = layers["RING_M"]
    TARGET_RING_M    = UNION_M.buffer(AVOID_KM * 1000.0)
    TARGET_BOUNDARY_M= TARGET_RING_M.boundary
    land_tree = build_land_strtree(LAND_PARTS_M)

    dbg(f"[LAND] polys={len(polys)} | layers ready (ring/union/prep/strtree)")

    # 起迄若在陸域緩衝內，外推到邊界
    origin_adj, moved_o = nudge_to_ring_if_inside_fast(origin, INNER_RING_M, TARGET_BOUNDARY_M)
    dest_adj,   moved_d = nudge_to_ring_if_inside_fast(dest,   INNER_RING_M, TARGET_BOUNDARY_M)

    # ---------------- (B) Features + sc 子網 ----------------
    bbox_union_ll = union_lonlat_bboxes(bboxes)

    # 原本特徵（凸峰、凸點）
    feat = extract_feature_points_bbox(
        shp_path=land_path,
        bbox_ll_polygon=bbox_union_ll,
        avoid_km=AVOID_KM
    )
    feature_nodes = list(feat.get("convex_peaks", [])) + list(feat.get("convex", []))

    # sc 子網（可走邊 + 海節點/轉折點）
    sc_network = None
    sc_nodes_extra: List[Tuple[float,float]] = []
    sc_edges_whitelist: List[Tuple[Tuple[float,float], Tuple[float,float]]] = []
    sc_od_path = None
    sc_used = False
    sc_error = None

    if use_scgraph:
        dbg("[SCGRAPH] start O→D path planning...")
        try:
            bbox_tuple = bbox_union_ll.bounds  # (minx, miny, maxx, maxy)
            aoi = (float(bbox_tuple[0]), float(bbox_tuple[1]), float(bbox_tuple[2]), float(bbox_tuple[3]))

            sc_network = sc_edges_in_bbox(
                aoi,
                edge_sample_ratio=1.0,
                max_sample_routes=60,
                node_snap_decimals=SC_WHITELIST_ROUNDING,
                simplify_epsilon_km=SCNET_EPS_KM,
            )
            sc_edges_whitelist = sc_network.get("edges", []) if sc_network else []
            # 轉折/交會點（高價值節點）
            sc_nodes_extra = sc_keypoints_in_bbox(
                aoi,
                edges=sc_edges_whitelist,
                node_snap_decimals=SC_WHITELIST_ROUNDING,
                bend_threshold_deg=12.0,
            )
            # 單一路徑（fallback 用）
            params = dict(output_units="km")
            if sc_kwargs: params.update(sc_kwargs)
            out = sc_shortest_path_lonlat(origin_adj, dest_adj, **params)
            if out and out.get("track"):
                sc_od_path = out["track"]
            sc_used = True
        except Exception as e:
            sc_error = str(e)
            if sc_error:
                dbg(f"[SCGRAPH] O→D path FAILED | error={sc_error}")

    # ---------------- (C) 組合 nodes + 白名單邊 ----------------
    base_nodes = [origin_adj, dest_adj] + feature_nodes
    # 併入 sc 的 keypoints（高價值節點）
    if sc_nodes_extra:
        base_nodes.extend(sc_nodes_extra)

    # 去重與上限
    seen = set()
    def _r(p, nd=SC_WHITELIST_ROUNDING): return (round(p[0], nd), round(p[1], nd))
    nodes: List[Tuple[float, float]] = []
    for p in base_nodes:
        rp = _r(p)
        if rp in seen: continue
        seen.add(rp)
        nodes.append((float(rp[0]), float(rp[1])))
        if len(nodes) >= LVS_MAX_NODES:
            break

    O_idx, D_idx = 0, 1

    # 白名單邊（雙向）
    SC_WHITELIST_EDGES = set()
    for (a, b) in sc_edges_whitelist or []:
        ra, rb = _r(a), _r(b)
        SC_WHITELIST_EDGES.add((ra, rb))
        SC_WHITELIST_EDGES.add((rb, ra))

    # 可視封裝：白名單邊免碰撞
    def visible_wrapper(a, b, cp, tree):
        if use_scgraph and SC_WHITELIST_EDGES:
            ra, rb = _r(a), _r(b)
            if (ra, rb) in SC_WHITELIST_EDGES:
                return True
        return visible(a, b, cp, tree)

    # LVS 需要的 inject
    inject_fn = make_inject_gateways_fn(
        UNION_M,
        {"convex_peaks": feat.get("convex_peaks", []), "convex": feat.get("convex", [])},
        take_each=3,
        inner_ring_m=INNER_RING_M,
        target_boundary_m=TARGET_BOUNDARY_M,
    )

    # ---------------- (D) 雙向 LVS → 簡化 → 比較；若失敗則 fallback ----------------
    def run_once(nodes_in: List[Tuple[float, float]], O: int, D: int):
        dbg(f"[LVS] run_once start | O={O} D={D} nodes={len(nodes_in)}")
        nodes_local = list(nodes_in)
        path = lazy_visibility_search(
            nodes_local, O, D, visible_wrapper, COLLISION_PREP_M, land_tree, inject_fn,
            max_iters=5000, progress=None
        )
        dbg(f"[LVS] run_once done  | O={O} D={D} path_len={len(path)}")
        return path, nodes_local

    results = []
    err_fwd = err_rev = None
    try:
        path_fwd, nodes_fwd = run_once(nodes, O_idx, D_idx)
        results.append(("fwd", path_fwd, nodes_fwd))
    except Exception as e:
        err_fwd = e
        if err_fwd:
            dbg(f"[LVS] forward FAILED | {repr(err_fwd)}")

    try:
        nodes_rev = list(nodes)
        nodes_rev[0], nodes_rev[1] = nodes_rev[1], nodes_rev[0]
        path_rev_do, nodes_do = run_once(nodes_rev, 0, 1)
        path_rev = list(reversed(path_rev_do))
        results.append(("rev", path_rev, nodes_do))
    except Exception as e:
        err_rev = e
        if err_rev:
            dbg(f"[LVS] reverse FAILED | {repr(err_rev)}")

    fallback_used = False
    fallback_reason = None

    if not results:
        # 雙向皆失敗 → 看要不要用 sc O→D 當保底
        if FALLBACK_USE_SC_OD_PATH and sc_od_path and len(sc_od_path) >= 2:
            # 用 sc_od_path 當作 chosen 路徑；再跑一次可視簡化（保持一致）
            chosen_nodes = sc_od_path[:]  # (lon,lat)
            chosen_path_idx = list(range(len(chosen_nodes)))
            chosen_simplified = simplify_path_gc(
                chosen_path_idx, chosen_nodes, visible_wrapper, COLLISION_PREP_M, land_tree,
                max_passes=FALLBACK_SIMPLIFY_PASSES
            )
            total_simple_best = None  # 之後會重算
            tag_best = "sc-fallback"
            fallback_used = True
            fallback_reason = f"lvs_failed: fwd={err_fwd}, rev={err_rev}"
        else:
            raise RuntimeError(f"LVS failed in both directions. fwd={err_fwd}, rev={err_rev}, sc_od_path={'yes' if sc_od_path else 'no'}")

    if not fallback_used:
        # 正常：在所有成功候選中挑距離最短
        cand = []
        for tag, p_idx, node_list in results:            
            simp = simplify_path_gc(
                p_idx, node_list, visible_wrapper, COLLISION_PREP_M, land_tree,
                max_passes=SIMPLIFY_MAX_PASSES
            )
            dbg(f"[SIMPLIFY] tag={tag} simp_pts={len(simp)}")
            cand.append((tag, p_idx, node_list, simp))

        def total_km_after_simplify(simplified_pts: List[Tuple[float,float]]) -> float:
            segs: List[Tuple[Tuple[float,float], Tuple[float,float]]] = []
            if moved_o and origin != origin_adj: segs.append((origin, origin_adj))
            if simplified_pts and len(simplified_pts) >= 2:
                segs.extend(zip(simplified_pts[:-1], simplified_pts[1:]))
            if moved_d and dest_adj != dest: segs.append((dest_adj, dest))
            return sum(gc_distance_km(a, b) for (a, b) in segs)

        best = None
        for tag, p_idx, node_list, simp in cand:
            dist = total_km_after_simplify(simp)
            if (best is None) or (dist < best[0]):
                best = (dist, tag, p_idx, node_list, simp)

        total_simple_best, tag_best, chosen_path_idx, chosen_nodes, chosen_simplified = best

    # ---------------- (E) folium 地圖 ----------------
    mid_lon, mid_lat = great_circle_midpoint(origin, dest)
    center_lon_pacific = normalize_lon_to_pacific_view(mid_lon)
    m = folium.Map(
        location=[mid_lat, center_lon_pacific],
        zoom_start=3,
        max_bounds=False, world_copy_jump=False, no_wrap=False, min_lon=0, max_lon=360
    )
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Ocean_Basemap/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Ocean Basemap', overlay=False, control=True, no_wrap=False
    ).add_to(m)
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='© OpenStreetMap', name='OpenStreetMap', overlay=False, control=True, no_wrap=False
    ).add_to(m)

    folium.GeoJson(
        convert_geom_to_pacific(land_raw_wgs),
        name="陸地",
        style_function=lambda x: {"color": "#2ca02c", "weight": 1, "fillOpacity": 0.15}
    ).add_to(m)
    folium.GeoJson(
        convert_geom_to_pacific(RING_WGS),
        name=f"航道緩衝區 {BUFFER_KM}km",
        style_function=lambda x: {"color": "#6a5acd", "weight": 2, "fillOpacity": 0.05}
    ).add_to(m)

    folium.Marker(
        [origin[1], normalize_lon_to_pacific_view(origin[0])],
        tooltip=f"起點: ({origin[0]:.2f}, {origin[1]:.2f})",
        icon=folium.Icon(color='green', icon='ship', prefix='fa')
    ).add_to(m)
    folium.Marker(
        [dest[1], normalize_lon_to_pacific_view(dest[0])],
        tooltip=f"終點: ({dest[0]:.2f}, {dest[1]:.2f})",
        icon=folium.Icon(color='red', icon='anchor', prefix='fa')
    ).add_to(m)

    # 參考大圓（灰）
    draw_gc_polyline_continuous(m, origin, dest, step_km=80.0, color='gray', weight=2, opacity=0.4, dash_array="8,4")

    # 候選特徵點（可選）
    if add_feature_layer:
        fg_feat = folium.FeatureGroup(name="候選特徵點（凸峰+凸點+sc轉折）", show=False)
        for (lon, lat) in feature_nodes:
            folium.CircleMarker([lat, normalize_lon_to_pacific_view(lon)], radius=3, color="#1f77b4", fill=True, fill_opacity=0.8, tooltip=f"Feature ({lon:.3f},{lat:.3f})").add_to(fg_feat)
        # sc 轉折點
        for (lon, lat) in (sc_nodes_extra or []):
            folium.CircleMarker([lat, normalize_lon_to_pacific_view(lon)], radius=3, color="#ff7f0e", fill=True, fill_opacity=0.9, tooltip=f"SC bend/junction ({lon:.3f},{lat:.3f})").add_to(fg_feat)
        fg_feat.add_to(m)

    # SC Network（橘色細虛線，可切換）
    if sc_network and sc_network.get("edges"):
        add_scgraph_network_layer(
            m, sc_network["edges"], name="SCGraph Network", show=False,
            weight=2, opacity=0.6, dash_array="4,6", color="#ff7f0e"
        )

    # 原始（被選擇之）路徑（藍）
    final_segments: List[Tuple[Tuple[float,float], Tuple[float,float]]] = []
    if moved_o and origin != origin_adj:
        final_segments.append((origin, origin_adj))
        draw_gc_polyline_continuous(m, origin, origin_adj, step_km=DRAW_STEP_KM, color='#1f77b4', weight=5, opacity=0.9)

    if fallback_used and tag_best == "sc-fallback":
        # 用 sc_od_path 畫原始藍線
        for a, b in zip(chosen_nodes[:-1], chosen_nodes[1:]):
            final_segments.append((a, b))
            draw_gc_polyline_continuous(m, a, b, step_km=DRAW_STEP_KM, color='#1f77b4', weight=5, opacity=0.9)
    else:
        # 正常：依索引於 chosen_nodes 畫
        for u, v in zip(chosen_path_idx[:-1], chosen_path_idx[1:]):
            a = chosen_nodes[u]; b = chosen_nodes[v]
            final_segments.append((a, b))
            draw_gc_polyline_continuous(m, a, b, step_km=DRAW_STEP_KM, color='#1f77b4', weight=5, opacity=0.9)

    if moved_d and dest_adj != dest:
        final_segments.append((dest_adj, dest))
        draw_gc_polyline_continuous(m, dest_adj, dest, step_km=DRAW_STEP_KM, color='#1f77b4', weight=5, opacity=0.9)

    # 簡化覆蓋（紅）
    if chosen_simplified and len(chosen_simplified) >= 2:
        fg_simplified = folium.FeatureGroup(name="簡化後航線 (可視直連)", show=True)
        if moved_o and origin != origin_adj:
            draw_gc_polyline_continuous(fg_simplified, origin, origin_adj, step_km=DRAW_STEP_KM, color='#d62728', weight=4, opacity=0.8, dash_array="6,4")
        for a, b in zip(chosen_simplified[:-1], chosen_simplified[1:]):
            draw_gc_polyline_continuous(fg_simplified, a, b, step_km=DRAW_STEP_KM, color='#d62728', weight=6, opacity=0.9)
        if moved_d and dest_adj != dest:
            draw_gc_polyline_continuous(fg_simplified, dest_adj, dest, step_km=DRAW_STEP_KM, color='#d62728', weight=4, opacity=0.8, dash_array="6,4")
        fg_simplified.add_to(m)

    # SC O→D 路徑（橘色粗虛線，可切換）
    if sc_od_path and len(sc_od_path) >= 2:
        add_scgraph_layer(
            m, sc_od_path,
            name="SCGraph O→D path",
            show=False,
            step_km=20.0, weight=5, opacity=0.9, dash_array="8,6", color="#ff7f0e"
        )

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)

    # ---------------- 指標與 meta ----------------
    total_km_original = sum(gc_distance_km(a, b) for (a, b) in final_segments)

    def _total_simplified(simplified_pts: List[Tuple[float,float]]):
        segs: List[Tuple[Tuple[float,float], Tuple[float,float]]] = []
        if moved_o and origin != origin_adj: segs.append((origin, origin_adj))
        if simplified_pts and len(simplified_pts) >= 2: segs += list(zip(simplified_pts[:-1], simplified_pts[1:]))
        if moved_d and dest_adj != dest: segs.append((dest_adj, dest))
        return sum(gc_distance_km(a, b) for (a, b) in segs)

    total_km_simplified = _total_simplified(chosen_simplified)

    # 組出「完整原始節點序列（含港口與外推接駁）」：track_ll
    track_ll: List[Tuple[float,float]] = []
    track_ll.append(origin)
    if moved_o and origin != origin_adj:
        track_ll.append(origin_adj)

    if fallback_used and tag_best == "sc-fallback":
        for p in chosen_nodes:
            if not track_ll or p != track_ll[-1]:
                track_ll.append(p)
    else:
        for idx in chosen_path_idx:
            p = chosen_nodes[idx]
            if not track_ll or p != track_ll[-1]:
                track_ll.append(p)

    if moved_d and dest_adj != dest:
        if not track_ll or track_ll[-1] != dest_adj:
            track_ll.append(dest_adj)
        track_ll.append(dest)
    else:
        if not track_ll or track_ll[-1] != dest:
            track_ll.append(dest)

    # 組出「簡化後最終軌跡（含港口與外推接駁）」：track_simplified_ll
    track_simplified_ll: List[Tuple[float,float]] = []
    track_simplified_ll.append(origin)
    if moved_o and origin != origin_adj:
        if track_simplified_ll[-1] != origin_adj:
            track_simplified_ll.append(origin_adj)
    for p in (chosen_simplified or []):
        if not track_simplified_ll or p != track_simplified_ll[-1]:
            track_simplified_ll.append(p)
    if moved_d and dest_adj != dest:
        if not track_simplified_ll or track_simplified_ll[-1] != dest_adj:
            track_simplified_ll.append(dest_adj)
        track_simplified_ll.append(dest)
    else:
        if not track_simplified_ll or track_simplified_ll[-1] != dest:
            track_simplified_ll.append(dest)

    meta = {
        "label": ("O→D（較短）" if not fallback_used else "SC O→D fallback"),
        "total_km_original": total_km_original,
        "total_km_simplified": total_km_simplified,
        "delta_km": total_km_original - total_km_simplified,
        "moved_o": moved_o, "moved_d": moved_d,
        "origin_adj": origin_adj, "dest_adj": dest_adj,
        "feature_count": len(feature_nodes),
        "track_ll": track_ll,
        "track_simplified_ll": track_simplified_ll,
        # scgraph 監測資訊
        "scgraph_used": sc_used,
        "scgraph_error": sc_error,
        "scgraph_network_nodes": (len(sc_network["nodes"]) if sc_network and sc_network.get("nodes") else 0),
        "scgraph_network_edges": (len(sc_network["edges"]) if sc_network and sc_network.get("edges") else 0),
        "scgraph_od_used": bool(fallback_used),
        "scgraph_od_len_km": None if not sc_od_path else None,  # 若上游提供長度可填
        "fallback_reason": fallback_reason,
    }

    return (chosen_simplified or []), total_km_simplified, str(out_html), meta
