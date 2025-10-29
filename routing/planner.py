# routing/planner.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import folium

from .config import (
    BUFFER_KM, LVS_MAX_NODES, PAD_DEG, AVOID_KM,
    DRAW_STEP_KM, SIMPLIFY_MAX_PASSES
)
from .geodesy import great_circle_midpoint, normalize_lon_to_pacific_view, gc_distance_km
from .land_layers import (
    dynamic_bboxes_idl, load_polys_in_bboxes, build_land_layers,
    build_land_strtree, nudge_to_ring_if_inside_fast
)
from .features import extract_feature_points_bbox
from .visibility import visible
from .lvs import lazy_visibility_search, make_inject_gateways_fn
from .draw import draw_gc_polyline_continuous, convert_geom_to_pacific, add_scgraph_layer
from .simplify import simplify_path_gc
from .scgraph_bridge import sc_shortest_path_lonlat


def plan_route(
    origin: Tuple[float, float],
    dest: Tuple[float, float],
    land_path: str | Path,
    out_html: str | Path,
    add_feature_layer: bool = True,
    # === scgraph 混合：白名單邊方案 ===
    use_scgraph: bool = True,
    sc_node_connect_k: int = 0,       # 目前不做 K 近鄰接點；保留參數（未用）
    sc_kwargs: Optional[Dict] = None, # 轉交 sc_shortest_path_lonlat 的參數
):
    """
    路徑規劃流程：
      1) 雙向 (O→D / D→O) Lazy Visibility Search
      2) 各自做可視直連簡化 simplify_path_gc
      3) 比較「含接駁段」之簡化後總距離，取較短者
      4) 產出 folium 地圖
      5) 回傳 (waypoints_ll, total_km_simplified, html_path, meta_dict)

    混合 scgraph（可選）：
      - 以 scgraph 先取一條 (lon,lat) polyline
      - 將該 polyline 加入 nodes，並把相鄰點對標記為白名單邊（不做碰撞檢查）
      - 不對 scgraph 給權重偏好；是否走該邊仍由 A*/啟發式決定
    """
    land_path = Path(land_path)
    out_html = Path(out_html)

    # ---------------- 1) Land & layers ----------------
    bboxes = dynamic_bboxes_idl(origin, dest, pad_deg=PAD_DEG)
    polys = load_polys_in_bboxes(land_path, bboxes)
    if not polys:
        raise RuntimeError("No land polygons found in bbox")
    layers = build_land_layers(polys)
    UNION_M = layers["UNION_M"]
    COLLISION_PREP_M = layers["COLLISION_PREP_M"]
    RING_WGS = layers["RING_WGS"]
    land_raw_wgs = layers["LAND_RAW_WGS"]
    LAND_PARTS_M = layers["LAND_PARTS_M"]
    INNER_RING_M = layers["RING_M"]
    TARGET_RING_M = UNION_M.buffer(AVOID_KM * 1000.0)
    TARGET_BOUNDARY_M = TARGET_RING_M.boundary
    land_tree = build_land_strtree(LAND_PARTS_M)

    # ---------------- 2) O/D nudge ----------------
    origin_adj, moved_o = nudge_to_ring_if_inside_fast(origin, INNER_RING_M, TARGET_BOUNDARY_M)
    dest_adj,   moved_d = nudge_to_ring_if_inside_fast(dest,   INNER_RING_M, TARGET_BOUNDARY_M)

    # ---------------- 3) Features ----------------
    bbox_union_ll = None
    try:
        from shapely.ops import unary_union  # 僅為穩妥；features 已內部 union
        bbox_union_ll = None
    except Exception:
        pass
    feat = extract_feature_points_bbox(
        land_path=land_path,
        bbox_ll_polygon=None,          # 模組內會處理 bbox 集合
        avoid_km=AVOID_KM
    )
    feature_nodes = list(feat.get("convex_peaks", [])) + list(feat.get("convex", []))

    # ---------------- 4) Nodes 基礎集 ----------------
    base_nodes = [origin_adj, dest_adj] + feature_nodes
    nodes: List[Tuple[float, float]] = [
        nudge_to_ring_if_inside_fast(p, INNER_RING_M, TARGET_BOUNDARY_M)[0] for p in base_nodes
    ]
    if len(nodes) > LVS_MAX_NODES:
        nodes = nodes[:LVS_MAX_NODES]
    O_idx, D_idx = 0, 1

    # ---------------- 5) inject gateways for LVS ----------------
    inject_fn = make_inject_gateways_fn(
        UNION_M,
        {"convex_peaks": feat.get("convex_peaks", []), "convex": feat.get("convex", [])},
        take_each=3,
        inner_ring_m=INNER_RING_M,
        target_boundary_m=TARGET_BOUNDARY_M,
    )

    # ---------------- [新增] 6) 併入 scgraph 白名單邊 ----------------
    
    SC_WHITELIST_EDGES = set()
    def _r(p, nd=6): return (round(p[0], nd), round(p[1], nd))

    #sc_track = None                # ← 供後續地圖圖層使用
    sc_used = False                # ← debug: 是否真的啟用過 scgraph
    sc_error = None                # ← debug: 錯誤訊息（若有）

    sc_track: Optional[List[Tuple[float,float]]] = None  # <== 新增：留給地圖層顯示

    if use_scgraph:
        params = dict(
            output_units="km",
            node_addition_lat_lon_bound=180.0,
            node_addition_type="quadrant",
            destination_node_addition_type="all",
            node_addition_circuity=10.0,
            cache=False,
        )
        if sc_kwargs:
            params.update(sc_kwargs)
        try:
            sc_out = sc_shortest_path_lonlat(origin_adj, dest_adj, **params)
            sc_track = sc_out["track"]  # [(lon,lat), ...]
            sc_used = True
            print(f"[SCGRAPH] track points = {len(sc_track)}, length_km = {sc_out.get('length_km'):.1f}")
            # 併入 nodes（去重）
            exist = set(_r(p) for p in nodes)
            for p in sc_track:
                rp = _r(p)
                if rp not in exist:
                    nodes.append(p)
                    exist.add(rp)
            # 白名單邊
            for a, b in zip(sc_track[:-1], sc_track[1:]):
                SC_WHITELIST_EDGES.add((_r(a), _r(b)))
                SC_WHITELIST_EDGES.add((_r(b), _r(a)))
        except Exception as e:
            sc_error = str(e)
            print(f"[WARN] scgraph 取得路徑失敗：{e}")

    # ---------------- 7) 可視封裝：白名單邊免碰撞 ----------------
    def visible_wrapper(a, b, cp, tree):
        if use_scgraph:
            ra, rb = _r(a), _r(b)
            if (ra, rb) in SC_WHITELIST_EDGES:
                return True
        return visible(a, b, cp, tree)

    # ---------------- 8) 單次搜尋 ----------------
    def run_once(nodes_in: List[Tuple[float, float]], O: int, D: int):
        nodes_local = list(nodes_in)
        path = lazy_visibility_search(
            nodes_local, O, D, visible_wrapper, COLLISION_PREP_M, land_tree, inject_fn,
            max_iters=5000, progress=None
        )
        return path, nodes_local

    # ---------------- 9) 雙向 LVS（允許一側失敗） ----------------
    results = []
    err_fwd = err_rev = None
    try:
        path_fwd, nodes_fwd = run_once(nodes, O_idx, D_idx)
        results.append(("fwd", path_fwd, nodes_fwd))
    except Exception as e:
        err_fwd = e

    try:
        nodes_rev = list(nodes)
        nodes_rev[0], nodes_rev[1] = nodes_rev[1], nodes_rev[0]
        path_rev_do, nodes_do = run_once(nodes_rev, 0, 1)
        path_rev = list(reversed(path_rev_do))
        results.append(("rev", path_rev, nodes_do))
    except Exception as e:
        err_rev = e

    if not results:
        raise RuntimeError(f"LVS failed in both directions. fwd={err_fwd}, rev={err_rev}")

    # ---------------- 10) 簡化並比較距離 ----------------
    cand = []
    for tag, p_idx, node_list in results:
        simp = simplify_path_gc(
            p_idx, node_list, visible_wrapper, COLLISION_PREP_M, land_tree,
            max_passes=SIMPLIFY_MAX_PASSES
        )
        cand.append((tag, p_idx, node_list, simp))

    def total_km_after_simplify(simplified_pts: List[Tuple[float,float]]) -> float:
        segs: List[Tuple[Tuple[float,float], Tuple[float,float]]] = []
        if moved_o and origin != origin_adj:
            segs.append((origin, origin_adj))
        if simplified_pts and len(simplified_pts) >= 2:
            segs.extend(zip(simplified_pts[:-1], simplified_pts[1:]))
        if moved_d and dest_adj != dest:
            segs.append((dest_adj, dest))
        return sum(gc_distance_km(a, b) for (a, b) in segs)

    best = None
    for tag, p_idx, node_list, simp in cand:
        dist = total_km_after_simplify(simp)
        if (best is None) or (dist < best[0]):
            best = (dist, tag, p_idx, node_list, simp)

    total_simple_best, tag_best, chosen_path_idx, chosen_nodes, chosen_simplified = best
    label = ("O→D（較短）" if tag_best == "fwd" else "D→O（較短）")

    # ---------------- 11) folium 地圖 ----------------
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

    # 參考大圓
    from .geodesy import geodesic_sample  # 僅用於畫參考
    draw_gc_polyline_continuous(
        m, origin, dest, step_km=80.0,
        color='gray', weight=2, opacity=0.4, dash_array="8,4"
    )

    # 候選特徵點（可選）
    if add_feature_layer:
        fg_feat = folium.FeatureGroup(name="候選特徵點（凸峰+凸點）", show=False)
        for (lon, lat) in (list(feat.get("convex_peaks", [])) + list(feat.get("convex", []))):
            folium.CircleMarker(
                [lat, normalize_lon_to_pacific_view(lon)], radius=3,
                color="#1f77b4", fill=True, fill_opacity=0.8,
                tooltip=f"Feature ({lon:.3f},{lat:.3f})"
            ).add_to(fg_feat)
        fg_feat.add_to(m)

    # 原始（被選擇之）LVS 路徑（藍）
    final_segments: List[Tuple[Tuple[float,float], Tuple[float,float]]] = []
    if moved_o and origin != origin_adj:
        final_segments.append((origin, origin_adj))
        draw_gc_polyline_continuous(
            m, origin, origin_adj, step_km=DRAW_STEP_KM,
            color='#1f77b4', weight=5, opacity=0.9
        )
    for u, v in zip(chosen_path_idx[:-1], chosen_path_idx[1:]):
        a = chosen_nodes[u]; b = chosen_nodes[v]
        final_segments.append((a, b))
        draw_gc_polyline_continuous(
            m, a, b, step_km=DRAW_STEP_KM,
            color='#1f77b4', weight=5, opacity=0.9
        )
    if moved_d and dest_adj != dest:
        final_segments.append((dest_adj, dest))
        draw_gc_polyline_continuous(
            m, dest_adj, dest, step_km=DRAW_STEP_KM,
            color='#1f77b4', weight=5, opacity=0.9
        )

    # 簡化覆蓋（紅）
    if chosen_simplified and len(chosen_simplified) >= 2:
        fg_simplified = folium.FeatureGroup(name="簡化後航線 (可視直連)", show=True)
        if moved_o and origin != origin_adj:
            draw_gc_polyline_continuous(
                fg_simplified, origin, origin_adj, step_km=DRAW_STEP_KM,
                color='#d62728', weight=4, opacity=0.8, dash_array="6,4"
            )
        for a, b in zip(chosen_simplified[:-1], chosen_simplified[1:]):
            draw_gc_polyline_continuous(
                fg_simplified, a, b, step_km=DRAW_STEP_KM,
                color='#d62728', weight=6, opacity=0.9
            )
        if moved_d and dest_adj != dest:
            draw_gc_polyline_continuous(
                fg_simplified, dest_adj, dest, step_km=DRAW_STEP_KM,
                color='#d62728', weight=4, opacity=0.8, dash_array="6,4"
            )
        fg_simplified.add_to(m)


        # 若有 scgraph 路徑，加入可切換圖層（橘色虛線）
    if sc_track and len(sc_track) >= 2:
        add_scgraph_layer(
            m, sc_track,
            name="SCGraph Path",
            show=True,            # 預設不開；你要預設顯示就改 True
            step_km=20.0,
            weight=4,
            opacity=0.9,
            dash_array="8,6",
            color="#ff7f0e",
        )
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)

    # ---------------- 12) 指標與 meta ----------------
    total_km_original = sum(gc_distance_km(a, b) for (a, b) in final_segments)

    def _total_simplified(simplified_pts: List[Tuple[float,float]]):
        segs: List[Tuple[Tuple[float,float], Tuple[float,float]]] = []
        if moved_o and origin != origin_adj:
            segs.append((origin, origin_adj))
        if simplified_pts and len(simplified_pts) >= 2:
            segs += list(zip(simplified_pts[:-1], simplified_pts[1:]))
        if moved_d and dest_adj != dest:
            segs.append((dest_adj, dest))
        return sum(gc_distance_km(a, b) for (a, b) in segs)

    total_km_simplified = _total_simplified(chosen_simplified)

    # 組出「完整原始節點序列（含港口與外推接駁）」：track_ll
    track_ll: List[Tuple[float,float]] = []
    track_ll.append(origin)
    if moved_o and origin != origin_adj:
        track_ll.append(origin_adj)
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
        "label": label,
        "total_km_original": total_km_original,
        "total_km_simplified": total_km_simplified,
        "delta_km": total_km_original - total_km_simplified,
        "moved_o": moved_o, "moved_d": moved_d,
        "origin_adj": origin_adj, "dest_adj": dest_adj,
        "feature_count": len(feature_nodes),
        "track_ll": track_ll,                       # 原始被選路徑（含接駁）
        "track_simplified_ll": track_simplified_ll, # 簡化後最終軌跡（含接駁）
        "scgraph_used": sc_used,
        "scgraph_error": sc_error,
        "scgraph_track_len": (len(sc_track) if sc_track else 0),
    }

    return chosen_simplified, total_km_simplified, str(out_html), meta
