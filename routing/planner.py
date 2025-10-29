# routing/planner.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Tuple, List, Dict
from pathlib import Path
import folium

from .config import (
    PAD_DEG, AVOID_KM, NEIGHBOR_K, LVS_MAX_NODES, SIMPLIFY_MAX_PASSES, DRAW_STEP_KM
)
from .geodesy import gc_distance_km, geodesic_midpoint, normalize_lon_to_pacific_view
from .land_layers import (
    dynamic_bboxes_idl, union_lonlat_bboxes, load_polys_in_bboxes,
    build_land_layers, build_land_strtree, nudge_to_ring_if_inside_fast
)
from .features import extract_feature_points_bbox
from .neighbors import MixedNeighbors
from .visibility import PreparedCollisionVisibility
from .simplify import simplify_path_gc
from .draw import convert_geom_to_pacific, draw_gc_polyline_continuous
from .lvs import lazy_visibility_search, make_inject_gateways_fn

def plan_route(
    origin: Tuple[float,float],
    dest: Tuple[float,float],
    land_path: str | Path,
    out_html: str | Path,
    add_feature_layer: bool = True,
):
    """
    雙向 (O→D / D→O) → 各自簡化 → 比較簡化後總距離 → 輸出較短者到地圖
    回傳: (waypoints_ll, total_km_simplified, html_path, meta_dict)
    """
    land_path = Path(land_path); out_html = Path(out_html)

    # 1) land & layers
    bboxes = dynamic_bboxes_idl(origin, dest, pad_deg=PAD_DEG)
    polys  = load_polys_in_bboxes(land_path, bboxes)
    if not polys: raise RuntimeError("No land polygons found in bbox")
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

    # 2) O/D nudge
    origin_adj, moved_o = nudge_to_ring_if_inside_fast(origin, INNER_RING_M, TARGET_BOUNDARY_M)
    dest_adj,   moved_d = nudge_to_ring_if_inside_fast(dest,   INNER_RING_M, TARGET_BOUNDARY_M)

    # 3) features
    bbox_union_ll = union_lonlat_bboxes(bboxes)
    feat = extract_feature_points_bbox(land_path, bbox_union_ll, avoid_km=AVOID_KM)
    feature_nodes = list(feat["convex_peaks"]) + list(feat["convex"])

    # 4) nodes
    base_nodes = [origin_adj, dest_adj] + feature_nodes
    nodes=[nudge_to_ring_if_inside_fast(p, INNER_RING_M, TARGET_BOUNDARY_M)[0] for p in base_nodes]
    if len(nodes) > LVS_MAX_NODES: nodes = nodes[:LVS_MAX_NODES]
    O_idx, D_idx = 0, 1

    # 5) neighbors + visibility + inject
    neighbors_obj = MixedNeighbors(nodes, D_idx=D_idx, k=NEIGHBOR_K, sc_adj=None)
    visibility = PreparedCollisionVisibility(
        COLLISION_PREP_M, land_tree=land_tree,
        to_metric=lambda g: __to_metric(g)  # 小包裝供 visibility 使用
    )
    inject_fn = make_inject_gateways_fn(
        UNION_M,
        {"convex_peaks": feat["convex_peaks"], "convex": feat["convex"]},
        take_each=3,
        inner_ring_m=INNER_RING_M,
        target_boundary_m=TARGET_BOUNDARY_M
    )

    # 6) bidirectional LVS (tolerate one-sided failure)
    results=[]; err_fwd=None; err_rev=None

    def run_once(nodes_in, O, D):
        nodes_local = list(nodes_in)
        nb = MixedNeighbors(nodes_local, D_idx=D, k=NEIGHBOR_K, sc_adj=None)
        vis = PreparedCollisionVisibility(COLLISION_PREP_M, land_tree=land_tree, to_metric=lambda g: __to_metric(g))
        path = lazy_visibility_search(
            nodes_local, O, D, nb.neighbors_of, vis, inject_fn, max_iters=5000, progress=None
        )
        return path, nodes_local

    # forward
    try:
        path_fwd, nodes_fwd = run_once(nodes, O_idx, D_idx)
        results.append(("fwd", path_fwd, nodes_fwd))
    except Exception as e:
        err_fwd = e
    # reverse
    try:
        nodes_rev = list(nodes); nodes_rev[0], nodes_rev[1] = nodes_rev[1], nodes_rev[0]
        path_rev_do, nodes_do = run_once(nodes_rev, 0, 1)
        path_rev = list(reversed(path_rev_do))
        results.append(("rev", path_rev, nodes_do))
    except Exception as e:
        err_rev = e

    if not results:
        raise RuntimeError(f"LVS failed in both directions. fwd={err_fwd}, rev={err_rev}")

    # 7) simplify & choose shorter
    def total_km_after_simplify(simplified_pts):
        segs=[]
        if moved_o and origin != origin_adj: segs.append((origin, origin_adj))
        if simplified_pts and len(simplified_pts)>=2: segs.extend(zip(simplified_pts[:-1], simplified_pts[1:]))
        if moved_d and dest_adj != dest: segs.append((dest_adj, dest))
        return sum(gc_distance_km(a,b) for (a,b) in segs)

    cand=[]
    for tag, p_idx, node_list in results:
        vis2 = PreparedCollisionVisibility(COLLISION_PREP_M, land_tree=land_tree, to_metric=lambda g: __to_metric(g))
        simp = simplify_path_gc(p_idx, node_list, visibility=vis2, max_passes=SIMPLIFY_MAX_PASSES)
        cand.append((tag, p_idx, node_list, simp))

    best=None
    for tag, p_idx, node_list, simp in cand:
        dist = total_km_after_simplify(simp)
        if (best is None) or (dist < best[0]):
            best = (dist, tag, p_idx, node_list, simp)

    total_simple_best, tag_best, chosen_path_idx, chosen_nodes, chosen_simplified = best
    label = ("O→D（較短）" if tag_best=="fwd" else "D→O（較短）")

    # 8) draw map
    mid_lon, mid_lat = geodesic_midpoint(origin, dest)
    center_lon_pacific = normalize_lon_to_pacific_view(mid_lon)
    m = folium.Map(location=[mid_lat, center_lon_pacific], zoom_start=3, max_bounds=False, world_copy_jump=False, no_wrap=False, min_lon=0, max_lon=360)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Ocean_Basemap/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Ocean Basemap', overlay=False, control=True, no_wrap=False
    ).add_to(m)
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='© OpenStreetMap', name='OpenStreetMap', overlay=False, control=True, no_wrap=False
    ).add_to(m)

    folium.GeoJson(convert_geom_to_pacific(land_raw_wgs), name="陸地", style_function=lambda x: {"color":"#2ca02c","weight":1,"fillOpacity":0.15}).add_to(m)
    folium.GeoJson(convert_geom_to_pacific(RING_WGS), name="航道緩衝區 5km", style_function=lambda x: {"color":"#6a5acd","weight":2,"fillOpacity":0.05}).add_to(m)

    # Markers
    folium.Marker([origin[1], normalize_lon_to_pacific_view(origin[0])], tooltip=f"起點: ({origin[0]:.2f}, {origin[1]:.2f})", icon=folium.Icon(color='green', icon='ship', prefix='fa')).add_to(m)
    folium.Marker([dest[1], normalize_lon_to_pacific_view(dest[0])], tooltip=f"終點: ({dest[0]:.2f}, {dest[1]:.2f})", icon=folium.Icon(color='red', icon='anchor', prefix='fa')).add_to(m)

    # Reference GC
    from .geodesy import geodesic_sample
    draw_gc_polyline_continuous(m, origin, dest, step_km=80.0, color='gray', weight=2, opacity=0.4, dash_array="8,4")

    # Feature layer (optional)
    if add_feature_layer:
        fg_feat = folium.FeatureGroup(name="候選特徵點（凸峰+凸點）", show=False)
        for (lon,lat) in (list(feat["convex_peaks"])+list(feat["convex"])):
            folium.CircleMarker([lat, normalize_lon_to_pacific_view(lon)], radius=3, color="#1f77b4", fill=True, fill_opacity=0.8, tooltip=f"Feature ({lon:.3f},{lat:.3f})").add_to(fg_feat)
        fg_feat.add_to(m)

    # blue: chosen original (with possible port<->adj segments)
    final_segments=[]
    if moved_o and origin != origin_adj:
        final_segments.append((origin, origin_adj))
        draw_gc_polyline_continuous(m, origin, origin_adj, step_km=DRAW_STEP_KM, color='#1f77b4', weight=5, opacity=0.9)
    for u,v in zip(chosen_path_idx[:-1], chosen_path_idx[1:]):
        a=chosen_nodes[u]; b=chosen_nodes[v]; final_segments.append((a,b))
        draw_gc_polyline_continuous(m, a, b, step_km=DRAW_STEP_KM, color='#1f77b4', weight=5, opacity=0.9)
    if moved_d and dest_adj != dest:
        final_segments.append((dest_adj, dest))
        draw_gc_polyline_continuous(m, dest_adj, dest, step_km=DRAW_STEP_KM, color='#1f77b4', weight=5, opacity=0.9)

    # red: simplified overlay
    if chosen_simplified and len(chosen_simplified)>=2:
        fg_simplified = folium.FeatureGroup(name="簡化後航線 (可視直連)", show=True)
        if moved_o and origin != origin_adj:
            draw_gc_polyline_continuous(fg_simplified, origin, origin_adj, step_km=DRAW_STEP_KM, color='#d62728', weight=4, opacity=0.8, dash_array="6,4")
        for a,b in zip(chosen_simplified[:-1], chosen_simplified[1:]):
            draw_gc_polyline_continuous(fg_simplified, a, b, step_km=DRAW_STEP_KM, color='#d62728', weight=6, opacity=0.9)
        if moved_d and dest_adj != dest:
            draw_gc_polyline_continuous(fg_simplified, dest_adj, dest, step_km=DRAW_STEP_KM, color='#d62728', weight=4, opacity=0.8, dash_array="6,4")
        fg_simplified.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)

    total_km_original = sum(gc_distance_km(a,b) for (a,b) in final_segments)
    def _total_simplified(chosen_simplified):
        segs=[]
        if moved_o and origin != origin_adj: segs.append((origin, origin_adj))
        if chosen_simplified and len(chosen_simplified)>=2: segs += list(zip(chosen_simplified[:-1], chosen_simplified[1:]))
        if moved_d and dest_adj != dest: segs.append((dest_adj, dest))
        return sum(gc_distance_km(a,b) for (a,b) in segs)
    total_km_simplified = _total_simplified(chosen_simplified)

    # 9) 組合 meta 與輸出座標（完整/簡化）
    meta = {
        "label": label,
        "total_km_original": total_km_original,
        "total_km_simplified": total_km_simplified,
        "delta_km": total_km_original - total_km_simplified,
        "moved_o": moved_o, "moved_d": moved_d,
        "origin_adj": origin_adj, "dest_adj": dest_adj,
        "feature_count": len(feature_nodes),
    }

    # 完整軌跡（含港口 <-> 外推點）
    track_ll=[]
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

    # 簡化後最終軌跡（含接駁）
    track_simplified_ll=[]
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

    meta["track_ll"] = track_ll
    meta["track_simplified_ll"] = track_simplified_ll

    return chosen_simplified, total_km_simplified, str(out_html), meta

# visibility 需要 metric 轉換；用 land_layers 的小工具（避免循環匯入）
def __to_metric(g):
    import shapely
    from .geodesy import to_m
    return shapely.ops.transform(lambda x,y: to_m(x,y), g)
