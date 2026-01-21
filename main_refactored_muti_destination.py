"""
Slab Routing Demo - Refactored
主要功能：讀取 AOI 地圖資料、計算航海路徑、並透過 GUI 介面展示結果。
"""

# ==============================================================================
# SECTION 1: IMPORTS (套件匯入)
# ==============================================================================

# 1.1 標準函式庫 (Standard Library)
import os
import sys
import pickle
import webbrowser
from pathlib import Path
from collections import Counter

# 1.2 第三方科學與地理運算庫 (Third-party Science/Geo)
import numpy as np
import pandas as pd
import networkx as nx
import folium
from shapely.geometry import LineString, box as shp_box
from shapely.prepared import prep
from shapely.ops import transform as shp_transform

# 1.3 GUI 介面庫 (PyQt6)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTextEdit, QFrame, QMessageBox)
from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

# 1.4 專案內部模組 (Local Modules - routing_map)
import routing_map
from routing_map import build_aoi, RoutingMapConfig
from routing_map.config import AoiConfig, LandConfig
from routing_map.path_simplifier import simplify_path_visibility
from routing_map.routing_graph import build_base_graph, haversine_km
from routing_map.c_gateb_connectors import (
    build_cnode_gateb_connectors_nearest,
    add_cnode_gateb_connectors_to_graph,
)
from routing_map.snap import snap_pair_component_aware, inject_point_edges
from routing_map.repairer import PathRepairer, RepairConfig
from routing_map.snap_link_repair import repair_snap_link_ll_if_needed
from routing_map.metrics import path_length_km_nm, format_distance


# ==============================================================================
# SECTION 2: CONFIGURATION (全域設定與參數)
# ==============================================================================

CACHE_FILE = "aoi_cache.pkl"

# 路由地圖建置設定
CFG = RoutingMapConfig(
    aoi=AoiConfig(
        # 格式：(最小經度-左邊界, 最小緯度-下邊界, 最大經度-右邊界, 最大緯度-上邊界)
        bbox_ll=(10, -50, 160, 50),
    ),
    land=LandConfig(
        # 請根據實際環境調整路徑
        shp_path=Path(r"C:\Users\slab\Desktop\Slab Project\Stage1\data\Land\ne_10m_land.shp"), 
        buffer_km=20.0,
        avoid_km=1,
        collision_safety_km=0.5,
    ),
)


# ==============================================================================
# SECTION 3: DATA LOADING & CACHING (資料載入邏輯)
# ==============================================================================

def load_or_build_data(config, cache_path):
    """
    嘗試從快取讀取 AOI 資料，若不存在則重新計算並存檔。
    處理了 Shapely Prepared Geometry 無法 Pickle 的問題。
    """
    if os.path.exists(cache_path):
        print(f"[{cache_path}] 存在，正在讀取快取資料...")
        with open(cache_path, "rb") as f:
            out = pickle.load(f)
        
        # 重建無法被儲存的物件 (Prepared Geometry)
        if "layers" in out and "COLLISION_M" in out["layers"]:
            print("正在重建 Collision Prep 物件...")
            out["collision_prep"] = prep(out["layers"]["COLLISION_M"])
            
        print("讀取完成！")
        return out

    else:
        print(f"[{cache_path}] 不存在，開始執行 build_aoi...")
        
        # 執行原本的建置流程
        out = build_aoi(config)
        
        # 儲存前的處理 (建立副本並移除不可 pickle 的物件)
        out_to_save = out.copy()
        if "collision_prep" in out_to_save:
            del out_to_save["collision_prep"]
        
        print(f"正在將資料儲存至 [{cache_path}] ...")
        with open(cache_path, "wb") as f:
            pickle.dump(out_to_save, f)
        print("儲存完成！")
        return out


# ==============================================================================
# SECTION 4: HELPER FUNCTIONS (通用工具函式)
# ==============================================================================

# --- 4.1 基礎幾何與資料處理工具 ---
def in_bbox(p, bbox_ll):
    if bbox_ll is None:
        return True
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_ll)
    lon, lat = float(p[0]), float(p[1])
    return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)

def safe_df(out, name):
    """安全地從 out 字典取得 DataFrame，若無資料回傳 None"""
    df = out.get(name, None)
    if df is None:
        return None
    try:
        return df if len(df) > 0 else None
    except Exception:
        return None

def unwrap_lon(lon, ref_lon):
    """處理經度跨越換日線的問題 (正規化到 -180~180 或連續區間)"""
    lon = float(lon); ref_lon = float(ref_lon)
    d = lon - ref_lon
    if d > 180: lon -= 360
    if d < -180: lon += 360
    return lon

def route_polyline_dateline_safe(path_ll):
    """將路徑點序列處理為適合繪圖的格式 (避免橫跨地圖邊緣的長線)"""
    if not path_ll:
        return []
    out = []
    ref = float(path_ll[0][0])
    for lon, lat in path_ll:
        lon_u = unwrap_lon(lon, ref)
        out.append((lon_u, float(lat)))
        ref = lon_u
    return out

# --- 4.2 Gate (港口/閘門) 資料處理工具 ---
def build_gate_xy(out):
    # 分開判斷，避免觸發 DataFrame 的布林轉換錯誤
    gate_df = safe_df(out, "Gate_all_cov")
    if gate_df is None:
        gate_df = safe_df(out, "Gate_all")

    if gate_df is None or "g_id" not in gate_df.columns:
        return {}
    return {int(r["g_id"]): (float(r["lon"]), float(r["lat"])) for _, r in gate_df.iterrows()}

def get_gateB_df(out, gate_xy):
    gb_obj = out.get("Gate_B_kept_gates", None)
    dfGB = None

    if isinstance(gb_obj, pd.DataFrame):
        dfGB = gb_obj.copy()
        if "lon" not in dfGB.columns or "lat" not in dfGB.columns:
            if "g_id" in dfGB.columns and gate_xy:
                dfGB["lon"] = dfGB["g_id"].map(lambda x: gate_xy.get(int(x), (np.nan, np.nan))[0])
                dfGB["lat"] = dfGB["g_id"].map(lambda x: gate_xy.get(int(x), (np.nan, np.nan))[1])
        dfGB = dfGB.dropna(subset=["lon", "lat"])
    elif isinstance(gb_obj, (list, set, tuple, np.ndarray)):
        gids = [int(x) for x in gb_obj]
        rows = [{"g_id": gid, "lon": gate_xy[gid][0], "lat": gate_xy[gid][1]} for gid in gids if gid in gate_xy]
        dfGB = pd.DataFrame(rows)

    if dfGB is None:
        gb2 = safe_df(out, "Gate_B")
        if gb2 is not None:
            dfGB = gb2.copy()
    return dfGB if (dfGB is not None and len(dfGB) > 0) else None

# --- 4.3 圖論 Edge 轉座標工具 ---
def parse_node_id_str(s):
    try:
        a, b = s.split(",")
        return float(a), float(b)
    except Exception:
        return None

def edge_to_lonlat(e, *, nodes_df=None, idx_to_lonlat_fn=None):
    if not isinstance(e, (list, tuple)) or len(e) < 2:
        return None
    a, b = e[0], e[1]
    # Handle Int IDs
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        if nodes_df is None or idx_to_lonlat_fn is None: return None
        try: return (idx_to_lonlat_fn(a), idx_to_lonlat_fn(b))
        except Exception: return None
    # Handle Tuple Coords
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        try: return ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
        except Exception: return None
    # Handle String IDs
    if isinstance(a, str) and isinstance(b, str):
        p1, p2 = parse_node_id_str(a), parse_node_id_str(b)
        if p1 and p2: return (p1, p2)
    return None

# --- 4.4 投影 (Projection) 與碰撞 (Collision) 工具 ---
def _make_ll_m_projectors_from_out(out):
    """從 out['proj'] 建立經緯度 <-> 公尺 (XY) 的轉換函數"""
    proj = out.get("proj", None)
    if proj is None:
        raise ValueError("out['proj'] not found.")

    def _apply(fn, a, b):
        if hasattr(fn, "transform") and callable(getattr(fn, "transform")):
            return fn.transform(a, b)
        if callable(fn):
            return fn(a, b) if hasattr(fn, "__call__") else fn((a, b))
        raise TypeError(f"Projector {type(fn)} invalid")

    candidates = [("ll_to_xy", "xy_to_ll"), ("ll_to_m", "m_to_ll"), ("to_m", "to_ll"), ("fwd", "inv")]
    for a, b in candidates:
        if hasattr(proj, a) and hasattr(proj, b):
            # -----
            print(f"\n[PROJECTOR] Using methods: {a} / {b}")
            # ----
            f, g = getattr(proj, a), getattr(proj, b)
            # ------------------------------------------
            print(f"[PROJECTOR] Type of f: {type(f)}")
            print(f"[PROJECTOR] f = {f}")
            try:
                if hasattr(f, "transform"):
                    test_result = f.transform(119.7734, 4.7502)
                else:
                    test_result = f(119.7734, 4.7502)
                print(f"[PROJECTOR] Direct call result = {test_result}")
            except Exception as e:
                print(f"[PROJECTOR] Direct call failed: {e}")
            # ------------------------------------------

            ll2m_xy = lambda lon, lat: tuple(map(float, _apply(f, float(lon), float(lat))))
            m2ll_xy = lambda x, y: tuple(map(float, _apply(g, float(x), float(y))))
            return ll2m_xy, m2ll_xy, lambda p: ll2m_xy(p[0], p[1]), lambda q: m2ll_xy(q[0], q[1])

    raise ValueError("Cannot infer projection methods.")

def _get_collision_metric(out):
    """取得公尺單位的碰撞層 (Collision Metric)"""
    layers = out.get("layers", None)
    if isinstance(layers, dict) and layers.get("COLLISION_M") is not None:
        return layers["COLLISION_M"]
    
    # 依序檢查，找到非 None 就回傳
    c = out.get("COLLISION_M")
    if c is not None: return c
    
    c = out.get("collision_m")
    if c is not None: return c
    
    if "collision_prep" in out:
        return out["collision_prep"].context
        
    return out.get("collision")

def _get_densified_metric_box(bbox_ll, ll2m_xy, step_deg=2.0, pad_m=50_000.0):
    """對 AOI 邊界進行密集取樣，建立正確的 Metric Bounding Box"""
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_ll)
    xs, ys = [], []
    # Sampling logic omitted for brevity, keeping original logic essentially
    lons = np.append(np.arange(min_lon, max_lon, step_deg), max_lon)
    lats = np.append(np.arange(min_lat, max_lat, step_deg), max_lat)
    
    for lon in lons:
        for lat in [min_lat, max_lat]:
            x, y = ll2m_xy(lon, lat)
            xs.append(x); ys.append(y)
    for lat in lats:
        for lon in [min_lon, max_lon]:
            x, y = ll2m_xy(lon, lat)
            xs.append(x); ys.append(y)
            
    return shp_box(min(xs), min(ys), max(xs), max(ys)).buffer(pad_m)

def _clip_collision_to_aoi_bbox(collision_m, bbox_ll, ll2m_xy, pad_m=80_000.0):
    """裁切碰撞層只保留 AOI 範圍內，優化效能"""
    if collision_m is None or bbox_ll is None: return collision_m
    # 簡化：不處理跨越換日線的複雜裁切
    aoi_box_m = _get_densified_metric_box(bbox_ll, ll2m_xy, pad_m=pad_m)
    try:
        c2 = collision_m.intersection(aoi_box_m)
        return c2 if c2 and not c2.is_empty else collision_m
    except Exception:
        return collision_m

def _geom_m_to_ll(geom_m, m2ll_xy):
    """將 Shapely Geometry 從公尺轉回經緯度"""
    def _xy_to_lonlat(x, y, z=None):
        a, b = m2ll_xy(float(x), float(y))
        # Heuristic to detect lat/lon swap
        if abs(a) <= 90 and abs(b) > 90: return (b, a) 
        return (a, b)
    return shp_transform(_xy_to_lonlat, geom_m)


# ==============================================================================
# SECTION 5: CORE ROUTING LOGIC (核心路徑計算)
# ==============================================================================

def open_routing_debug_map_p2p(
    out,
    origin_ll,
    dest_ll,
    *,
    html_path="aoi_p2p_map.html",
    zoom_start=5,
    c_sample=8000, s_sample=3000, max_sea_edges_viz=6000,
    include_sea=True, include_cc=True, include_gateb_sea=True,
    use_c_gateb_bridge=True, c_to_gateB_max_deg_dist=None,
    k_near=30, r_max_km_snap=150.0, k_inject=4,
    do_repair=True, do_simplify=True,
    extra_paths=None  # <--- NEW: 新增參數，用於接收額外要畫的路徑
):
    """
    核心函式：執行從 A 到 B 的完整路徑規劃與視覺化流程。
    修改版：支援 extra_paths 繪圖，並回傳路徑座標。
    """
    
    # --- 5.1 準備資料與變數 ---
    bbox_ll = out.get("bbox_ll", None)
    if bbox_ll is None: raise ValueError("out['bbox_ll'] is required.")
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_ll)
    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]

    origin_ll = (float(origin_ll[0]), float(origin_ll[1]))
    dest_ll   = (float(dest_ll[0]), float(dest_ll[1]))

    C_nodes = safe_df(out, "C_nodes")
    S_nodes = safe_df(out, "S_nodes")
    S_edges = out.get("S_edges", None)
    
    gate_xy = build_gate_xy(out)
    dfGB = get_gateB_df(out, gate_xy)

    # --- 5.2 建立基礎圖 (Base Graph) ---
    G, stats = build_base_graph(
        out,
        include_sea=bool(include_sea),
        include_cc=bool(include_cc),
        include_gateb_sea=bool(include_gateb_sea),
        include_c_gateb=False,
        bbox_ll=bbox_ll,
        weight_unit="km",
    )
    print(f"[graph] base stats: {stats}")

    # 加入 C↔GateB 連接
    if use_c_gateb_bridge and isinstance(C_nodes, pd.DataFrame) and isinstance(dfGB, pd.DataFrame):
        cgb_df = build_cnode_gateb_connectors_nearest(
            C_nodes, dfGB[["g_id", "lon", "lat"]], bbox_ll=bbox_ll, max_deg_dist=c_to_gateB_max_deg_dist,
        )
        added = add_cnode_gateb_connectors_to_graph(G, cgb_df, etype="c_gb", weight_col="dist_deg")
        print(f"[graph] C↔GateB bridge edges added: {added}")

    # --- 5.3 吸附起終點 (Snap & Inject) ---
    pair = snap_pair_component_aware(
        out, origin_ll, dest_ll,
        k_near=int(k_near), r_max_km=float(r_max_km_snap), k_inject=int(k_inject),
        prefer_ok_set=True, allow_fallback_non_ok=True, allow_radius_fallback=True, do_nudge=True,
    )

    start_key = (float(getattr(pair.start, "p_used_ll", origin_ll)[0]), float(getattr(pair.start, "p_used_ll", origin_ll)[1]))
    end_key   = (float(getattr(pair.end, "p_used_ll", dest_ll)[0]), float(getattr(pair.end, "p_used_ll", dest_ll)[1]))

    path, path_ll_for_simplify, path_simplified = None, None, None
    final_ll = None
    
    collision_used = None
    ll2m_xy, m2ll_xy = None, None

    if len(pair.start_pick) > 0 and len(pair.end_pick) > 0:
        inject_point_edges(G, start_key, pair.start_pick, k_inject=int(k_inject), etype="start_inject")
        inject_point_edges(G, end_key,   pair.end_pick,   k_inject=int(k_inject), etype="end_inject")
        print(f"[snap] OK: {pair.reason}")
        proj = out.get("proj", None)
        print("[DEBUG] proj type:", type(proj))
        print("[DEBUG] proj attrs:", dir(proj))

        # --- 5.4 A* 路徑搜尋 ---
        try:
            path = nx.astar_path(G, start_key, end_key, heuristic=lambda n1, n2: haversine_km(n1, n2), weight="weight")
            etypes = [G[u][v].get("etype") for u, v in zip(path, path[1:])]
            print(f"[route] OK | nodes={len(path)}", "| etypes=", dict(Counter(etypes)))
        except Exception as e:
            print(f"[route] FAIL: {repr(e)}")

        # --- 5.5 路徑修復 (Repair) 與 簡化 (Simplify) ---
        if path and len(path) >= 2:
            ll2m_xy, m2ll_xy, _, _ = _make_ll_m_projectors_from_out(out)
            collision = _get_collision_metric(out)
            
            _ll_to_m = lambda a, b=None: ll2m_xy(float(a[0]), float(a[1])) if b is None else ll2m_xy(float(a), float(b))
            _m_to_ll = lambda a, b=None: m2ll_xy(float(a[0]), float(a[1])) if b is None else m2ll_xy(float(a), float(b))

            if collision is None:
                print("[collision] not found -> skip repair")
                path_ll_for_simplify = [(float(p[0]), float(p[1])) for p in path]
            else:
                collision = _clip_collision_to_aoi_bbox(collision, bbox_ll, ll2m_xy)
                collision_used = collision 
                
                path_ll = [(float(p[0]), float(p[1])) for p in path]

                if do_repair:
                    repairer = PathRepairer(RepairConfig(debug=True, rb_n_samples=25, rb_max_iter=60, rb_push_step_m=250.0))
                    
                    u0, u1 = path[0], path[1]
                    v1, v0 = path[-2], path[-1]
                    is_start_inj = G[u0][u1].get("etype") == "start_inject" if G.has_edge(u0, u1) else False
                    is_end_inj   = G[v1][v0].get("etype") == "end_inject"   if G.has_edge(v1, v0) else False

                    inject_start_ll = repair_snap_link_ll_if_needed(u0, u1, collision_m=collision, ll_to_m=_ll_to_m, m_to_ll=_m_to_ll, repairer_obj=repairer) if is_start_inj else [u0, u1]
                    inject_end_ll   = repair_snap_link_ll_if_needed(v1, v0, collision_m=collision, ll_to_m=_ll_to_m, m_to_ll=_m_to_ll, repairer_obj=repairer) if is_end_inj else [v1, v0]

                    core_nodes = path[1:-1]
                    core_repaired_ll = []
                    if len(core_nodes) >= 2:
                        core_rep = repairer.repair_path(G, core_nodes, collision_m=collision, ll_to_m=_ll_to_m, m_to_ll=_m_to_ll)
                        core_repaired_ll = core_rep.path_ll

                    full = []
                    for seg in [inject_start_ll, core_repaired_ll, inject_end_ll]:
                        if not seg: continue
                        if full and full[-1] == seg[0]: full.extend(seg[1:])
                        else: full.extend(seg)
                    path_ll_for_simplify = full
                else:
                    path_ll_for_simplify = path_ll
                
                print(len(path_ll_for_simplify))
                #for i, (lon, lat) in enumerate(path_ll_for_simplify):
                    #print(f"[dump] {i:03d}: ({float(lon):.10f}, {float(lat):.10f})")
                
                p = (119.7734, 4.7502)
                print("[dbg] ll_to_m(p) =", _ll_to_m(p))


                if do_simplify and path_ll_for_simplify and len(path_ll_for_simplify) >= 2:
                    path_simplified, simp_stats = simplify_path_visibility(
                        path_ll_for_simplify, collision_m=collision, ll_to_m=_ll_to_m, m_to_ll=_m_to_ll,
                        window_size=80, max_tries=300, use_prepared_collision=True, dateline_unwrap=True
                    )
                    print(f"[simplify] {simp_stats}")
                    #for i, (lon, lat) in enumerate(path_simplified):
                        #print(f"[dump] {i:03d}: ({float(lon):.10f}, {float(lat):.10f})")

                    core_final = path_simplified if (path_simplified and len(path_simplified) >=2) else path_ll_for_simplify

                    snap_start = repair_snap_link_ll_if_needed(origin_ll, start_key, collision_m=collision, ll_to_m=_ll_to_m, m_to_ll=_m_to_ll, repairer_obj=repairer) if origin_ll != start_key else [origin_ll]
                    snap_end   = repair_snap_link_ll_if_needed(end_key, dest_ll, collision_m=collision, ll_to_m=_ll_to_m, m_to_ll=_m_to_ll, repairer_obj=repairer) if end_key != dest_ll else [dest_ll]
                    
                    merged = list(snap_start)
                    if core_final:
                        if merged and merged[-1] == core_final[0]: merged.extend(core_final[1:])
                        else: merged.extend(core_final)
                    
                    snap_end = [(float(p[0]), float(p[1])) for p in snap_end]
                    if snap_end and len(snap_end) >= 1:
                         if merged and merged[-1] == snap_end[0]: merged.extend(snap_end[1:])
                         else: merged.extend(snap_end)
                    
                    final_ll = merged
                    
                    if final_ll and len(final_ll) >= 2:
                        km, nm = path_length_km_nm(final_ll, dateline_unwrap=True)
                        print(f"[distance] {format_distance(km, nm)}")
                        out["_p2p_last_distance"] = {"km": km, "nm": nm}

    # --- 5.6 視覺化 (Folium Map) ---
    m = folium.Map(location=center, zoom_start=zoom_start, control_scale=True)
    folium.Rectangle([[min_lat, min_lon], [max_lat, max_lon]], color="black", weight=2, fill=False).add_to(m)

    # 1. Collision Layer
    if collision_used is not None and m2ll_xy is not None:
        try:
            col_ll = _geom_m_to_ll(collision_used, m2ll_xy)
            viz_box = shp_box(min_lon, min_lat, max_lon, max_lat)
            col_ll = col_ll.intersection(viz_box)
            fgCol = folium.FeatureGroup(name="Collision", show=False)
            folium.GeoJson(
                data=col_ll.__geo_interface__,
                style_function=lambda _: {"fillColor": "#3b82f6", "color": "#3b82f6", "weight": 2, "fillOpacity": 0.15},
            ).add_to(fgCol)
            fgCol.add_to(m)
        except Exception as e:
            print(f"[viz] collision layer failed: {repr(e)}")

    # 2. Edges Layer (S_edges)
    if isinstance(S_nodes, pd.DataFrame) and S_edges:
        fgE = folium.FeatureGroup(name="S_edges", show=True)
        take = S_edges[:max_sea_edges_viz]
        for e in take:
            seg = edge_to_lonlat(e, nodes_df=S_nodes, idx_to_lonlat_fn=lambda i: (float(S_nodes.iloc[int(i)].lon), float(S_nodes.iloc[int(i)].lat)))
            if seg and (in_bbox(seg[0], bbox_ll) or in_bbox(seg[1], bbox_ll)):
                folium.PolyLine([[p[1], p[0]] for p in seg], color="#3352ff", weight=1, opacity=0.5).add_to(fgE)
        fgE.add_to(m)

    # --- NEW: Extra Paths (繪製傳入的額外路徑，如：已航行路徑、廢棄路徑) ---
    if extra_paths:
        for ep in extra_paths:
            coords = ep.get('coords')
            if coords and len(coords) >= 2:
                # 處理跨越換日線
                coords_u = route_polyline_dateline_safe(coords)
                fgEx = folium.FeatureGroup(name=ep.get('name', 'Extra Path'), show=True)
                folium.PolyLine(
                    [[p[1], p[0]] for p in coords_u],
                    color=ep.get('color', 'gray'),
                    weight=ep.get('weight', 4),
                    dash_array=ep.get('dash_array', None),
                    opacity=ep.get('opacity', 0.8)
                ).add_to(fgEx)
                fgEx.add_to(m)

    # 3. Path Layers (Current Route)
    if path and len(path) >= 2:
        # A* Raw
        fgA = folium.FeatureGroup(name="Path: A* Raw", show=False)
        folium.PolyLine([[p[1], p[0]] for p in route_polyline_dateline_safe(path)], color="red", weight=4, opacity=0.6).add_to(fgA)
        fgA.add_to(m)

        fgR = folium.FeatureGroup(name="Route: Repaired (FULL, pre-simplify)", show=True)
        repaired_ll = [(float(p[0]), float(p[1])) for p in path_ll_for_simplify]
        path_ru = route_polyline_dateline_safe(repaired_ll)
        folium.PolyLine([[p[1], p[0]] for p in path_ru], color="#2ca02c", weight=5, opacity=0.90).add_to(fgR)
        fgR.add_to(m)
        
        # FINAL
        if final_ll:
             fgF = folium.FeatureGroup(name="Path: FINAL", show=True)
             folium.PolyLine([[p[1], p[0]] for p in route_polyline_dateline_safe(final_ll)], color="orange", weight=6, opacity=0.9).add_to(fgF)
             fgF.add_to(m)

    # 4. Markers
    folium.Marker([origin_ll[1], origin_ll[0]], tooltip="Start Input", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([dest_ll[1], dest_ll[0]], tooltip="End Input", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    folium.LayerControl().add_to(m)
    
    html_path = Path(html_path).resolve()
    m.save(str(html_path))
    
    # --- NEW: 修改回傳值，多回傳 final_ll ---
    return html_path, final_ll

# ==============================================================================
# SECTION 6: GUI APPLICATION (使用者介面)
# ==============================================================================

class RoutingDemoApp(QMainWindow):
    def __init__(self, routing_data_out):
        super().__init__()
        self.setWindowTitle("Slab Routing Demo - Local UI")
        self.resize(1920, 1080)
        self.out = routing_data_out
        self.default_origin = "127.09912, 13.3102"
        self.default_dest = "17.47887, 42.36624"
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- Left Panel ---
        left_panel = QFrame()
        #left_panel.setStyleSheet("background-color: #f0f0f0; border-right: 1px solid #ccc;")
        left_panel.setStyleSheet("""
            QFrame {
                background-color: #2b2f36;
                border-right: 1px solid #3a404a;
                color: #e6e6e6;
            }
            QLabel { color: #e6e6e6; }
            QLineEdit, QTextEdit {
                background-color: #1f2329;
                color: #e6e6e6;
                border: 1px solid #3a404a;
                border-radius: 4px;
                padding: 6px;
                selection-background-color: #3d6fb6;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        
        left_layout.addWidget(QLabel("<h2>Routing Control</h2>"))
        
        # Origin
        left_layout.addWidget(QLabel("<b>Origin (Lon, Lat):</b>"))
        self.input_origin = QLineEdit(self.default_origin)
        left_layout.addWidget(self.input_origin)

        # Destination A
        left_layout.addWidget(QLabel("<b>Destination A (Lon, Lat):</b>"))
        self.input_dest_a = QLineEdit(self.default_dest)
        left_layout.addWidget(self.input_dest_a)

        # Destination B (NEW)
        left_layout.addWidget(QLabel("<b>Destination B (Optional - Reroute):</b>"))
        self.input_dest_b = QLineEdit()
        self.input_dest_b.setPlaceholderText("Leave empty for normal routing")
        left_layout.addWidget(self.input_dest_b)

        self.btn_calc = QPushButton("Calculate Route")
        self.btn_calc.setStyleSheet("background-color: #007bff; color: white; padding: 10px; font-weight: bold;")
        self.btn_calc.clicked.connect(self.run_routing)
        left_layout.addWidget(self.btn_calc)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        left_layout.addWidget(line)

        # Port Reference
        left_layout.addWidget(QLabel("<b>Port Reference (Lon, Lat):</b>"))
        self.port_box = QTextEdit()
        self.port_box.setReadOnly(True)
        self.port_box.setStyleSheet("font-family: Consolas, Monospace; font-size: 11px;")
        
        port_data = """AUPWL: 117.1879, -20.6045
AUPHE: 118.5760, -20.3165
AUDAM: 116.6732, -20.6327
CNZHU: 122.2000, 29.9500
CNZUH: 113.2000, 21.9198
JPFUK: 133.4325, 34.4546
JPFNB: 139.9591, 35.6673
JPKKJ: 130.7842, 33.9400
JPKSM: 140.6833, 35.9094
JPMIZ: 133.7141, 34.5047
JPOSA: 135.4404, 34.6141
JPUKB: 135.2671, 34.6867
KRMAS: 128.5909, 35.1799
PHACY: 123.5105, 9.7090
PHMNL: 120.9445, 14.6165
PHCEB: 123.9465, 10.3150
PHCAA: 120.8098, 13.9192
TWKEL: 121.7532, 25.1452
TWKHH: 120.3181, 22.5843
TWTXG: 120.5075, 24.2550
TWTPE: 121.3910, 25.1590
TWSUO: 121.8530, 24.5985
VNDQT: 108.7873, 15.4052
VNPHU: 107.0125, 10.6097
VNNGH: 105.8149, 19.3117"""
        self.port_box.setPlainText(port_data)
        left_layout.addWidget(self.port_box, stretch=1)

        # Log
        left_layout.addWidget(QLabel("<b>Status Log:</b>"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        left_layout.addWidget(self.log_box, stretch=1)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.setStyleSheet("background-color: #dc3545; color: white; padding: 10px;")
        self.btn_exit.clicked.connect(self.close)
        left_layout.addWidget(self.btn_exit)

        # --- Right Panel ---
        self.web_view = QWebEngineView()
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        self.web_view.setHtml("<html><body style='display:flex;justify-content:center;align-items:center;font-family:sans-serif;color:#555;'><h1>Ready</h1></body></html>")

        main_layout.addWidget(left_panel, stretch=15)
        main_layout.addWidget(self.web_view, stretch=85)

    def log(self, msg):
        self.log_box.append(f">> {msg}")
        QApplication.processEvents()

    def parse_latlon(self, text):
        try:
            p = text.split(',')
            return (float(p[0].strip()), float(p[1].strip())) if len(p)==2 else None
        except: return None

    def run_routing(self):
        # 1. 讀取輸入
        origin_ll = self.parse_latlon(self.input_origin.text())
        dest_a_ll = self.parse_latlon(self.input_dest_a.text())
        dest_b_str = self.input_dest_b.text().strip()
        dest_b_ll = self.parse_latlon(dest_b_str) if dest_b_str else None

        if not origin_ll or not dest_a_ll:
            self.log("Error: Origin and Destination A are required.")
            return

        self.btn_calc.setEnabled(False)
        self.log("Computing...")

        try:
            # 2. 情境判斷
            if dest_b_ll:
                # === 情境 B: 中途改道 (Reroute) ===
                self.log(f"Mode: Rerouting Simulation")
                self.log(f"1. Calculating Path A: {origin_ll} -> {dest_a_ll}")
                
                # 第一步：計算原始路徑 (Origin -> Dest A)
                # 我們只想要取得座標 (path_a)，暫時不需要顯示 HTML
                _, path_a = open_routing_debug_map_p2p(
                    self.out, origin_ll, dest_a_ll, html_path="temp_a.html",
                    do_repair=True, do_simplify=True
                )
                
                if not path_a or len(path_a) < 2:
                    self.log("Error: Failed to calculate Path A.")
                    return

                # 第二步：找出 40% 的中斷點
                split_idx = int(len(path_a) * 0.4)
                midpoint_ll = path_a[split_idx]
                
                # 分割路徑以利視覺化
                # sailed_path: 0% ~ 40% (已航行，綠色實線)
                sailed_path = path_a[:split_idx+1]
                # abandoned_path: 40% ~ 100% (已放棄，灰色虛線)
                abandoned_path = path_a[split_idx:]
                
                self.log(f"2. Split at 40% (idx={split_idx}/{len(path_a)})")
                self.log(f"   Midpoint: {midpoint_ll}")
                
                self.log(f"3. Calculating Path B: Midpoint -> {dest_b_ll}")

                # 第三步：計算新路徑 (Midpoint -> Dest B)，並傳入舊路徑進行疊圖
                extra_layers = [
                    {
                        'coords': sailed_path, 'name': 'Sailed Part (0-40%)', 
                        'color': 'green', 'weight': 5, 'opacity': 0.8
                    },
                    {
                        'coords': abandoned_path, 'name': 'Abandoned Part', 
                        'color': 'gray', 'weight': 3, 'dash_array': '5, 5', 'opacity': 0.6
                    }
                ]

                # 呼叫主函式生成最終地圖
                html, final_path_b = open_routing_debug_map_p2p(
                    self.out, midpoint_ll, dest_b_ll, html_path="aoi_p2p_map_ui.html",
                    do_repair=True, do_simplify=True,
                    extra_paths=extra_layers  # <--- 將舊路徑傳進去畫
                )
                
                self.log("Reroute calculation complete.")
                self.web_view.setUrl(QUrl.fromLocalFile(str(html)))

            else:
                # === 情境 A: 一般路徑規劃 (Normal) ===
                self.log(f"Mode: Normal Routing ({origin_ll} -> {dest_a_ll})")
                html, _ = open_routing_debug_map_p2p(
                    self.out, origin_ll, dest_a_ll, html_path="aoi_p2p_map_ui.html",
                    do_repair=True, do_simplify=True
                )
                self.web_view.setUrl(QUrl.fromLocalFile(str(html)))
                self.log("Done.")

        except Exception as e:
            self.log(f"Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.btn_calc.setEnabled(True)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Exit', "Quit?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if os.path.exists("aoi_p2p_map_ui.html"):
                try: os.remove("aoi_p2p_map_ui.html")
                except: pass
            if os.path.exists("temp_a.html"):
                try: os.remove("temp_a.html")
                except: pass
            event.accept()
        else:
            event.ignore()


# ==============================================================================
# SECTION 7: MAIN ENTRY POINT (程式進入點)
# ==============================================================================

if __name__ == "__main__":
    print("=== System Starting ===")
    
    # 1. 載入或重建地圖資料
    try:
        out_data = load_or_build_data(CFG, CACHE_FILE)
    except Exception as e:
        print(f"Critical Error loading data: {e}")
        sys.exit(1)

    # 2. 啟動 GUI
    print("Starting GUI...")
    app = QApplication(sys.argv)
    window = RoutingDemoApp(out_data)
    window.show()
    sys.exit(app.exec())