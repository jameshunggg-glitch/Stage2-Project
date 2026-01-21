from pathlib import Path
import routing_map
from routing_map import build_aoi, RoutingMapConfig
from routing_map.config import AoiConfig, LandConfig
import folium, webbrowser
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

import os
import pickle
from shapely.prepared import prep # 讀取後重建需要用到

from shapely.geometry import LineString, box
from shapely.ops import transform as shp_transform

# === modules ===
from routing_map.path_simplifier import simplify_path_visibility
from routing_map.routing_graph import build_base_graph, haversine_km
from routing_map.c_gateb_connectors import (
    build_cnode_gateb_connectors_nearest,
    add_cnode_gateb_connectors_to_graph,
)
from routing_map.snap import snap_pair_component_aware, inject_point_edges
from routing_map.repairer import PathRepairer, RepairConfig

#  snap-link repair helper (you said it's already ready)
from routing_map.snap_link_repair import repair_snap_link_ll_if_needed

from routing_map.metrics import path_length_km_nm, format_distance

cfg = RoutingMapConfig(
    aoi=AoiConfig(
        bbox_ll=(10, -50, 150, 50),
    ),
    land=LandConfig(
        # shp_path=Path(r"C:\Users\slab\Desktop\Slab Project\Stage1\data\Land\ne_10m_land.shp"),
        shp_path=Path(r"C:\Users\slab\Desktop\Slab Project\Stage1\data\Land\ne_10m_land.shp"),
        
        buffer_km=20.0,
        avoid_km= 1,
        collision_safety_km=0.5,
    ),
)

# out = build_aoi(cfg)

CACHE_FILE = "aoi_cache.pkl"

# 1. 嘗試讀取快取
if os.path.exists(CACHE_FILE):
    print(f"[{CACHE_FILE}] 存在，正在讀取快取資料...")
    with open(CACHE_FILE, "rb") as f:
        out = pickle.load(f)
    
    # 2. 重建無法被儲存的物件 (Prepared Geometry)
    # 因為 collision_prep 無法被 pickle，所以我們讀取後要用 layers["COLLISION_M"] 重建它
    if "layers" in out and "COLLISION_M" in out["layers"]:
        print("正在重建 Collision Prep 物件...")
        out["collision_prep"] = prep(out["layers"]["COLLISION_M"])
        
    print("讀取完成！")

else:
    print(f"[{CACHE_FILE}] 不存在，開始執行 build_aoi...")
    
    # 執行原本的建置流程
    out = build_aoi(cfg)
    
    # 3. 儲存前的處理 (移除不可 pickle 的物件)
    # 建立一個淺拷貝 (Shallow Copy)，以免影響到記憶體中正在用的 out
    out_to_save = out.copy()
    
    # 移除 collision_prep，避免 pickle 報錯
    if "collision_prep" in out_to_save:
        del out_to_save["collision_prep"]
    
    # 寫入檔案
    print(f"正在將資料儲存至 [{CACHE_FILE}] ...")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(out_to_save, f)
    print("儲存完成！")



# ---------------------------
# helpers
# ---------------------------
def in_bbox(p, bbox_ll):
    if bbox_ll is None:
        return True
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_ll)
    lon, lat = float(p[0]), float(p[1])
    return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)

def safe_df(out, name):
    df = out.get(name, None)
    if df is None:
        return None
    try:
        return df if len(df) > 0 else None
    except Exception:
        return None

def build_gate_xy(out):
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
        rows = []
        for gid in gids:
            if gid in gate_xy:
                lon, lat = gate_xy[gid]
                rows.append({"g_id": gid, "lon": lon, "lat": lat})
        dfGB = pd.DataFrame(rows)

    if dfGB is None:
        gb2 = safe_df(out, "Gate_B")
        if gb2 is not None:
            dfGB = gb2.copy()

    return dfGB if (dfGB is not None and len(dfGB) > 0) else None

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

    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        if nodes_df is None or idx_to_lonlat_fn is None:
            return None
        try:
            return (idx_to_lonlat_fn(a), idx_to_lonlat_fn(b))
        except Exception:
            return None

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) >= 2 and len(b) >= 2:
        try:
            return ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
        except Exception:
            return None

    if isinstance(a, str) and isinstance(b, str):
        p1 = parse_node_id_str(a)
        p2 = parse_node_id_str(b)
        if p1 and p2:
            return (p1, p2)

    return None

def unwrap_lon(lon, ref_lon):
    lon = float(lon); ref_lon = float(ref_lon)
    d = lon - ref_lon
    if d > 180: lon -= 360
    if d < -180: lon += 360
    return lon

def route_polyline_dateline_safe(path_ll):
    if not path_ll:
        return []
    out = []
    ref = float(path_ll[0][0])
    for lon, lat in path_ll:
        lon_u = unwrap_lon(lon, ref)
        out.append((lon_u, float(lat)))
        ref = lon_u
    return out


# --- projection utilities (USE out["proj"] to match layers CRS) ---
def _make_ll_m_projectors_from_out(out):
    """
    Return:
      ll2m_xy(lon,lat)->(x,y)      # for repairer (positional lon,lat)
      m2ll_xy(x,y)->(lon,lat)
      ll2m_tuple((lon,lat))->(x,y) # for simplifier (tuple)
      m2ll_tuple((x,y))->(lon,lat)
    """
    proj = out.get("proj", None)
    if proj is None:
        raise ValueError("out['proj'] not found. build_aoi() should return 'proj'.")

    def _apply(fn, a, b):
        if hasattr(fn, "transform") and callable(getattr(fn, "transform")):
            return fn.transform(a, b)
        if callable(fn):
            try:
                return fn(a, b)
            except TypeError:
                return fn((a, b))
        raise TypeError(f"Projector {type(fn)} not callable and no .transform")

    candidates = [
        ("ll_to_xy", "xy_to_ll"),
        ("ll_to_m", "m_to_ll"),
        ("to_m", "to_ll"),
        ("fwd", "inv"),
        ("forward", "inverse"),
    ]

    for a, b in candidates:
        if hasattr(proj, a) and hasattr(proj, b):
            f = getattr(proj, a)
            g = getattr(proj, b)

            def ll2m_xy(lon, lat, _f=f):
                x, y = _apply(_f, float(lon), float(lat))
                return (float(x), float(y))

            def m2ll_xy(x, y, _g=g):
                lon, lat = _apply(_g, float(x), float(y))
                return (float(lon), float(lat))

            ll2m_tuple = lambda p: ll2m_xy(p[0], p[1])
            m2ll_tuple = lambda q: m2ll_xy(q[0], q[1])
            return ll2m_xy, m2ll_xy, ll2m_tuple, m2ll_tuple

    if hasattr(proj, "transform") and callable(getattr(proj, "transform")):
        raise ValueError("proj.transform exists but inverse isn't inferable; please provide proj with inverse method.")

    raise ValueError("Cannot infer projection methods from out['proj'].")


def _get_collision_metric(out):
    layers = out.get("layers", None)
    if isinstance(layers, dict) and layers.get("COLLISION_M") is not None:
        return layers["COLLISION_M"]

    c = out.get("COLLISION_M", None)
    if c is None:
        c = out.get("collision_m", None)
    if c is not None:
        return c

    cp = out.get("collision_prep", None)
    if cp is not None and hasattr(cp, "context") and cp.context is not None:
        return cp.context

    return out.get("collision", None)


from shapely.geometry import box
import numpy as np
from shapely.geometry import box

def _get_densified_metric_box(bbox_ll, ll2m_xy, step_deg=2.0, pad_m=50_000.0):
    """
    沿著 AOI 的四個邊界進行密集取樣，然後找出 Metric 空間中真正的 min/max xy。
    解決大範圍投影造成的「弧線切除」問題。
    """
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_ll)
    
    xs, ys = [], []
    
    # 建立經度與緯度的取樣點
    lons = np.arange(min_lon, max_lon + step_deg, step_deg)
    if lons[-1] != max_lon: lons = np.append(lons, max_lon)
    
    lats = np.arange(min_lat, max_lat + step_deg, step_deg)
    if lats[-1] != max_lat: lats = np.append(lats, max_lat)
    
    # 定義要檢查的邊界點 (上緣、下緣、左緣、右緣)
    # 1. Top & Bottom edges (沿著經度走)
    for lon in lons:
        # Bottom edge
        x, y = ll2m_xy(lon, min_lat)
        xs.append(x); ys.append(y)
        # Top edge
        x, y = ll2m_xy(lon, max_lat)
        xs.append(x); ys.append(y)
        
    # 2. Left & Right edges (沿著緯度走)
    for lat in lats:
        # Left edge
        x, y = ll2m_xy(min_lon, lat)
        xs.append(x); ys.append(y)
        # Right edge
        x, y = ll2m_xy(max_lon, lat)
        xs.append(x); ys.append(y)
        
    # 找出真正的 Metric 邊界
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # 加上 Padding 並回傳 Metric Box
    return box(min_x, min_y, max_x, max_y).buffer(pad_m)

def _clip_collision_to_aoi_bbox(collision_m, bbox_ll, ll2m_xy, pad_m=80_000.0):
    """
    Clip metric collision to AOI bbox (also in metric), robust to nonlinear/local projections:
    - Project ALL 4 bbox corners and take min/max in metric space.
    - Then build a metric box (+ optional pad) and intersect.

    NOTE:
      bbox_ll is (min_lon, min_lat, max_lon, max_lat) in lon/lat degrees.
    """
    if collision_m is None or bbox_ll is None:
        return collision_m

    min_lon, min_lat, max_lon, max_lat = map(float, bbox_ll)

    # If bbox crosses dateline in [-180,180] convention (rare in your current AOI),
    # you can either skip clip or handle split-box. For now, handle the simple case only.
    # (Your current bbox (10..150) doesn't cross, so this won't trigger.)
    if min_lon > max_lon:
        # conservative fallback: don't clip (avoid generating a wrong box)
        # You can implement split-box clipping later if you ever need dateline AOI.
        print("[collision] bbox crosses dateline; skip clip to avoid wrong AOI box")
        return collision_m

    # --- Project 4 corners (IMPORTANT) ---
    corners_ll = [
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
    ]

    xs, ys = [], []
    for lon, lat in corners_ll:
        try:
            x, y = ll2m_xy(lon, lat)
        except Exception as e:
            print("[collision] clip: ll2m failed on corner", (lon, lat), "-> keep original:", repr(e))
            return collision_m
        xs.append(float(x))
        ys.append(float(y))

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    # Build AOI metric window (+ pad)
    #aoi_box_m = box(minx, miny, maxx, maxy).buffer(float(pad_m))
    aoi_box_m = _get_densified_metric_box(bbox_ll, ll2m_xy, step_deg=2.0, pad_m=pad_m)

    try:
        c2 = collision_m.intersection(aoi_box_m)
        if c2 is None or c2.is_empty:
            print("[collision] clip empty -> keep original")
            return collision_m
        return c2
    except Exception as e:
        print("[collision] clip failed -> keep original:", repr(e))
        return collision_m




def _geom_m_to_ll(geom_m, m2ll_xy):
    def _norm_lon(lon):
        lon = float(lon)
        # normalize to [-180, 180]
        while lon > 180.0:
            lon -= 360.0
        while lon < -180.0:
            lon += 360.0
        return lon

    def _xy_to_lonlat(x, y, z=None):
        a, b = m2ll_xy(float(x), float(y))

        a = float(a); b = float(b)

        # --- Auto-fix axis order if inverse returns (lat, lon) ---
        # Heuristic: lat must be [-90,90], lon must be [-180,180] (or 0..360 before norm)
        if abs(a) <= 90.0 and abs(b) <= 360.0 and abs(b) > 90.0:
            # looks like (lat, lon)
            lat, lon = a, b
        else:
            # assume (lon, lat)
            lon, lat = a, b

        lon = _norm_lon(lon)
        lat = max(-90.0, min(90.0, float(lat)))  # clamp safety
        return (lon, lat)

    return shp_transform(_xy_to_lonlat, geom_m)



# ---------------------------
# main
# ---------------------------
def open_routing_debug_map_p2p(
    out,
    origin_ll,
    dest_ll,
    *,
    html_path="aoi_p2p_map.html",
    zoom_start=5,

    c_sample=8000,
    s_sample=3000,
    max_sea_edges_viz=6000,

    include_sea=True,
    include_cc=True,
    include_gateb_sea=True,

    use_c_gateb_bridge=True,
    c_to_gateB_max_deg_dist=None,

    k_near=30,
    r_max_km_snap=150.0,
    k_inject=4,

    do_repair=True,
    do_simplify=True,
):
    bbox_ll = out.get("bbox_ll", None)
    if bbox_ll is None:
        raise ValueError("out['bbox_ll'] is required.")
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_ll)
    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]

    origin_ll = (float(origin_ll[0]), float(origin_ll[1]))
    dest_ll   = (float(dest_ll[0]), float(dest_ll[1]))

    C_nodes = safe_df(out, "C_nodes")
    S_nodes = safe_df(out, "S_nodes")
    S_edges = out.get("S_edges", None)
    C_edges_df = safe_df(out, "C_edges")

    gate_xy = build_gate_xy(out)
    dfGB = get_gateB_df(out, gate_xy)
    dfGB_conn = safe_df(out, "gateB_connectors")

    # --- build base graph ---
    G, stats = build_base_graph(
        out,
        include_sea=bool(include_sea),
        include_cc=bool(include_cc),
        include_gateb_sea=bool(include_gateb_sea),
        include_c_gateb=False,
        bbox_ll=bbox_ll,
        weight_unit="km",
    )
    print("[graph] base stats:", stats)

    # --- optional C↔GateB bridge ---
    cgb_df = None
    if use_c_gateb_bridge and isinstance(C_nodes, pd.DataFrame) and isinstance(dfGB, pd.DataFrame):
        cgb_df = build_cnode_gateb_connectors_nearest(
            C_nodes,
            dfGB[["g_id", "lon", "lat"]],
            bbox_ll=bbox_ll,
            max_deg_dist=c_to_gateB_max_deg_dist,
        )
        added = add_cnode_gateb_connectors_to_graph(G, cgb_df, etype="c_gb", weight_col="dist_deg")
        print(f"[graph] C↔GateB bridge edges added: {added}")

    # --- snap pair ---
    pair = snap_pair_component_aware(
        out,
        origin_ll, dest_ll,
        k_near=int(k_near),
        r_max_km=float(r_max_km_snap),
        k_inject=int(k_inject),
        prefer_ok_set=True,
        allow_fallback_non_ok=True,
        allow_radius_fallback=True,
        do_nudge=True,
    )

    start_key = getattr(pair.start, "p_used_ll", origin_ll)
    end_key   = getattr(pair.end,   "p_used_ll", dest_ll)
    start_key = (float(start_key[0]), float(start_key[1]))
    end_key   = (float(end_key[0]), float(end_key[1]))

    path = None
    path_ll_for_simplify = None
    path_simplified, simp_stats = None, None

    #  store inject-edge polylines (these are the "temporary point" links you care about)
    inject_start_ll = None   # path[0] -> path[1] (repaired if start_inject)
    inject_end_ll   = None   # path[-2] -> path[-1] (repaired if end_inject)

    if len(pair.start_pick) == 0 or len(pair.end_pick) == 0:
        print("[snap] FAIL:", pair.reason, pair.debug)
    else:
        inject_point_edges(G, start_key, pair.start_pick, k_inject=int(k_inject), etype="start_inject")
        inject_point_edges(G, end_key,   pair.end_pick,   k_inject=int(k_inject), etype="end_inject")
        print("[snap] OK:", pair.reason, pair.debug)

        # --- A* ---
        try:
            path = nx.astar_path(
                G,
                start_key,
                end_key,
                heuristic=lambda n1, n2: haversine_km(n1, n2),
                weight="weight",
            )
            etypes = [G[u][v].get("etype") for u, v in zip(path, path[1:])]
            print("[route] OK | points=", len(path), "| etypes=", dict(Counter(etypes)))
        except Exception as e:
            path = None
            print("[route] FAIL:", repr(e))

        # --- repair + simplify ---
        if path is None or len(path) < 2:
            print("[repair/simplify] skip: no valid path")
        else:
            ll2m_xy, m2ll_xy, ll2m_tuple, m2ll_tuple = _make_ll_m_projectors_from_out(out)

            def _ll_to_m_any(a, b=None):
                if b is None:
                    return ll2m_xy(float(a[0]), float(a[1]))
                return ll2m_xy(float(a), float(b))

            def _m_to_ll_any(a, b=None):
                if b is None:
                    return m2ll_xy(float(a[0]), float(a[1]))
                return m2ll_xy(float(a), float(b))

            collision = _get_collision_metric(out)
            if collision is None:
                print("[collision] not found -> skip repair/simplify")
                path_ll_for_simplify = [(float(p[0]), float(p[1])) for p in path]
            else:
                collision = _clip_collision_to_aoi_bbox(collision, bbox_ll, ll2m_xy, pad_m=80000.0)
                print("[collision] bounds:", getattr(collision, "bounds", None), "type:", getattr(collision, "geom_type", type(collision)))
                collision_used = collision

                path_ll = [(float(p[0]), float(p[1])) for p in path]

                if do_repair:
                    repairer_obj = PathRepairer(RepairConfig(
                        debug=True,
                        rb_n_samples=25,
                        rb_max_iter=60,
                        rb_push_step_m=250.0,
                        rb_smooth_lambda=0.35,
                    ))

                    # ----------------------------
                    # (1) Repair inject edges ONLY:
                    #     - start: path[0] -> path[1] if etype == "start_inject"
                    #     - end  : path[-2] -> path[-1] if etype == "end_inject"
                    # ----------------------------
                    u0, u1 = path[0], path[1]
                    v1, v0 = path[-2], path[-1]

                    et0 = G[u0][u1].get("etype") if G.has_edge(u0, u1) else None
                    et1 = G[v1][v0].get("etype") if G.has_edge(v1, v0) else None
                    print("[inject] first edge etype:", et0, "| last edge etype:", et1)

                    # start inject polyline
                    if et0 == "start_inject":
                        inject_start_ll = repair_snap_link_ll_if_needed(
                            (float(u0[0]), float(u0[1])),
                            (float(u1[0]), float(u1[1])),
                            collision_m=collision,
                            ll_to_m=_ll_to_m_any,
                            m_to_ll=_m_to_ll_any,
                            repairer_obj=repairer_obj,
                        )
                    else:
                        inject_start_ll = [(float(u0[0]), float(u0[1])), (float(u1[0]), float(u1[1]))]

                    # end inject polyline
                    if et1 == "end_inject":
                        inject_end_ll = repair_snap_link_ll_if_needed(
                            (float(v1[0]), float(v1[1])),
                            (float(v0[0]), float(v0[1])),
                            collision_m=collision,
                            ll_to_m=_ll_to_m_any,
                            m_to_ll=_m_to_ll_any,
                            repairer_obj=repairer_obj,
                        )
                    else:
                        inject_end_ll = [(float(v1[0]), float(v1[1])), (float(v0[0]), float(v0[1]))]

                    # ----------------------------
                    # (2) Repair CORE path only: path[1:-1]
                    #     This excludes inject edges completely.
                    # ----------------------------
                    core_nodes = path[1:-1]  # from u1 .. v1
                    if core_nodes is None or len(core_nodes) < 2:
                        core_repaired_ll = [(float(u1[0]), float(u1[1]))] if len(path) >= 2 else []
                        print("[core] skip repair: core_nodes<2")
                    else:
                        core_rep = repairer_obj.repair_path(
                            G,
                            core_nodes,
                            collision_m=collision,
                            ll_to_m=_ll_to_m_any,
                            m_to_ll=_m_to_ll_any,
                        )
                        print("[core-repair]", core_rep.stats)
                        core_repaired_ll = [(float(p[0]), float(p[1])) for p in core_rep.path_ll]

                    # ----------------------------
                    # (3) Stitch: inject_start + core + inject_end
                    # ----------------------------
                    def _extend_no_dup(dst, src):
                        if not src:
                            return
                        if not dst:
                            dst.extend(src)
                            return
                        if dst[-1] == src[0]:
                            dst.extend(src[1:])
                        else:
                            dst.extend(src)

                    full_ll = []
                    _extend_no_dup(full_ll, inject_start_ll)
                    _extend_no_dup(full_ll, core_repaired_ll)
                    _extend_no_dup(full_ll, inject_end_ll)

                    path_ll_for_simplify = full_ll
                    print("[collision used for debug] bounds:", collision.bounds, "type:", collision.geom_type)

                else:
                    path_ll_for_simplify = path_ll

                # simplify
                if do_simplify and path_ll_for_simplify is not None and len(path_ll_for_simplify) >= 2:
                    path_simplified, simp_stats = simplify_path_visibility(
                        path_ll_for_simplify,
                        collision_m=collision,
                        ll_to_m=_ll_to_m_any,
                        m_to_ll=_m_to_ll_any,
                        window_size=80,
                        max_tries=300,
                        use_prepared_collision=True,
                        dateline_unwrap=True,
                    )
                    print("[simplify]", simp_stats)

                    # ---------------------------
                    # 1) Decide CORE final polyline (priority: simplified > repaired_full > original A*)
                    # ---------------------------
                    core_final_ll = None
                    if path_simplified is not None and len(path_simplified) >= 2:
                        core_final_ll = [(float(p[0]), float(p[1])) for p in path_simplified]
                    elif path_ll_for_simplify is not None and len(path_ll_for_simplify) >= 2:
                        core_final_ll = [(float(p[0]), float(p[1])) for p in path_ll_for_simplify]
                    elif path is not None and len(path) >= 2:
                        core_final_ll = [(float(p[0]), float(p[1])) for p in path]

                    # ---------------------------
                    # 2) Build snap-links (origin->start_key, end_key->dest) and MERGE into FINAL
                    # ---------------------------
                    final_ll = core_final_ll

                    if core_final_ll is not None and len(core_final_ll) >= 2:
                        # snap-link start
                        if origin_ll != start_key:
                            snap_start_ll = repair_snap_link_ll_if_needed(
                                origin_ll, start_key,
                                collision_m=collision,
                                ll_to_m=_ll_to_m_any,
                                m_to_ll=_m_to_ll_any,
                                repairer_obj=repairer_obj,   # 你前面已經建過 PathRepairer 了，直接重用
                            )
                        else:
                            snap_start_ll = [origin_ll]

                        # snap-link end
                        if end_key != dest_ll:
                            snap_end_ll = repair_snap_link_ll_if_needed(
                                end_key, dest_ll,
                                collision_m=collision,
                                ll_to_m=_ll_to_m_any,
                                m_to_ll=_m_to_ll_any,
                                repairer_obj=repairer_obj,
                            )
                        else:
                            snap_end_ll = [dest_ll]

                        # merge without duplicate joints
                        merged = []
                        if snap_start_ll and len(snap_start_ll) >= 2:
                            merged.extend([(float(p[0]), float(p[1])) for p in snap_start_ll])
                        else:
                            merged.append((float(origin_ll[0]), float(origin_ll[1])))

                        # connect to core
                        if merged and core_final_ll:
                            if merged[-1] == core_final_ll[0]:
                                merged.extend(core_final_ll[1:])
                            else:
                                merged.extend(core_final_ll)

                        # connect end snap-link
                        if snap_end_ll and len(snap_end_ll) >= 2:
                            snap_end_ll = [(float(p[0]), float(p[1])) for p in snap_end_ll]
                            if merged and merged[-1] == snap_end_ll[0]:
                                merged.extend(snap_end_ll[1:])
                            else:
                                merged.extend(snap_end_ll)

                        final_ll = merged

                    # ---------------------------
                    # 3) Distance on FINAL (includes snap-links!)
                    # ---------------------------
                    if final_ll is not None and len(final_ll) >= 2:
                        total_km, total_nm = path_length_km_nm(final_ll, dateline_unwrap=True)
                        print(f"[distance] final route = {format_distance(total_km, total_nm)}")

                        try:
                            out["_p2p_last_distance"] = {"km": float(total_km), "nm": float(total_nm)}
                        except Exception:
                            pass
                    else:
                        print("[distance] skip: no final polyline")
                else:
                    print("[simplify] skip")

    # ---------------------------
    # folium map + layers
    # ---------------------------
    m = folium.Map(location=center, zoom_start=zoom_start, control_scale=True)
    folium.Rectangle(bounds=[[min_lat, min_lon], [max_lat, max_lon]], fill=False, weight=3, opacity=0.9).add_to(m)
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    # draw collision used (optional)
    try:
        collision_viz = collision_used
        if collision_viz is not None:
            print("[collision viz ] bounds:", collision_viz.bounds, "type:", collision_viz.geom_type)
            ll2m_xy, m2ll_xy, _, _ = _make_ll_m_projectors_from_out(out)
            collision_viz = _clip_collision_to_aoi_bbox(collision_viz, bbox_ll, ll2m_xy, pad_m=50_000.0)
            col_ll = _geom_m_to_ll(collision_viz, m2ll_xy)
            from shapely.geometry import box as shp_box
            min_lon, min_lat, max_lon, max_lat = map(float, bbox_ll)
            # 建立一個完美的經緯度矩形
            viz_box = shp_box(min_lon, min_lat, max_lon, max_lat)
            # 只保留在這個矩形內的圖形
            col_ll = col_ll.intersection(viz_box)
            fgCol = folium.FeatureGroup(name="Collision (USED, approx ll)", show=False)
            folium.GeoJson(
                data=col_ll.__geo_interface__,
                style_function=lambda _: {"fillColor": "#3b82f6", "color": "#3b82f6", "weight": 2, "fillOpacity": 0.15},
            ).add_to(fgCol)
            fgCol.add_to(m)
    except Exception as e:
        print("[viz] collision layer failed:", repr(e))

    def circle_layer(df, name, radius, show=True):
        fg = folium.FeatureGroup(name=name, show=show)
        for _, r in df.iterrows():
            p = (float(r["lon"]), float(r["lat"]))
            if not in_bbox(p, bbox_ll):
                continue
            folium.CircleMarker([p[1], p[0]], radius=radius).add_to(fg)
        fg.add_to(m)

    # --- node layers ---
    if isinstance(C_nodes, pd.DataFrame) and len(C_nodes) > 0:
        nC = len(C_nodes)
        dfC_plot = C_nodes.sample(min(int(c_sample), nC), random_state=7) if nC > c_sample else C_nodes
        circle_layer(dfC_plot, f"C_nodes (sample {len(dfC_plot)}/{nC})", radius=1, show=False)

    if isinstance(dfGB, pd.DataFrame) and len(dfGB) > 0:
        circle_layer(dfGB, f"Gate_B ({len(dfGB)})", radius=6, show=False)

    if isinstance(S_nodes, pd.DataFrame) and len(S_nodes) > 0:
        nS = len(S_nodes)
        if s_sample is not None and nS > int(s_sample):
            dfS_plot = S_nodes.sample(int(s_sample), random_state=7)
            title = f"S_nodes (sample {len(dfS_plot)}/{nS})"
        else:
            dfS_plot = S_nodes
            title = f"S_nodes ({nS})"
        circle_layer(dfS_plot, title, radius=3, show=False)

    # --- sea edges (viz) ---
    def sea_lonlat_by_idx(i):
        s = S_nodes.iloc[int(i)]
        return (float(s["lon"]), float(s["lat"]))

    if isinstance(S_nodes, pd.DataFrame) and S_edges is not None and len(S_edges) > 0:
        fgE = folium.FeatureGroup(name=f"S_edges (show {min(len(S_edges), max_sea_edges_viz)}/{len(S_edges)})", show=True)
        take = S_edges[:max_sea_edges_viz] if len(S_edges) > max_sea_edges_viz else S_edges
        drawn = 0
        for e in take:
            seg = edge_to_lonlat(e, nodes_df=S_nodes, idx_to_lonlat_fn=sea_lonlat_by_idx)
            if seg is None:
                continue
            u, v = seg
            if not (in_bbox(u, bbox_ll) or in_bbox(v, bbox_ll)):
                continue
            folium.PolyLine([[u[1], u[0]], [v[1], v[0]]], color="#3352ff", weight=2, opacity=0.6).add_to(fgE)
            drawn += 1
        fgE.add_to(m)
        print(f"[viz] sea edges drawn: {drawn}/{len(take)}")

    # --- GateB→Sea connectors (viz) ---
    if isinstance(dfGB_conn, pd.DataFrame) and isinstance(S_nodes, pd.DataFrame) and gate_xy:
        fgConn = folium.FeatureGroup(name=f"GateB→Sea connectors ({len(dfGB_conn)})", show=False)
        drawn = 0
        for _, r in dfGB_conn.iterrows():
            try:
                gid = int(r["g_id"])
                if gid not in gate_xy:
                    continue
                gb = gate_xy[gid]
                if not in_bbox(gb, bbox_ll):
                    continue
                sea = sea_lonlat_by_idx(int(r["sea_idx"]))
                if not (in_bbox(gb, bbox_ll) or in_bbox(sea, bbox_ll)):
                    continue
                folium.PolyLine([[gb[1], gb[0]], [sea[1], sea[0]]], weight=2, opacity=0.7).add_to(fgConn)
                drawn += 1
            except Exception:
                continue
        fgConn.add_to(m)
        print(f"[viz] GateB→Sea connectors drawn: {drawn}/{len(dfGB_conn)}")

    # --- C↔GateB bridge connectors (viz) ---
    if isinstance(cgb_df, pd.DataFrame) and len(cgb_df) > 0:
        fgCGB = folium.FeatureGroup(name=f"C↔GateB bridge (nearest) ({len(cgb_df)})", show=False)
        for _, r in cgb_df.iterrows():
            c = (float(r["c_lon"]), float(r["c_lat"]))
            gb = (float(r["g_lon"]), float(r["g_lat"]))
            folium.PolyLine([[c[1], c[0]], [gb[1], gb[0]]], weight=2, opacity=0.7).add_to(fgCGB)
        fgCGB.add_to(m)

    # --- start/end markers + candidates ---
    fgSE = folium.FeatureGroup(name="Start/End + snapped candidates", show=True)
    folium.Marker([origin_ll[1], origin_ll[0]], tooltip="START (input)").add_to(fgSE)
    folium.Marker([dest_ll[1], dest_ll[0]], tooltip="END (input)").add_to(fgSE)

    if start_key != origin_ll:
        folium.CircleMarker([start_key[1], start_key[0]], radius=7, opacity=0.9, tooltip="START (used / nudged)").add_to(fgSE)
    if end_key != dest_ll:
        folium.CircleMarker([end_key[1], end_key[0]], radius=7, opacity=0.9, tooltip="END (used / nudged)").add_to(fgSE)

    for i, c in enumerate(pair.start_pick):
        folium.CircleMarker([c.node_ll[1], c.node_ll[0]], color="#6f42c1", radius=6, tooltip=f"start_cand#{i} d={c.dist_km:.1f}km").add_to(fgSE)
    for i, c in enumerate(pair.end_pick):
        folium.CircleMarker([c.node_ll[1], c.node_ll[0]], color="#6f42c1", radius=6, tooltip=f"end_cand#{i} d={c.dist_km:.1f}km").add_to(fgSE)
    fgSE.add_to(m)

    # --- route layers ---
    if path is not None and len(path) >= 2:
        fgA = folium.FeatureGroup(name="Route: A* (original core)", show=True)
        path_u = route_polyline_dateline_safe([(float(p[0]), float(p[1])) for p in path])
        folium.PolyLine([[p[1], p[0]] for p in path_u], color="#d62728", weight=6, opacity=0.95).add_to(fgA)
        fgA.add_to(m)

        #  Inject edges (the two edges you care about)
        has_inj_start = (inject_start_ll is not None and len(inject_start_ll) >= 2 and inject_start_ll[0] != inject_start_ll[-1])
        has_inj_end   = (inject_end_ll   is not None and len(inject_end_ll)   >= 2 and inject_end_ll[0]   != inject_end_ll[-1])
        if has_inj_start or has_inj_end:
            fgSL = folium.FeatureGroup(name="Route: Inject edges (repaired if needed)", show=True)
            if has_inj_start:
                sl1 = route_polyline_dateline_safe([(float(p[0]), float(p[1])) for p in inject_start_ll])
                folium.PolyLine([[p[1], p[0]] for p in sl1], color="#9467bd", weight=5, opacity=0.92).add_to(fgSL)
            if has_inj_end:
                sl2 = route_polyline_dateline_safe([(float(p[0]), float(p[1])) for p in inject_end_ll])
                folium.PolyLine([[p[1], p[0]] for p in sl2], color="#9467bd", weight=5, opacity=0.92).add_to(fgSL)
            fgSL.add_to(m)

        # repaired full (pre-simplify)
        if path_ll_for_simplify is not None and len(path_ll_for_simplify) >= 2:
            fgR = folium.FeatureGroup(name="Route: Repaired (FULL, pre-simplify)", show=True)
            repaired_ll = [(float(p[0]), float(p[1])) for p in path_ll_for_simplify]
            path_ru = route_polyline_dateline_safe(repaired_ll)
            folium.PolyLine([[p[1], p[0]] for p in path_ru], color="#2ca02c", weight=5, opacity=0.90).add_to(fgR)
            fgR.add_to(m)

        # simplified
        if path_simplified is not None and len(path_simplified) >= 2:
            fgS = folium.FeatureGroup(name="Route: Simplified (visibility)", show=True)
            path_su = route_polyline_dateline_safe([(float(p[0]), float(p[1])) for p in path_simplified])
            folium.PolyLine([[p[1], p[0]] for p in path_su], color="#ff7f0e", weight=5, opacity=0.95).add_to(fgS)
            fgS.add_to(m)
            
        # --- FINAL route (with snap-links) ---
        if final_ll is not None and len(final_ll) >= 2:
            fgF = folium.FeatureGroup(name="Route: FINAL (simplified + snap-links)", show=True)
            final_u = route_polyline_dateline_safe(final_ll)
            folium.PolyLine([[p[1], p[0]] for p in final_u], weight=6, opacity=0.95).add_to(fgF)
            fgF.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    html_path = Path(html_path).resolve()
    m.save(str(html_path))
    # webbrowser.open(html_path.as_uri())
    return html_path


# =========================
# === call
# =========================
origin_ll = (120.325625, 22.548649)    #(127.09912, 13.3102)
dest_ll   = (135.134173, 34.200334)    # (17.47887, 42.36624)

# open_routing_debug_map_p2p(
#     out,
#     origin_ll=origin_ll,
#     dest_ll=dest_ll,
#     html_path="aoi_p2p_map.html",
#     zoom_start=5,
#     include_sea=True,
#     include_cc=True,
#     include_gateb_sea=True,
#     use_c_gateb_bridge=True,
#     c_to_gateB_max_deg_dist=None,
#     k_near=30,
#     r_max_km_snap=150,
#     k_inject=4,
#     do_repair=True,
#     do_simplify=True,
# )

# ==========================================
# === 以下為新增的 UI 程式碼 (請貼在原檔最下方) ===
# ==========================================
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTextEdit, QFrame)
from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView

class RoutingDemoApp(QMainWindow):
    def __init__(self, routing_data_out):
        super().__init__()
        self.setWindowTitle("Slab Routing Demo - Local UI")
        self.resize(1920, 1080) # 設定 FHD 初始大小
        
        # 保存預先計算好的 AOI 資料 (out)
        self.out = routing_data_out
        
        # 預設座標 (範例)
        self.default_origin = "120.325625, 22.548649"
        self.default_dest = "135.134173, 34.200334"

        # 初始化 UI
        self.init_ui()

    def init_ui(self):
        # 主容器
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主佈局：水平分割 (左：控制面板, 右：地圖)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- 左側控制面板 (15%) ---
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
        left_layout.setSpacing(15)

        # 標題
        title_label = QLabel("<h2>Routing Control</h2>")
        left_layout.addWidget(title_label)

        # Origin 輸入
        left_layout.addWidget(QLabel("<b>Origin (Lon, Lat):</b>"))
        self.input_origin = QLineEdit(self.default_origin)
        self.input_origin.setPlaceholderText("Lon, Lat")
        left_layout.addWidget(self.input_origin)

        # Destination 輸入
        left_layout.addWidget(QLabel("<b>Destination (Lon, Lat):</b>"))
        self.input_dest = QLineEdit(self.default_dest)
        self.input_dest.setPlaceholderText("Lon, Lat")
        left_layout.addWidget(self.input_dest)

        # 計算按鈕
        self.btn_calc = QPushButton("Calculate Route")
        self.btn_calc.setStyleSheet("""
            QPushButton {
                background-color: #007bff; color: white; 
                font-weight: bold; padding: 10px; border-radius: 5px;
            }
            QPushButton:hover { background-color: #0056b3; }
        """)
        self.btn_calc.clicked.connect(self.run_routing)
        left_layout.addWidget(self.btn_calc)

        # 訊息輸出框 (Log)
        left_layout.addWidget(QLabel("<b>Status Log:</b>"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        left_layout.addWidget(self.log_box)

        # 底部填充 (讓元件靠上)
        left_layout.addStretch()
        
        # 分隔線
        # from PyQt6.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        left_layout.addWidget(line)

        # 離開按鈕
        self.btn_exit = QPushButton("Exit Demo")
        self.btn_exit.setStyleSheet("""
            QPushButton {
                background-color: #dc3545; color: white; 
                font-weight: bold; padding: 10px; border-radius: 5px;
            }
            QPushButton:hover { background-color: #c82333; }
        """)
        # 連接到視窗的 close() 方法
        self.btn_exit.clicked.connect(self.close)
        left_layout.addWidget(self.btn_exit)

        # --- 右側地圖視窗 (85%) ---
        self.web_view = QWebEngineView()
        
        # === 新增設定：允許本地檔案存取遠端資源 & 開啟 JavaScript ===
        from PyQt6.QtWebEngineCore import QWebEngineSettings
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        # =======================================================
        
        # 預設先顯示空白或說明
        self.web_view.setHtml("<html><body style='display:flex;justify-content:center;align-items:center;height:100%;font-family:sans-serif;color:#555;'><h1>Please Click Calculate</h1></body></html>")

        # --- 加入主佈局並設定比例 ---
        main_layout.addWidget(left_panel, stretch=15) # 左邊 15%
        main_layout.addWidget(self.web_view, stretch=85) # 右邊 85%

    def log(self, message):
        self.log_box.append(f">> {message}")
        # 強制刷新 UI 以顯示 log
        QApplication.processEvents()

    def parse_latlon(self, text):
        try:
            parts = text.split(',')
            if len(parts) != 2:
                raise ValueError
            lon = float(parts[0].strip())
            lat = float(parts[1].strip())
            return (lon, lat)
        except:
            return None

    def run_routing(self):
        # 1. 取得輸入
        origin_str = self.input_origin.text()
        dest_str = self.input_dest.text()
        
        origin_ll = self.parse_latlon(origin_str)
        dest_ll = self.parse_latlon(dest_str)

        if not origin_ll or not dest_ll:
            self.log("Error: Invalid Coordinate Format. Use 'lon, lat'")
            return

        self.log(f"Computing route...")
        self.log(f"From: {origin_ll}")
        self.log(f"To:   {dest_ll}")
        self.btn_calc.setEnabled(False) # 鎖定按鈕避免重複點擊

        try:
            # 2. 呼叫原本的 Routing 函式
            # 注意：這裡呼叫您原本定義的函式 open_routing_debug_map_p2p
            # 為了避免它自動開啟瀏覽器，我們稍微依賴它產生的檔案，
            # 建議您在原函式把 webbrowser.open 註解掉，或者就讓它開著也沒關係，這裡我們會重新載入。
            
            output_html = "aoi_p2p_map_ui.html" # 指定一個專用的檔名以免衝突
            
            # 呼叫邏輯 (傳入 out 與座標)
            generated_path = open_routing_debug_map_p2p(
                self.out,
                origin_ll=origin_ll,
                dest_ll=dest_ll,
                html_path=output_html, # 覆寫檔名
                zoom_start=5,
                include_sea=True,
                include_cc=True,
                include_gateb_sea=True,
                use_c_gateb_bridge=True,
                c_to_gateB_max_deg_dist=None,
                k_near=30,
                r_max_km_snap=150,
                k_inject=4,
                do_repair=True,
                do_simplify=True,
            )
            
            self.log(f"Map generated: {generated_path}")

            # 3. 將生成的 HTML 載入右側 WebView
            # 必須使用絕對路徑
            file_path = str(Path(generated_path).resolve())
            self.web_view.setUrl(QUrl.fromLocalFile(file_path))
            
            self.log("Display updated.")

        except Exception as e:
            self.log(f"Error during calculation: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.btn_calc.setEnabled(True)
            
    def closeEvent(self, event):
        """
        當使用者按下視窗右上角的 X 或介面上的 Exit 按鈕時，會觸發此事件。
        我們在這裡詢問是否確定離開，並清理暫存檔案。
        """
        from PyQt6.QtWidgets import QMessageBox
        
        # 1. (選用) 跳出確認視窗
        reply = QMessageBox.question(self, 'Exit Confirmation',
                                     "Are you sure you want to quit?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            # 2. 清理暫存的 HTML 檔案
            import os
            temp_file = "aoi_p2p_map_ui.html"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    print(f"Warning: Could not delete temp file: {e}")
            
            # 接受關閉事件 (程式結束)
            event.accept()
        else:
            # 取消關閉 (回到視窗)
            event.ignore()

# ==========================================
# === 主程式進入點修改 ===
# ==========================================

if __name__ == "__main__":
    # 這裡假設您的 main_disassembly.py 上方已經跑完了資料讀取 (out = ...)
    # 如果 out 還沒讀取，請確保上面的程式碼有執行到 out = pickle.load(...) 或是 build_aoi(...)
    
    print("啟動 UI...")
    
    # 建立 Qt Application
    app = QApplication(sys.argv)
    
    # 建立視窗並傳入已讀取的資料 out
    window = RoutingDemoApp(out)
    window.show()
    
    # 執行主迴圈
    sys.exit(app.exec())