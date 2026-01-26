import folium, webbrowser
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

from shapely.geometry import LineString, box
from shapely.ops import transform as shp_transform

# === modules ===
from routing_map.path_simplifier import simplify_path_visibility
from routing_map.routing_graph import build_base_graph, haversine_km
from routing_map.c_gateb_connectors import (
    build_cnode_gateb_connectors_nearest,
    add_cnode_gateb_connectors_to_graph,
)

from routing_map.e_t_transfer import (
    build_et_shared_edges,
    add_et_shared_edges_to_graph,
    ETTransferParams,
)
from routing_map.t_gate_connectors import (
    build_tgate_sea_connectors,
    add_tgate_sea_connectors_to_graph,
    TGateSeaConnectorParams,
)

from routing_map.snap import snap_pair_component_aware, inject_point_edges
from routing_map.repairer import PathRepairer, RepairConfig

#  snap-link repair helper (you said it's already ready)
from routing_map.snap_link_repair import repair_snap_link_ll_if_needed

from routing_map.metrics import path_length_km_nm, format_distance
from routing_map.snap_link_repair import repair_snap_link_ll_if_needed



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


# ---------------------------
# ring / t-gate graph helpers
# ---------------------------

def guess_sea_node_key_fn(G: nx.Graph):
    """Guess how sea nodes are keyed in the networkx graph."""
    if G is None:
        return lambda i: int(i)
    # common patterns
    if 0 in G:
        return lambda i: int(i)
    if ("S", 0) in G:
        return lambda i: ("S", int(i))
    if ("sea", 0) in G:
        return lambda i: ("sea", int(i))
    return lambda i: int(i)

def add_ring_nodes_edges_to_graph(
    G: nx.Graph,
    out: dict,
    *,
    e_node_key_fn=lambda eid: ("E", int(eid)),
    t_node_key_fn=lambda tid: ("T", int(tid)),
    weight_attr="weight_km",
):
    """Add E/T ring nodes and their cycle edges into the graph."""
    rg = out.get("ring_graph", {}) or {}
    E_nodes = rg.get("E_nodes", None)
    T_nodes = rg.get("T_nodes", None)
    E_edges = rg.get("E_edges", None)
    T_edges = rg.get("T_edges", None)

    added_nodes = 0
    added_edges = 0

    # --- nodes ---
    if isinstance(E_nodes, pd.DataFrame) and len(E_nodes) > 0:
        for r in E_nodes.itertuples(index=False):
            k = e_node_key_fn(int(getattr(r, "node_id")))
            if k not in G:
                G.add_node(k)
                added_nodes += 1
            # attach attrs for viz/debug
            for col in ("lon", "lat", "x_m", "y_m", "ring_id", "seq", "s_km"):
                if hasattr(r, col):
                    G.nodes[k][col] = float(getattr(r, col)) if col in ("lon","lat","x_m","y_m","s_km") else int(getattr(r, col))

    if isinstance(T_nodes, pd.DataFrame) and len(T_nodes) > 0:
        for r in T_nodes.itertuples(index=False):
            k = t_node_key_fn(int(getattr(r, "node_id")))
            if k not in G:
                G.add_node(k)
                added_nodes += 1
            for col in ("lon", "lat", "x_m", "y_m", "ring_id", "seq", "s_km"):
                if hasattr(r, col):
                    G.nodes[k][col] = float(getattr(r, col)) if col in ("lon","lat","x_m","y_m","s_km") else int(getattr(r, col))
            if hasattr(r, "is_gate_candidate"):
                G.nodes[k]["is_gate_candidate"] = bool(getattr(r, "is_gate_candidate"))

    # --- fast lookup metric xy ---
    e_xy = {}
    if isinstance(E_nodes, pd.DataFrame) and len(E_nodes) > 0 and "x_m" in E_nodes.columns:
        for r in E_nodes.itertuples(index=False):
            e_xy[int(r.node_id)] = (float(r.x_m), float(r.y_m))

    t_xy = {}
    if isinstance(T_nodes, pd.DataFrame) and len(T_nodes) > 0 and "x_m" in T_nodes.columns:
        for r in T_nodes.itertuples(index=False):
            t_xy[int(r.node_id)] = (float(r.x_m), float(r.y_m))

    def _dist_km(a, b):
        ax, ay = a
        bx, by = b
        return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / 1000.0

    # --- edges ---
    if isinstance(E_edges, pd.DataFrame) and len(E_edges) > 0:
        for r in E_edges.itertuples(index=False):
            u = int(getattr(r, "u"))
            v = int(getattr(r, "v"))
            ku = e_node_key_fn(u)
            kv = e_node_key_fn(v)
            w = _dist_km(e_xy[u], e_xy[v]) if (u in e_xy and v in e_xy) else 0.0
            G.add_edge(ku, kv, etype=str(getattr(r, "etype", "E_RING")), ring_id=int(getattr(r, "ring_id", -1)), seq=int(getattr(r, "seq", -1)), **{weight_attr: w, "weight": w})
            added_edges += 1

    if isinstance(T_edges, pd.DataFrame) and len(T_edges) > 0:
        for r in T_edges.itertuples(index=False):
            u = int(getattr(r, "u"))
            v = int(getattr(r, "v"))
            ku = t_node_key_fn(u)
            kv = t_node_key_fn(v)
            w = _dist_km(t_xy[u], t_xy[v]) if (u in t_xy and v in t_xy) else 0.0
            G.add_edge(ku, kv, etype=str(getattr(r, "etype", "T_RING")), ring_id=int(getattr(r, "ring_id", -1)), seq=int(getattr(r, "seq", -1)), **{weight_attr: w, "weight": w})
            added_edges += 1

    return {"nodes_added": added_nodes, "edges_added": added_edges}

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
            # debug msg
            print(f"\n[PROJECTOR] Using methods: {a} / {b}")
            # debug msg
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


# --- add ring graph (E/T) + E↔T transfer + T-gate→Sea connectors ---
try:
    rg = out.get("ring_graph", {}) or {}
    if isinstance(rg.get("E_nodes"), pd.DataFrame) and isinstance(rg.get("T_nodes"), pd.DataFrame):
        # namespace ring node ids to avoid collisions with sea indices
        e_key_fn = lambda eid: ("E", int(eid))
        t_key_fn = lambda tid: ("T", int(tid))

        sea_key_fn = guess_sea_node_key_fn(G)

        st = add_ring_nodes_edges_to_graph(G, out, e_node_key_fn=e_key_fn, t_node_key_fn=t_key_fn)
        print(f"[graph] ring nodes/edges added: {st}")

        # E↔T shared transfer edges (0-cost)
        df_et = build_et_shared_edges(out, params=ETTransferParams(weight_km=0.0, bidirectional=True))
        added_et = add_et_shared_edges_to_graph(G, df_et, e_node_key_fn=e_key_fn, t_node_key_fn=t_key_fn)
        out["et_shared_edges"] = df_et
        print(f"[graph] E↔T shared transfer edges added: {added_et}")

        # T gate → Sea connectors
        tparams = TGateSeaConnectorParams(
            k_connect=2,
            topN=60,
            r_connect_km=120.0,
            enable_sector_filter=True,
            sector_deg=110.0,
            do_collision_check=True,
            do_repair=True,
        )
        df_tconn = build_tgate_sea_connectors(out, params=tparams)
        added_tconn = add_tgate_sea_connectors_to_graph(G, df_tconn, t_node_key_fn=t_key_fn, sea_node_key_fn=sea_key_fn)
        out["tgate_sea_connectors"] = df_tconn
        print(f"[graph] T-gate→Sea connector edges added: {added_tconn}")
    else:
        print("[graph] ring_graph not found or empty; skip ring/tgate integration")
except Exception as e:
    print("[graph] ring/tgate integration failed:", repr(e))


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
    collision_used = None
    ll2m_xy, m2ll_xy = None, None

    #  store inject-edge polylines (these are the "temporary point" links you care about)
    inject_start_ll = None   # path[0] -> path[1] (repaired if start_inject)
    inject_end_ll   = None   # path[-2] -> path[-1] (repaired if end_inject)

    if len(pair.start_pick) == 0 or len(pair.end_pick) == 0:
        print("[snap] FAIL:", pair.reason, pair.debug)
    else:
        inject_point_edges(G, start_key, pair.start_pick, k_inject=int(k_inject), etype="start_inject")
        inject_point_edges(G, end_key,   pair.end_pick,   k_inject=int(k_inject), etype="end_inject")
        print("[snap] OK:", pair.reason, pair.debug)
        print("[snap][start] local_entrance_aug:", pair.start.debug.get("local_entrance_aug"))
        print("[snap][end  ] local_entrance_aug:", pair.end.debug.get("local_entrance_aug"))


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
            collision = _get_collision_metric(out)
            b = collision.bounds
            print("[crs] collision.bounds =", b)
            print("[crs] collision looks like",
                "DEGREES(lon/lat)" if max(map(abs, b)) < 1000 else "METERS")
            # ----- debug
            p = (119.7734, 4.7502)
            x, y = ll2m_xy(p[0], p[1])
            print("[crs] ll2m(p) =", (x, y))
            b = collision.bounds
            print("[crs] point-in-collision-bounds?", (b[0] <= x <= b[2] and b[1] <= y <= b[3]))
            print("[crs] collision.bounds =", b)
            

            # ----- debug 

            _ll_to_m = lambda a, b=None: ll2m_xy(float(a[0]), float(a[1])) if b is None else ll2m_xy(float(a), float(b))
            _m_to_ll = lambda a, b=None: m2ll_xy(float(a[0]), float(a[1])) if b is None else m2ll_xy(float(a), float(b))

            
            if collision is None:
                print("[collision] not found -> skip repair/simplify")
                path_ll_for_simplify = [(float(p[0]), float(p[1])) for p in path]
            else:
                collision = _clip_collision_to_aoi_bbox(collision, bbox_ll, ll2m_xy)
                #print("[collision_full.bounds]", collision_full.bounds)
                #print("[collision_repair.bounds]", collision_repair.bounds)
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
                            ll_to_m=_ll_to_m,
                            m_to_ll=_m_to_ll,
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
                            ll_to_m=_ll_to_m,
                            m_to_ll=_m_to_ll,
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
                            ll_to_m=_ll_to_m,
                            m_to_ll=_m_to_ll,
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
                # simplify
                print(len(path_ll_for_simplify))
                #for i, (lon, lat) in enumerate(path_ll_for_simplify):
                    #print(f"[dump] {i:03d}: ({float(lon):.10f}, {float(lat):.10f})")
                if do_simplify and path_ll_for_simplify is not None and len(path_ll_for_simplify) >= 2:
                    path_simplified, simp_stats = simplify_path_visibility(
                        path_ll_for_simplify,
                        collision_m=collision,
                        ll_to_m=_ll_to_m,
                        m_to_ll=_m_to_ll,
                        window_size=80,
                        max_tries=300,
                        use_prepared_collision=True,
                        dateline_unwrap=True,
                    )
                    print("[simplify]", simp_stats)
                    #for i, (lon, lat) in enumerate(path_simplified):
                       # print(f"[dump] {i:03d}: ({float(lon):.10f}, {float(lat):.10f})")
                    p = (119.7734, 4.7502)
                    print("[crs] _ll_to_m(p)     =", _ll_to_m(p))
                    # 如果你程式裡還存在 _ll_to_m_any，也一起印
                    try:
                        print("[crs] _ll_to_m_any(p) =", _ll_to_m_any(p))
                    except NameError:
                        pass

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
                                ll_to_m=_ll_to_m,
                                m_to_ll=_m_to_ll,
                                repairer_obj=repairer_obj,   # 你前面已經建過 PathRepairer 了，直接重用
                            )
                        else:
                            snap_start_ll = [origin_ll]

                        # snap-link end
                        if end_key != dest_ll:
                            snap_end_ll = repair_snap_link_ll_if_needed(
                                end_key, dest_ll,
                                collision_m=collision,
                                ll_to_m=_ll_to_m,
                                m_to_ll=_m_to_ll,
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
            print("[collision viz EXACT] bounds:", getattr(collision_viz, "bounds", None),
                  "type:", getattr(collision_viz, "geom_type", type(collision_viz)))

            # Use the same projector as the algorithm
            _, m2ll_xy, _, _ = _make_ll_m_projectors_from_out(out)

            # IMPORTANT: no extra clipping here, no viz_box intersection.
            # This ensures what you see == what simplify used.
            col_ll = _geom_m_to_ll(collision_viz, m2ll_xy)

            fgCol = folium.FeatureGroup(name="Collision (USED, exact ll)", show=False)
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



# ---------------------------
# Ring graph + T-gates + connectors (viz)
# ---------------------------
rg = out.get("ring_graph", {}) or {}
E_nodes = rg.get("E_nodes", None)
T_nodes = rg.get("T_nodes", None)
E_edges_rg = rg.get("E_edges", None)
T_edges_rg = rg.get("T_edges", None)
Shared_nodes_rg = rg.get("Shared_nodes", None)

df_et = safe_df(out, "et_shared_edges")
df_tconn = safe_df(out, "tgate_sea_connectors")

def _lonlat_dict(df):
    d = {}
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        for r in df.itertuples(index=False):
            try:
                d[int(getattr(r, "node_id"))] = (float(getattr(r, "lon")), float(getattr(r, "lat")))
            except Exception:
                pass
    return d

e_ll = _lonlat_dict(E_nodes)
t_ll = _lonlat_dict(T_nodes)

# E edges
if isinstance(E_edges_rg, pd.DataFrame) and len(E_edges_rg) > 0:
    fg = folium.FeatureGroup(name=f"E_edges ({len(E_edges_rg)})", show=False)
    drawn = 0
    for r in E_edges_rg.itertuples(index=False):
        u, v = int(getattr(r, "u")), int(getattr(r, "v"))
        if u in e_ll and v in e_ll:
            a, b = e_ll[u], e_ll[v]
            folium.PolyLine([[a[1], a[0]], [b[1], b[0]]], color="#ff6a00", weight=2, opacity=0.75).add_to(fg)
            drawn += 1
    fg.add_to(m)
    print(f"[viz] E_edges drawn: {drawn}/{len(E_edges_rg)}")

# T edges
if isinstance(T_edges_rg, pd.DataFrame) and len(T_edges_rg) > 0:
    fg = folium.FeatureGroup(name=f"T_edges ({len(T_edges_rg)})", show=True)
    drawn = 0
    for r in T_edges_rg.itertuples(index=False):
        u, v = int(getattr(r, "u")), int(getattr(r, "v"))
        if u in t_ll and v in t_ll:
            a, b = t_ll[u], t_ll[v]
            folium.PolyLine([[a[1], a[0]], [b[1], b[0]]], color="#111111", weight=3, opacity=0.85).add_to(fg)
            drawn += 1
    fg.add_to(m)
    print(f"[viz] T_edges drawn: {drawn}/{len(T_edges_rg)}")

# T gate candidates (nodes)
if isinstance(T_nodes, pd.DataFrame) and len(T_nodes) > 0 and "is_gate_candidate" in T_nodes.columns:
    fg = folium.FeatureGroup(name=f"T_gates ({int(T_nodes['is_gate_candidate'].sum())})", show=True)
    dfG = T_nodes[T_nodes["is_gate_candidate"] == True]
    for _, r in dfG.iterrows():
        p = (float(r["lon"]), float(r["lat"]))
        if not in_bbox(p, bbox_ll):
            continue
        folium.CircleMarker([p[1], p[0]], radius=7, color="#8b5cf6", fill=True, fill_opacity=0.9, tooltip=str(r.get("gate_reason","t_gate"))).add_to(fg)
    fg.add_to(m)

# E↔T shared transfer edges (viz)
if isinstance(df_et, pd.DataFrame) and len(df_et) > 0:
    fg = folium.FeatureGroup(name=f"E↔T shared edges ({len(df_et)})", show=False)
    drawn = 0
    for r in df_et.itertuples(index=False):
        if getattr(r, "u_kind", "") == "E" and getattr(r, "v_kind", "") == "T":
            e_id = int(getattr(r, "u"))
            t_id = int(getattr(r, "v"))
        elif getattr(r, "u_kind", "") == "T" and getattr(r, "v_kind", "") == "E":
            t_id = int(getattr(r, "u"))
            e_id = int(getattr(r, "v"))
        else:
            continue
        if e_id in e_ll and t_id in t_ll:
            a, b = e_ll[e_id], t_ll[t_id]
            folium.PolyLine([[a[1], a[0]], [b[1], b[0]]], color="#94a3b8", weight=2, opacity=0.7).add_to(fg)
            drawn += 1
    fg.add_to(m)
    print(f"[viz] E↔T edges drawn: {drawn}/{len(df_et)}")

# T gate → Sea connectors (viz)
if isinstance(df_tconn, pd.DataFrame) and len(df_tconn) > 0 and isinstance(S_nodes, pd.DataFrame) and len(S_nodes) > 0:
    fg = folium.FeatureGroup(name=f"Tgate→Sea connectors ({len(df_tconn)})", show=True)
    drawn = 0
    for r in df_tconn.itertuples(index=False):
        tid = int(getattr(r, "t_node_id"))
        sid = int(getattr(r, "sea_idx"))
        if tid not in t_ll:
            continue
        try:
            s = S_nodes.iloc[sid]
            b = (float(s["lon"]), float(s["lat"]))
        except Exception:
            continue
        a = t_ll[tid]
        if not (in_bbox(a, bbox_ll) or in_bbox(b, bbox_ll)):
            continue
        folium.PolyLine([[a[1], a[0]], [b[1], b[0]]], color="#ef4444", weight=2, opacity=0.85).add_to(fg)
        drawn += 1
    fg.add_to(m)
    print(f"[viz] Tgate→Sea connectors drawn: {drawn}/{len(df_tconn)}")

    folium.LayerControl(collapsed=False).add_to(m)

    html_path = Path(html_path).resolve()
    m.save(str(html_path))
    webbrowser.open(html_path.as_uri())
    return html_path


# =========================
# === call
# =========================
origin_ll = (128.52636, -30.38197)
dest_ll   = (119.82013, 13.85607)

open_routing_debug_map_p2p(
    out,
    origin_ll=origin_ll,
    dest_ll=dest_ll,
    html_path="aoi_p2p_map.html",
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