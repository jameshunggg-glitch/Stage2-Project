"""routing_map.routing_graph

Minimal utilities to assemble a base routing graph (networkx) from build_aoi() outputs.

Design goals (dev-stage):
- Small, explicit, easy to debug.
- Robust to a few common edge formats.
- Store per-edge metadata: etype, length_km.

Expected `out` keys (from your build_aoi pipeline):
- S_nodes: pd.DataFrame with columns ['lon','lat', ...]
- S_edges: list of edges, typically [((lon,lat),(lon,lat)), ...]
- C_nodes: pd.DataFrame with columns ['c_id','lon','lat', ...]
- C_edges: pd.DataFrame with columns ['u','v', ...] and optionally 'length_km'
- gateB_connectors: pd.DataFrame with columns ['g_id', 'sea_idx', ...] or ['gate_uid', ...]
- Gate_all_cov or Gate_all: pd.DataFrame with columns ['g_id','lon','lat']

Notes
-----
- Node IDs in the assembled graph are lon/lat tuples: (lon, lat) as float.
  (This matches your notebook debug graph.)
- Edge weights are stored under attribute 'weight' (NetworkX convention) and
  are in kilometers when weight_unit='km'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

import networkx as nx

LonLat = Tuple[float, float]


# -------------------------
# Distance (dateline-safe)
# -------------------------
EARTH_R_KM = 6371.0088


def _wrap_dlon_deg(lon2: float, lon1: float) -> float:
    """Shortest signed longitude difference in degrees in [-180, 180]."""
    return (float(lon2) - float(lon1) + 180.0) % 360.0 - 180.0


def haversine_km(p1: LonLat, p2: LonLat) -> float:
    """Dateline-safe haversine distance between (lon,lat) points, in km."""
    lon1, lat1 = map(float, p1)
    lon2, lat2 = map(float, p2)

    dlon = math.radians(_wrap_dlon_deg(lon2, lon1))
    dlat = math.radians(lat2 - lat1)
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)

    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2.0) ** 2
    a = min(1.0, max(0.0, a))
    return 2.0 * EARTH_R_KM * math.asin(math.sqrt(a))


# -------------------------
# Helpers
# -------------------------

def _is_df(x: Any) -> bool:
    return pd is not None and isinstance(x, pd.DataFrame)


def _safe_df(out: Dict[str, Any], name: str):
    df = out.get(name)
    if df is None:
        return None
    try:
        return df if len(df) > 0 else None
    except Exception:
        return None


def _to_lonlat(p: Any) -> Optional[LonLat]:
    if isinstance(p, (tuple, list)) and len(p) >= 2:
        try:
            return (float(p[0]), float(p[1]))
        except Exception:
            return None
    return None


def _sea_lonlat_by_idx(S_nodes, idx: int) -> LonLat:
    row = S_nodes.iloc[int(idx)]
    return (float(row["lon"]), float(row["lat"]))


def _edge_to_lonlat(
    e: Any,
    *,
    nodes_df=None,
    idx_to_lonlat_fn=None,
) -> Optional[Tuple[LonLat, LonLat]]:
    """Return ((lon1,lat1),(lon2,lat2)) for an edge or None.

    Supports:
    - (u_idx, v_idx) with nodes_df + idx_to_lonlat_fn
    - ((lon,lat),(lon,lat))
    - ("lon,lat","lon,lat")
    """
    if not isinstance(e, (list, tuple)) or len(e) < 2:
        return None
    a, b = e[0], e[1]

    # index form
    if isinstance(a, (int,)) and isinstance(b, (int,)):
        if nodes_df is None or idx_to_lonlat_fn is None:
            return None
        try:
            return (idx_to_lonlat_fn(int(a)), idx_to_lonlat_fn(int(b)))
        except Exception:
            return None

    # lonlat tuple form
    p1 = _to_lonlat(a)
    p2 = _to_lonlat(b)
    if p1 and p2:
        return (p1, p2)

    # string "lon,lat" form
    if isinstance(a, str) and isinstance(b, str):
        try:
            lon1, lat1 = a.split(",")
            lon2, lat2 = b.split(",")
            return ((float(lon1), float(lat1)), (float(lon2), float(lat2)))
        except Exception:
            return None

    return None


@dataclass
class GraphBuildStats:
    sea_edges_added: int = 0
    cc_edges_added: int = 0
    gateb_sea_edges_added: int = 0
    c_gateb_edges_added: int = 0


def build_gate_xy(out: Dict[str, Any]) -> Dict[int, LonLat]:
    """Map g_id -> (lon,lat) from Gate_all_cov or Gate_all."""
    gate_df = _safe_df(out, "Gate_all_cov")
    if gate_df is None:
        gate_df = _safe_df(out, "Gate_all")
    if gate_df is None:
        return {}
    if "g_id" not in gate_df.columns or "lon" not in gate_df.columns or "lat" not in gate_df.columns:
        return {}
    xy: Dict[int, LonLat] = {}
    for _, r in gate_df.iterrows():
        try:
            gid = int(r["g_id"])
            xy[gid] = (float(r["lon"]), float(r["lat"]))
        except Exception:
            continue
    return xy


def build_c_map(C_nodes) -> Dict[int, LonLat]:
    """Map c_id -> (lon,lat)."""
    if C_nodes is None or not _is_df(C_nodes):
        return {}
    if "c_id" not in C_nodes.columns:
        return {}
    out: Dict[int, LonLat] = {}
    for _, r in C_nodes.iterrows():
        try:
            out[int(r["c_id"])] = (float(r["lon"]), float(r["lat"]))
        except Exception:
            continue
    return out


def build_base_graph(
    out: Dict[str, Any],
    *,
    include_sea: bool = True,
    include_cc: bool = True,
    include_gateb_sea: bool = True,
    include_c_gateb: bool = False,
    c_gateb_df=None,
    max_sea_edges: Optional[int] = None,
    max_cc_edges: Optional[int] = None,
    weight_unit: str = "km",
    bbox_ll: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[nx.Graph, GraphBuildStats]:
    """Assemble a NetworkX undirected graph from `out`.

    Parameters
    ----------
    include_sea:
        Add sea edges from out['S_edges'].
    include_cc:
        Add coastal chain edges from out['C_edges'].
    include_gateb_sea:
        Add GateB->Sea connectors from out['gateB_connectors'].
    include_c_gateb:
        Add C<->GateB bridge edges (optional, if you have a df from c_gateb_connectors).
    c_gateb_df:
        Optional DataFrame with columns like ['c_lon','c_lat','g_lon','g_lat'] and optionally
        a distance column in degrees or km.
    weight_unit:
        'km' (default) uses haversine_km, else 'deg' uses Euclidean degrees (debug only).
    bbox_ll:
        Optional (min_lon,min_lat,max_lon,max_lat) filter: keep edges if either endpoint in bbox.

    Returns
    -------
    (G, stats)
    """

    def in_bbox(p: LonLat) -> bool:
        if bbox_ll is None:
            return True
        x0, y0, x1, y1 = map(float, bbox_ll)
        return (x0 <= p[0] <= x1) and (y0 <= p[1] <= y1)

    def edge_keep(u: LonLat, v: LonLat) -> bool:
        if bbox_ll is None:
            return True
        return in_bbox(u) or in_bbox(v)

    def dist(u: LonLat, v: LonLat) -> float:
        if weight_unit == "km":
            return haversine_km(u, v)
        # debug: degree euclidean
        return float(math.hypot(u[0] - v[0], u[1] - v[1]))

    stats = GraphBuildStats()
    G = nx.Graph()

    # --- sea edges ---
    if include_sea:
        S_nodes = _safe_df(out, "S_nodes")
        S_edges = out.get("S_edges")

        if S_edges is not None and isinstance(S_edges, (list, tuple)) and len(S_edges) > 0:
            take = S_edges
            if max_sea_edges is not None and len(take) > int(max_sea_edges):
                take = take[: int(max_sea_edges)]

            idx_to_ll = None
            if _is_df(S_nodes):
                idx_to_ll = lambda i: _sea_lonlat_by_idx(S_nodes, i)

            for e in take:
                seg = _edge_to_lonlat(e, nodes_df=S_nodes, idx_to_lonlat_fn=idx_to_ll)
                if seg is None:
                    continue
                u, v = seg
                if not edge_keep(u, v):
                    continue
                w = dist(u, v)
                G.add_edge(u, v, weight=w, length_km=(w if weight_unit == "km" else None), etype="sea")
                stats.sea_edges_added += 1

    # --- coastal chain edges ---
    if include_cc:
        C_nodes = _safe_df(out, "C_nodes")
        C_edges = _safe_df(out, "C_edges")
        c_map = build_c_map(C_nodes)

        if _is_df(C_edges) and c_map:
            cc_df = C_edges
            if max_cc_edges is not None and len(cc_df) > int(max_cc_edges):
                cc_df = cc_df.iloc[: int(max_cc_edges)]

            # best effort length
            has_len = "length_km" in cc_df.columns

            for _, e in cc_df.iterrows():
                try:
                    p1 = c_map.get(int(e["u"]))
                    p2 = c_map.get(int(e["v"]))
                except Exception:
                    continue
                if p1 is None or p2 is None:
                    continue
                if not edge_keep(p1, p2):
                    continue

                if weight_unit == "km":
                    w = float(e["length_km"]) if has_len and e.get("length_km") is not None else haversine_km(p1, p2)
                    G.add_edge(p1, p2, weight=w, length_km=w, etype="cc")
                else:
                    w = dist(p1, p2)
                    G.add_edge(p1, p2, weight=w, length_km=None, etype="cc")

                stats.cc_edges_added += 1

    # --- GateB -> Sea connectors ---
    if include_gateb_sea:
        df_conn = _safe_df(out, "gateB_connectors")
        gate_xy = build_gate_xy(out)
        S_nodes = _safe_df(out, "S_nodes")

        if _is_df(df_conn) and gate_xy and _is_df(S_nodes):
            # support columns: sea_idx OR s_idx
            sea_col = "sea_idx" if "sea_idx" in df_conn.columns else ("s_idx" if "s_idx" in df_conn.columns else None)
            gid_col = "g_id" if "g_id" in df_conn.columns else None

            if sea_col and gid_col:
                for _, r in df_conn.iterrows():
                    try:
                        gid = int(r[gid_col])
                        if gid not in gate_xy:
                            continue
                        gb = gate_xy[gid]
                        sea = _sea_lonlat_by_idx(S_nodes, int(r[sea_col]))
                    except Exception:
                        continue
                    if not edge_keep(gb, sea):
                        continue
                    w = dist(gb, sea)
                    G.add_edge(gb, sea, weight=w, length_km=(w if weight_unit == "km" else None), etype="gb_sea")
                    stats.gateb_sea_edges_added += 1

    # --- C <-> GateB bridge connectors (optional) ---
    if include_c_gateb:
        df = c_gateb_df
        if df is None:
            df = _safe_df(out, "c_gateb_connectors")

        if _is_df(df):
            # expect columns: c_lon,c_lat,g_lon,g_lat and optionally dist_km / dist_deg
            for _, r in df.iterrows():
                try:
                    c = (float(r.get("c_lon", r.get("lon_c"))), float(r.get("c_lat", r.get("lat_c"))))
                    g = (float(r.get("g_lon", r.get("lon_g"))), float(r.get("g_lat", r.get("lat_g"))))
                except Exception:
                    continue
                if not edge_keep(c, g):
                    continue

                if weight_unit == "km":
                    w = float(r["dist_km"]) if "dist_km" in df.columns and r.get("dist_km") is not None else haversine_km(c, g)
                    G.add_edge(c, g, weight=w, length_km=w, etype="c_gb")
                else:
                    w = float(r["dist_deg"]) if "dist_deg" in df.columns and r.get("dist_deg") is not None else dist(c, g)
                    G.add_edge(c, g, weight=w, length_km=None, etype="c_gb")

                stats.c_gateb_edges_added += 1

    return G, stats


__all__ = [
    "LonLat",
    "GraphBuildStats",
    "haversine_km",
    "build_base_graph",
    "build_gate_xy",
    "build_c_map",
]
