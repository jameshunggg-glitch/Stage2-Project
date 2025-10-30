# routing/scgraph_bridge.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import math, random

try:
    import numpy as np
except Exception:  # numpy 非硬性需求，但建議安裝
    np = None

# -- 嘗試導入 scgraph（marnet） --
_MARNET = None
try:
    from scgraph.geographs.marnet import marnet_geograph as _MARNET  # 官方入口
except Exception:
    _MARNET = None


# ----------------- 基礎工具 -----------------
def _snap_key(pt: Tuple[float, float], decimals: int = 6) -> Tuple[float, float]:
    return (round(float(pt[0]), decimals), round(float(pt[1]), decimals))

def _bbox_tuple_from_polygon_bounds(bounds: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    # shapely polygon.bounds -> (minx, miny, maxx, maxy)
    x0, y0, x1, y1 = [float(v) for v in bounds]
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
    return (x0, y0, x1, y1)

def _bearing_deg(p_from: Tuple[float,float], p_to: Tuple[float,float]) -> float:
    dx = float(p_to[0]) - float(p_from[0])
    dy = float(p_to[1]) - float(p_from[1])
    ang = math.degrees(math.atan2(dy, dx))  # [-180,180]
    return (ang + 360.0) % 360.0

def _angle_deviation_from_straight(b1: float, b2: float) -> float:
    # 相差 0 表示同向；此函數回傳與「直線(180°)」的偏差量（彎越大→值越大）
    diff = abs(b1 - b2)
    diff = 360.0 - diff if diff > 180.0 else diff  # [0,180]
    return 180.0 - diff  # 直線=0，越彎越大


# ----------------- 解析 marnet 物件 -----------------
def _segments_from_edges_list(edges, nodes_lookup=None) -> List[Tuple[Tuple[float,float], Tuple[float,float]]]:
    """
    嘗試把 scgraph 各種 edge 物件展開成「線段端點 pair」清單。
    回傳：[( (lon1,lat1), (lon2,lat2) ), ...]
    """
    segs = []
    for e in edges or []:
        try:
            # 常見形式：tuple/list: (u,v, data)
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                u, v = e[0], e[1]
                if nodes_lookup and u in nodes_lookup and v in nodes_lookup:
                    pu, pv = nodes_lookup[u], nodes_lookup[v]
                    segs.append(( (float(pu[0]), float(pu[1])), (float(pv[0]), float(pv[1])) ))
            # 字典形式：geojson-like
            elif isinstance(e, dict):
                if "geometry" in e and hasattr(e["geometry"], "coords"):
                    coords = list(e["geometry"].coords)
                elif "coordinates" in e and isinstance(e["coordinates"], (list, tuple)):
                    coords = list(e["coordinates"])
                else:
                    coords = None
                if coords and len(coords) >= 2:
                    for i in range(len(coords)-1):
                        a, b = coords[i], coords[i+1]
                        segs.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
        except Exception:
            continue
    return segs


def _try_graph_like_and_build_segments() -> Optional[List[Tuple[Tuple[float,float], Tuple[float,float]]]]:
    """
    嘗試在 marnet 物件裡找 nodes/edges，直接抽出全域線段。
    """
    if _MARNET is None:
        return None
    # 嘗試幾個常見屬性名稱
    for attr in ("graph", "_graph", "geograph", "network", "G", "_G"):
        G = getattr(_MARNET, attr, None)
        if G is None:
            continue

        # 讀 nodes：不同版本可能是 G.nodes 或 dict-like
        nodes = None
        for nattr in ("nodes", "node", "vertices", "_nodes"):
            try:
                obj = getattr(G, nattr)
            except Exception:
                obj = None
            if obj is None:
                continue
            # networkx-like: G.nodes(data=True)
            try:
                tmp = {}
                it = obj(data=True) if callable(obj) else obj
                for item in it:
                    if isinstance(item, tuple) and len(item) == 2:
                        nid, data = item
                        lon = data.get("longitude") or data.get("lon") or data.get("x")
                        lat = data.get("latitude")  or data.get("lat")  or data.get("y")
                        if lon is not None and lat is not None:
                            tmp[nid] = (float(lon), float(lat))
                if tmp:
                    nodes = tmp
                    break
            except Exception:
                # dict-like
                try:
                    tmp = {}
                    for nid, data in obj.items():
                        if isinstance(data, dict):
                            lon = data.get("longitude") or data.get("lon") or data.get("x")
                            lat = data.get("latitude")  or data.get("lat")  or data.get("y")
                            if lon is not None and lat is not None:
                                tmp[nid] = (float(lon), float(lat))
                    if tmp:
                        nodes = tmp
                        break
                except Exception:
                    pass

        # 讀 edges
        for eattr in ("edges", "_edges", "links"):
            edges = getattr(G, eattr, None)
            if edges is None:
                continue
            try:
                edges = edges() if callable(edges) else edges
            except Exception:
                pass
            segs = _segments_from_edges_list(edges, nodes_lookup=nodes)
            if segs:
                return segs
    return None


def _fallback_segments_by_sampling(aoi: Optional[Tuple[float,float,float,float]], n_paths: int = 60) -> List[Tuple[Tuple[float,float], Tuple[float,float]]]:
    """
    如果抓不到原生邊，就在 AOI 內做多對隨機端點的最短路，疊出一個近似子網。
    """
    segs = []
    if _MARNET is None:
        return segs
    if aoi is None:
        aoi = (60, -20, 150, 40)  # 安全預設，避免抽全地球

    x0, y0, x1, y1 = aoi
    nx, ny = 6, 5
    xs = [x0 + i*(x1-x0)/(nx-1) for i in range(nx)]
    ys = [y0 + j*(y1-y0)/(ny-1) for j in range(ny)]
    pts = [(float(x), float(y)) for x in xs for y in ys]

    for _ in range(max(1, int(n_paths))):
        (lon1, lat1), (lon2, lat2) = random.sample(pts, 2)
        try:
            out = _MARNET.get_shortest_path(
                origin_node={"longitude": lon1, "latitude": lat1},
                destination_node={"longitude": lon2, "latitude": lat2},
                output_units="km",
            )
            path = out.get("coordinate_path", [])
            coords = []
            for p in path:
                if isinstance(p, dict):
                    lon = p.get("longitude") or p.get("lon") or p.get("x")
                    lat = p.get("latitude")  or p.get("lat")  or p.get("y")
                    if lon is not None and lat is not None:
                        coords.append((float(lon), float(lat)))
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    a, b = p[0], p[1]
                    # 嘗試判斷 (lon,lat) or (lat,lon)
                    if -180 <= a <= 180 and -90 <= b <= 90:
                        coords.append((float(a), float(b)))
                    elif -90 <= a <= 90 and -180 <= b <= 180:
                        coords.append((float(b), float(a)))
            if len(coords) >= 2:
                for i in range(len(coords)-1):
                    segs.append((coords[i], coords[i+1]))
        except Exception:
            continue
    return segs


# ----------------- 對外 API -----------------
def sc_edges_in_bbox(
    bbox_ll: Tuple[float,float,float,float],
    edge_sample_ratio: float = 1.0,
    max_sample_routes: int = 60,
    node_snap_decimals: int = 6,
    simplify_epsilon_km: float = 0.0,
    **kwargs,
) -> Dict:
    """
    取得 AOI（lon/lat bbox）內的 scgraph 子網：nodes + edges。
    返回：
      {
        "nodes": [(lon,lat), ...],
        "edges": [ ((lon1,lat1),(lon2,lat2)), ... ],
        "stats": {"edge_count": N, "node_count": M}
      }
    """
    aoi = bbox_ll
    segs = _try_graph_like_and_build_segments()  # 全域線段（若拿得到）
    if not segs:
        segs = _fallback_segments_by_sampling(aoi, n_paths=max_sample_routes)

    # AOI 篩選與抽稀
    x0, y0, x1, y1 = aoi
    def _in_aoi(pt):
        x, y = pt
        return (x0 <= x <= x1) and (y0 <= y <= y1)

    edges = []
    nodes_set = set()
    import random as _rnd

    for (a, b) in segs or []:
        if not (_in_aoi(a) or _in_aoi(b)):
            continue
        if edge_sample_ratio < 1.0 and _rnd.random() > edge_sample_ratio:
            continue
        ra, rb = _snap_key(a, node_snap_decimals), _snap_key(b, node_snap_decimals)
        nodes_set.add(ra); nodes_set.add(rb)
        edges.append((ra, rb))

    return {
        "nodes": [(float(x), float(y)) for (x, y) in nodes_set],
        "edges": edges,
        "stats": {"edge_count": len(edges), "node_count": len(nodes_set)}
    }


def sc_keypoints_in_bbox(
    bbox_ll: Tuple[float,float,float,float],
    edges: Optional[List[Tuple[Tuple[float,float], Tuple[float,float]]]] = None,
    node_snap_decimals: int = 5,
    bend_threshold_deg: float = 12.0,
    **kwargs,
) -> List[Tuple[float,float]]:
    """
    從 sc 子網邊集中抽取「交會點（度數≠2）」與「轉折點（度數=2 且彎折>=門檻）」。
    若未提供 edges，會自行呼叫 sc_edges_in_bbox() 取得。
    """
    if edges is None:
        got = sc_edges_in_bbox(bbox_ll, node_snap_decimals=node_snap_decimals, **kwargs)
        edges = got.get("edges", [])

    from collections import defaultdict
    neighbors = defaultdict(set)
    coord_accum = defaultdict(lambda: [0.0, 0.0, 0])

    for (a, b) in edges:
        if a == b:  # 去除零長
            continue
        neighbors[a].add(b)
        neighbors[b].add(a)
        sx, sy, c = coord_accum[a]; coord_accum[a] = [sx + a[0], sy + a[1], c + 1]
        sx, sy, c = coord_accum[b]; coord_accum[b] = [sx + b[0], sy + b[1], c + 1]

    # 平均同 key 的代表座標
    node_xy = {}
    for k, (sx, sy, c) in coord_accum.items():
        node_xy[k] = (sx / c, sy / c) if c > 0 else (k[0], k[1])

    keypoints: List[Tuple[float,float]] = []
    for k, ns in neighbors.items():
        xy = node_xy[k]
        deg = len(ns)
        if deg != 2:
            keypoints.append(xy)
        else:
            n1, n2 = list(ns)
            b1 = _bearing_deg(xy, node_xy[n1])
            b2 = _bearing_deg(xy, node_xy[n2])
            bend = _angle_deviation_from_straight(b1, b2)
            if bend >= bend_threshold_deg:
                keypoints.append(xy)

    # 去重
    seen = set()
    out = []
    for p in keypoints:
        sp = _snap_key(p, node_snap_decimals)
        if sp in seen:
            continue
        seen.add(sp)
        out.append(sp)
    return out


def sc_shortest_path_lonlat(
    origin: Tuple[float,float],
    dest: Tuple[float,float],
    **kwargs
) -> Dict:
    """
    取得 scgraph O→D 路徑（lon/lat polyline）。若不可得，回傳 {}。
    """
    if _MARNET is None:
        return {}
    try:
        out = _MARNET.get_shortest_path(
            origin_node={"longitude": float(origin[0]), "latitude": float(origin[1])},
            destination_node={"longitude": float(dest[0]), "latitude": float(dest[1])},
            **({"output_units": "km"} | kwargs),
        )
        coords = []
        for p in out.get("coordinate_path", []):
            if isinstance(p, dict):
                lon = p.get("longitude") or p.get("lon") or p.get("x")
                lat = p.get("latitude")  or p.get("lat")  or p.get("y")
                if lon is not None and lat is not None:
                    coords.append((float(lon), float(lat)))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                a, b = p[0], p[1]
                if -180 <= a <= 180 and -90 <= b <= 90:
                    coords.append((float(a), float(b)))
                elif -90 <= a <= 90 and -180 <= b <= 180:
                    coords.append((float(b), float(a)))

        length_km = None
        for key in ("length_km", "length", "route_length", "distance"):
            if key in out and out[key] is not None:
                try:
                    length_km = float(out[key])
                    break
                except Exception:
                    pass

        return {"track": coords, "length_km": length_km}
    except Exception:
        return {}
