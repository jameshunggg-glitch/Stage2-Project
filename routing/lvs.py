# routing/lvs.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import heapq, itertools

from shapely.geometry import LineString, Point
from shapely.ops import transform as shp_transform

from .config import NEIGHBOR_K, STEP_KM_GEODESIC
from .geodesy import gc_distance_km, geodesic_sample, to_m
from .land_layers import nudge_to_ring_if_inside_fast

# ----------------- 基本成本與啟發 -----------------
def edge_cost(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return gc_distance_km(a, b)

def heuristic(p: Tuple[float,float], D: Tuple[float,float]) -> float:
    return gc_distance_km(p, D)

def neighbors_of(u_idx: int, nodes: List[Tuple[float,float]], D_idx: int, k: int = NEIGHBOR_K) -> List[int]:
    """貪心式近鄰：對每個候選 v 以 du + 0.5*dD 排序，取前 k 並確保包含終點"""
    u = nodes[u_idx]; D = nodes[D_idx]
    scored = []
    for v_idx, q in enumerate(nodes):
        if v_idx == u_idx:
            continue
        du = gc_distance_km(u, q)
        dD = gc_distance_km(q, D)
        scored.append((du + 0.5 * dD, v_idx))
    scored.sort(key=lambda t: t[0])
    out = [v for _, v in itertools.islice(scored, 0, k)]
    if D_idx not in out:
        out.append(D_idx)
    return out

# ----------------- 進度物件 -----------------
def create_lvs_progress() -> Dict:
    return {
        "candidate_path": None,
        "free_prefix_len": 0,
        "free_edges": [],
        "nodes_ref": None,
        "iter": 0,
    }

# ----------------- 主演算法 -----------------
def lazy_visibility_search(
    nodes: List[Tuple[float,float]],
    O_idx: int,
    D_idx: int,
    visible_fn,
    COLLISION_PREP_M,
    land_tree,
    inject_gateways_fn,
    max_iters: int = 5000,
    progress: Optional[Dict] = None,
) -> List[int]:
    """
    懶惰可視搜尋（LVS）：
      - 以鄰近啟發建圖（不預先驗證可視性）
      - A* 找到候選 path 後，從前綴開始逐邊驗證可視性
      - 若遇 BLOCKED：標記該邊，呼叫 gateway 注入更多節點，然後重規劃
    """
    EDGE_STATE: Dict[Tuple[int,int], str] = {}   # ('FREE' / 'BLOCKED')
    adj_cache: Dict[int, List[int]] = {}

    if progress is None:
        progress = create_lvs_progress()
    progress["nodes_ref"] = nodes

    def get_neighbors(u: int) -> List[int]:
        if u not in adj_cache:
            adj_cache[u] = neighbors_of(u, nodes, D_idx, k=NEIGHBOR_K)
        return adj_cache[u]

    def a_star() -> Optional[List[int]]:
        N = len(nodes)
        INF = 1e18
        g = [INF] * N
        parent = [-1] * N
        open_heap = []

        g[O_idx] = 0.0
        h0 = heuristic(nodes[O_idx], nodes[D_idx])
        heapq.heappush(open_heap, (g[O_idx] + h0, O_idx))
        closed = set()

        while open_heap:
            _, u = heapq.heappop(open_heap)
            if u in closed:
                continue
            if u == D_idx:
                path = [u]
                while parent[u] != -1:
                    u = parent[u]
                    path.append(u)
                return list(reversed(path))
            closed.add(u)

            for v in get_neighbors(u):
                if v >= len(nodes):
                    continue
                if EDGE_STATE.get((u, v)) == "BLOCKED":
                    continue
                c = edge_cost(nodes[u], nodes[v])
                alt = g[u] + c
                if alt < g[v]:
                    g[v] = alt
                    parent[v] = u
                    f = alt + heuristic(nodes[v], nodes[D_idx])
                    heapq.heappush(open_heap, (f, v))
        return None

    it = 0
    while it < max_iters:
        it += 1
        path = a_star()
        if not path:
            raise RuntimeError("LVS: path not found (graph fully blocked).")

        progress["iter"] = it
        progress["candidate_path"] = list(path)
        progress["free_prefix_len"] = 0

        all_valid = True
        prefix_ok = 0
        for u, v in zip(path[:-1], path[1:]):
            a = nodes[u]; b = nodes[v]
            st = EDGE_STATE.get((u, v))

            # 快取命中
            if st == "FREE":
                progress["free_edges"].append((u, v))
                prefix_ok += 1
                continue
            if st == "BLOCKED":
                all_valid = False
                break

            # 現場驗證
            if visible_fn(a, b, COLLISION_PREP_M, land_tree):
                EDGE_STATE[(u, v)] = EDGE_STATE[(v, u)] = "FREE"
                progress["free_edges"].append((u, v))
                prefix_ok += 1
            else:
                EDGE_STATE[(u, v)] = EDGE_STATE[(v, u)] = "BLOCKED"

                # 注入 gateways
                new_nodes = inject_gateways_fn(a, b)
                if new_nodes:
                    seen = set((round(x, 5), round(y, 5)) for x, y in nodes)
                    appended = False
                    for q in new_nodes:
                        key = (round(q[0], 5), round(q[1], 5))
                        if key not in seen:
                            nodes.append(q)
                            seen.add(key)
                            appended = True
                    if appended:
                        adj_cache.clear()
                all_valid = False
                break

        progress["free_prefix_len"] = prefix_ok
        if all_valid:
            return path

    raise RuntimeError("LVS: exceeded max_iters without a valid path.")


# ----------------- Gateway 注入 -----------------
def make_inject_gateways_fn(
    UNION_M,
    features_index_ll: Dict[str, List[Tuple[float,float]]],
    take_each: int = 3,
    inner_ring_m=None,
    target_boundary_m=None,
):
    """
    從特徵點池（凸峰/凸點）挑選「靠近被阻邊」的點作為 gateway，
    並用 nudge_to_ring_if_inside_fast 推到外圍緩衝，避免靠岸過近。
    """
    pool = list(features_index_ll.get("convex_peaks", [])) + list(features_index_ll.get("convex", []))

    def _to_m_geom(g):
        return shp_transform(lambda x, y, z=None: to_m(x, y), g)

    def _dedup(seq, nd=5):
        seen, out = set(), []
        for (lon, lat) in seq:
            key = (round(lon, nd), round(lat, nd))
            if key in seen:
                continue
            seen.add(key)
            out.append((lon, lat))
        return out

    def f(u_ll: Tuple[float,float], v_ll: Tuple[float,float]) -> List[Tuple[float,float]]:
        if not pool:
            return []

        seg_ll = LineString(geodesic_sample(u_ll, v_ll, step_km=STEP_KM_GEODESIC))
        seg_m  = _to_m_geom(seg_ll)

        cand = []
        for (lon, lat) in pool:
            px, py = to_m(lon, lat)
            d = seg_m.distance(Point(px, py))
            cand.append((d, (lon, lat)))
        cand.sort(key=lambda t: t[0])

        chosen = [p for _, p in itertools.islice(cand, 0, take_each)]
        nudged = [nudge_to_ring_if_inside_fast(p, inner_ring_m, target_boundary_m)[0] for p in chosen]
        return _dedup(nudged, nd=5)

    return f
