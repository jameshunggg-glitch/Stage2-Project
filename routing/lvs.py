# routing/lvs.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable
import heapq
from shapely.geometry import LineString

from .geodesy import gc_distance_km
from .config import NEIGHBOR_K, STEP_KM_GEODESIC
from .visibility import Visibility
from .geodesy import geodesic_sample

def edge_cost(a,b) -> float:
    return gc_distance_km(a,b)

def heuristic(p, D):
    return gc_distance_km(p, D)

def lazy_visibility_search(
    nodes: List[Tuple[float,float]],
    O_idx: int,
    D_idx: int,
    neighbors_fn: Callable[[int], List[int]],
    visibility: Visibility,
    inject_gateways_fn: Callable[[Tuple[float,float], Tuple[float,float]], List[Tuple[float,float]]],
    max_iters: int = 5000,
    progress: Optional[Dict]=None,
) -> List[int]:
    EDGE_STATE: Dict[Tuple[int,int], str] = {}
    adj_cache: Dict[int, List[int]] = {}

    def get_neighbors(u: int) -> List[int]:
        if u not in adj_cache:
            adj_cache[u] = neighbors_fn(u)
        return adj_cache[u]

    def a_star() -> Optional[List[int]]:
        N_local = len(nodes)
        open_heap = []
        INF = 1e18
        g = [INF] * N_local
        parent = [-1] * N_local
        g[O_idx] = 0.0
        heapq.heappush(open_heap, (g[O_idx] + heuristic(nodes[O_idx], nodes[D_idx]), O_idx))
        closed = set()
        while open_heap:
            _, u = heapq.heappop(open_heap)
            if u in closed: continue
            if u == D_idx:
                path=[u]
                while parent[u] != -1:
                    u = parent[u]; path.append(u)
                return list(reversed(path))
            closed.add(u)
            for v in get_neighbors(u):
                if v >= len(nodes): continue
                if EDGE_STATE.get((u,v)) == 'BLOCKED': continue
                c = edge_cost(nodes[u], nodes[v])
                alt = g[u] + c
                if alt < g[v]:
                    g[v] = alt
                    parent[v] = u
                    f = alt + heuristic(nodes[v], nodes[D_idx])
                    heapq.heappush(open_heap, (f, v))
        return None

    it=0
    while it < max_iters:
        it += 1
        path = a_star()
        if not path:
            raise RuntimeError("LVS: path not found.")

        all_valid = True
        for u, v in zip(path[:-1], path[1:]):
            st = EDGE_STATE.get((u, v))
            a = nodes[u]; b = nodes[v]
            if st == 'FREE':
                continue
            if st == 'BLOCKED':
                all_valid = False; break
            # 檢查可視
            if visibility.is_visible(a, b):
                EDGE_STATE[(u, v)] = EDGE_STATE[(v, u)] = 'FREE'
            else:
                EDGE_STATE[(u, v)] = EDGE_STATE[(v, u)] = 'BLOCKED'
                # 注入 gateway
                new_nodes = inject_gateways_fn(a, b)
                if new_nodes:
                    existing = set((round(lon,5), round(lat,5)) for (lon,lat) in nodes)
                    filtered=[]
                    for q in new_nodes:
                        key=(round(q[0],5), round(q[1],5))
                        if key not in existing:
                            filtered.append(q); existing.add(key)
                    if filtered:
                        nodes.extend(filtered)
                        adj_cache.clear()
                all_valid = False
                break

        if all_valid:
            return path

    raise RuntimeError("LVS: exceeded max iters")

# --- Gateway injection（沿你原本的寫法） ---
def make_inject_gateways_fn(
    UNION_M,
    features_index_ll,
    take_each=3,
    inner_ring_m=None,
    target_boundary_m=None,
):
    # 使用 convex_peaks + convex 作為候選 gateway
    pool = list(features_index_ll.get("convex_peaks", [])) + list(features_index_ll.get("convex", []))

    from .geodesy import to_m, geodesic_sample
    import shapely
    from shapely.geometry import Point, LineString

    def nudge_to_ring_if_inside_fast(pt_ll, inner_ring_m, target_boundary_m):
        px, py = to_m(pt_ll[0], pt_ll[1])
        p_m = Point(px, py)
        if inner_ring_m.contains(p_m):
            q_m = shapely.ops.nearest_points(p_m, target_boundary_m)[1]
            from .geodesy import to_ll
            q_llx, q_lly = to_ll(q_m.x, q_m.y)
            return (q_llx, q_lly), True
        return pt_ll, False

    def f(u_ll, v_ll):
        if not pool: return []
        seg_ll = LineString(geodesic_sample(u_ll, v_ll, step_km=STEP_KM_GEODESIC))
        seg_m  = shapely.ops.transform(lambda x,y: to_m(x,y), seg_ll)
        # 依距離簇取最近的幾個特徵點
        cand=[]
        for (lon,lat) in pool:
            px,py = to_m(lon,lat)
            p = Point(px,py)
            cand.append((seg_m.distance(p), (lon,lat)))
        cand.sort(key=lambda t: t[0])
        new=[]
        for _,pt in cand[:take_each]:
            q,_ = nudge_to_ring_if_inside_fast(pt, inner_ring_m, target_boundary_m)
            new.append(q)
        # 去重
        out=[]; seen=set()
        for lon,lat in new:
            key=(round(lon,5), round(lat,5))
            if key in seen: continue
            seen.add(key); out.append((lon,lat))
        return out

    return f
