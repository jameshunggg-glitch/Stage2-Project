# routing/neighbors.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import itertools
from .geodesy import gc_distance_km

def knn_neighbors(u_idx: int, nodes: List[Tuple[float,float]], D_idx: int, k: int) -> List[int]:
    u = nodes[u_idx]; D = nodes[D_idx]
    scored=[]
    for v_idx in range(len(nodes)):
        if v_idx == u_idx: continue
        du = gc_distance_km(u, nodes[v_idx])
        dD = gc_distance_km(nodes[v_idx], D)
        scored.append((du + 0.5*dD, v_idx))
    scored.sort(key=lambda t: t[0])
    out = [v for _,v in itertools.islice(scored, 0, k)]
    if D_idx not in out: out.append(D_idx)
    return out

class MixedNeighbors:
    """KNN（LVS） +（未來）scgraph 鄰接；目前只用 KNN。"""
    def __init__(self, nodes: List[Tuple[float,float]], D_idx: int, k: int, sc_adj: Optional[Dict[int, List[int]]]=None):
        self.nodes = nodes
        self.D_idx = D_idx
        self.k = k
        self.sc_adj = sc_adj or {}

    def neighbors_of(self, u_idx: int) -> List[int]:
        n_lvs = knn_neighbors(u_idx, self.nodes, self.D_idx, self.k)
        n_sc  = self.sc_adj.get(u_idx, [])
        merged = list(dict.fromkeys([*n_sc, *n_lvs]))
        if self.D_idx not in merged: merged.append(self.D_idx)
        return merged
