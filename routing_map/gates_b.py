from __future__ import annotations
import numpy as np
import pandas as pd
from shapely.prepared import PreparedGeometry
from sklearn.neighbors import KDTree

from .visibility import segment_clear

def gate_to_sea_candidates(
    gate_xy: np.ndarray,
    *,
    S_nodes: pd.DataFrame,
    kdt: KDTree,
    top_n: int,
    r_max_km: float,
) -> list[int]:
    """Return candidate sea node indices near gate."""
    r = r_max_km * 1000.0
    dist, idx = kdt.query([gate_xy], k=top_n, return_distance=True)
    dist = dist[0]; idx = idx[0]
    out = []
    for d, i in zip(dist, idx):
        if float(d) <= r:
            out.append(int(i))
    return out

def build_gateB_connectors(
    Gate_all: pd.DataFrame,
    S_nodes: pd.DataFrame,
    *,
    sea_ok_set: set[int],
    kdt: KDTree,
    collision_prep: PreparedGeometry,
    top_n: int,
    r_max_km: float,
    k_connect: int,
) -> pd.DataFrame:
    """For each gate, pick up to k_connect sea nodes with collision-free segments."""
    rows = []
    for _, g in Gate_all.iterrows():
        gx, gy = float(g["x_m"]), float(g["y_m"])
        cand = gate_to_sea_candidates(np.array([gx,gy], dtype=float), S_nodes=S_nodes, kdt=kdt, top_n=top_n, r_max_km=r_max_km)
        cand = [i for i in cand if i in sea_ok_set]
        scored = []
        for i in cand:
            s = S_nodes.iloc[i]
            sx, sy = float(s["x_m"]), float(s["y_m"])
            ok = segment_clear((gx,gy), (sx,sy), collision_prep=collision_prep)
            if not ok:
                continue
            d_km = float(np.hypot(sx-gx, sy-gy) / 1000.0)
            scored.append((d_km, i))
        scored.sort(key=lambda t: t[0])
        for rank, (d_km, i) in enumerate(scored[:k_connect]):
            s = S_nodes.iloc[i]
            rows.append({
                "g_id": int(g["g_id"]),
                "sea_idx": int(i),
                "sea_node_id": str(s["node_id"]),
                "dist_km": float(d_km),
                "rank": int(rank),
            })
    return pd.DataFrame(rows)
