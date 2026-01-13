"""routing_map.snap

Endpoint snapping / injection helpers for point-to-point routing.

This file is intentionally "dev-stage":
- Implements a small, usable baseline (sea-first snap to nearest sea nodes).
- Leaves hooks for later improvements (component-aware, nudge-out-of-land, etc.).

Key idea:
- You build a base graph from AOI outputs (routing_graph.build_base_graph).
- Then you "inject" start/end points by adding temporary edges from the point to
  selected nearby network nodes.

We will iterate the flow later; for now we provide:
- dateline-safe haversine_km
- candidate generation using KDTree from out['sea_kdt'] + out['S_nodes']
- optional filtering by out['sea_ok_set'] (largest component, degree filter)
- injection helper that adds edges into an existing NetworkX graph

Expected `out` keys (from build_aoi):
- S_nodes: pd.DataFrame with lon/lat columns
- sea_kdt: sklearn.neighbors.KDTree built on (x_m,y_m)
- proj: AOIProjector with to_m.transform(lon,lat) for metric conversion
- sea_ok_set: set[int] of allowed S_nodes indices (optional)
- layers["COLLISION_M"] prepared geometry (optional, only for future land checks)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math

import networkx as nx

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


LonLat = Tuple[float, float]

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


def _is_df(x: Any) -> bool:
    return pd is not None and isinstance(x, pd.DataFrame)


def _project_ll_to_xy(out: Dict[str, Any], p_ll: LonLat) -> Tuple[float, float]:
    """Project lon/lat to metric x/y using out['proj']."""
    proj = out.get("proj")
    if proj is None or not hasattr(proj, "to_m"):
        raise ValueError("out['proj'] with .to_m.transform(lon,lat) is required for KDTree snapping")
    lon, lat = map(float, p_ll)
    # AOIProjector: proj.to_m is pyproj Transformer
    return proj.to_m.transform(lon, lat)


@dataclass
class SnapCandidate:
    sea_idx: int
    node_ll: LonLat
    dist_km: float
    component: Optional[int] = None
    ok: bool = True  # passes sea_ok_set filter


@dataclass
class SnapResult:
    p_input: LonLat
    p_used: LonLat
    candidates: List[SnapCandidate]
    reason: str = "ok"


def snap_to_sea_candidates(
    out: Dict[str, Any],
    p_ll: LonLat,
    *,
    k_near: int = 30,
    r_max_km: float = 150.0,
    prefer_ok_set: bool = True,
    allow_fallback_non_ok: bool = True,
) -> SnapResult:
    """Sea-first: get nearby sea-node candidates for an arbitrary point.

    This is a baseline implementation we can refine later.

    Parameters
    ----------
    k_near:
        KDTree kNN size to query.
    r_max_km:
        Maximum allowed distance from p_ll to candidate sea node.
    prefer_ok_set:
        If True and out['sea_ok_set'] exists, prefer candidates inside it.
    allow_fallback_non_ok:
        If True, when no ok_set candidate exists within r_max_km, allow non-ok candidates.

    Returns
    -------
    SnapResult with candidate list ordered by preference then distance.
    """
    if not _is_df(out.get("S_nodes")):
        return SnapResult(p_input=p_ll, p_used=p_ll, candidates=[], reason="missing_S_nodes")
    if out.get("sea_kdt") is None:
        return SnapResult(p_input=p_ll, p_used=p_ll, candidates=[], reason="missing_sea_kdt")

    S_nodes = out["S_nodes"]
    kdt = out["sea_kdt"]
    sea_ok_set = out.get("sea_ok_set")
    has_ok = isinstance(sea_ok_set, set) and len(sea_ok_set) > 0

    # Query KDTree in metric space
    x, y = _project_ll_to_xy(out, p_ll)
    # sklearn KDTree expects 2D array
    dists_m, idxs = kdt.query([[x, y]], k=min(int(k_near), len(S_nodes)))
    idxs = idxs[0]

    # Build candidates
    cands: List[SnapCandidate] = []
    for i in idxs:
        try:
            ii = int(i)
            node = S_nodes.iloc[ii]
            node_ll = (float(node["lon"]), float(node["lat"]))
            dk = haversine_km(p_ll, node_ll)
            if dk > float(r_max_km):
                continue
            comp = int(node["component"]) if "component" in S_nodes.columns and not pd.isna(node["component"]) else None
            ok = True
            if prefer_ok_set and has_ok:
                ok = (ii in sea_ok_set)
            cands.append(SnapCandidate(sea_idx=ii, node_ll=node_ll, dist_km=float(dk), component=comp, ok=ok))
        except Exception:
            continue

    if not cands:
        return SnapResult(p_input=p_ll, p_used=p_ll, candidates=[], reason="no_candidate_within_rmax")

    # If prefer_ok_set and we found ok candidates, keep ok first.
    if prefer_ok_set and has_ok:
        ok_cands = [c for c in cands if c.ok]
        if ok_cands:
            cands = ok_cands + sorted([c for c in cands if not c.ok], key=lambda c: c.dist_km)
        else:
            # no ok candidates; allow fallback?
            if not allow_fallback_non_ok:
                return SnapResult(p_input=p_ll, p_used=p_ll, candidates=[], reason="no_ok_candidate")
            cands = sorted(cands, key=lambda c: c.dist_km)
    else:
        cands = sorted(cands, key=lambda c: c.dist_km)

    return SnapResult(p_input=p_ll, p_used=p_ll, candidates=cands, reason="ok")


def inject_point_edges(
    G: nx.Graph,
    p_ll: LonLat,
    candidates: Sequence[SnapCandidate],
    *,
    k_inject: int = 4,
    etype: str = "inject",
    weight_attr: str = "weight",
) -> int:
    """Inject edges from an arbitrary point node `p_ll` to selected candidate nodes.

    Adds the point node if not present.

    Returns
    -------
    number of injected edges.
    """
    if k_inject <= 0:
        return 0

    # Ensure point node exists
    if p_ll not in G:
        G.add_node(p_ll)

    added = 0
    for c in list(candidates)[: int(k_inject)]:
        u = p_ll
        v = c.node_ll
        w = float(c.dist_km)
        G.add_edge(u, v, **{weight_attr: w}, length_km=w, etype=etype)
        added += 1

    return added


__all__ = [
    "LonLat",
    "SnapCandidate",
    "SnapResult",
    "haversine_km",
    "snap_to_sea_candidates",
    "inject_point_edges",
]
