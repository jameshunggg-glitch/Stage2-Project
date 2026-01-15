# routing_map/snap.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.prepared import prep
from shapely.ops import nearest_points

from .routing_graph import haversine_km

LonLat = Tuple[float, float]


# -------------------------
# Small utilities
# -------------------------
def normalize_lonlat(p: LonLat) -> LonLat:
    """Normalize lon into [-180,180] range."""
    lon, lat = float(p[0]), float(p[1])
    lon = (lon + 180.0) % 360.0 - 180.0
    return (lon, lat)


def _get_proj(out: Dict[str, Any]):
    proj = out.get("proj", None)
    if proj is None:
        raise ValueError("out['proj'] is required for snap (AOIProjector).")
    return proj


def _get_collision_geom_m(out: Dict[str, Any]):
    layers = out.get("layers", None)
    if not isinstance(layers, dict) or "COLLISION_M" not in layers:
        return None
    return layers["COLLISION_M"]


def _point_ll_to_m(out: Dict[str, Any], p_ll: LonLat) -> Tuple[float, float]:
    proj = _get_proj(out)
    x_m, y_m = proj.to_m.transform(float(p_ll[0]), float(p_ll[1]))
    return float(x_m), float(y_m)


def _point_m_to_ll(out: Dict[str, Any], p_m: Tuple[float, float]) -> LonLat:
    proj = _get_proj(out)
    lon, lat = proj.to_ll.transform(float(p_m[0]), float(p_m[1]))
    return (float(lon), float(lat))


def _is_in_collision(out: Dict[str, Any], p_ll: LonLat, collision_prep=None) -> bool:
    """Check if lon/lat point falls in collision geometry (meters)."""
    geom_m = _get_collision_geom_m(out)
    if geom_m is None:
        return False
    if collision_prep is None:
        collision_prep = prep(geom_m)
    x, y = _point_ll_to_m(out, p_ll)
    pt = Point(x, y)
    return bool(collision_prep.contains(pt) or collision_prep.intersects(pt))


def _guess_nudge_buffer_m(out: Dict[str, Any], default_km: float = 0.5) -> float:
    """Use cfg.land.collision_safety_km if present, else default."""
    cfg = out.get("cfg", None)
    try:
        km = float(getattr(getattr(cfg, "land"), "collision_safety_km"))
        if km > 0:
            return km * 1000.0
    except Exception:
        pass
    return float(default_km) * 1000.0


def _kdt_query_indices(kdt, x: float, y: float, k: int):
    """
    Return list of indices for k nearest points.
    Compatible with sklearn.neighbors.KDTree and scipy.spatial.cKDTree.
    """
    k = max(1, int(k))
    try:
        # sklearn KDTree
        dists, idxs = kdt.query(np.array([[x, y]], dtype=float), k=k)
        return idxs[0].tolist()
    except Exception:
        # scipy cKDTree
        dists, idxs = kdt.query([x, y], k=k)
        idxs = np.atleast_1d(idxs)
        return [int(i) for i in idxs.tolist()]


# -------------------------
# Coastal KDTree helpers
# -------------------------
def _get_coast_nodes_df(out):
    C_nodes = out.get("C_nodes", None)
    if C_nodes is None:
        return None
    if not hasattr(C_nodes, "columns"):
        return None
    need = {"lon", "lat"}
    if not need.issubset(set(C_nodes.columns)):
        return None
    return C_nodes


def _get_or_build_coast_kdt(out):
    """Build & cache KDTree for coastal nodes in metric xy."""
    if out.get("coast_kdt", None) is not None and out.get("coast_xy_m", None) is not None:
        return out["coast_kdt"], out["coast_xy_m"]

    C_nodes = _get_coast_nodes_df(out)
    if C_nodes is None or len(C_nodes) == 0:
        return None, None

    xs = C_nodes["lon"].astype(float).to_numpy()
    ys = C_nodes["lat"].astype(float).to_numpy()
    proj = _get_proj(out)
    xm, ym = proj.to_m.transform(xs, ys)
    xy = np.column_stack([np.asarray(xm, dtype=float), np.asarray(ym, dtype=float)])

    # KDTree (prefer sklearn, fallback scipy)
    try:
        from sklearn.neighbors import KDTree  # type: ignore
        kdt = KDTree(xy, leaf_size=40)
    except Exception:
        from scipy.spatial import cKDTree  # type: ignore
        kdt = cKDTree(xy)

    out["coast_xy_m"] = xy
    out["coast_kdt"] = kdt
    return kdt, xy


def nudge_to_nearest_coastal_node(
    out: Dict[str, Any],
    p_ll: LonLat,
    *,
    k_near: int = 80,
    r_max_km: float = 150.0,
    collision_prep=None,
) -> Tuple[LonLat, bool, Dict[str, Any]]:
    """
    If p_ll is in collision, move it to nearest coastal node (C_nodes) that is outside collision.
    """
    p_ll = normalize_lonlat(p_ll)

    geom_m = _get_collision_geom_m(out)
    if geom_m is not None and collision_prep is None:
        collision_prep = prep(geom_m)

    if geom_m is None or (not _is_in_collision(out, p_ll, collision_prep=collision_prep)):
        return p_ll, False, {"inside": False}

    kdt, _xy = _get_or_build_coast_kdt(out)
    C_nodes = _get_coast_nodes_df(out)
    if kdt is None or C_nodes is None or len(C_nodes) == 0:
        return p_ll, False, {"inside": True, "fail": "missing_C_nodes_or_coast_kdt"}

    x, y = _point_ll_to_m(out, p_ll)
    idxs = _kdt_query_indices(kdt, x, y, k=min(int(k_near), len(C_nodes)))

    best = None
    for i in idxs:
        row = C_nodes.iloc[int(i)]
        cand = (float(row["lon"]), float(row["lat"]))
        d_km = float(haversine_km(p_ll, cand))
        if d_km > float(r_max_km):
            continue
        if _is_in_collision(out, cand, collision_prep=collision_prep):
            continue
        best = (cand, d_km, int(i))
        break

    if best is None:
        return p_ll, False, {
            "inside": True,
            "fail": "no_valid_coastal_node_within_radius",
            "k": len(idxs),
            "r_max_km": float(r_max_km),
        }

    cand, d_km, idx = best
    return normalize_lonlat(cand), True, {"inside": True, "picked_idx": idx, "dist_km": d_km}


# -------------------------
# GateB KDTree + rank0 info helpers
# -------------------------
def _get_or_build_gateb_kdt(out):
    """
    Build & cache:
    - gateb_df: Gate_B_kept_gates dataframe
    - gateb_kdt: KDTree over gateb xy_m
    - gate_uid -> (lon,lat,g_id)
    - gate_uid -> reachable sea components set (via connectors->sea_idx->S_nodes.component)
    """
    if (
        out.get("gateb_kdt") is not None
        and out.get("gateb_uid_to_ll") is not None
        and out.get("gateb_uid_to_comps") is not None
        and out.get("gateb_uid_to_gid") is not None
    ):
        return out["gateb_kdt"], out["gateb_uid_to_ll"], out["gateb_uid_to_comps"], out["gateb_uid_to_gid"]

    gate_df = out.get("Gate_B_kept_gates", None)
    conn_df = out.get("gateB_connectors", None)
    S_nodes = out.get("S_nodes", None)

    if gate_df is None or conn_df is None or S_nodes is None:
        return None, None, None, None
    if not (hasattr(gate_df, "columns") and hasattr(conn_df, "columns") and hasattr(S_nodes, "columns")):
        return None, None, None, None

    need_gate = {"gate_uid", "lon", "lat", "g_id"}
    need_conn = {"gate_uid", "sea_idx"}
    if not need_gate.issubset(set(gate_df.columns)) or not need_conn.issubset(set(conn_df.columns)):
        return None, None, None, None
    if "component" not in S_nodes.columns:
        return None, None, None, None

    proj = _get_proj(out)

    # gate_uid -> ll, g_id
    uid_to_ll: Dict[int, LonLat] = {}
    uid_to_gid: Dict[int, int] = {}
    for _, r in gate_df.iterrows():
        uid = int(r["gate_uid"])
        uid_to_ll[uid] = (float(r["lon"]), float(r["lat"]))
        uid_to_gid[uid] = int(r["g_id"])

    # build KDTree on gate points (metric)
    uids_list = list(uid_to_ll.keys())
    xs = np.array([uid_to_ll[uid][0] for uid in uids_list], dtype=float)
    ys = np.array([uid_to_ll[uid][1] for uid in uids_list], dtype=float)
    xm, ym = proj.to_m.transform(xs, ys)
    xy = np.column_stack([np.asarray(xm, dtype=float), np.asarray(ym, dtype=float)])

    uids = np.array(uids_list, dtype=int)  # index -> gate_uid

    try:
        from sklearn.neighbors import KDTree  # type: ignore
        kdt = KDTree(xy, leaf_size=40)
    except Exception:
        from scipy.spatial import cKDTree  # type: ignore
        kdt = cKDTree(xy)

    # gate_uid -> reachable components
    uid_to_comps: Dict[int, set] = {}
    for uid, g in conn_df.groupby("gate_uid"):
        uid = int(uid)
        sea_idxs = g["sea_idx"].astype(int).tolist()
        comps = set(int(S_nodes.iloc[i]["component"]) for i in sea_idxs if 0 <= i < len(S_nodes))
        if len(comps) > 0:
            uid_to_comps[uid] = comps

    # cache
    out["gateb_kdt"] = (kdt, xy, uids)  # store uids array too
    out["gateb_uid_to_ll"] = uid_to_ll
    out["gateb_uid_to_gid"] = uid_to_gid
    out["gateb_uid_to_comps"] = uid_to_comps

    return out["gateb_kdt"], uid_to_ll, uid_to_comps, uid_to_gid


def _get_gateb_rank0_map(out: Dict[str, Any]) -> Dict[int, Tuple[int, Optional[int]]]:
    """
    Build & cache gate_uid -> (rank0_sea_idx, rank0_component).
    rank0 = smallest 'rank' if present; else smallest dist_km.
    """
    if out.get("gateb_uid_to_rank0", None) is not None:
        return out["gateb_uid_to_rank0"]

    conn_df = out.get("gateB_connectors", None)
    S_nodes = out.get("S_nodes", None)
    if conn_df is None or S_nodes is None:
        out["gateb_uid_to_rank0"] = {}
        return out["gateb_uid_to_rank0"]

    if not (hasattr(conn_df, "columns") and hasattr(S_nodes, "columns")):
        out["gateb_uid_to_rank0"] = {}
        return out["gateb_uid_to_rank0"]

    if "gate_uid" not in conn_df.columns or "sea_idx" not in conn_df.columns:
        out["gateb_uid_to_rank0"] = {}
        return out["gateb_uid_to_rank0"]

    has_rank = "rank" in conn_df.columns
    has_dist = "dist_km" in conn_df.columns
    has_comp = "component" in S_nodes.columns

    m: Dict[int, Tuple[int, Optional[int]]] = {}
    for uid, g in conn_df.groupby("gate_uid"):
        uid = int(uid)
        gg = g.copy()
        if has_rank:
            gg = gg.sort_values(["rank", "dist_km"] if has_dist else ["rank"], ascending=True)
        elif has_dist:
            gg = gg.sort_values(["dist_km"], ascending=True)

        if len(gg) == 0:
            continue

        sea_idx = int(gg.iloc[0]["sea_idx"])
        comp = None
        if has_comp and 0 <= sea_idx < len(S_nodes):
            try:
                comp = int(S_nodes.iloc[sea_idx]["component"])
            except Exception:
                comp = None
        m[uid] = (sea_idx, comp)

    out["gateb_uid_to_rank0"] = m
    return m


def gateb_candidates_from_coastal(
    out: Dict[str, Any],
    coastal_ll: LonLat,
    *,
    k_near_gateb: int = 30,
    r_max_km_gateb: float = 200.0,
    require_connectors: bool = True,
) -> List[Dict[str, Any]]:
    """
    Given a coastal point (lon,lat), find nearby GateB that can connect to sea components.

    Returns list of dicts:
      {
        gate_uid, g_id, gate_ll, dist_km,
        rank0_sea_idx, component, comps:set[int]
      }
    """
    res: List[Dict[str, Any]] = []

    pack, uid_to_ll, uid_to_comps, uid_to_gid = _get_or_build_gateb_kdt(out)
    if pack is None:
        return res

    (kdt, _xy, uids) = pack
    x, y = _point_ll_to_m(out, coastal_ll)

    idxs = _kdt_query_indices(kdt, x, y, k=min(int(k_near_gateb), len(uids)))

    uid_to_rank0 = _get_gateb_rank0_map(out)

    for j in idxs:
        uid = int(uids[int(j)])
        gate_ll = uid_to_ll.get(uid)
        if gate_ll is None:
            continue
        d_km = float(haversine_km(coastal_ll, gate_ll))
        if d_km > float(r_max_km_gateb):
            continue

        comps = uid_to_comps.get(uid, set())
        if require_connectors and len(comps) == 0:
            continue

        rank0 = uid_to_rank0.get(uid, None)
        rank0_sea_idx = int(rank0[0]) if rank0 is not None else None
        rank0_comp = rank0[1] if rank0 is not None else None

        res.append(
            {
                "gate_uid": uid,
                "g_id": int(uid_to_gid.get(uid, -1)),
                "gate_ll": gate_ll,
                "dist_km": d_km,
                "rank0_sea_idx": rank0_sea_idx,
                "component": rank0_comp,
                "comps": comps,
            }
        )

    res.sort(key=lambda r: float(r["dist_km"]))
    return res


# -------------------------
# Dataclasses
# -------------------------
@dataclass
class SnapCandidate:
    node_idx: int
    node_ll: LonLat
    dist_km: float
    component: Optional[int]
    ok: bool


@dataclass
class SnapResult:
    p_input_ll: LonLat
    p_used_ll: LonLat
    was_nudged: bool
    in_collision_input: bool
    candidates: List[SnapCandidate]
    reason: str
    debug: Dict[str, Any]


@dataclass
class SnapPairResult:
    start: SnapResult
    end: SnapResult
    chosen_common_component: Optional[int]
    largest_component: Optional[int]
    start_pick: List[SnapCandidate]
    end_pick: List[SnapCandidate]
    reason: str
    debug: Dict[str, Any]


# -------------------------
# Nudge: push point out of collision (boundary-based)
# -------------------------
def nudge_out_of_collision(
    out: Dict[str, Any],
    p_ll: LonLat,
    *,
    buffer_m: Optional[float] = None,
    max_step_m: float = 50_000.0,
    collision_prep=None,
) -> Tuple[LonLat, bool, Dict[str, Any]]:
    """
    If p_ll is inside collision geom, push it to outside:
    - Find nearest point on collision boundary in meters.
    - Move from p towards boundary and go beyond by (distance_to_boundary + buffer).
    Returns (p_ll_new, moved?, debug)
    """
    p_ll = normalize_lonlat(p_ll)
    geom_m = _get_collision_geom_m(out)
    if geom_m is None:
        return p_ll, False, {"note": "no_collision_geom"}

    if collision_prep is None:
        collision_prep = prep(geom_m)

    x, y = _point_ll_to_m(out, p_ll)
    pt = Point(x, y)

    inside = bool(collision_prep.contains(pt) or collision_prep.intersects(pt))
    if not inside:
        return p_ll, False, {"inside": False}

    if buffer_m is None:
        buffer_m = _guess_nudge_buffer_m(out, default_km=0.5)

    # distance from point to boundary (meters)
    try:
        boundary = geom_m.boundary
        d_to_boundary = float(boundary.distance(pt))
    except Exception:
        boundary = geom_m
        d_to_boundary = float(boundary.distance(pt))

    # find nearest boundary point
    try:
        _, nb = nearest_points(pt, boundary)
        bx, by = float(nb.x), float(nb.y)
    except Exception:
        bx, by = x + 1.0, y  # fallback

    vx, vy = (bx - x), (by - y)
    norm = float(np.hypot(vx, vy))

    dbg = {
        "inside": True,
        "buffer_m": float(buffer_m),
        "d_to_boundary_m": float(d_to_boundary),
        "nearest_boundary_xy": (bx, by),
        "pt_xy": (x, y),
    }

    if norm < 1e-6:
        # Degenerate: just step east by buffer
        new_xy = (x + float(buffer_m), y)
        new_ll = _point_m_to_ll(out, new_xy)
        step = float(buffer_m)
        while step < float(max_step_m):
            if not _is_in_collision(out, new_ll, collision_prep=collision_prep):
                return normalize_lonlat(new_ll), True, {**dbg, "degenerate": True, "step_m": step}
            step *= 1.5
            new_ll = _point_m_to_ll(out, (x + step, y))
        return p_ll, False, {**dbg, "degenerate": True, "fail": "cannot_escape"}

    ux, uy = vx / norm, vy / norm
    step_m = min(d_to_boundary + float(buffer_m), float(max_step_m))

    new_xy = (x + ux * step_m, y + uy * step_m)
    new_ll = normalize_lonlat(_point_m_to_ll(out, new_xy))

    step = step_m
    while step < float(max_step_m) and _is_in_collision(out, new_ll, collision_prep=collision_prep):
        step *= 1.5
        new_xy = (x + ux * step, y + uy * step)
        new_ll = normalize_lonlat(_point_m_to_ll(out, new_xy))

    ok = not _is_in_collision(out, new_ll, collision_prep=collision_prep)
    dbg.update({"step_m": step, "escaped": ok})

    return (new_ll if ok else p_ll), bool(ok), dbg


# -------------------------
# Snap decision tree (Improved)
# -------------------------
def snap_to_sea_candidates(
    out: Dict[str, Any],
    p_ll: LonLat,
    *,
    k_near: int = 30,
    r_max_km: float = 150.0,
    prefer_ok_set: bool = True,
    allow_fallback_non_ok: bool = True,
    allow_radius_fallback: bool = True,
    r_fallback_km: Optional[float] = None,
    do_nudge: bool = True,
    # coastal nudge params
    k_near_coast: int = 80,
    r_max_km_coast: Optional[float] = None,
    # gateB params (used only in collision mode)
    k_near_gateb: int = 30,
    r_max_km_gateb: float = 200.0,
    gateb_debug_topn: int = 10,
) -> SnapResult:
    """
    改良版決策樹：
    - 若點不在 collision：原本 sea-first（找 sea candidates）
    - 若點在 collision 且 do_nudge=True：
        1) 先 nudge 到 coastal node（合法）
        2) 從 coastal 找 GateB candidates（必須有 connectors）
        3) 回傳 candidates=GateB（含 g_id/gate_uid、rank0 sea_idx、component）
        4) 若 GateB candidates 找不到 -> fallback 回 sea-first（從 coastal 點去找 sea nodes）
    """
    p_ll0 = normalize_lonlat(p_ll)
    dbg: Dict[str, Any] = {}

    # Needed objects for sea-first
    S_nodes = out.get("S_nodes", None)
    sea_kdt = out.get("sea_kdt", None)
    sea_ok_set = out.get("sea_ok_set", None)

    if S_nodes is None or sea_kdt is None:
        return SnapResult(
            p_input_ll=p_ll0,
            p_used_ll=p_ll0,
            was_nudged=False,
            in_collision_input=False,
            candidates=[],
            reason="missing_S_nodes_or_sea_kdt",
            debug={"has_S_nodes": S_nodes is not None, "has_kdt": sea_kdt is not None},
        )

    # collision check
    collision_prep = None
    geom_m = _get_collision_geom_m(out)
    if geom_m is not None:
        collision_prep = prep(geom_m)

    in_collision = _is_in_collision(out, p_ll0, collision_prep=collision_prep) if geom_m is not None else False

    # -------------------------
    # CASE 1) Not in collision -> normal sea-first
    # -------------------------
    if (not in_collision) or (not do_nudge) or (geom_m is None):
        dbg["mode"] = "sea_first"
        return _sea_first_candidates(
            out,
            p_input_ll=p_ll0,
            p_used_ll=p_ll0,
            was_nudged=False,
            in_collision_input=in_collision,
            S_nodes=S_nodes,
            sea_kdt=sea_kdt,
            sea_ok_set=sea_ok_set,
            k_near=k_near,
            r_max_km=r_max_km,
            prefer_ok_set=prefer_ok_set,
            allow_fallback_non_ok=allow_fallback_non_ok,
            allow_radius_fallback=allow_radius_fallback,
            r_fallback_km=r_fallback_km,
            extra_debug=dbg,
        )

    # -------------------------
    # CASE 2) In collision -> coast -> GateB -> sea
    # -------------------------
    dbg["mode"] = "coast_gateb"

    # 2.1 nudge to coastal
    p_coast, ok_coast, dbg_coast = nudge_to_nearest_coastal_node(
        out,
        p_ll0,
        k_near=int(k_near_coast),
        r_max_km=float(r_max_km if r_max_km_coast is None else r_max_km_coast),
        collision_prep=collision_prep,
    )

    dbg["chosen_coastal"] = {
        "ok": bool(ok_coast),
        "p_coast_ll": (float(p_coast[0]), float(p_coast[1])),
        **(dbg_coast if isinstance(dbg_coast, dict) else {"dbg": str(dbg_coast)}),
    }

    # If coastal nudge fails, fallback to boundary nudge then sea-first
    if not ok_coast:
        dbg["fallback_reason"] = "coastal_nudge_failed"
        p_b, ok_b, dbg_b = nudge_out_of_collision(out, p_ll0, collision_prep=collision_prep)
        dbg["boundary_fallback"] = {"ok": bool(ok_b), **(dbg_b if isinstance(dbg_b, dict) else {"dbg": str(dbg_b)})}

        return _sea_first_candidates(
            out,
            p_input_ll=p_ll0,
            p_used_ll=p_b,
            was_nudged=bool(ok_b),
            in_collision_input=True,
            S_nodes=S_nodes,
            sea_kdt=sea_kdt,
            sea_ok_set=sea_ok_set,
            k_near=k_near,
            r_max_km=r_max_km,
            prefer_ok_set=prefer_ok_set,
            allow_fallback_non_ok=allow_fallback_non_ok,
            allow_radius_fallback=allow_radius_fallback,
            r_fallback_km=r_fallback_km,
            extra_debug=dbg,
        )

    # 2.2 gateB candidates from coastal
    gateb_rows = gateb_candidates_from_coastal(
        out,
        p_coast,
        k_near_gateb=int(k_near_gateb),
        r_max_km_gateb=float(r_max_km_gateb),
        require_connectors=True,
    )

    # debug gateB list
    topn = max(0, int(gateb_debug_topn))
    dbg["chosen_gateB_list"] = [
        {
            "gate_uid": int(r.get("gate_uid", -1)),
            "g_id": int(r.get("g_id", -1)),
            "dist_km": float(r.get("dist_km", np.nan)),
            "rank0_sea_idx": (None if r.get("rank0_sea_idx", None) is None else int(r["rank0_sea_idx"])),
            "component": (None if r.get("component", None) is None else int(r["component"])),
        }
        for r in gateb_rows[:topn]
    ]

    # 2.3 If gateB exists: return GateB candidates (NOT sea nodes)
    if len(gateb_rows) > 0:
        cands_gateb: List[SnapCandidate] = []
        for r in gateb_rows:
            uid = int(r["gate_uid"])
            gate_ll = r["gate_ll"]
            d_km = float(r["dist_km"])
            comp = r.get("component", None)
            comp_int = int(comp) if comp is not None else None

            # GateB candidate: node_idx use gate_uid (stable)
            cands_gateb.append(
                SnapCandidate(
                    node_idx=uid,
                    node_ll=(float(gate_ll[0]), float(gate_ll[1])),
                    dist_km=d_km,
                    component=comp_int,
                    ok=True,
                )
            )

        cands_gateb.sort(key=lambda c: c.dist_km)

        return SnapResult(
            p_input_ll=p_ll0,
            p_used_ll=p_coast,             # IMPORTANT: inject point should be coastal
            was_nudged=True,
            in_collision_input=True,
            candidates=cands_gateb,
            reason="coast_gateb_candidates",
            debug=dbg,
        )

    # 2.4 No gateB candidates -> fallback to sea-first (from coastal)
    dbg["fallback_reason"] = "no_gateb_candidates"
    return _sea_first_candidates(
        out,
        p_input_ll=p_ll0,
        p_used_ll=p_coast,
        was_nudged=True,
        in_collision_input=True,
        S_nodes=S_nodes,
        sea_kdt=sea_kdt,
        sea_ok_set=sea_ok_set,
        k_near=k_near,
        r_max_km=r_max_km,
        prefer_ok_set=prefer_ok_set,
        allow_fallback_non_ok=allow_fallback_non_ok,
        allow_radius_fallback=allow_radius_fallback,
        r_fallback_km=r_fallback_km,
        extra_debug=dbg,
    )


def _sea_first_candidates(
    out: Dict[str, Any],
    *,
    p_input_ll: LonLat,
    p_used_ll: LonLat,
    was_nudged: bool,
    in_collision_input: bool,
    S_nodes: pd.DataFrame,
    sea_kdt,
    sea_ok_set,
    k_near: int,
    r_max_km: float,
    prefer_ok_set: bool,
    allow_fallback_non_ok: bool,
    allow_radius_fallback: bool,
    r_fallback_km: Optional[float],
    extra_debug: Optional[Dict[str, Any]] = None,
) -> SnapResult:
    """
    Original sea-first candidate selection (refactored out for reuse).
    p_used_ll is the point we query sea_kdt from (can be original, coastal, or boundary-nudged).
    """
    dbg = extra_debug if isinstance(extra_debug, dict) else {}
    dbg.setdefault("mode", "sea_first")

    # query
    x, y = _point_ll_to_m(out, p_used_ll)
    kq = max(1, int(k_near))
    try:
        idxs = _kdt_query_indices(sea_kdt, x, y, k=min(kq, len(S_nodes)))
    except Exception as e:
        return SnapResult(
            p_input_ll=p_input_ll,
            p_used_ll=p_used_ll,
            was_nudged=was_nudged,
            in_collision_input=in_collision_input,
            candidates=[],
            reason="kdt_query_failed",
            debug={**dbg, "error": repr(e)},
        )

    # decide radius fallback
    if r_fallback_km is None:
        cfg = out.get("cfg", None)
        try:
            r_fallback_km = float(getattr(getattr(cfg, "sea"), "r_max_km"))
        except Exception:
            r_fallback_km = float(r_max_km)

    def make_candidate(i: int) -> SnapCandidate:
        row = S_nodes.iloc[int(i)]
        ll = (float(row["lon"]), float(row["lat"]))
        dist = float(haversine_km(p_used_ll, ll))
        comp = None
        try:
            comp = int(row["component"])
        except Exception:
            comp = None
        ok = True
        if isinstance(sea_ok_set, set):
            ok = (int(i) in sea_ok_set)
        return SnapCandidate(node_idx=int(i), node_ll=ll, dist_km=dist, component=comp, ok=ok)

    cands_all = [make_candidate(i) for i in idxs]
    dbg["k_near_returned"] = len(cands_all)

    def filter_by_radius(cands: List[SnapCandidate], r: float) -> List[SnapCandidate]:
        rr = float(r)
        return [c for c in cands if c.dist_km <= rr]

    cands_r = filter_by_radius(cands_all, float(r_max_km))
    dbg["within_r_max_km"] = len(cands_r)

    cands_ok = [c for c in cands_r if c.ok] if prefer_ok_set else cands_r
    dbg["within_r_ok"] = len(cands_ok)

    chosen = cands_ok
    reason = "ok_within_radius" if chosen else "no_ok_within_radius"

    if (not chosen) and allow_fallback_non_ok:
        chosen = cands_r
        reason = "fallback_non_ok_within_radius" if chosen else "no_candidates_within_radius"

    if (not chosen) and allow_radius_fallback and float(r_fallback_km) > float(r_max_km):
        cands_r2 = filter_by_radius(cands_all, float(r_fallback_km))
        dbg["within_r_fallback_km"] = len(cands_r2)
        cands_ok2 = [c for c in cands_r2 if c.ok] if prefer_ok_set else cands_r2
        if cands_ok2:
            chosen = cands_ok2
            reason = "ok_within_radius_fallback"
        elif allow_fallback_non_ok and cands_r2:
            chosen = cands_r2
            reason = "fallback_non_ok_within_radius_fallback"

    chosen = sorted(chosen, key=lambda c: c.dist_km)

    return SnapResult(
        p_input_ll=p_input_ll,
        p_used_ll=p_used_ll,
        was_nudged=was_nudged,
        in_collision_input=in_collision_input,
        candidates=chosen,
        reason=reason,
        debug=dbg,
    )


# -------------------------
# Pair: component-aware pick (unchanged)
# -------------------------
def snap_pair_component_aware(
    out: Dict[str, Any],
    start_ll: LonLat,
    end_ll: LonLat,
    *,
    k_near: int = 30,
    r_max_km: float = 150.0,
    k_inject: int = 4,
    prefer_ok_set: bool = True,
    allow_fallback_non_ok: bool = True,
    allow_radius_fallback: bool = True,
    do_nudge: bool = True,
    k_near_coast: int = 80,
    r_max_km_coast: Optional[float] = None,
    # GateB params forwarded into snap_to_sea_candidates
    k_near_gateb: int = 30,
    r_max_km_gateb: float = 200.0,
) -> SnapPairResult:
    """
    NOTE: now candidates may be either:
      - Sea nodes (normal sea-first), or
      - GateB nodes (collision mode, coast->gateB->sea)
    Component-aware logic still works because GateB candidates carry component from rank0 sea_idx.
    """
    sres = snap_to_sea_candidates(
        out,
        start_ll,
        k_near=k_near,
        r_max_km=r_max_km,
        prefer_ok_set=prefer_ok_set,
        allow_fallback_non_ok=allow_fallback_non_ok,
        allow_radius_fallback=allow_radius_fallback,
        do_nudge=do_nudge,
        k_near_coast=k_near_coast,
        r_max_km_coast=r_max_km_coast,
        k_near_gateb=k_near_gateb,
        r_max_km_gateb=r_max_km_gateb,
    )
    eres = snap_to_sea_candidates(
        out,
        end_ll,
        k_near=k_near,
        r_max_km=r_max_km,
        prefer_ok_set=prefer_ok_set,
        allow_fallback_non_ok=allow_fallback_non_ok,
        allow_radius_fallback=allow_radius_fallback,
        do_nudge=do_nudge,
        k_near_coast=k_near_coast,
        r_max_km_coast=r_max_km_coast,
        k_near_gateb=k_near_gateb,
        r_max_km_gateb=r_max_km_gateb,
    )

    dbg: Dict[str, Any] = {
        "start_reason": sres.reason,
        "end_reason": eres.reason,
        "start_mode": sres.debug.get("mode") if isinstance(sres.debug, dict) else None,
        "end_mode": eres.debug.get("mode") if isinstance(eres.debug, dict) else None,
        "start_fallback": sres.debug.get("fallback_reason") if isinstance(sres.debug, dict) else None,
        "end_fallback": eres.debug.get("fallback_reason") if isinstance(eres.debug, dict) else None,
    }

    if len(sres.candidates) == 0 or len(eres.candidates) == 0:
        return SnapPairResult(
            start=sres,
            end=eres,
            chosen_common_component=None,
            largest_component=None,
            start_pick=[],
            end_pick=[],
            reason="snap_failed",
            debug=dbg,
        )

    start_comps = {c.component for c in sres.candidates if c.component is not None}
    end_comps = {c.component for c in eres.candidates if c.component is not None}
    common = sorted(list(start_comps.intersection(end_comps)))

    chosen_common = None
    if common:
        best = None
        for comp in common:
            ds = min([c.dist_km for c in sres.candidates if c.component == comp], default=np.inf)
            de = min([c.dist_km for c in eres.candidates if c.component == comp], default=np.inf)
            score = ds + de
            if best is None or score < best[0]:
                best = (score, comp)
        chosen_common = best[1] if best else common[0]

    # infer largest component from sea_ok_set distribution
    largest_comp = None
    try:
        S_nodes_df = out.get("S_nodes")
        sea_ok_set = out.get("sea_ok_set")
        if (
            isinstance(S_nodes_df, pd.DataFrame)
            and isinstance(sea_ok_set, set)
            and len(sea_ok_set) > 0
            and "component" in S_nodes_df.columns
        ):
            comps = S_nodes_df.loc[list(sea_ok_set), "component"].value_counts()
            if len(comps):
                largest_comp = int(comps.index[0])
    except Exception:
        largest_comp = None

    dbg.update(
        {
            "common_components": common,
            "chosen_common_component": chosen_common,
            "largest_component": largest_comp,
        }
    )

    def order_candidates(cands: List[SnapCandidate]) -> List[SnapCandidate]:
        def key(c: SnapCandidate):
            pri = 2
            if chosen_common is not None and c.component == chosen_common:
                pri = 0
            elif largest_comp is not None and c.component == largest_comp:
                pri = 1
            return (pri, c.dist_km)

        return sorted(cands, key=key)

    spick = order_candidates(sres.candidates)[: max(1, int(k_inject))]
    epick = order_candidates(eres.candidates)[: max(1, int(k_inject))]

    reason = "common_component_preferred" if chosen_common is not None else "largest_component_preferred"

    return SnapPairResult(
        start=sres,
        end=eres,
        chosen_common_component=chosen_common,
        largest_component=largest_comp,
        start_pick=spick,
        end_pick=epick,
        reason=reason,
        debug=dbg,
    )


# -------------------------
# Inject edges into graph
# -------------------------
def inject_point_edges(
    G,
    p_ll: LonLat,
    candidates: Sequence[SnapCandidate],
    *,
    k_inject: int = 4,
    etype: str = "inject",
    weight_attr: str = "weight",
):
    """
    Inject p_ll into graph by adding edges to top-k candidates.
    weight = dist_km (km)
    """
    p_ll = normalize_lonlat(p_ll)
    if not hasattr(G, "add_edge"):
        raise TypeError("G must be a networkx-like graph with add_edge.")

    use = list(candidates)[: max(1, int(k_inject))]
    for c in use:
        u = p_ll
        v = (float(c.node_ll[0]), float(c.node_ll[1]))
        w = float(c.dist_km)
        G.add_edge(u, v, **{weight_attr: w, "etype": etype, "inject": True})
    return len(use)


def inject_coastal_to_gateb(
    G,
    coastal_ll: LonLat,
    gateb_rows: List[Dict[str, Any]],
    *,
    k_inject: int = 4,
    weight_attr: str = "weight",
    etype: str = "inject_coast_gateb",
):
    """
    Optional helper if you want to explicitly inject coastal->GateB edges.
    (Often unnecessary if you already have c_gateb connectors in your base graph.)
    """
    coastal_ll = normalize_lonlat(coastal_ll)
    use = gateb_rows[: max(1, int(k_inject))]
    added = 0
    for r in use:
        gate_ll = (float(r["gate_ll"][0]), float(r["gate_ll"][1]))
        w = float(r["dist_km"])
        G.add_edge(coastal_ll, gate_ll, **{weight_attr: w, "etype": etype, "inject": True})
        added += 1
    return added


__all__ = [
    "SnapCandidate",
    "SnapResult",
    "SnapPairResult",
    "nudge_out_of_collision",
    "nudge_to_nearest_coastal_node",
    "gateb_candidates_from_coastal",
    "snap_to_sea_candidates",
    "snap_pair_component_aware",
    "inject_point_edges",
    "inject_coastal_to_gateb",
]
