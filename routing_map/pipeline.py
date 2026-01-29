from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .routing_graph import build_base_graph, haversine_km
from .snap import snap_pair_component_aware, inject_point_edges
from .repairer import PathRepairer, RepairConfig
from .path_simplifier import simplify_path_visibility
from .geom_utils import get_projector, get_collision_metric

LonLat = Tuple[float, float]
BBoxLL = Tuple[float, float, float, float]


@dataclass
class GraphConfig:
    bbox_ll: Optional[BBoxLL] = None
    include_sea: bool = True
    include_rings: bool = True
    include_et: bool = True
    include_tgate_sea: bool = True
    # legacy (kept for compatibility)
    include_cc: bool = False
    include_gateb_sea: bool = False
    include_c_gateb: bool = False

    max_sea_edges: Optional[int] = None
    max_ring_edges: Optional[int] = None
    weight_unit: str = "km"


@dataclass
class SnapConfig:
    k_near: int = 30
    r_max_km: float = 150.0
    k_inject: int = 4
    prefer_ok_set: bool = True
    allow_fallback_non_ok: bool = True
    allow_radius_fallback: bool = True
    do_nudge: bool = True
    k_near_coast: int = 80
    r_max_km_coast: Optional[float] = None
    enable_local_entrance_aug: bool = True
    aug_dist_trigger_km: float = 60.0
    aug_delta_end_km: float = 120.0
    aug_angle_trigger_deg: float = 110.0
    aug_seed_neighbors_cap: int = 12
    aug_seed_count: int = 1
    # === multiworld snap ===
    R_NEAR_COAST_KM: float = 120.0
    S_MAX_SNAP_KM: float = 200.0
    # per-run override (new): allow wrapper to force ring/sea worldview
    force_start_policy: Optional[str] = None   # "R" or "S" or None
    force_end_policy: Optional[str] = None     # "R" or "S" or None


@dataclass
class SimplifyConfig:
    enabled: bool = True
    window_size: int = 80
    max_tries: int = 300
    use_prepared_collision: bool = True
    dateline_unwrap: bool = True
    wrap_output_lon: bool = True
    strategy: str = "linear_backscan"


@dataclass
class RunConfig:
    do_repair: bool = True
    do_simplify: bool = True
    debug: bool = True


@dataclass
class RouteResult:
    # inputs
    origin_ll: Optional[LonLat] = None
    dest_ll: Optional[LonLat] = None

    # snap
    start_ll_snap: Optional[LonLat] = None
    end_ll_snap: Optional[LonLat] = None
    snap_debug: Optional[Dict[str, Any]] = None

    # graph
    G: Optional[Any] = None
    graph_stats: Optional[Any] = None

    # A*
    path_nodes: Optional[List[Any]] = None
    path_ll_raw: Optional[List[LonLat]] = None

    # repair / simplify
    path_ll_repaired: Optional[List[LonLat]] = None
    path_ll_simplified: Optional[List[LonLat]] = None
    path_ll_final: Optional[List[LonLat]] = None

    lengths_km: Optional[Dict[str, float]] = None

    # errors
    error: Optional[str] = None


def _path_len_km(path_ll: Optional[List[LonLat]]) -> float:
    if not path_ll or len(path_ll) < 2:
        return 0.0
    s = 0.0
    for a, b in zip(path_ll, path_ll[1:]):
        s += float(haversine_km(a, b))
    return float(s)


def run_p2p(
    out: Dict[str, Any],
    origin_ll: LonLat,
    dest_ll: LonLat,
    *,
    graph_cfg: Optional[GraphConfig] = None,
    snap_cfg: Optional[SnapConfig] = None,
    repair_cfg: Optional[RepairConfig] = None,
    simplify_cfg: Optional[SimplifyConfig] = None,
    run_cfg: Optional[RunConfig] = None,
    G_in: Optional[Any] = None,
) -> RouteResult:
    """End-to-end routing runner for the new Sea + E/T ring graph.

    Assumptions:
    - `out` already comes from build_aoi.
    - `routing_graph.build_base_graph` is the rings-compatible version.
    - snap.py already snaps to sea candidates and injects edges.
    """

    graph_cfg = graph_cfg or GraphConfig(bbox_ll=out.get("bbox_ll"))
    snap_cfg = snap_cfg or SnapConfig()
    simplify_cfg = simplify_cfg or SimplifyConfig()
    run_cfg = run_cfg or RunConfig()

    res = RouteResult(origin_ll=origin_ll, dest_ll=dest_ll)

    bbox_ll = graph_cfg.bbox_ll or out.get("bbox_ll")

    # projector + collision
    try:
        proj = get_projector(out, bbox_ll=bbox_ll)
    except Exception:
        proj = out.get("proj", None)

    collision_m, _is_prep = get_collision_metric(out, prefer_prepared=True)

    # build graph
    try:
        if G_in is not None:
            G, stats = G_in, None
        else:
            G, stats = build_base_graph(
                out,
                include_sea=graph_cfg.include_sea,
                include_cc=graph_cfg.include_cc,
                include_gateb_sea=graph_cfg.include_gateb_sea,
                include_c_gateb=graph_cfg.include_c_gateb,
                include_rings=graph_cfg.include_rings,
                include_et=graph_cfg.include_et,
                include_tgate_sea=graph_cfg.include_tgate_sea,
                max_sea_edges=graph_cfg.max_sea_edges,
                max_ring_edges=graph_cfg.max_ring_edges,
                weight_unit=graph_cfg.weight_unit,
                bbox_ll=bbox_ll,
            )
        res.G = G
        res.graph_stats = stats
        if run_cfg.debug:
            try:
                print(f"[pipeline][graph] nodes={G.number_of_nodes()} edges={G.number_of_edges()} stats={stats}")
            except Exception:
                print("[pipeline][graph] built")
    except Exception as e:
        res.error = f"graph_build_error: {e}"
        return res
    
    extra = {}
    if getattr(snap_cfg, "force_start_policy", None) is not None:
        extra["start_policy"] = snap_cfg.force_start_policy
    if getattr(snap_cfg, "force_end_policy", None) is not None:
        extra["end_policy"] = snap_cfg.force_end_policy


    # snap pair
    try:
        pair = snap_pair_component_aware(
            out,
            origin_ll,
            dest_ll,
            k_near=snap_cfg.k_near,
            r_max_km=snap_cfg.r_max_km,
            k_inject=snap_cfg.k_inject,
            prefer_ok_set=snap_cfg.prefer_ok_set,
            allow_fallback_non_ok=snap_cfg.allow_fallback_non_ok,
            allow_radius_fallback=snap_cfg.allow_radius_fallback,
            do_nudge=snap_cfg.do_nudge,
            k_near_coast=snap_cfg.k_near_coast,
            r_max_km_coast=snap_cfg.r_max_km_coast,
            enable_local_entrance_aug=snap_cfg.enable_local_entrance_aug,
            aug_dist_trigger_km=snap_cfg.aug_dist_trigger_km,
            aug_delta_end_km=snap_cfg.aug_delta_end_km,
            aug_angle_trigger_deg=snap_cfg.aug_angle_trigger_deg,
            aug_seed_neighbors_cap=snap_cfg.aug_seed_neighbors_cap,
            aug_seed_count=snap_cfg.aug_seed_count,
            **extra,
        )

        # choose candidates (these are EXISTING graph nodes we will connect to)
        start_pick = pair.start_pick or (pair.start.candidates[: snap_cfg.k_inject] if pair.start else [])
        end_pick = pair.end_pick or (pair.end.candidates[: snap_cfg.k_inject] if pair.end else [])

        # base "keys" that will be inserted into the graph:
        # use the nudged/normalized point (p_used_ll) instead of raw input.
        # Otherwise A* will fail with "Source ... is not in G".
        start_key = (float(pair.start.p_used_ll[0]), float(pair.start.p_used_ll[1]))
        end_key = (float(pair.end.p_used_ll[0]), float(pair.end.p_used_ll[1]))

        # nearest picked graph node (for debug / viz)
        start_snap = (float(start_pick[0].node_ll[0]), float(start_pick[0].node_ll[1])) if start_pick else None
        end_snap = (float(end_pick[0].node_ll[0]), float(end_pick[0].node_ll[1])) if end_pick else None

        res.start_ll_snap = start_snap  # closest existing node (not the injected key)
        res.end_ll_snap = end_snap
        res.snap_debug = getattr(pair, "debug", None) if hasattr(pair, "debug") else None

        if run_cfg.debug:
            print(
                f"[pipeline][snap] start_in={origin_ll} used={start_key} -> pick={start_snap} | "
                f"end_in={dest_ll} used={end_key} -> pick={end_snap} | {res.snap_debug}"
            )

        if not start_pick or not end_pick:
            res.error = "snap_failed"
            return res

        # inject (IMPORTANT: inject using the same node keys we will use for A*)
        inject_point_edges(G, start_key, start_pick, k_inject=snap_cfg.k_inject, etype="inject")
        inject_point_edges(G, end_key, end_pick, k_inject=snap_cfg.k_inject, etype="inject")

        # overwrite the keys for downstream A*
        origin_ll = start_key
        dest_ll = end_key

    except Exception as e:
        res.error = f"snap_inject_error: {e}"
        return res

    # A*
    try:
        path_nodes = nx.astar_path(
            G,
            start_key,
            end_key,
            heuristic=lambda a, b: haversine_km(a, b),
            weight="weight",
        )
        if run_cfg.debug:
            print("[pipeline][astar] start_key in G?", start_key in G)
            print("[pipeline][astar] end_key in G?", end_key in G)
        res.path_nodes = list(path_nodes)
        res.path_ll_raw = [(float(p[0]), float(p[1])) for p in path_nodes]
        if run_cfg.debug:
            print(f"[pipeline][A*] n_nodes={len(path_nodes)}")
    except Exception as e:
        res.error = f"astar_error: {e}"
        return res

    # repair
    path_ll_work = res.path_ll_raw
    if run_cfg.do_repair and repair_cfg is not None and collision_m is not None:
        try:
            rep = PathRepairer(repair_cfg)
            out_rep = rep.repair_path(G, res.path_nodes, collision_m=collision_m, proj=proj)
            res.path_ll_repaired = out_rep.path_ll
            path_ll_work = res.path_ll_repaired
            if run_cfg.debug:
                print(f"[pipeline][repair] repaired_edges={out_rep.stats.repaired_edges} colliding={out_rep.stats.colliding_edges}")
        except Exception as e:
            res.path_ll_repaired = path_ll_work
            if run_cfg.debug:
                print(f"[pipeline][repair][warn] {e}")

    # simplify
    if run_cfg.do_simplify and simplify_cfg.enabled and collision_m is not None:
        try:
            simp_ll, simp_stats = simplify_path_visibility(
                path_ll_work,
                collision_m=collision_m,
                proj=proj,
                window_size=simplify_cfg.window_size,
                max_tries=simplify_cfg.max_tries,
                use_prepared_collision=simplify_cfg.use_prepared_collision,
                dateline_unwrap=simplify_cfg.dateline_unwrap,
                wrap_output_lon=simplify_cfg.wrap_output_lon,
                strategy=simplify_cfg.strategy,
            )
            res.path_ll_simplified = simp_ll
            if run_cfg.debug:
                print(f"[pipeline][simplify] {simp_stats.n_in}->{simp_stats.n_out} checks={simp_stats.n_checks}")
        except Exception as e:
            res.path_ll_simplified = path_ll_work
            if run_cfg.debug:
                print(f"[pipeline][simplify][warn] {e}")

    # final (ALWAYS include input->used and used->input legs for visualization)
    core = res.path_ll_simplified or res.path_ll_repaired or res.path_ll_raw or []

    path_final: List[LonLat] = []

    origin_in = res.origin_ll  # original user input
    dest_in = res.dest_ll      # original user input

    start_used = start_key     # injected key (p_used_ll)
    end_used = end_key         # injected key (p_used_ll)

    def _push(pt: LonLat):
        pt2 = (float(pt[0]), float(pt[1]))
        if not path_final or path_final[-1] != pt2:
            path_final.append(pt2)

    # prepend: input -> used
    if origin_in is not None and start_used is not None:
        _push(origin_in)
        _push(start_used)

    # core path (used -> ... -> used)
    for p in core:
        _push(p)

    # append: used -> input
    if dest_in is not None and end_used is not None:
        _push(end_used)
        _push(dest_in)

    res.path_ll_final = path_final


    lengths = {
        "raw": _path_len_km(res.path_ll_raw),
        "repaired": _path_len_km(res.path_ll_repaired),
        "simplified": _path_len_km(res.path_ll_simplified),
        "final": _path_len_km(res.path_ll_final),
    }
    res.lengths_km = lengths

    if run_cfg.debug:
        print(f"[pipeline][done] lengths_km={lengths}")

    return res

def run_p2p_multiworld(
    out: Dict[str, Any],
    origin_ll: LonLat,
    dest_ll: LonLat,
    *,
    graph_cfg: Optional[GraphConfig] = None,
    snap_cfg: Optional[SnapConfig] = None,
    repair_cfg: Optional[RepairConfig] = None,
    simplify_cfg: Optional[SimplifyConfig] = None,
    run_cfg: Optional[RunConfig] = None,
) -> RouteResult:
    """
    Multi-worldview runner:
    - Each endpoint chooses policy in {R,S} after pruning:
        R_NEAR_COAST_KM=120km, S_MAX_SNAP_KM=200km (from snap_cfg)
    - Runs up to 4 combos: RR/RS/SR/SS
    - Each combo runs full pipeline: snap -> inject -> A* -> repair -> simplify
    - Select winner by final simplified length (res.lengths_km["final"])
    """
    graph_cfg = graph_cfg or GraphConfig(bbox_ll=out.get("bbox_ll"))
    snap_cfg = snap_cfg or SnapConfig()
    simplify_cfg = simplify_cfg or SimplifyConfig()
    run_cfg = run_cfg or RunConfig()

    bbox_ll = graph_cfg.bbox_ll or out.get("bbox_ll")

    # --- projector + collision (for "in_collision" pruning safety) ---
    try:
        proj = get_projector(out, bbox_ll=bbox_ll)
    except Exception:
        proj = out.get("proj", None)

    collision_m, _is_prep = get_collision_metric(out, prefer_prepared=True)

    # --- build base graph ONCE ---
    try:
        G_base, stats = build_base_graph(
            out,
            include_sea=graph_cfg.include_sea,
            include_cc=graph_cfg.include_cc,
            include_gateb_sea=graph_cfg.include_gateb_sea,
            include_c_gateb=graph_cfg.include_c_gateb,
            include_rings=graph_cfg.include_rings,
            include_et=graph_cfg.include_et,
            include_tgate_sea=graph_cfg.include_tgate_sea,
            max_sea_edges=graph_cfg.max_sea_edges,
            max_ring_edges=graph_cfg.max_ring_edges,
            weight_unit=graph_cfg.weight_unit,
            bbox_ll=bbox_ll,
        )
        if run_cfg.debug:
            try:
                print(f"[pipeline][graph] nodes={G_base.number_of_nodes()} edges={G_base.number_of_edges()} stats={stats}")
            except Exception:
                print("[pipeline][graph] built (multiworld)")
    except Exception as e:
        return RouteResult(origin_ll=origin_ll, dest_ll=dest_ll, error=f"graph_build_error: {e}")

    # --- helper: min distance (km) from point to node dataframe using x_m/y_m ---
    def _min_df_dist_km(p_ll: LonLat, df) -> Optional[float]:
        if df is None:
            return None
        cols = getattr(df, "columns", [])
        if "x_m" not in cols or "y_m" not in cols:
            return None
        if proj is None or not hasattr(proj, "ll2m"):
            return None
        x0, y0 = proj.ll2m(float(p_ll[0]), float(p_ll[1]))

        # Fast path with numpy if available
        try:
            import numpy as np  # type: ignore
            xs = df["x_m"].to_numpy(dtype=float)
            ys = df["y_m"].to_numpy(dtype=float)
            if xs.size == 0:
                return None
            d2 = (xs - x0) ** 2 + (ys - y0) ** 2
            return float(np.sqrt(d2.min()) / 1000.0)
        except Exception:
            # Pure python fallback
            best = None
            for r in df.itertuples(index=False):
                try:
                    dx = float(getattr(r, "x_m")) - x0
                    dy = float(getattr(r, "y_m")) - y0
                    d = (dx * dx + dy * dy) ** 0.5 / 1000.0
                    best = d if best is None else min(best, d)
                except Exception:
                    continue
            return best

    def _in_collision(p_ll: LonLat) -> bool:
        if collision_m is None or proj is None or not hasattr(proj, "ll2m"):
            return False
        try:
            from shapely.geometry import Point
            x, y = proj.ll2m(float(p_ll[0]), float(p_ll[1]))
            return bool(collision_m.contains(Point(x, y)))
        except Exception:
            return False

    # --- fetch node dfs (robust keys) ---
    sea_nodes = out.get("sea_nodes", None)
    e_nodes = out.get("e_nodes", out.get("E_nodes", None))
    t_nodes = out.get("t_nodes", out.get("T_nodes", None))

    # distances for pruning (approx via nearest nodes, good enough for pruning)
    d_sea_o = _min_df_dist_km(origin_ll, sea_nodes)
    d_sea_d = _min_df_dist_km(dest_ll, sea_nodes)

    d_e_o = _min_df_dist_km(origin_ll, e_nodes)
    d_t_o = _min_df_dist_km(origin_ll, t_nodes)
    d_ring_o = min([v for v in [d_e_o, d_t_o] if v is not None], default=None)

    d_e_d = _min_df_dist_km(dest_ll, e_nodes)
    d_t_d = _min_df_dist_km(dest_ll, t_nodes)
    d_ring_d = min([v for v in [d_e_d, d_t_d] if v is not None], default=None)

    # --- pruning rules (necessary pruning only) ---
    R_NEAR = float(getattr(snap_cfg, "R_NEAR_COAST_KM", 120.0))
    S_MAX = float(getattr(snap_cfg, "S_MAX_SNAP_KM", 200.0))

    def _allowed_policies(p_ll: LonLat, d_ring: Optional[float], d_sea: Optional[float]) -> List[str]:
        allow_R = True
        allow_S = True

        # Always keep R if in collision (safety)
        if _in_collision(p_ll):
            allow_R = True
        else:
            # If ring is far, we can prune R (but only if we can compute d_ring)
            if d_ring is not None and d_ring > R_NEAR:
                allow_R = False

        # If nearest sea node is too far, prune S (only if we can compute d_sea)
        if d_sea is not None and d_sea > S_MAX:
            allow_S = False

        # Never end up with empty set: fallback to R
        out_p = []
        if allow_R:
            out_p.append("R")
        if allow_S:
            out_p.append("S")
        if not out_p:
            out_p = ["R"]
        return out_p

    P_start = _allowed_policies(origin_ll, d_ring_o, d_sea_o)
    P_end = _allowed_policies(dest_ll, d_ring_d, d_sea_d)

    combos: List[Tuple[str, str]] = [(a, b) for a in P_start for b in P_end]
    if run_cfg.debug:
        print(
            f"[pipeline][multiworld][prune] "
            f"start d_ring={d_ring_o} d_sea={d_sea_o} -> {P_start} | "
            f"end d_ring={d_ring_d} d_sea={d_sea_d} -> {P_end} | "
            f"combos={[''.join(c) for c in combos]}"
        )

    # --- run each combo and select best by final length ---
    best: Optional[RouteResult] = None
    best_len: Optional[float] = None

    for sp, ep in combos:
        combo = f"{sp}{ep}"

        # copy base graph because inject mutates the graph
        try:
            G_run = G_base.copy()
        except Exception:
            # in case graph copy fails for any reason, rebuild (slower but safe)
            G_run, _ = build_base_graph(
                out,
                include_sea=graph_cfg.include_sea,
                include_cc=graph_cfg.include_cc,
                include_gateb_sea=graph_cfg.include_gateb_sea,
                include_c_gateb=graph_cfg.include_c_gateb,
                include_rings=graph_cfg.include_rings,
                include_et=graph_cfg.include_et,
                include_tgate_sea=graph_cfg.include_tgate_sea,
                max_sea_edges=graph_cfg.max_sea_edges,
                max_ring_edges=graph_cfg.max_ring_edges,
                weight_unit=graph_cfg.weight_unit,
                bbox_ll=bbox_ll,
            )

        snap_cfg_run = replace(snap_cfg, force_start_policy=sp, force_end_policy=ep)
        if run_cfg.debug:
            print(f"[pipeline][multiworld][run] combo={combo} start_policy={sp} end_policy={ep}")

        res = run_p2p(
            out,
            origin_ll,
            dest_ll,
            graph_cfg=graph_cfg,
            snap_cfg=snap_cfg_run,
            repair_cfg=repair_cfg,
            simplify_cfg=simplify_cfg,
            run_cfg=run_cfg,
            G_in=G_run,  # reuse built graph (copied)
        )

        if res.error is None and res.lengths_km is not None:
            L = float(res.lengths_km.get("final", 0.0))
            if run_cfg.debug:
                print(f"[pipeline][multiworld][result] combo={combo} final_km={L}")
            if best is None or best_len is None or L < best_len:
                best = res
                best_len = L
        else:
            if run_cfg.debug:
                print(f"[pipeline][multiworld][result] combo={combo} FAIL: {res.error}")

    if best is None:
        # all failed: return a representative failure (prefer last)
        return RouteResult(origin_ll=origin_ll, dest_ll=dest_ll, error="multiworld_all_failed")

    # attach a small hint in snap_debug
    if best.snap_debug is None:
        best.snap_debug = {}
    best.snap_debug["multiworld_selected"] = True
    best.snap_debug["R_NEAR_COAST_KM"] = R_NEAR
    best.snap_debug["S_MAX_SNAP_KM"] = S_MAX

    return best


__all__ = [
    "GraphConfig",
    "SnapConfig",
    "SimplifyConfig",
    "RunConfig",
    "RouteResult",
    "run_p2p",
    "run_p2p_multiworld",
]
