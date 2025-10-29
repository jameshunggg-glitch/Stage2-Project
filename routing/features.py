# routing/features.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, List, Tuple
import math
import shapely
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from .geodesy import to_m, to_ll

# 這份是你單檔版本的濃縮移植（保留凸點/peaks）
def _resample_linestring_m(ls_m: LineString, step_m: float) -> LineString:
    L = ls_m.length
    if L == 0:
        return ls_m
    n = max(4, int(round(L / step_m)))
    d = L / n
    pts = [ls_m.interpolate(i*d) for i in range(n)]
    if pts[0].distance(pts[-1]) > 1e-6:
        pts.append(pts[0])
    return LineString(pts)

def _bearing_deg(a, b):
    ax, ay = a; bx, by = b
    return (math.degrees(math.atan2(by - ay, bx - ax)) + 360.0) % 360.0

def _angdiff(a, b):
    return (a - b + 540.0) % 360.0 - 180.0

def _local_maxima(seq, radius):
    n = len(seq); peaks=[]
    for i in range(radius, n-radius):
        v = seq[i]
        if all(v > seq[i-k] for k in range(1,radius+1)) and all(v >= seq[i+k] for k in range(1,radius+1)):
            peaks.append(i)
    return peaks

def extract_convex_peaks_from_buffer(
    union_ll,
    avoid_km=15.0,
    resample_step_m=300.0,
    window_km=8.0,
    peak_radius_pts=3,
    min_turn_deg=18.0,
    dedup_m=1200.0
):
    union_m = shapely.ops.transform(lambda x,y: to_m(x,y), union_ll)
    buf_m   = union_m.buffer(avoid_km * 1000.0)
    polys_m = [buf_m] if buf_m.geom_type=="Polygon" else [p for p in buf_m.geoms if p.geom_type=="Polygon"]
    out=[]
    for poly in polys_m:
        ring_raw = LineString(list(poly.exterior.coords))
        ring = _resample_linestring_m(ring_raw, resample_step_m)
        coords = list(ring.coords)
        if len(coords) < 8: 
            continue
        ccw = Polygon(coords).exterior.is_ccw
        W = max(1, int(round(window_km * 1000.0 / resample_step_m)))
        dturn=[0.0]*len(coords)
        for i in range(W, len(coords)-W):
            h1 = _bearing_deg(coords[i-W], coords[i])
            h2 = _bearing_deg(coords[i], coords[i+W])
            dturn[i] = _angdiff(h2, h1)
        score = [max(0.0,v) if ccw else max(0.0,-v) for v in dturn]
        peaks = _local_maxima(score, peak_radius_pts)
        peaks = [i for i in peaks if score[i] >= min_turn_deg]
        kept=[]
        for i in sorted(peaks, key=lambda j: -score[j]):
            pi = Point(coords[i])
            if all(pi.distance(Point(coords[k])) > dedup_m for k in kept):
                kept.append(i)
        for i in kept:
            x,y = coords[i]; lon,lat = to_ll(x,y)
            out.append((lon,lat))
    return out

def _dedup_points_geom(pts: List[Point], tol_m: float) -> List[Point]:
    out=[]
    for p in pts:
        if all(p.distance(q) > tol_m for q in out):
            out.append(p)
    return out

def score_convex_concave_on_ring(
    ring_ls_m: LineString,
    scales_km=(5, 10, 20, 40),
    angle_convex_max=170.0,
    angle_concave_min=210.0,
    min_prom_convex=0.002,
    min_prom_concave=0.002,
    simplify_before=False, simplify_m=800.0,
    dedup_m=800.0,
):
    ls = ring_ls_m
    if simplify_before:
        ls = ring_ls_m.simplify(simplify_m, preserve_topology=False)
        if not ls.is_ring:
            ls = LineString(list(ls.coords)+[ls.coords[0]])
    coords = list(ls.coords)
    n = len(coords)
    if n < 5: return [], [], coords
    ccw = Polygon(coords).exterior.is_ccw

    # 預先計算各點的最小角與最大顯著度
    def _cum_lengths(cs):
        L=[0.0]
        for (x1,y1),(x2,y2) in zip(cs, cs[1:]):
            L.append(L[-1] + ((x2-x1)**2 + (y2-y1)**2)**0.5)
        return L
    def _index_at_arclen(L, idx, s):
        # 簡化版掃描
        target_back = L[idx] - s; i_back = idx
        while i_back>0 and L[i_back-1] > target_back: i_back -= 1
        target_fwd = L[idx] + s; i_fwd = idx
        n2 = len(L)-1
        while i_fwd<n2 and L[i_fwd+1] < target_fwd: i_fwd += 1
        return i_back, i_fwd
    def _angle_and_prominence(a,b,c):
        ax,ay=a; bx,by=b; cx,cy=c
        v1x,v1y = ax-bx, ay-by
        v2x,v2y = cx-bx, cy-by
        n1 = (v1x*v1x+v1y*v1y)**0.5
        n2 = (v2x*v2x+v2y*v2y)**0.5
        if n1==0 or n2==0: return 180.0, 0.0
        cosang = max(-1.0, min(1.0, (v1x*v2x+v1y*v2y)/(n1*n2)))
        ang = math.degrees(math.acos(cosang))
        vx, vy = cx-ax, cy-ay
        vlen = (vx*vx+vy*vy)**0.5
        if vlen==0: prom=0.0
        else:
            abx,aby = bx-ax, by-ay
            cross = abs(abx*vy - aby*vx)
            prom = cross / (vlen*vlen)
        return ang, prom

    L = _cum_lengths(coords)
    scales_m = [s*1000.0 for s in scales_km]
    min_angle=[180.0]*n; max_prom=[0.0]*n
    for i in range(n):
        for s in scales_m:
            i_back, i_fwd = _index_at_arclen(L, i, s)
            if i_back==i or i_fwd==i: continue
            ang, prom = _angle_and_prominence(coords[i_back], coords[i], coords[i_fwd])
            if ang < min_angle[i]: min_angle[i] = ang
            if prom > max_prom[i]: max_prom[i] = prom

    convex_idx=[]; concave_idx=[]
    for i in range(1,n-1):
        a=coords[i-1]; b=coords[i]; c=coords[i+1]
        cross = (a[0]-b[0])*(c[1]-b[1]) - (a[1]-b[1])*(c[0]-b[0])
        is_concave = (cross < 0) if ccw else (cross > 0)
        ang  = min_angle[i]; prom = max_prom[i]
        if not is_concave:
            if ang < angle_convex_max and prom >= min_prom_convex: convex_idx.append(i)
        else:
            if ang > angle_concave_min and prom >= min_prom_concave: concave_idx.append(i)

    def _dedup_by_spacing(idxs):
        kept=[]; taken=[False]*n
        for i in sorted(idxs, key=lambda j: -max_prom[j]):
            if taken[i]: continue
            kept.append(i)
            xi,yi=coords[i]
            for j in range(n):
                if taken[j]: continue
                xj,yj=coords[j]
                if ((xj-xi)**2+(yj-yi)**2)**0.5 <= 800.0:
                    taken[j]=True
        return kept

    return _dedup_by_spacing(convex_idx), _dedup_by_spacing(concave_idx), coords

def extract_feature_points_bbox(
    shp_path: str | Path,
    bbox_ll_polygon: Polygon,
    avoid_km=15.0,
    simplify_m=1000.0,
    ANGLE_CONVEX_MAX=170.0,
    ANGLE_CONCAVE_MIN=210.0,
    MIN_PROM_CONVEX=0.002,
    MIN_PROM_CONCAVE=0.002,
    DEDUP_CONVEX_M=800.0,
    DEDUP_CONCAVE_M=800.0,
    ENABLE_UNIFORM=False,
    TARGET_SPACING_KM=25.0,
    N_MIN=8, N_MAX=64,
    PERIM_MIN_KM=20.0,
    AREA_MIN_KM2=5.0,
) -> Dict[str, List[Tuple[float,float]]]:
    import fiona
    from shapely.geometry import shape
    polys=[]
    with fiona.open(shp_path) as src:
        for feat in src:
            g = shape(feat["geometry"])
            if g.is_empty: continue
            if g.intersects(bbox_ll_polygon):
                gi = g.intersection(bbox_ll_polygon)
                if not gi.is_empty: polys.append(gi)
    if not polys:
        return {"convex":[], "concave":[], "uniform":[], "convex_peaks":[]}

    union_ll = unary_union(polys)
    union_m  = shapely.ops.transform(lambda x,y: to_m(x,y), union_ll)
    buf_m    = union_m.buffer(avoid_km * 1000.0)
    polys_m  = [buf_m] if buf_m.geom_type=="Polygon" else [p for p in buf_m.geoms if p.geom_type=="Polygon"]

    convex_pts_m=[]; concave_pts_m=[]; uniform_pts_m=[]
    for poly in polys_m:
        perim_km = poly.exterior.length / 1000.0
        area_km2 = poly.area / 1e6
        if perim_km < PERIM_MIN_KM or area_km2 < AREA_MIN_KM2:
            continue
        ring_raw = shapely.LineString(poly.exterior.coords)
        ring_s   = ring_raw.simplify(simplify_m, preserve_topology=False)
        if not ring_s.is_ring:
            ring_s = shapely.LineString(list(ring_s.coords)+[ring_s.coords[0]])
        convex_idx, concave_idx, coords_used = score_convex_concave_on_ring(
            ring_s,
            scales_km=(5,10,20,40),
            angle_convex_max=ANGLE_CONVEX_MAX,
            angle_concave_min=ANGLE_CONCAVE_MIN,
            min_prom_convex=MIN_PROM_CONVEX,
            min_prom_concave=MIN_PROM_CONCAVE,
            simplify_before=False,
            dedup_m=max(DEDUP_CONVEX_M, DEDUP_CONCAVE_M),
        )
        for i in convex_idx:
            x,y = coords_used[i]; convex_pts_m.append(Point(x,y))
        for i in concave_idx:
            x,y = coords_used[i]; concave_pts_m.append(Point(x,y))

        if ENABLE_UNIFORM:
            n_uniform = max(N_MIN, min(N_MAX, int(perim_km / TARGET_SPACING_KM)))
            L = ring_raw.length
            for i in range(n_uniform):
                uniform_pts_m.append(ring_raw.interpolate(i * L / n_uniform))

    convex_peaks_ll = extract_convex_peaks_from_buffer(
        union_ll=union_ll, avoid_km=avoid_km, resample_step_m=300.0,
        window_km=8.0, peak_radius_pts=3, min_turn_deg=18.0, dedup_m=1200.0
    )
    def _to_ll_list(pts):
        out=[]
        for p in _dedup_points_geom(pts, max(DEDUP_CONVEX_M, DEDUP_CONCAVE_M)):
            lon,lat = to_ll(p.x,p.y); out.append((lon,lat))
        return out

    return {
        "convex": _to_ll_list(convex_pts_m),
        "concave": _to_ll_list(concave_pts_m),
        "uniform": _to_ll_list(uniform_pts_m) if ENABLE_UNIFORM else [],
        "convex_peaks": convex_peaks_ll
    }
