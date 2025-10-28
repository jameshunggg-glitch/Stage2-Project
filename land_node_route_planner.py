# land_node_route_planner.py
from __future__ import annotations
from pathlib import Path
import math, heapq, itertools
from typing import List, Tuple, Dict, Optional

import fiona, folium, shapely
from shapely.geometry import shape, Polygon, Point, LineString, GeometryCollection
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree
from pyproj import Transformer, Geod

# ---------------- Default params ----------------
BUFFER_KM   = 5.0
COLLISION_SAFETY_KM = 0.25
PAD_DEG     = 6.0
STEP_KM_GEODESIC = 3.0
DRAW_STEP_KM = 20.0
AVOID_KM    = 15.0
NEIGHBOR_K = 24
LVS_MAX_NODES = 4000
SIMPLIFY_MAX_PASSES = 8

# ---------------- CRS & geodesy ----------------
to_m  = Transformer.from_crs("EPSG:4326","EPSG:3857", always_xy=True).transform
to_ll = Transformer.from_crs("EPSG:3857","EPSG:4326", always_xy=True).transform
def to_metric(g): return shapely.ops.transform(to_m, g)
def to_wgs(g):    return shapely.ops.transform(to_ll, g)
GEOD = Geod(ellps="WGS84")

def geodesic_sample(a: Tuple[float,float], b: Tuple[float,float], step_km: float=STEP_KM_GEODESIC) -> List[Tuple[float,float]]:
    lon1,lat1=a; lon2,lat2=b
    _,_,dist_m = GEOD.inv(lon1,lat1,lon2,lat2)
    n = max(1,int(dist_m/(step_km*1000)))
    pts = GEOD.npts(lon1,lat1,lon2,lat2,n)
    return [(lon1,lat1)] + pts + [(lon2,lat2)]

def gc_distance_km(a,b) -> float:
    _,_,d = GEOD.inv(a[0],a[1],b[0],b[1])
    return d/1000.0

def great_circle_midpoint(a,b):
    pts = geodesic_sample(a,b,step_km=500.0)
    return pts[len(pts)//2]

def normalize_lon_to_pacific_view(lon: float) -> float:
    return lon if lon>=0 else lon+360

def draw_gc_polyline_continuous(m, a, b, step_km=DRAW_STEP_KM, **style):
    pts = geodesic_sample(a, b, step_km=step_km)
    folium_coords = []
    for lon, lat in pts:
        lon_pacific = normalize_lon_to_pacific_view(lon)
        folium_coords.append([lat, lon_pacific])
    folium.PolyLine(folium_coords, **style).add_to(m)

# ---------------- BBox / land loading ----------------
def dynamic_bboxes_idl(origin, dest, pad_deg: float) -> List[Polygon]:
    def _pv(lon): return lon if lon>=0 else lon+360
    o_lon = _pv(origin[0]); d_lon = _pv(dest[0])
    min_lon = min(o_lon, d_lon) - pad_deg
    max_lon = max(o_lon, d_lon) + pad_deg
    min_lat = min(origin[1], dest[1]) - pad_deg
    max_lat = max(origin[1], dest[1]) + pad_deg
    bboxes=[]
    if min_lon < 0:
        bboxes.append(Polygon([(min_lon,min_lat),(0,min_lat),(0,max_lat),(min_lon,max_lat)])); min_lon=0
    if max_lon > 360:
        bboxes.append(Polygon([(0,min_lat),(max_lon-360,min_lat),(max_lon-360,max_lat),(0,max_lat)])); max_lon=360
    lon_min_std = min_lon if min_lon <= 180 else min_lon-360
    lon_max_std = max_lon if max_lon <= 180 else max_lon-360
    if lon_min_std <= lon_max_std:
        bboxes.append(Polygon([(lon_min_std,min_lat),(lon_max_std,min_lat),(lon_max_std,max_lat),(lon_min_std,max_lat)]))
    else:
        bboxes.append(Polygon([(lon_min_std,min_lat),(180,min_lat),(180,max_lat),(lon_min_std,max_lat)]))
        bboxes.append(Polygon([(-180,min_lat),(lon_max_std,min_lat),(lon_max_std,max_lat),(-180,max_lat)]))
    return bboxes

def union_lonlat_bboxes(bboxes: List[Polygon]) -> Polygon:
    u = unary_union(bboxes)
    if isinstance(u, GeometryCollection): return u.envelope
    return u

def load_polys_in_bboxes(shp_path: Path, bbox_polys: List[Polygon]) -> List[Polygon]:
    polys=[]
    with fiona.open(shp_path) as src:
        for feat in src:
            g = shape(feat["geometry"])
            for box in bbox_polys:
                if g.intersects(box):
                    gi = g.intersection(box)
                    try:
                        parts = list(shapely.get_parts(gi))
                    except Exception:
                        parts = list(gi.geoms) if gi.geom_type=="MultiPolygon" else [gi]
                    for p in parts:
                        if not p.is_empty:
                            polys.append(p)
                    break
    return polys

def build_land_layers(polys: List[Polygon]):
    parts_m = [to_metric(p).buffer(0) for p in polys]
    union_m = unary_union(parts_m)
    collision_m = union_m.buffer(COLLISION_SAFETY_KM * 1000.0)
    ring_m      = union_m.buffer(BUFFER_KM * 1000.0)
    return {
        "UNION_M":          union_m,
        "COLLISION_PREP_M": prep(collision_m),
        "COLLISION_WGS":    to_wgs(collision_m),
        "RING_M":           ring_m,
        "RING_WGS":         to_wgs(ring_m),
        "LAND_RAW_WGS":     to_wgs(union_m),
        "LAND_PARTS_M":     parts_m,
        "COLLISION_M":      collision_m,
    }

def build_land_strtree(parts_m: List[shapely.geometry.base.BaseGeometry]) -> STRtree:
    return STRtree(parts_m)

# ---------------- Nudge & visibility ----------------
def nudge_to_ring_if_inside_fast(pt_ll, inner_ring_m, target_boundary_m):
    px, py = to_m(pt_ll[0], pt_ll[1])
    p_m = Point(px, py)
    if inner_ring_m.contains(p_m):
        q_m = shapely.ops.nearest_points(p_m, target_boundary_m)[1]
        q_llx, q_lly = to_ll(q_m.x, q_m.y)
        return (q_llx, q_lly), True
    return pt_ll, False

def visible(a, b, COLLISION_PREP_M, land_tree: Optional[STRtree]=None) -> bool:
    ls_ll = LineString(geodesic_sample(a, b, step_km=STEP_KM_GEODESIC))
    ls_m  = to_metric(ls_ll)
    if land_tree is not None:
        if len(land_tree.query(ls_m)) == 0:
            return True
    return not COLLISION_PREP_M.intersects(ls_m)

# ---------------- Feature extraction（精簡版） ----------------
def _bearing_deg(a, b):
    ax, ay = a; bx, by = b
    return (math.degrees(math.atan2(by - ay, bx - ax)) + 360.0) % 360.0

def _angdiff(a, b):
    return (a - b + 540.0) % 360.0 - 180.0

def _resample_linestring_m(ls_m: LineString, step_m: float) -> LineString:
    L = ls_m.length
    if L == 0: return ls_m
    n = max(4, int(round(L / step_m)))
    d = L / n
    pts = [ls_m.interpolate(i * d) for i in range(n)]
    if pts[0].distance(pts[-1]) > 1e-6: pts.append(pts[0])
    return LineString(pts)

def _local_maxima(seq, radius):
    n = len(seq); peaks=[]
    for i in range(radius, n - radius):
        v = seq[i]
        if all(v > seq[i - k] for k in range(1, radius + 1)) and all(v >= seq[i + k] for k in range(1, radius + 1)):
            peaks.append(i)
    return peaks

def extract_convex_peaks_from_buffer(union_ll, avoid_km=AVOID_KM, resample_step_m=300.0, window_km=8.0, peak_radius_pts=3, min_turn_deg=18.0, dedup_m=1200.0):
    union_m = to_metric(union_ll); buf_m = union_m.buffer(avoid_km * 1000.0)
    polys_m = [buf_m] if buf_m.geom_type=="Polygon" else [p for p in buf_m.geoms if p.geom_type=="Polygon"]
    out_pts=[]
    for poly in polys_m:
        ring_raw = LineString(list(poly.exterior.coords))
        ring     = _resample_linestring_m(ring_raw, resample_step_m)
        coords   = list(ring.coords); n = len(coords)
        if n < 8: continue
        ccw = Polygon(coords).exterior.is_ccw
        W = max(1, int(round(window_km * 1000.0 / resample_step_m)))
        dturn = [0.0]*n
        for i in range(W, n - W):
            h1 = _bearing_deg(coords[i - W], coords[i])
            h2 = _bearing_deg(coords[i],     coords[i + W])
            dturn[i] = _angdiff(h2, h1)
        score = [max(0.0, v) if ccw else max(0.0, -v) for v in dturn]
        peaks = [i for i in _local_maxima(score, peak_radius_pts) if score[i] >= min_turn_deg]
        kept=[]
        for i in sorted(peaks, key=lambda j: -score[j]):
            pi = Point(coords[i])
            if all(pi.distance(Point(coords[k])) > dedup_m for k in kept):
                kept.append(i)
        for i in kept:
            x, y = coords[i]; lon, lat = to_ll(x,y); out_pts.append((lon,lat))
    return out_pts

def score_convex_concave_on_ring(ring_ls_m: LineString, scales_km=(5,10,20,40), angle_convex_max=170.0, angle_concave_min=210.0, min_prom_convex=0.002, min_prom_concave=0.002, simplify_before=False, simplify_m=800.0, dedup_m=800.0):
    ls = ring_ls_m
    if simplify_before:
        ls = ring_ls_m.simplify(simplify_m, preserve_topology=False)
        if not ls.is_ring: ls = LineString(list(ls.coords)+[ls.coords[0]])
    coords = list(ls.coords); n = len(coords)
    if n < 5: return [], [], coords
    L=[0.0]
    for (x1,y1),(x2,y2) in zip(coords, coords[1:]): L.append(L[-1] + math.hypot(x2-x1, y2-y1))
    def _index_at_arclen(L, idx, s):
        target_back = L[idx]-s; i_back=idx
        while i_back>0 and L[i_back-1]>target_back: i_back-=1
        target_fwd = L[idx]+s; i_fwd=idx; n=len(L)-1
        while i_fwd<n and L[i_fwd+1]<target_fwd: i_fwd+=1
        return i_back, i_fwd
    min_angle=[180.0]*n; max_prom=[0.0]*n
    scales_m=[s*1000.0 for s in scales_km]
    for i in range(n):
        for s in scales_m:
            i_back,i_fwd=_index_at_arclen(L,i,s)
            if i_back==i or i_fwd==i: continue
            a=coords[i_back]; b=coords[i]; c=coords[i_fwd]
            ax,ay=a; bx,by=b; cx,cy=c
            v1x,v1y=ax-bx,ay-by; v2x,v2y=cx-bx,cy-by
            n1=math.hypot(v1x,v1y); n2=math.hypot(v2x,v2y)
            if n1==0 or n2==0: ang=180.0; prom=0.0
            else:
                cosang=max(-1.0,min(1.0,(v1x*v2x+v1y*v2y)/(n1*n2)))
                ang=math.degrees(math.acos(cosang))
                vx,vy=cx-ax,cy-ay; vlen=math.hypot(vx,vy)
                prom=0.0 if vlen==0 else abs((bx-ax)*vy-(by-ay)*vx)/(vlen*vlen)
            if ang<min_angle[i]: min_angle[i]=ang
            if prom>max_prom[i]: max_prom[i]=prom
    ccw = Polygon(coords).exterior.is_ccw
    convex_idx=[]; concave_idx=[]
    for i in range(1,n-1):
        a=coords[i-1]; b=coords[i]; c=coords[i+1]
        v1x,v1y=a[0]-b[0],a[1]-b[1]; v2x,v2y=c[0]-b[0],c[1]-b[1]
        cross=v1x*v2y-v1y*v2x; is_concave=(cross<0) if ccw else (cross>0)
        ang=min_angle[i]; prom=max_prom[i]
        if not is_concave:
            if ang<angle_convex_max and prom>=min_prom_convex: convex_idx.append(i)
        else:
            if ang>angle_concave_min and prom>=min_prom_concave: concave_idx.append(i)
    def _dedup(idxs):
        kept=[]; taken=[False]*n
        for i in sorted(idxs, key=lambda j: -max_prom[j]):
            if taken[i]: continue
            kept.append(i); xi,yi=coords[i]
            for j in range(n):
                if taken[j]: continue
                xj,yj=coords[j]
                if math.hypot(xj-xi,yj-yi)<=dedup_m: taken[j]=True
        return kept
    return _dedup(convex_idx), _dedup(concave_idx), coords

def extract_feature_points_bbox(shp_path: Path, bbox_ll_polygon: Polygon, avoid_km=AVOID_KM, simplify_m=1000.0, ANGLE_CONVEX_MAX=170.0, ANGLE_CONCAVE_MIN=210.0, MIN_PROM_CONVEX=0.002, MIN_PROM_CONCAVE=0.002, DEDUP_CONVEX_M=800.0, DEDUP_CONCAVE_M=800.0, ENABLE_UNIFORM=False, TARGET_SPACING_KM=25.0, N_MIN=8, N_MAX=64, PERIM_MIN_KM=20.0, AREA_MIN_KM2=5.0) -> Dict[str, List[Tuple[float,float]]]:
    polys=[]
    with fiona.open(shp_path) as src:
        for feat in src:
            g = shape(feat["geometry"])
            if g.is_empty: continue
            if g.intersects(bbox_ll_polygon):
                gi = g.intersection(bbox_ll_polygon)
                if not gi.is_empty: polys.append(gi)
    if not polys: return {"convex":[], "concave":[], "uniform":[], "convex_peaks":[]}
    union_ll = unary_union(polys); union_m = to_metric(union_ll); buf_m = union_m.buffer(avoid_km * 1000.0)
    polys_m = [buf_m] if buf_m.geom_type=="Polygon" else [p for p in buf_m.geoms if p.geom_type=="Polygon"]
    convex_pts_m, concave_pts_m, uniform_pts_m = [], [], []
    for poly in polys_m:
        perim_km = poly.exterior.length / 1000.0; area_km2 = poly.area / 1e6
        if perim_km < PERIM_MIN_KM or area_km2 < AREA_MIN_KM2: continue
        ring_raw = shapely.LineString(poly.exterior.coords)
        ring_s   = ring_raw.simplify(simplify_m, preserve_topology=False)
        if not ring_s.is_ring: ring_s = shapely.LineString(list(ring_s.coords)+[ring_s.coords[0]])
        convex_idx, concave_idx, coords_used = score_convex_concave_on_ring(ring_s, scales_km=(5,10,20,40), angle_convex_max=ANGLE_CONVEX_MAX, angle_concave_min=ANGLE_CONCAVE_MIN, min_prom_convex=MIN_PROM_CONVEX, min_prom_concave=MIN_PROM_CONCAVE, simplify_before=False, dedup_m=max(DEDUP_CONVEX_M, DEDUP_CONCAVE_M))
        for i in convex_idx:  x,y = coords_used[i]; convex_pts_m.append(Point(x,y))
        for i in concave_idx: x,y = coords_used[i]; concave_pts_m.append(Point(x,y))
        if ENABLE_UNIFORM:
            n_uniform = max(N_MIN, min(N_MAX, int(perim_km / TARGET_SPACING_KM)))
            L = ring_raw.length
            for i in range(n_uniform): uniform_pts_m.append(ring_raw.interpolate(i * L / n_uniform))
    convex_peaks_ll = extract_convex_peaks_from_buffer(union_ll=union_ll, avoid_km=avoid_km)
    def _dedup_points_geom(pts: List[Point], tol_m: float) -> List[Point]:
        out=[]; 
        for p in pts:
            if all(p.distance(q) > tol_m for q in out): out.append(p)
        return out
    convex_pts_m  = _dedup_points_geom(convex_pts_m,  DEDUP_CONVEX_M)
    concave_pts_m = _dedup_points_geom(concave_pts_m, DEDUP_CONCAVE_M)
    def _to_ll_list(pts): return [to_ll(p.x,p.y) for p in pts]
    convex_ll   = [(lon,lat) for (lon,lat) in _to_ll_list(convex_pts_m)]
    concave_ll  = [(lon,lat) for (lon,lat) in _to_ll_list(concave_pts_m)]
    uniform_ll  = []  # disabled
    return {"convex": convex_ll, "concave": concave_ll, "uniform": uniform_ll, "convex_peaks": convex_peaks_ll}

# ---------------- LVS & simplify ----------------
def neighbors_of(u_idx: int, nodes: List[Tuple[float,float]], D_idx: int, k=NEIGHBOR_K) -> List[int]:
    u = nodes[u_idx]; D = nodes[D_idx]; scored=[]
    for v_idx in range(len(nodes)):
        if v_idx == u_idx: continue
        du = gc_distance_km(u, nodes[v_idx]); dD = gc_distance_km(nodes[v_idx], D)
        scored.append((du + 0.5*dD, v_idx))
    scored.sort(key=lambda t:t[0])
    out = [v for _,v in itertools.islice(scored, 0, k)]
    if D_idx not in out: out.append(D_idx)
    return out

def lazy_visibility_search(nodes: List[Tuple[float,float]], O_idx: int, D_idx: int, visible_fn, COLLISION_PREP_M, land_tree: Optional[STRtree], inject_gateways_fn, max_iters: int = 5000, progress: Optional[Dict]=None):
    EDGE_STATE: Dict[Tuple[int,int], str] = {}; adj_cache: Dict[int, List[int]] = {}
    if progress is None: progress = {"iter":0, "candidate_path":[], "free_prefix_len":0, "free_edges":[], "nodes_ref": None}
    progress["nodes_ref"] = nodes
    def get_neighbors(u: int) -> List[int]:
        if u not in adj_cache: adj_cache[u] = neighbors_of(u, nodes, D_idx, k=NEIGHBOR_K)
        return adj_cache[u]
    def a_star() -> Optional[List[int]]:
        N_local=len(nodes); open_heap=[]; INF=1e18; g=[INF]*N_local; parent=[-1]*N_local
        g[O_idx]=0.0; h0 = gc_distance_km(nodes[O_idx], nodes[D_idx]); heapq.heappush(open_heap, (g[O_idx]+h0, O_idx)); closed=set()
        while open_heap:
            _, u = heapq.heappop(open_heap)
            if u in closed: continue
            if u == D_idx:
                path=[u]; 
                while parent[u]!=-1: u=parent[u]; path.append(u)
                return list(reversed(path))
            closed.add(u)
            for v in get_neighbors(u):
                if v>=len(nodes): continue
                if EDGE_STATE.get((u,v))=='BLOCKED': continue
                c = gc_distance_km(nodes[u], nodes[v])
                alt = g[u] + c
                if alt < g[v]:
                    g[v]=alt; parent[v]=u
                    f = alt + gc_distance_km(nodes[v], nodes[D_idx])
                    heapq.heappush(open_heap, (f, v))
        return None
    it=0
    while it<max_iters:
        it+=1; path=a_star()
        if not path: raise RuntimeError("LVS: path not found.")
        progress["iter"]=it; progress["candidate_path"]=list(path); progress["free_prefix_len"]=0
        all_valid=True; prefix_ok=0
        for u, v in zip(path[:-1], path[1:]):
            st = EDGE_STATE.get((u,v)); a=nodes[u]; b=nodes[v]
            if st=='FREE': prefix_ok+=1; progress["free_edges"].append((u,v)); continue
            if st=='BLOCKED': all_valid=False; break
            if visible_fn(a,b, COLLISION_PREP_M, land_tree):
                EDGE_STATE[(u,v)]=EDGE_STATE[(v,u)]='FREE'; prefix_ok+=1; progress["free_edges"].append((u,v))
            else:
                EDGE_STATE[(u,v)]=EDGE_STATE[(v,u)]='BLOCKED'
                new_nodes = inject_gateways_fn(a,b)
                if new_nodes:
                    existing=set((round(lon,5),round(lat,5)) for (lon,lat) in nodes)
                    filtered=[]
                    for q in new_nodes:
                        key=(round(q[0],5),round(q[1],5))
                        if key not in existing: filtered.append(q); existing.add(key)
                    if filtered:
                        nodes.extend(filtered); adj_cache.clear()
                all_valid=False; break
        progress["free_prefix_len"]=prefix_ok
        if all_valid: return path
    raise RuntimeError("LVS: exceeded max iters")

def make_inject_gateways_fn(UNION_M, features_index_ll, take_each=3, inner_ring_m=None, target_boundary_m=None):
    pool = list(features_index_ll.get("convex_peaks", [])) + list(features_index_ll.get("convex", []))
    def f(u_ll, v_ll):
        if not pool: return []
        seg_ll = LineString(geodesic_sample(u_ll, v_ll, step_km=STEP_KM_GEODESIC)); seg_m = to_metric(seg_ll)
        cand=[]
        for (lon,lat) in pool:
            px,py = to_m(lon,lat); p=Point(px,py); cand.append((seg_m.distance(p),(lon,lat)))
        cand.sort(key=lambda t:t[0])
        new=[]
        for _,pt in itertools.islice(cand,0,take_each):
            q,_ = nudge_to_ring_if_inside_fast(pt, inner_ring_m, target_boundary_m)
            new.append(q)
        out=[]; seen=set()
        for lon,lat in new:
            key=(round(lon,5),round(lat,5))
            if key in seen: continue
            seen.add(key); out.append((lon,lat))
        return out
    return f

def simplify_path_gc(path_idx: List[int], nodes: List[Tuple[float,float]], visible_fn, COLLISION_PREP_M, land_tree: Optional[STRtree], max_passes: int = SIMPLIFY_MAX_PASSES) -> List[Tuple[float,float]]:
    if not path_idx or len(path_idx)<2: return [nodes[i] for i in path_idx]
    pts=[nodes[i] for i in path_idx]; passes=0
    while passes<max_passes:
        passes+=1; changed=False; new_pts=[pts[0]]; i=0
        while i < len(pts)-1:
            jumped=False
            for j in range(len(pts)-1, i+1, -1):
                if visible_fn(pts[i], pts[j], COLLISION_PREP_M, land_tree):
                    if j>i+1:
                        new_pts.append(pts[j]); i=j; changed=True; jumped=True; break
            if not jumped:
                new_pts.append(pts[i+1]); i+=1
        if len(new_pts)>=2 and new_pts[-1]==new_pts[-2]: new_pts.pop()
        pts=new_pts
        if not changed: break
    return pts

# ---------------- Public API ----------------
def plan_route(
    origin: Tuple[float,float],
    dest: Tuple[float,float],
    land_path: str | Path,
    out_html: str | Path,
    add_feature_layer: bool = True,
):
    """
    路徑規劃：永遠雙向 (O→D / D→O) → 各自簡化 → 比較簡化後總距離 → 輸出較短者到地圖
    回傳: (waypoints_ll, total_km_simplified, html_path, meta_dict)
    """
    land_path = Path(land_path); out_html = Path(out_html)

    # 1) land & layers
    bboxes = dynamic_bboxes_idl(origin, dest, pad_deg=PAD_DEG)
    polys  = load_polys_in_bboxes(land_path, bboxes)
    if not polys: raise RuntimeError("No land polygons found in bbox")
    layers = build_land_layers(polys)
    UNION_M           = layers["UNION_M"]
    COLLISION_PREP_M  = layers["COLLISION_PREP_M"]
    RING_WGS          = layers["RING_WGS"]
    land_raw_wgs      = layers["LAND_RAW_WGS"]
    LAND_PARTS_M      = layers["LAND_PARTS_M"]
    INNER_RING_M      = layers["RING_M"]
    TARGET_RING_M     = UNION_M.buffer(AVOID_KM * 1000.0)
    TARGET_BOUNDARY_M = TARGET_RING_M.boundary
    land_tree = build_land_strtree(LAND_PARTS_M)

    # 2) O/D nudge
    origin_adj, moved_o = nudge_to_ring_if_inside_fast(origin, INNER_RING_M, TARGET_BOUNDARY_M)
    dest_adj,   moved_d = nudge_to_ring_if_inside_fast(dest,   INNER_RING_M, TARGET_BOUNDARY_M)

    # 3) features
    bbox_union_ll = union_lonlat_bboxes(bboxes)
    feat = extract_feature_points_bbox(land_path, bbox_union_ll, avoid_km=AVOID_KM)
    feature_nodes = list(feat["convex_peaks"]) + list(feat["convex"])

    # 4) nodes
    base_nodes = [origin_adj, dest_adj] + feature_nodes
    nodes = [nudge_to_ring_if_inside_fast(p, INNER_RING_M, TARGET_BOUNDARY_M)[0] for p in base_nodes]
    if len(nodes) > LVS_MAX_NODES:
        nodes = nodes[:LVS_MAX_NODES]
    O_idx, D_idx = 0, 1

    # 5) inject / visible
    inject_fn = make_inject_gateways_fn(
        UNION_M,
        {"convex_peaks": feat["convex_peaks"], "convex": feat["convex"]},
        take_each=3,
        inner_ring_m=INNER_RING_M,
        target_boundary_m=TARGET_BOUNDARY_M
    )
    visible_wrapper = lambda a,b,cp,tree: visible(a,b,cp,tree)

    # 6) bidirectional LVS (tolerate one-sided failure) —— 不再使用 run_once
    results = []  # tuples: ("fwd"/"rev", path_idx, nodes_list)
    err_fwd = err_rev = None

    # forward O→D
    try:
        nodes_fwd = list(nodes)
        path_fwd = lazy_visibility_search(
            nodes_fwd, O_idx, D_idx, visible_wrapper, COLLISION_PREP_M, land_tree, inject_fn,
            max_iters=5000, progress=None
        )
        results.append(("fwd", path_fwd, nodes_fwd))
    except Exception as e:
        err_fwd = e

    # reverse D→O; swap 0/1 then reverse result to O→D
    try:
        nodes_do = list(nodes)
        nodes_do[0], nodes_do[1] = nodes_do[1], nodes_do[0]
        path_rev_do = lazy_visibility_search(
            nodes_do, 0, 1, visible_wrapper, COLLISION_PREP_M, land_tree, inject_fn,
            max_iters=5000, progress=None
        )
        path_rev = list(reversed(path_rev_do))
        results.append(("rev", path_rev, nodes_do))
    except Exception as e:
        err_rev = e

    if not results:
        raise RuntimeError(f"LVS failed in both directions. fwd={err_fwd}, rev={err_rev}")

    # 7) simplify any successful result(s)
    cand = []
    for tag, p_idx, node_list in results:
        simp = simplify_path_gc(
            p_idx, node_list, visible_wrapper, COLLISION_PREP_M, land_tree,
            max_passes=SIMPLIFY_MAX_PASSES
        )
        cand.append((tag, p_idx, node_list, simp))

    # 8) distance helper
    def total_km_after_simplify(simplified_pts):
        segs = []
        if moved_o and origin != origin_adj:
            segs.append((origin, origin_adj))
        if simplified_pts and len(simplified_pts) >= 2:
            segs.extend(zip(simplified_pts[:-1], simplified_pts[1:]))
        if moved_d and dest_adj != dest:
            segs.append((dest_adj, dest))
        return sum(gc_distance_km(a, b) for (a, b) in segs)

    # 9) choose shorter among the successful ones
    best = None
    for tag, p_idx, node_list, simp in cand:
        dist = total_km_after_simplify(simp)
        if (best is None) or (dist < best[0]):
            best = (dist, tag, p_idx, node_list, simp)

    total_simple_best, tag_best, chosen_path_idx, chosen_nodes, chosen_simplified = best
    label = ("O→D（較短）" if tag_best == "fwd" else "D→O（較短）")

    # 10) draw map
    mid_lon, mid_lat = great_circle_midpoint(origin, dest)
    center_lon_pacific = normalize_lon_to_pacific_view(mid_lon)
    m = folium.Map(
        location=[mid_lat, center_lon_pacific],
        zoom_start=3, max_bounds=False, world_copy_jump=False, no_wrap=False, min_lon=0, max_lon=360
    )
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Ocean_Basemap/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Ocean Basemap', overlay=False, control=True, no_wrap=False
    ).add_to(m)
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='© OpenStreetMap', name='OpenStreetMap', overlay=False, control=True, no_wrap=False
    ).add_to(m)
    folium.GeoJson(_to_pacific(land_raw_wgs), name="陸地",
                   style_function=lambda x: {"color":"#2ca02c","weight":1,"fillOpacity":0.15}).add_to(m)
    folium.GeoJson(_to_pacific(RING_WGS), name=f"航道緩衝區 {BUFFER_KM}km",
                   style_function=lambda x: {"color":"#6a5acd","weight":2,"fillOpacity":0.05}).add_to(m)

    # markers
    folium.Marker([origin[1], normalize_lon_to_pacific_view(origin[0])],
                  tooltip=f"起點: ({origin[0]:.2f}, {origin[1]:.2f})",
                  icon=folium.Icon(color='green', icon='ship', prefix='fa')).add_to(m)
    folium.Marker([dest[1], normalize_lon_to_pacific_view(dest[0])],
                  tooltip=f"終點: ({dest[0]:.2f}, {dest[1]:.2f})",
                  icon=folium.Icon(color='red', icon='anchor', prefix='fa')).add_to(m)

    # reference GC
    draw_gc_polyline_continuous(m, origin, dest, step_km=80.0,
                                color='gray', weight=2, opacity=0.4, dash_array="8,4")

    # features (optional)
    if add_feature_layer:
        fg_feat = folium.FeatureGroup(name="候選特徵點（凸峰+凸點）", show=False)
        for (lon, lat) in (list(feat["convex_peaks"]) + list(feat["convex"])):
            folium.CircleMarker(
                [lat, normalize_lon_to_pacific_view(lon)], radius=3,
                color="#1f77b4", fill=True, fill_opacity=0.8,
                tooltip=f"Feature ({lon:.3f},{lat:.3f})"
            ).add_to(fg_feat)
        fg_feat.add_to(m)

    # chosen original (blue) —— 原始選路徑（含接駁），供比對
    final_segments = []
    if moved_o and origin != origin_adj:
        final_segments.append((origin, origin_adj))
        draw_gc_polyline_continuous(m, origin, origin_adj, step_km=DRAW_STEP_KM,
                                    color='#1f77b4', weight=5, opacity=0.9)
    for u, v in zip(chosen_path_idx[:-1], chosen_path_idx[1:]):
        a = chosen_nodes[u]; b = chosen_nodes[v]
        final_segments.append((a, b))
        draw_gc_polyline_continuous(m, a, b, step_km=DRAW_STEP_KM,
                                    color='#1f77b4', weight=5, opacity=0.9)
    if moved_d and dest_adj != dest:
        final_segments.append((dest_adj, dest))
        draw_gc_polyline_continuous(m, dest_adj, dest, step_km=DRAW_STEP_KM,
                                    color='#1f77b4', weight=5, opacity=0.9)

    # simplified overlay (red) —— 對外應用的最終軌跡
    if chosen_simplified and len(chosen_simplified) >= 2:
        fg_simplified = folium.FeatureGroup(name="簡化後航線 (可視直連)", show=True)
        if moved_o and origin != origin_adj:
            draw_gc_polyline_continuous(fg_simplified, origin, origin_adj, step_km=DRAW_STEP_KM,
                                        color='#d62728', weight=4, opacity=0.8, dash_array="6,4")
        for a, b in zip(chosen_simplified[:-1], chosen_simplified[1:]):
            draw_gc_polyline_continuous(fg_simplified, a, b, step_km=DRAW_STEP_KM,
                                        color='#d62728', weight=6, opacity=0.9)
        if moved_d and dest_adj != dest:
            draw_gc_polyline_continuous(fg_simplified, dest_adj, dest, step_km=DRAW_STEP_KM,
                                        color='#d62728', weight=4, opacity=0.8, dash_array="6,4")
        fg_simplified.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)

    # totals
    total_km_original = sum(gc_distance_km(a, b) for (a, b) in final_segments)
    def _total_simplified(chosen_simplified):
        segs = []
        if moved_o and origin != origin_adj:
            segs.append((origin, origin_adj))
        if chosen_simplified and len(chosen_simplified) >= 2:
            segs += list(zip(chosen_simplified[:-1], chosen_simplified[1:]))
        if moved_d and dest_adj != dest:
            segs.append((dest_adj, dest))
        return sum(gc_distance_km(a, b) for (a, b) in segs)
    total_km_simplified = _total_simplified(chosen_simplified)

    meta = {
        "label": label,
        "total_km_original": total_km_original,
        "total_km_simplified": total_km_simplified,
        "delta_km": total_km_original - total_km_simplified,
        "moved_o": moved_o, "moved_d": moved_d,
        "origin_adj": origin_adj, "dest_adj": dest_adj,
        "feature_count": len(feature_nodes),
    }

    # --- 軌跡輸出 ---
    # 原始選路徑（含港口/外推接駁）
    track_ll = [origin]
    if moved_o and origin != origin_adj:
        track_ll.append(origin_adj)
    for idx in chosen_path_idx:
        p = chosen_nodes[idx]
        if not track_ll or p != track_ll[-1]:
            track_ll.append(p)
    if moved_d and dest_adj != dest:
        if not track_ll or track_ll[-1] != dest_adj:
            track_ll.append(dest_adj)
        track_ll.append(dest)
    else:
        if not track_ll or track_ll[-1] != dest:
            track_ll.append(dest)

    # 簡化後最終軌跡（含港口/外推接駁）——建議對外使用
    track_simplified_ll = [origin]
    if moved_o and origin != origin_adj:
        if track_simplified_ll[-1] != origin_adj:
            track_simplified_ll.append(origin_adj)
    for p in (chosen_simplified or []):
        if not track_simplified_ll or p != track_simplified_ll[-1]:
            track_simplified_ll.append(p)
    if moved_d and dest_adj != dest:
        if not track_simplified_ll or track_simplified_ll[-1] != dest_adj:
            track_simplified_ll.append(dest_adj)
        track_simplified_ll.append(dest)
    else:
        if not track_simplified_ll or track_simplified_ll[-1] != dest:
            track_simplified_ll.append(dest)

    meta["track_ll"] = track_ll
    meta["track_simplified_ll"] = track_simplified_ll

    return chosen_simplified, total_km_simplified, str(out_html), meta


def _to_pacific(geom):
    from shapely.geometry import mapping
    geom_dict = mapping(geom)
    def _convert(coords):
        if isinstance(coords[0], (list, tuple)): return [_convert(c) for c in coords]
        lon,lat = coords[0], coords[1]
        return [normalize_lon_to_pacific_view(lon), lat]
    if geom_dict['type'] == 'Polygon':
        geom_dict['coordinates'] = [_convert(ring) for ring in geom_dict['coordinates']]
    elif geom_dict['type'] == 'MultiPolygon':
        geom_dict['coordinates'] = [[_convert(ring) for ring in poly] for poly in geom_dict['coordinates']]
    return geom_dict
