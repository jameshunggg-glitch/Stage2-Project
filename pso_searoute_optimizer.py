"""
Visibility + 陸地特徵點繞路 + 簡化版海上路徑優化（前 5 條航線）

流程：
1. 起迄點 nudge 到海面（若在陸地上）
2. searoute 生歷史骨幹路徑 γ_sr
3. 4.1 風格修補器處理小穿陸 → γ0
4. 對仍然撞陸的 segments[Pi, Pj]：
   - 在 bbox 建 AOI
   - 取陸地特徵點（優先用 routing.features.extract_feature_points_bbox）
   - 建 local visibility graph + Dijkstra A→B → 替換原段 → γ1
5. 對 γ1 做 visibility shortcut 全線簡化 → γ2
6. 用 folium 繪圖：
   - 藍：searoute
   - 橘：簡化後路徑 γ2
   - 紅：若 γ2 仍有撞陸 segment，畫紅線做 debug

依賴套件：
    pip install pandas geopandas shapely searoute folium
（若要用你的 routing.features，請確保 routing 套件在 PYTHONPATH 內）
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import folium
import geopandas as gpd
import pandas as pd
import searoute as sr
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree
import webbrowser

# 嘗試匯入你原本的陸地特徵點函式
try:
    from routing.features import extract_feature_points_bbox  # type: ignore
    HAS_FEATURES = True
    print("[info] 成功匯入 routing.features.extract_feature_points_bbox")
except Exception as e:  # noqa: F841
    HAS_FEATURES = False
    print("[warn] 無法匯入 routing.features，將使用簡化版 AOI 節點。")

LonLat = Tuple[float, float]  # (lon, lat)


# ========================= 幾何工具 =========================

def haversine_km(lon1, lat1, lon2, lat2) -> float:
    """計算兩點之間的球面距離（公里）。"""
    R = 6371.0  # 地球半徑 km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return R * c


def polyline_length_km(coords: Sequence[LonLat]) -> float:
    """計算一條 polyline 的總長度（公里）。"""
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coords[:-1], coords[1:]):
        total += haversine_km(lon1, lat1, lon2, lat2)
    return total


# ========================= LandMask 與基本檢查 =========================

@dataclass
class LandMask:
    """
    包裝 shapely 的 STRtree，方便做「segment 是否與陸地相交」與「點是否在陸地內」檢查。
    land_geoms: List[Polygon/MultiPolygon]
    """
    land_geoms: List
    land_index: STRtree

    @classmethod
    def from_geoms(cls, land_geoms: List):
        return cls(land_geoms=land_geoms, land_index=STRtree(land_geoms))

    def segment_intersects_land(self, p1: LonLat, p2: LonLat) -> bool:
        line = LineString([p1, p2])
        idxs = self.land_index.query(line)
        for idx in idxs:
            geom = self.land_geoms[int(idx)]
            if line.intersects(geom):
                return True
        return False

    def point_in_land(self, p: LonLat) -> bool:
        pt = Point(p)
        idxs = self.land_index.query(pt)
        for idx in idxs:
            geom = self.land_geoms[int(idx)]
            if pt.within(geom):
                return True
        return False


def polyline_has_land_intersection(coords: Sequence[LonLat], landmask: LandMask) -> bool:
    """檢查整條 polyline 是否有任何 segment 與陸地相交。"""
    if len(coords) < 2:
        return False
    for a, b in zip(coords[:-1], coords[1:]):
        if landmask.segment_intersects_land(a, b):
            return True
    return False


# ========================= 起迄點 nudge =========================

def nudge_point_off_land(
    p: LonLat,
    landmask: LandMask,
    max_steps: int = 24,
    step_deg: float = 0.05,
) -> Tuple[LonLat, bool]:
    """
    若點在陸地上，嘗試以放射狀往外海推，直到不在陸地上為止。
    回傳 (新座標, 是否有做 nudge)。
    """
    if not landmask.point_in_land(p):
        return p, False

    lon0, lat0 = p
    angles = [2 * math.pi * k / 16.0 for k in range(16)]  # 16 方向

    for r_step in range(1, max_steps + 1):
        r = step_deg * r_step
        for ang in angles:
            cand = (lon0 + r * math.cos(ang), lat0 + r * math.sin(ang))
            if not landmask.point_in_land(cand):
                return cand, True

    # 找不到就回傳原點（理論上不太會）
    return p, False


# ========================= 4.1 風格修補器（小穿陸） =========================

def repair_polyline(
    coords: Sequence[LonLat],
    landmask: LandMask,
    max_outer_iters: int = 20,
    offset_step_deg: float = 0.2,
) -> Tuple[List[LonLat], bool]:
    """
    改良版 4.1 修補器，用來修「輕微穿陸」：
    - 對每個撞陸 segment [a, b]：
      1) 取中點 m
      2) 兩個法向方向 n1, n2
      3) scale ∈ {1,2,4} * offset_step_deg
      4) 找到 cand = m + scale*n，且：
         * cand 不在陸地
         * [a, cand]、[cand, b] 都不撞陸
    - 某輪若沒有任何 segment 被替換 → 視為收斂 (True)
    - 跑到 max_outer_iters 還在修 → 回傳目前結果, False
    """
    if len(coords) < 2:
        return list(coords), True

    curr = list(coords)

    for _ in range(max_outer_iters):
        changed = False
        new_coords: List[LonLat] = [curr[0]]

        for i in range(len(curr) - 1):
            a = new_coords[-1]
            b = curr[i + 1]

            # 不撞陸就照抄
            if not landmask.segment_intersects_land(a, b):
                new_coords.append(b)
                continue

            changed = True

            ax, ay = a
            bx, by = b
            mx = (ax + bx) / 2.0
            my = (ay + by) / 2.0

            dx = bx - ax
            dy = by - ay
            seg_len = math.hypot(dx, dy)
            if seg_len == 0:
                new_coords.append(b)
                continue

            nx1, ny1 = -dy / seg_len, dx / seg_len
            nx2, ny2 = dy / seg_len, -dx / seg_len

            dirs = [(nx1, ny1), (nx2, ny2)]
            m_prime: Optional[LonLat] = None

            for nx, ny in dirs:
                for scale in (1.0, 2.0, 4.0):
                    cand = (
                        mx + nx * offset_step_deg * scale,
                        my + ny * offset_step_deg * scale,
                    )
                    if landmask.point_in_land(cand):
                        continue
                    if landmask.segment_intersects_land(a, cand):
                        continue
                    if landmask.segment_intersects_land(cand, b):
                        continue
                    m_prime = cand
                    break
                if m_prime is not None:
                    break

            if m_prime is None:
                new_coords.append(b)
            else:
                new_coords.append(m_prime)
                new_coords.append(b)

        curr = new_coords
        if not changed:
            return curr, True

    return curr, False


# ========================= Hard segments：local visibility detour =========================

@dataclass
class DetourResult:
    coords: List[LonLat]
    all_fixed: bool


def _dijkstra_shortest_path(
    nodes: List[LonLat],
    landmask: LandMask,
    start_idx: int,
    end_idx: int
) -> Optional[List[int]]:
    """
    在 small visibility graph 上跑 Dijkstra，回傳節點 index path。
    nodes: List[(lon, lat)]
    """
    import heapq

    n = len(nodes)
    if n < 2:
        return None

    # 建 adjacency matrix：O(n^2) 小圖沒關係
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not landmask.segment_intersects_land(nodes[i], nodes[j]):
                d = haversine_km(nodes[i][0], nodes[i][1], nodes[j][0], nodes[j][1])
                adj[i].append((j, d))
                adj[j].append((i, d))

    INF = 1e18
    dist = [INF] * n
    prev = [-1] * n
    dist[start_idx] = 0.0

    pq: List[Tuple[float, int]] = [(0.0, start_idx)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == end_idx:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dist[end_idx] >= INF:
        return None

    # 回溯 path
    path_idx: List[int] = []
    cur = end_idx
    while cur != -1:
        path_idx.append(cur)
        cur = prev[cur]
    path_idx.reverse()
    return path_idx


def _collect_feature_nodes_for_segment(
    a: LonLat,
    b: LonLat,
    landmask: LandMask,
    land_shp_path: str,
    margin_deg: float = 2.0,
    max_nodes: int = 80,
) -> List[LonLat]:
    """
    建立某一撞陸 segment 的 AOI 節點：
    - a, b 一定包含
    - 若可用 routing.features，就取 convex/concave/convex_peaks
    - 若不能，就用 bbox 四角 + 中點們
    """
    min_lon = min(a[0], b[0]) - margin_deg
    max_lon = max(a[0], b[0]) + margin_deg
    min_lat = min(a[1], b[1]) - margin_deg
    max_lat = max(a[1], b[1]) + margin_deg

    bbox_poly = Polygon(
        [
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
        ]
    )

    nodes: List[LonLat] = [a, b]

    if HAS_FEATURES:
        try:
            feats = extract_feature_points_bbox(
                land_shp_path,
                bbox_poly,
                ENABLE_UNIFORM=False,
            )
            cand = (
                feats.get("convex", [])
                + feats.get("concave", [])
                + feats.get("convex_peaks", [])
                + feats.get("uniform", [])
            )
            # 過濾掉在 bbox 外或在陸上的點
            filtered: List[LonLat] = []
            for lon, lat in cand:
                if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                    continue
                if landmask.point_in_land((lon, lat)):
                    continue
                filtered.append((lon, lat))
            # 限制最多 max_nodes-2 個
            if len(filtered) > max_nodes - 2:
                step = max(1, len(filtered) // (max_nodes - 2))
                filtered = filtered[::step]
            nodes.extend(filtered)
        except Exception as e:  # noqa: F841
            # 若 features 裡面出錯，就用簡化版
            pass

    if len(nodes) <= 2:
        # fallback：用 bbox 四角 + 中線幾個點
        corners = [
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
        ]
        for c in corners:
            if not landmask.point_in_land(c):
                nodes.append(c)

        # 中線若干點
        for t in [0.25, 0.5, 0.75]:
            lon = a[0] + t * (b[0] - a[0])
            lat = a[1] + t * (b[1] - a[1])
            nodes.append((lon, lat))

    # 去重
    uniq: List[LonLat] = []
    seen = set()
    for lon, lat in nodes:
        key = (round(lon, 4), round(lat, 4))
        if key not in seen:
            seen.add(key)
            uniq.append((lon, lat))
    return uniq


def detour_hard_segments(
    coords: Sequence[LonLat],
    landmask: LandMask,
    land_shp_path: str,
) -> DetourResult:
    """
    對仍撞陸的 segments，用 local visibility graph + Dijkstra 做繞路。
    回傳 (新 polyline, all_fixed)
    """
    if len(coords) < 2:
        return DetourResult(coords=list(coords), all_fixed=True)

    new_coords: List[LonLat] = [coords[0]]
    all_fixed = True

    for i in range(len(coords) - 1):
        a = new_coords[-1]
        b = coords[i + 1]

        if not landmask.segment_intersects_land(a, b):
            new_coords.append(b)
            continue

        # 為這一段 [a, b] 建 AOI 節點 + visibility graph
        nodes = _collect_feature_nodes_for_segment(a, b, landmask, land_shp_path)
        # 確保 a/b 在 nodes[0], nodes[1]
        nodes[0] = a
        nodes[1] = b

        path_idx = _dijkstra_shortest_path(nodes, landmask, 0, 1)
        if path_idx is None or len(path_idx) <= 2:
            # 找不到繞路 → 這一段就留原本的 b，但標記為沒有完全修好
            all_fixed = False
            new_coords.append(b)
        else:
            # 轉成座標，發現第一個點就是 a，所以從 1 開始加進來避免重複
            for k in path_idx[1:]:
                new_coords.append(nodes[k])

    return DetourResult(coords=new_coords, all_fixed=all_fixed)


# ========================= 全線 visibility shortcut 簡化 =========================

def simplify_polyline_visibility(
    coords: Sequence[LonLat],
    landmask: LandMask,
) -> List[LonLat]:
    """
    使用「只要 segment(Pi, Pj) 不撞陸，就可以直接連接」的 shortcut 簡化。
    結果會是保證不撞陸的前提下，刪掉中間多餘折點。
    """
    if len(coords) <= 2:
        return list(coords)

    simplified: List[LonLat] = [coords[0]]
    i = 0
    n = len(coords) - 1

    while i < n:
        # 找最遠可視點
        last_visible = i + 1
        j = i + 1
        while j <= n:
            if not landmask.segment_intersects_land(coords[i], coords[j]):
                last_visible = j
                j += 1
            else:
                break
        simplified.append(coords[last_visible])
        i = last_visible

    return simplified


# ========================= searoute 包裝 =========================

def searoute_path(start: LonLat, end: LonLat) -> Optional[List[LonLat]]:
    """
    呼叫 searoute.searoute，回傳 [(lon, lat), ...] 或 None。
    """
    try:
        feature = sr.searoute(list(start), list(end), units="km")
        if feature is None:
            return None
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [])
        if not coords:
            return None
        return [(float(lon), float(lat)) for lon, lat in coords]
    except Exception as e:
        print(f"[searoute error] {e!r}")
        return None


# ========================= I/O 與視覺化 =========================

PORTS_FILE = r"C:\Users\slab\Desktop\Slab Project\Stage1\data\ports70.csv"
ROUTES_FILE = r"C:\Users\slab\Desktop\Slab Project\Stage1\data\port2port_70.csv"
LAND_SHP = r"C:\Users\slab\Desktop\Slab Project\Stage1\data\Land\ne_10m_land.shp"

PORT_CODE_COL = "locode"
PORT_LON_COL = "lon"
PORT_LAT_COL = "lat"

ROUTE_ORIGIN_COL = "From_code"
ROUTE_DEST_COL = "To_code"


def build_port_lookup() -> Dict[str, LonLat]:
    df_ports = pd.read_csv(PORTS_FILE)
    df_ports = df_ports.dropna(subset=[PORT_CODE_COL, PORT_LON_COL, PORT_LAT_COL])
    df_ports = df_ports.drop_duplicates(subset=[PORT_CODE_COL], keep="first")

    port_lookup: Dict[str, LonLat] = {}
    for _, row in df_ports.iterrows():
        code = str(row[PORT_CODE_COL]).strip()
        lon = float(row[PORT_LON_COL])
        lat = float(row[PORT_LAT_COL])
        port_lookup[code] = (lon, lat)

    print(f"[info] 建立港口查詢表，共 {len(port_lookup)} 個港口。")
    return port_lookup


def load_land_geoms_from_shp(shp_path: str) -> List:
    print(f"[info] 讀取陸地 shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path)
    geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    print(f"[info] 取得 {len(geoms)} 個 land geometry。")
    return geoms


def make_route_map(
    start: LonLat,
    end: LonLat,
    sr_coords: Sequence[LonLat],
    final_coords: Sequence[LonLat],
    landmask: LandMask,
    origin_code: str,
    dest_code: str,
    idx: int,
):
    """
    用 folium 畫出：
        - searoute 原始路徑（藍）
        - final 簡化路徑（橘）
        - 撞陸段（紅）
    """
    sr_latlon = [(lat, lon) for lon, lat in sr_coords]
    final_latlon = [(lat, lon) for lon, lat in final_coords]

    all_pts = list(sr_coords) + list(final_coords)
    avg_lat = sum(lat for lon, lat in all_pts) / len(all_pts)
    avg_lon = sum(lon for lon, lat in all_pts) / len(all_pts)

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=4, tiles="OpenStreetMap")

    folium.Marker(
        [start[1], start[0]],
        popup=f"Start {origin_code}",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)
    folium.Marker(
        [end[1], end[0]],
        popup=f"End {dest_code}",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    fg_sr = folium.FeatureGroup(name="searoute")
    fg_final = folium.FeatureGroup(name="optimized (visibility)")
    fg_collide = folium.FeatureGroup(name="collision segments (debug)")

    folium.PolyLine(sr_latlon, weight=3, color="blue", opacity=0.6).add_to(fg_sr)
    folium.PolyLine(final_latlon, weight=4, color="orange", opacity=0.8).add_to(fg_final)

    # 如果 final 還有撞陸段，畫紅線
    for a, b in zip(final_coords[:-1], final_coords[1:]):
        if landmask.segment_intersects_land(a, b):
            folium.PolyLine(
                [(a[1], a[0]), (b[1], b[0])],
                weight=5,
                color="red",
                opacity=0.9,
            ).add_to(fg_collide)

    fg_sr.add_to(m)
    fg_final.add_to(m)
    fg_collide.add_to(m)

    folium.LayerControl().add_to(m)

    outfile = Path(f"route_vis_{idx}_{origin_code}_{dest_code}.html")
    m.save(str(outfile))
    print(f"  已輸出地圖到: {outfile}")
    try:
        webbrowser.open(outfile.resolve().as_uri())
        print("  已嘗試在預設瀏覽器開啟地圖。")
    except Exception as e:  # noqa: F841
        print("  開啟瀏覽器失敗，請手動點選 HTML 檔。")

    return outfile


# ========================= 主程式 =========================

def main():
    # landmask
    land_geoms = load_land_geoms_from_shp(LAND_SHP)
    if not land_geoms:
        raise RuntimeError("無法從 shapefile 載入任何陸地幾何。")
    landmask = LandMask.from_geoms(land_geoms)

    port_lookup = build_port_lookup()
    df_routes = pd.read_csv(ROUTES_FILE).head(5)  # 只跑前五條

    records = []

    for idx, row in df_routes.iterrows():
        origin_code = str(row[ROUTE_ORIGIN_COL]).strip()
        dest_code = str(row[ROUTE_DEST_COL]).strip()
        key = f"{origin_code}->{dest_code}"

        print("=" * 80)
        print(f"[Route {idx}] {key}")

        if origin_code not in port_lookup or dest_code not in port_lookup:
            print("  [SKIP] 港口座標缺失")
            continue

        start_raw = port_lookup[origin_code]
        end_raw = port_lookup[dest_code]

        # 起迄點 nudge
        start, nudged_s = nudge_point_off_land(start_raw, landmask)
        end, nudged_e = nudge_point_off_land(end_raw, landmask)
        if nudged_s:
            print("  [nudge] 起點在陸地上，已外推到海面。")
        if nudged_e:
            print("  [nudge] 終點在陸地上，已外推到海面。")

        # 1) searoute
        print("  1) 呼叫 searoute...")
        sr_coords = searoute_path(start, end)
        if not sr_coords:
            print("  searoute 失敗，跳過此航線。")
            continue
        len_sr = polyline_length_km(sr_coords)
        print(f"  searoute points: {len(sr_coords)}, length = {len_sr:.1f} km")

        # 2) 修補小穿陸
        print("  2) 修補 searoute 路徑（4.1 風格）...")
        repaired, converged = repair_polyline(
            sr_coords,
            landmask,
            max_outer_iters=25,
            offset_step_deg=0.25,
        )
        len_rep = polyline_length_km(repaired)
        print(f"  repaired length = {len_rep:.1f} km, converged = {converged}")
        print(f"  repaired has collision? {polyline_has_land_intersection(repaired, landmask)}")

        # 3) 對 hard segments 做陸地節點繞路
        print("  3) 對仍撞陸 segments 用 local visibility 繞路...")
        detour_res = detour_hard_segments(repaired, landmask, LAND_SHP)
        len_detour = polyline_length_km(detour_res.coords)
        print(f"  detour length = {len_detour:.1f} km, all_fixed = {detour_res.all_fixed}")
        print(f"  detour has collision? {polyline_has_land_intersection(detour_res.coords, landmask)}")

        # 4) 全線 shortcut 簡化
        print("  4) visibility shortcut 簡化整條路徑...")
        simplified = simplify_polyline_visibility(detour_res.coords, landmask)
        len_simplified = polyline_length_km(simplified)
        has_collision_final = polyline_has_land_intersection(simplified, landmask)
        print(f"  simplified length = {len_simplified:.1f} km")
        print(f"  simplified has collision? {has_collision_final}")

        # 5) folium 地圖
        print("  5) 繪製地圖（searoute vs visibility-route）...")
        make_route_map(
            start=start,
            end=end,
            sr_coords=sr_coords,
            final_coords=simplified,
            landmask=landmask,
            origin_code=origin_code,
            dest_code=dest_code,
            idx=idx,
        )

        records.append(
            {
                "index": idx,
                "origin": origin_code,
                "dest": dest_code,
                "len_searoute_km": len_sr,
                "len_repaired_km": len_rep,
                "len_detour_km": len_detour,
                "len_simplified_km": len_simplified,
                "repaired_collide": polyline_has_land_intersection(repaired, landmask),
                "detour_all_fixed": detour_res.all_fixed,
                "simplified_collide": has_collision_final,
            }
        )

    df_out = pd.DataFrame(records)
    df_out.to_csv("searoute_5_vis_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== 完成前 5 條航線 visibility 優化 ===")
    print(df_out)


if __name__ == "__main__":
    main()
