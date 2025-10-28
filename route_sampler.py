# route_sampler.py  (drop-in replacement)
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from pyproj import Geod
import bisect

def _parse_start_time(ts: Optional[str | datetime]) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    # simple ISO8601 parser (assume Z or naive -> UTC)
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts[:-1]).replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        raise ValueError("start_time must be ISO8601 string (e.g., '2025-10-27T08:00:00Z') or datetime")

def sample_along_route(
    waypoints_ll: List[Tuple[float,float]],
    *,
    speed_kmh: Optional[float] = None,
    speed_knots: Optional[float] = None,
    every_h: float = 1.0,
    start_time: Optional[str | datetime] = None,
    ellps: str = "WGS84",
    include_waypoints: bool = True,
    snap_km: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    對由 (lon,lat) 航點組成的路徑進行「等速」取樣，並可確保全部航點也被取到。

    參數：
      - waypoints_ll: [(lon,lat), ...] 至少 2 點（建議用「簡化後航點」）
      - speed_kmh / speed_knots: 二擇一（km/h 或 節）；內部統一轉成 km/h
      - every_h: 等時間取樣間隔（小時）
      - start_time: ISO8601 字串或 datetime（可省略；若給，輸出含時間戳）
      - ellps: 大地基準，預設 WGS84
      - include_waypoints: 若 True，強制把所有航點插入取樣序列
      - snap_km: 若某等時間節點距某航點的弧長差 < snap_km，則吸附到該航點

    回傳：
      List[dict]，每個元素至少包含：
        {"idx":int, "time":ISO8601|None, "lon":float, "lat":float,
         "cum_km":float, "seg_idx":int, "kind": "grid"|"waypoint"|"snap"}
    """
    if not waypoints_ll or len(waypoints_ll) < 2:
        raise ValueError("waypoints_ll must contain at least two points")
    if (speed_kmh is None) == (speed_knots is None):
        raise ValueError("Provide exactly one of speed_kmh or speed_knots")
    v_kmh = speed_kmh if speed_kmh is not None else speed_knots * 1.852
    if v_kmh <= 0:
        raise ValueError("speed must be positive")
    if every_h <= 0:
        raise ValueError("every_h must be positive")

    geod = Geod(ellps=ellps)

    # 1) 預先算每段長度與從該段起點的 forward azimuth
    seg_len_km: List[float] = []
    fwd_az: List[float] = []
    for (lon1,lat1),(lon2,lat2) in zip(waypoints_ll[:-1], waypoints_ll[1:]):
        az12, az21, dist_m = geod.inv(lon1,lat1,lon2,lat2)
        seg_len_km.append(dist_m/1000.0)
        fwd_az.append(az12)

    # 累積長度 C[i] = 從起點到第 i 個航點的總長（km）
    C = [0.0]
    for L in seg_len_km:
        C.append(C[-1] + L)
    Ltot = C[-1]

    # 2) 生成「等時間格網」的距離序列（km）
    step_km = v_kmh * every_h
    grid_distances: List[float] = []
    n = 0
    while True:
        S = n * step_km
        if S >= Ltot:
            grid_distances.append(Ltot)  # 確保最終終點
            break
        grid_distances.append(S)
        n += 1

    # 3) 準備「航點距離」序列（km），含起終點
    waypoint_distances: List[float] = C[:] if include_waypoints else [0.0, Ltot]

    # 4) 對等時間節點做「吸附」至最近的航點（距離差 < snap_km）
    #    吸附後的節點標記為 "snap"，未吸附則為 "grid"
    combined: List[Tuple[float, str, Optional[int]]] = []  # (S_km, kind, waypoint_index)
    if include_waypoints:
        # 用集合快速查找最近航點：這裡用二分在 C 上找鄰近
        for S in grid_distances:
            # 在 C 中找 S 的插入點
            j = bisect.bisect_left(waypoint_distances, S)
            nearest_idx = None
            nearest_diff = float("inf")
            # 檢查左、右鄰
            for cand in (j-1, j):
                if 0 <= cand < len(waypoint_distances):
                    diff = abs(waypoint_distances[cand] - S)
                    if diff < nearest_diff:
                        nearest_diff = diff
                        nearest_idx = cand
            if nearest_idx is not None and nearest_diff <= snap_km:
                combined.append((waypoint_distances[nearest_idx], "snap", nearest_idx))
            else:
                combined.append((S, "grid", None))
        # 再把所有航點本身也加入
        for idx, S_wp in enumerate(waypoint_distances):
            combined.append((S_wp, "waypoint", idx))
    else:
        # 不加入航點，只用格網
        for S in grid_distances:
            combined.append((S, "grid", None))

    # 5) 依距離排序 + 去重（距離容差）
    combined.sort(key=lambda t: t[0])
    deduped: List[Tuple[float, str, Optional[int]]] = []
    # 允許非常小的距離容差（公分級），避免重複點
    EPS_KM = 1e-6  # 1e-6 km = 1 mm
    for S, kind, idx_wp in combined:
        if not deduped:
            deduped.append((S, kind, idx_wp))
            continue
        S_prev, kind_prev, idx_prev = deduped[-1]
        if abs(S - S_prev) <= EPS_KM:
            # 同一距離的點只保留「優先級」較高的類型
            # 優先順序：snap > waypoint > grid
            pri = {"snap": 3, "waypoint": 2, "grid": 1}
            keep_cur = pri.get(kind, 0) > pri.get(kind_prev, 0)
            if keep_cur:
                deduped[-1] = (S, kind, idx_wp)
        else:
            deduped.append((S, kind, idx_wp))

    # 6) 將每個距離 S 轉成點位（正算），並附上時間/屬性
    t0 = _parse_start_time(start_time)
    out: List[Dict[str, Any]] = []
    for i, (S, kind, idx_wp) in enumerate(deduped):
        # 邊界：Ltot -> 直接終點
        if S >= Ltot:
            lon, lat = waypoints_ll[-1]
            cum_km = Ltot
            time_iso = (t0 + timedelta(hours=(cum_km/v_kmh))).isoformat().replace("+00:00","Z") if t0 else None
            out.append({
                "idx": i,
                "time": time_iso,
                "lon": lon, "lat": lat,
                "cum_km": cum_km,
                "seg_idx": len(seg_len_km)-1,
                "kind": "waypoint" if include_waypoints else kind  # 終點通常也是航點
            })
            continue

        # 找到所在段：C[k] <= S < C[k+1]
        k = bisect.bisect_right(C, S) - 1
        if k < 0:
            k = 0
        if k >= len(seg_len_km):
            k = len(seg_len_km)-1

        d_on_m = (S - C[k]) * 1000.0
        lon0, lat0 = waypoints_ll[k]
        az = fwd_az[k]
        lonN, latN, _ = geod.fwd(lon0, lat0, az, d_on_m)

        time_iso = (t0 + timedelta(hours=(S / v_kmh))).isoformat().replace("+00:00","Z") if t0 else None
        out.append({
            "idx": i,
            "time": time_iso,
            "lon": lonN, "lat": latN,
            "cum_km": S,
            "seg_idx": k,
            "kind": kind
        })

    return out
