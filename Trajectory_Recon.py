"""Trajectory reconstruction utilities for voyages with missing AIS segments.

This module merges the small- and mid-gap prototypes into a reusable helper
that can be imported from ``main.py``. It inspects voyage QA results and fills
in gaps marked as ``R5_missing_time_gap`` using grid-based or density-guided
routing, then injects interpolated points back into the voyage track.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

try:  # Optional smoothing; module still works without SciPy.
    from scipy.signal import savgol_filter  # type: ignore
except ImportError:  # pragma: no cover - SciPy is optional at runtime.
    savgol_filter = None  # type: ignore

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

DEFAULT_LAND_SHP = Path(
    r"C:\Users\slab\Desktop\Slab Project\Stage1\data\Land\ne_10m_land.shp"
)
DEFAULT_DENSITY_TIF = Path(
    r"C:\Users\slab\Desktop\Slab Project\Stage1\data\ShipDensity_Commercial\ShipDensity_Commercial1.tif"
)

SMALL_GAP_STEP_DEG = 0.01
SMALL_GAP_PAD_DEG = 0.4
SMALL_GAP_NEIGHBOR_HOPS: Tuple[int, ...] = (1, 2)
SMALL_GAP_SMOOTH_WINDOW = 7
SMALL_GAP_SMOOTH_POLY = 3
SMALL_GAP_DP_EPS_KM = 0.9
SAFE_DISTANCE_KM = 3.0  # 船需要離陸地的安全距離

MID_GAP_PAD_DEG = 0.8
LARGE_GAP_PAD_DEG = 2.0
DENSITY_PERCENTILES: Tuple[int, ...] = (95, 90, 80, 70, 60, 40, 20)
DENSITY_CORRIDOR_HALF_WIDTH_KM = 200
DENSITY_MIN_AREA_PIXELS = 50
DENSITY_BRIDGE_KM = 30

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometres."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(max(a, 0.0)))


def _segment_lengths_km(xs: Sequence[float], ys: Sequence[float]) -> np.ndarray:
    if len(xs) < 2:
        return np.array([], dtype=float)
    segs = [haversine(xs[i - 1], ys[i - 1], xs[i], ys[i]) for i in range(1, len(xs))]
    return np.array(segs, dtype=float)


def _linear_time_expand(
    xs: Sequence[float],
    ys: Sequence[float],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> List[pd.Timestamp]:
    if len(xs) < 2:
        return [pd.Timestamp(start_time), pd.Timestamp(end_time)]

    segs = _segment_lengths_km(xs, ys)
    segs = np.nan_to_num(segs, nan=0.0, posinf=0.0, neginf=0.0)
    cum = np.concatenate([[0.0], np.cumsum(segs)])
    total = float(cum[-1])
    if total <= 0:
        return [pd.Timestamp(start_time)] * len(xs)

    duration = (pd.Timestamp(end_time) - pd.Timestamp(start_time)).total_seconds()
    if duration < 0:
        duration = abs(duration)
    return [
        pd.Timestamp(start_time) + pd.to_timedelta((s / total) * duration, unit="s")
        for s in cum
    ]


def _round_coord(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


@dataclass
class GapResult:
    voyage_id: int
    gap_index: int
    gap_type: str
    gap_hours: float
    inserted_points: int
    status: str
    message: str


# ---------------------------------------------------------------------------
# Core reconstructor
# ---------------------------------------------------------------------------


class TrajectoryReconstructor:
    """Reconstruct missing trajectory segments for voyages with time gaps."""

    def __init__(
        self,
        land_shp_path: Path | str | None = DEFAULT_LAND_SHP,
        density_tif_path: Path | str | None = DEFAULT_DENSITY_TIF,
    ) -> None:
        self.land_shp_path = Path(land_shp_path) if land_shp_path else None
        self.density_tif_path = Path(density_tif_path) if density_tif_path else None
        self._land_gdf: Optional[gpd.GeoDataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconstruct_voyages(
        self,
        df: pd.DataFrame,
        voyages_summary: pd.DataFrame,
        target_reason: str = "R5_missing_time_gap",
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Reconstruct all voyages flagged with ``target_reason``.

        Returns the augmented dataframe and a per-gap summary table."""

        if df.empty:
            result = df.copy()
            result["Reconstruction"] = []
            return result, pd.DataFrame(columns=list(GapResult.__annotations__.keys()))

        if "invalid_reason" not in voyages_summary.columns:
            raise ValueError("voyages_summary must contain an 'invalid_reason' column.")
        if "gap_list" not in voyages_summary.columns:
            raise ValueError("voyages_summary must contain a 'gap_list' column.")

        work_df = df.copy()
        time_col = self._ensure_time_column(work_df)
        lat_cols = [col for col in ("Lat", "Latitude") if col in work_df.columns]
        lon_cols = [col for col in ("Long", "Longitude") if col in work_df.columns]
        if not lat_cols or not lon_cols:
            raise ValueError("Dataframe must include latitude and longitude columns (Lat/Long).")
        long360_col = "Long_360" if "Long_360" in work_df.columns else None

        if "Reconstruction" in work_df.columns:
            prev_mask = work_df["Reconstruction"].fillna(False).astype(bool)
            if prev_mask.any():
                work_df = work_df.loc[~prev_mask].copy()
        work_df["Reconstruction"] = False

        inserted_rows: List[Dict[str, object]] = []
        summary: List[Dict[str, object]] = []

        targets = voyages_summary[voyages_summary["invalid_reason"] == target_reason]
        if targets.empty and verbose:
            print("TrajectoryReconstructor: no voyages require reconstruction.")

        for _, voy_row in targets.iterrows():
            voyage_id = voy_row.get("voyage_id")
            if pd.isna(voyage_id):
                continue
            voyage_id = int(voyage_id)

            gap_list = self._normalize_gap_list(voy_row.get("gap_list"))
            if not gap_list:
                continue

            mmsi_value = self._extract_mmsi(work_df, voyage_id)

            for gap_index, gap in enumerate(gap_list):
                result = self._reconstruct_single_gap(
                    work_df,
                    time_col,
                    lat_cols,
                    lon_cols,
                    long360_col,
                    voyage_id,
                    gap,
                    mmsi_value,
                    gap_index,
                )

                if result is None:
                    summary.append(
                        GapResult(
                            voyage_id=voyage_id,
                            gap_index=gap_index,
                            gap_type=gap.get("gap_type", "unknown"),
                            gap_hours=float(gap.get("gap_hr", float("nan"))),
                            inserted_points=0,
                            status="skipped",
                            message="path search failed",
                        ).__dict__
                    )
                    if verbose:
                        print(f"[warn] voyage {voyage_id} gap#{gap_index}: reconstruction failed.")
                    continue

                inserted_rows.extend(result["rows"])
                summary.append(result["summary"])
                if verbose:
                    inserted_pts = result["summary"]["inserted_points"]
                    print(
                        f"[ok] voyage {voyage_id} gap#{gap_index} ({result['summary']['gap_type']}) -> {inserted_pts} pts"
                    )

        if inserted_rows:
            new_rows_df = pd.DataFrame(inserted_rows)
            for col in work_df.columns:
                if col not in new_rows_df.columns:
                    new_rows_df[col] = np.nan
            new_rows_df = new_rows_df[work_df.columns]
            combined = pd.concat([work_df, new_rows_df], ignore_index=True)
        else:
            combined = work_df.reset_index(drop=True)

        combined.sort_values(time_col, inplace=True)
        combined.reset_index(drop=True, inplace=True)
        combined["Reconstruction"] = combined["Reconstruction"].astype(bool)
        return combined, pd.DataFrame(summary)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_time_column(self, df: pd.DataFrame) -> str:
        candidates = [
            "Timestamp",
            "TimeStamp",
            "CreateTime",
            "Create_Time",
            "Time",
            "datetime",
            "time",
        ]
        for col in candidates:
            if col in df.columns:
                if np.issubdtype(df[col].dtype, np.datetime64):
                    return col
                try:
                    df[col] = pd.to_datetime(df[col], errors="raise")
                    return col
                except Exception:
                    continue
        raise ValueError("Cannot locate a usable timestamp column in dataframe.")

    def _normalize_gap_list(self, raw_gap_list: object) -> List[Dict[str, object]]:
        if raw_gap_list is None or (isinstance(raw_gap_list, float) and math.isnan(raw_gap_list)):
            return []
        if isinstance(raw_gap_list, str):
            try:
                gap_list = json.loads(raw_gap_list)
            except json.JSONDecodeError:
                return []
        else:
            gap_list = list(raw_gap_list)

        normalized: List[Dict[str, object]] = []
        for gap in gap_list:
            if not isinstance(gap, dict):
                continue
            A = gap.get("A") or {}
            B = gap.get("B") or {}
            try:
                start_time = pd.Timestamp(A.get("t"))
                end_time = pd.Timestamp(B.get("t"))
            except Exception:
                continue
            if pd.isna(start_time) or pd.isna(end_time):
                continue
            gap_hr = gap.get("gap_hr")
            if gap_hr is None or (isinstance(gap_hr, float) and math.isnan(gap_hr)):
                gap_hr = (end_time - start_time).total_seconds() / 3600.0
            gap_type = gap.get("gap_type")
            if not gap_type:
                gap_type = self._classify_gap(float(gap_hr))
            normalized.append(
                {
                    "gap_type": gap_type,
                    "gap_hr": float(gap_hr),
                    "A_idx": gap.get("A_idx"),
                    "B_idx": gap.get("B_idx"),
                    "A": {
                        "lon": A.get("lon"),
                        "lat": A.get("lat"),
                        "time": start_time,
                    },
                    "B": {
                        "lon": B.get("lon"),
                        "lat": B.get("lat"),
                        "time": end_time,
                    },
                }
            )
        return normalized

    def _classify_gap(self, gap_hours: float) -> str:
        if gap_hours < 1.5 and gap_hours >= 0.5:
            return "small_time_gap"
        if gap_hours < 4.0 and gap_hours >= 1.5:
            return "mid_time_gap"
        return "large_time_gap"

    def _extract_mmsi(self, df: pd.DataFrame, voyage_id: int) -> Optional[int]:
        if "MMSI" not in df.columns:
            return None
        subset = df.loc[df["voyage_id"] == voyage_id, "MMSI"].dropna()
        if subset.empty:
            return None
        try:
            return int(subset.iloc[0])
        except Exception:
            return None

    def _reconstruct_single_gap(
        self,
        df: pd.DataFrame,
        time_col: str,
        lat_cols: Sequence[str],
        lon_cols: Sequence[str],
        long360_col: Optional[str],
        voyage_id: int,
        gap: Dict[str, object],
        mmsi_value: Optional[int],
        gap_index: int,
    ) -> Optional[Dict[str, object]]:
        
        print("\n" + "="*60)
        print(f"🔧 [DEBUG] 開始處理 voyage {voyage_id}, gap#{gap_index}")
        print("="*60)
        
        gap_type = gap.get("gap_type", "unknown")
        gap_hours = float(gap.get("gap_hr", float("nan")))

        print(f"Gap 類型: {gap_type}, 時長: {gap_hours:.2f} 小時")

        A_info = gap.get("A", {})
        B_info = gap.get("B", {})
        a_idx = gap.get("A_idx")
        b_idx = gap.get("B_idx")

        start_time = A_info.get("time")
        end_time = B_info.get("time")
        if (start_time is None or pd.isna(start_time)) and a_idx in df.index:
            start_time = df.at[a_idx, time_col]
        if (end_time is None or pd.isna(end_time)) and b_idx in df.index:
            end_time = df.at[b_idx, time_col]
        if start_time is None or end_time is None:
            return None

        lon_col = lon_cols[0]
        lat_col = lat_cols[0]

        def _fallback(idx: object, column: str) -> Optional[float]:
            if idx in df.index and column in df.columns:
                value = df.at[idx, column]
                return None if pd.isna(value) else float(value)
            return None

        start_lon = A_info.get("lon")
        start_lat = A_info.get("lat")
        end_lon = B_info.get("lon")
        end_lat = B_info.get("lat")

        if start_lon is None or pd.isna(start_lon):
            start_lon = _fallback(a_idx, lon_col)
        if start_lat is None or pd.isna(start_lat):
            start_lat = _fallback(a_idx, lat_col)
        if end_lon is None or pd.isna(end_lon):
            end_lon = _fallback(b_idx, lon_col)
        if end_lat is None or pd.isna(end_lat):
            end_lat = _fallback(b_idx, lat_col)

        if None in (start_lon, start_lat, end_lon, end_lat):
            return None
        
        print(f"📍 起點: ({start_lon:.6f}, {start_lat:.6f})")
        print(f"📍 終點: ({end_lon:.6f}, {end_lat:.6f})")
        direct_dist = haversine(start_lon, start_lat, end_lon, end_lat)
        print(f"📏 直線距離: {direct_dist:.2f} km")

        # === 新增：靠岸跳過判定 ===
        too_close_A = self._check_land_obstruction(
            start_lon, start_lat, start_lon, start_lat, safe_distance_km=SAFE_DISTANCE_KM
        )
        too_close_B = self._check_land_obstruction(
            end_lon, end_lat, end_lon, end_lat, safe_distance_km=SAFE_DISTANCE_KM
        )
        if too_close_A or too_close_B:
            print(f"⚠️ 起點或終點距離陸地 < {SAFE_DISTANCE_KM} km，跳過修復")
            return None

        print(f"\n🔍 開始路徑搜尋 (方法: {gap_type})...")

        print(f"\n🔍 開始路徑搜尋 (方法: {gap_type})...")

        if gap_type == "small_time_gap":
            route = self._small_gap_path(start_lon, start_lat, end_lon, end_lat, gap_hours)
        else:
            route = self._density_gap_path(start_lon, start_lat, end_lon, end_lat, gap_type)

        print(f"✅ 路徑搜尋完成: {'成功' if route else '失敗'}")
        if route:
            print(f"📈 最終路徑點數: {len(route)}")
        
        if not route or len(route) < 2:
            print(f"❌ 路徑點數不足，放棄")
            return None

        route = self._ensure_endpoints(route, start_lon, start_lat, end_lon, end_lat)
        print(f"🔧 確保端點後: {len(route)} 個點")

        xs = [pt[0] for pt in route]
        ys = [pt[1] for pt in route]
        ts = _linear_time_expand(xs, ys, pd.Timestamp(start_time), pd.Timestamp(end_time))

        new_rows: List[Dict[str, object]] = []
        for idx in range(1, len(route) - 1):
            lon, lat = route[idx]
            timestamp = ts[idx]
            row = {col: np.nan for col in df.columns}
            row[time_col] = timestamp
            for col in lat_cols:
                if col in df.columns:
                    row[col] = float(lat)
            for col in lon_cols:
                if col in df.columns:
                    row[col] = float(lon)
            if long360_col and long360_col in df.columns:
                row[long360_col] = float(lon) % 360.0
            row["voyage_id"] = voyage_id
            row["Reconstruction"] = True
            if mmsi_value is not None and "MMSI" in df.columns:
                row["MMSI"] = int(mmsi_value)
            new_rows.append(row)
        
        print(f"\n✨ 實際插入: {len(new_rows)} 個點")
        print("="*60 + "\n")
        
        summary = GapResult(
            voyage_id=voyage_id,
            gap_index=gap_index,
            gap_type=gap_type,
            gap_hours=gap_hours,
            inserted_points=len(new_rows),
            status="ok" if new_rows else "no_insert",
            message="" if new_rows else "path produced endpoints only",
        ).__dict__

        return {"rows": new_rows, "summary": summary}

    def _ensure_endpoints(
        self,
        route: Sequence[Tuple[float, float]],
        lon_a: float,
        lat_a: float,
        lon_b: float,
        lat_b: float,
    ) -> List[Tuple[float, float]]:
        if not route:
            return []
        adjusted = list(route)
        adjusted[0] = (float(lon_a), float(lat_a))
        adjusted[-1] = (float(lon_b), float(lat_b))
        return adjusted

    # ------------------------------------------------------------------
    # Small-gap reconstruction (grid-based A* with land checking)
    # ------------------------------------------------------------------

    def _small_gap_path(
        self,
        lon_a: float,
        lat_a: float,
        lon_b: float,
        lat_b: float,
        gap_hours: float,
    ) -> Optional[List[Tuple[float, float]]]:
        
        direct_dist = haversine(lon_a, lat_a, lon_b, lat_b)

                # === 新增：檢查起訖點是否太靠近陸地 ===
        if self._is_near_land(lon_a, lat_a, safe_distance_km=SAFE_DISTANCE_KM) or \
           self._is_near_land(lon_b, lat_b, safe_distance_km=SAFE_DISTANCE_KM):
            print(f"  ⚠️ 起訖點靠近陸地，跳過此 gap 修復")
            return None

        # === 步驟 1：檢查陸地阻擋 ===
        has_land = self._check_land_obstruction(
            lon_a, lat_a, lon_b, lat_b,
            safe_distance_km=SAFE_DISTANCE_KM
        )
        
        if not has_land:
            # === 情況 A：開闊海域 → 直線插值 ===
            print(f"  ✅ 開闊海域，使用直線路徑")
            num_points = self._calculate_target_points(direct_dist, gap_hours)
            print(f"  → 目標點數: {num_points}")
            xs = np.linspace(lon_a, lon_b, num_points)
            ys = np.linspace(lat_a, lat_b, num_points)
            return [(float(x), float(y)) for x, y in zip(xs, ys)]
        
        # === 情況 B：有陸地 → A* 繞行 ===
        print(f"  ⚠️ 陸地阻擋，使用 A* 繞行")
        
        land = self._load_land()
        sea_points, point_index, neighbor_steps = self._build_local_sea_graph(
            lon_a, lat_a, lon_b, lat_b, land,
            step=SMALL_GAP_STEP_DEG,
            hops=SMALL_GAP_NEIGHBOR_HOPS,
            pad_deg=SMALL_GAP_PAD_DEG,
        )
        print(f"  🌊 海洋網格點數: {len(sea_points)}")
        
        xs, ys = self._astar_on_graph(
            sea_points, point_index, neighbor_steps,
            (lon_a, lat_a), (lon_b, lat_b),
        )
        
        if xs is None or ys is None or len(xs) == 0:
            print(f"  ❌ A* 搜尋失敗，退回直線")
            num_points = self._calculate_target_points(direct_dist, gap_hours)
            xs = np.linspace(lon_a, lon_b, num_points)
            ys = np.linspace(lat_a, lat_b, num_points)
            return [(float(x), float(y)) for x, y in zip(xs, ys)]
        
        print(f"  → A* 路徑: {len(xs)} 個節點")
        
        # === 步驟 2：DP 簡化 ===
        original_len = len(xs)
        xs, ys = self._dp_simplify(xs, ys, eps_km=SMALL_GAP_DP_EPS_KM)
        print(f"  → DP 簡化: {original_len} → {len(xs)} 個點 (eps={SMALL_GAP_DP_EPS_KM}km)")
        
        # === 步驟 3：確保最小點數 ===
        path_length = sum(haversine(xs[i-1], ys[i-1], xs[i], ys[i]) 
                          for i in range(1, len(xs)))
        min_points = self._calculate_target_points(path_length, gap_hours)
        
        if len(xs) < min_points:
            print(f"  → 補點: {len(xs)} → {min_points}")
            xs, ys = self._interpolate_along_path(xs, ys, min_points)
        
        # === 步驟 4：Savgol 平滑 ===
        if savgol_filter is not None and len(xs) >= SMALL_GAP_SMOOTH_WINDOW:
            xs = savgol_filter(xs, SMALL_GAP_SMOOTH_WINDOW, SMALL_GAP_SMOOTH_POLY, mode="interp")
            ys = savgol_filter(ys, SMALL_GAP_SMOOTH_WINDOW, SMALL_GAP_SMOOTH_POLY, mode="interp")
            print(f"  ✨ Savgol 平滑完成")
        
        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    def _calculate_target_points(self, distance_km: float, gap_hours: float) -> int:
        """計算目標點數：根據距離和時間"""
        # small_time_gap 範圍：0.5~1.5 小時
        points_from_dist = int(distance_km / 2.0) + 2  # 每 2km 一個點
        points_from_time = int(gap_hours * 8) + 2      # 每小時 8 個點
        return max(5, points_from_dist, points_from_time)
    
    def _is_near_land(
        self,
        lon: float,
        lat: float,
        safe_distance_km: float = SAFE_DISTANCE_KM,
    ) -> bool:
        """檢查單一點是否太靠近陸地邊界（距離小於安全距離）"""

        land_gdf = self._load_land()
        point = Point(lon, lat)

     # 只取局部區域
        pad = max(0.3, safe_distance_km / 50.0)
        land_nearby = land_gdf.cx[
            lon - pad : lon + pad,
            lat - pad : lat + pad,
        ]
        if land_nearby.empty:
            return False

    # 計算與最近陸地的距離（度轉 km）
        union_land = land_nearby.unary_union
        dist_deg = point.distance(union_land)
        dist_km = dist_deg * 111  # 1 度 ≈ 111 km
        return dist_km < safe_distance_km

    def _check_land_obstruction(
        self,
        lon_a: float,
        lat_a: float,
        lon_b: float,
        lat_b: float,
        safe_distance_km: float = 1.0,
    ) -> bool:
        """檢查直線路徑是否會穿越陸地（含安全距離）"""
        land_gdf = self._load_land()
        
        # 建立直線
        direct_line = LineString([(lon_a, lat_a), (lon_b, lat_b)])
        
        # 只檢查路徑附近的陸地
        buffer = max(0.5, safe_distance_km / 50.0)
        land_nearby = land_gdf.cx[
            min(lon_a, lon_b) - buffer : max(lon_a, lon_b) + buffer,
            min(lat_a, lat_b) - buffer : max(lat_a, lat_b) + buffer
        ]
        
        if land_nearby.empty:
            return False
        
        # 加入安全距離緩衝
        buffer_deg = safe_distance_km / 111.0
        direct_line_buffered = direct_line.buffer(buffer_deg)
        
        # 檢查相交
        union_land = land_nearby.unary_union
        return direct_line_buffered.intersects(union_land)

    def _interpolate_along_path(
        self,
        xs: List[float],
        ys: List[float],
        target_points: int,
    ) -> Tuple[List[float], List[float]]:
        """沿著路徑進行等距插值"""
        if len(xs) < 2:
            return xs, ys
        
        # 計算累積距離
        distances = [0.0]
        for i in range(1, len(xs)):
            seg_dist = haversine(xs[i-1], ys[i-1], xs[i], ys[i])
            distances.append(distances[-1] + seg_dist)
        
        total_dist = distances[-1]
        if total_dist <= 0:
            return xs, ys
        
        # 在累積距離上均勻取樣
        sample_distances = np.linspace(0, total_dist, target_points)
        
        # 插值得到新座標
        xs_new = np.interp(sample_distances, distances, xs)
        ys_new = np.interp(sample_distances, distances, ys)
        
        return list(xs_new), list(ys_new)

    def _build_local_sea_graph(
        self,
        lon_a: float,
        lat_a: float,
        lon_b: float,
        lat_b: float,
        land_gdf: gpd.GeoDataFrame,
        step: float,
        hops: Sequence[int],
        pad_deg: float,
    ) -> Tuple[gpd.GeoDataFrame, Dict[Tuple[float, float], int], List[Tuple[float, float]]]:
        minx = min(lon_a, lon_b) - pad_deg
        maxx = max(lon_a, lon_b) + pad_deg
        miny = min(lat_a, lat_b) - pad_deg
        maxy = max(lat_a, lat_b) + pad_deg

        xs = np.arange(minx, maxx + 1e-9, step)
        ys = np.arange(miny, maxy + 1e-9, step)
        grid_points = [Point(x, y) for x in xs for y in ys]
        gdf_points = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:4326")

        land_clip = land_gdf.cx[minx:maxx, miny:maxy]
        if land_clip.empty:
            sea_mask = np.ones(len(gdf_points), dtype=bool)
        else:
            union = land_clip.unary_union
            sea_mask = ~gdf_points.within(union)
        sea_points = gdf_points[sea_mask].copy()
        sea_points.reset_index(drop=True, inplace=True)

        point_index = {
            (_round_coord(p.x), _round_coord(p.y)): idx for idx, p in enumerate(sea_points.geometry)
        }

        neighbor_steps: List[Tuple[float, float]] = []
        for h in hops:
            neighbor_steps.extend(
                [
                    (h * step, 0.0),
                    (-h * step, 0.0),
                    (0.0, h * step),
                    (0.0, -h * step),
                    (h * step, h * step),
                    (h * step, -h * step),
                    (-h * step, h * step),
                    (-h * step, -h * step),
                ]
            )
        return sea_points, point_index, neighbor_steps

    def _find_nearest_node_idx(
        self,
        sea_points: gpd.GeoDataFrame,
        lon: float,
        lat: float,
    ) -> Optional[int]:
        if sea_points.empty:
            return None
        best_idx = None
        best_dist = float("inf")
        for idx, geom in enumerate(sea_points.geometry):
            dist = haversine(lon, lat, geom.x, geom.y)
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        return best_idx

    def _astar_on_graph(
        self,
        sea_points: gpd.GeoDataFrame,
        point_index: Dict[Tuple[float, float], int],
        neighbor_steps: Sequence[Tuple[float, float]],
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
    ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        start_idx = self._find_nearest_node_idx(sea_points, *start_xy)
        goal_idx = self._find_nearest_node_idx(sea_points, *goal_xy)
        if start_idx is None or goal_idx is None:
            return None, None

        start = sea_points.geometry[start_idx]
        goal = sea_points.geometry[goal_idx]

        open_set: List[Tuple[float, int]] = [(0.0, start_idx)]
        came_from: Dict[int, int] = {}
        gscore = {start_idx: 0.0}
        fscore = {start_idx: haversine(start.x, start.y, goal.x, goal.y)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_idx:
                path: List[int] = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_idx)
                path.reverse()
                xs = [sea_points.geometry[i].x for i in path]
                ys = [sea_points.geometry[i].y for i in path]
                return xs, ys

            x0 = _round_coord(sea_points.geometry[current].x)
            y0 = _round_coord(sea_points.geometry[current].y)
            for dx, dy in neighbor_steps:
                x1 = _round_coord(x0 + dx)
                y1 = _round_coord(y0 + dy)
                neighbor = point_index.get((x1, y1))
                if neighbor is None:
                    continue
                tentative = gscore[current] + haversine(x0, y0, x1, y1)
                if tentative < gscore.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative
                    fscore[neighbor] = tentative + haversine(x1, y1, goal.x, goal.y)
                    heapq.heappush(open_set, (fscore[neighbor], neighbor))
        return None, None

    def _dp_simplify(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        eps_km: float,
    ) -> Tuple[List[float], List[float]]:
        if len(xs) < 3:
            return list(xs), list(ys)
        
        eps_deg = float(eps_km) / 111.0
        line = LineString(zip(xs, ys))
        simplified = line.simplify(eps_deg, preserve_topology=False)
        sx, sy = zip(*simplified.coords)
        sx_list = list(sx)
        sy_list = list(sy)

        # 確保端點
        if (sx_list[0], sy_list[0]) != (xs[0], ys[0]):
            sx_list.insert(0, xs[0])
            sy_list.insert(0, ys[0])
        if (sx_list[-1], sy_list[-1]) != (xs[-1], ys[-1]):
            sx_list.append(xs[-1])
            sy_list.append(ys[-1])
        
        return sx_list, sy_list

    # ------------------------------------------------------------------
    # Mid / large gap reconstruction (density map guided)
    # ------------------------------------------------------------------

    def _density_gap_path(
        self,
        lon_a: float,
        lat_a: float,
        lon_b: float,
        lat_b: float,
        gap_type: str,
    ) -> Optional[List[Tuple[float, float]]]:
        if self.density_tif_path is None:
            raise FileNotFoundError(
                "Density TIFF path is not configured; cannot repair mid/large gaps."
            )
        try:
            import rasterio
            from rasterio.windows import from_bounds as window_from_bounds
            from skimage import measure, morphology
            from skimage.morphology import dilation, skeletonize, square
            import networkx as nx  # type: ignore  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover - optional deps.
            raise ImportError(
                "Mid/large gap reconstruction requires rasterio, scikit-image, and networkx."
            ) from exc

        pad_deg = LARGE_GAP_PAD_DEG if gap_type == "large_time_gap" else MID_GAP_PAD_DEG
        bbox = (
            min(lon_a, lon_b) - pad_deg,
            min(lat_a, lat_b) - pad_deg,
            max(lon_a, lon_b) + pad_deg,
            max(lat_a, lat_b) + pad_deg,
        )

        with rasterio.open(self.density_tif_path) as src:
            window = window_from_bounds(*bbox, transform=src.transform)
            data = src.read(1, window=window)
            transform = src.window_transform(window)

        if data.size == 0:
            return None

        lon2d, lat2d = self._affine_lonlat_grid(transform, data.shape[0], data.shape[1])
        work = np.log1p(data.astype(float))
        mask_corridor = self._corridor_mask_from_ab(
            lon2d,
            lat2d,
            lon_a,
            lat_a,
            lon_b,
            lat_b,
            DENSITY_CORRIDOR_HALF_WIDTH_KM,
        )

        corridor_vals = work[(mask_corridor) & (work > 0)]
        if corridor_vals.size == 0:
            return None

        p90 = np.percentile(corridor_vals, 90)
        p999 = np.percentile(corridor_vals, 99.9)
        if (p999 + 1e-6) / (p90 + 1e-6) > 2.0:
            cap = np.percentile(corridor_vals, 99.5)
            work = np.minimum(work, cap)

        ar, ac = self._pixel_of_lonlat(transform, lon_a, lat_a, data.shape)
        br, bc = self._pixel_of_lonlat(transform, lon_b, lat_b, data.shape)

        binary = None
        for q in DENSITY_PERCENTILES:
            thr = np.percentile(work[(mask_corridor) & (work > 0)], q)
            cand = (work >= thr) & mask_corridor
            cand = morphology.opening(cand, square(3))
            cand = morphology.closing(cand, square(3))
            labels = measure.label(cand)
            if labels.max() > 0:
                props = measure.regionprops(labels)
                keep_labels = [p.label for p in props if p.area >= DENSITY_MIN_AREA_PIXELS]
                cand = np.isin(labels, keep_labels)
            if self._check_connected(cand, ar, ac, br, bc):
                binary = cand
                break
        if binary is None:
            thr = np.percentile(work[(mask_corridor) & (work > 0)], DENSITY_PERCENTILES[-1])
            binary = (work >= thr) & mask_corridor
            binary = morphology.opening(binary, square(3))
            binary = morphology.closing(binary, square(3))

        skeleton = skeletonize(binary)
        G = self._skeleton_to_graph(skeleton)

        resampled_lines: List[List[Tuple[int, int]]] = []
        for comp in nx.connected_components(G):
            subG = G.subgraph(comp)
            endpoints = [n for n in subG.nodes if subG.degree[n] == 1]
            if len(endpoints) >= 2:
                path_nodes = nx.shortest_path(subG, source=endpoints[0], target=endpoints[-1])
                coords = [subG.nodes[n]["pos"] for n in path_nodes]
                resampled_lines.append(self._resample_polyline(coords, step_px=15))

        if not resampled_lines:
            return None

        lonlat_lines = [self._pixel_to_lonlat(transform, line) for line in resampled_lines]
        backbone = self._build_backbone_graph(lonlat_lines)
        backbone = self._bridge_nearby_nodes(backbone, max_gap_km=DENSITY_BRIDGE_KM)

        start_node, dist_start = self._find_nearest_backbone_node(backbone, lon_a, lat_a)
        end_node, dist_end = self._find_nearest_backbone_node(backbone, lon_b, lat_b)
        if start_node is None or end_node is None:
            return None

        backbone.add_node("__start__", pos=(lon_a, lat_a))
        backbone.add_edge("__start__", start_node, weight=dist_start)
        backbone.add_node("__end__", pos=(lon_b, lat_b))
        backbone.add_edge("__end__", end_node, weight=dist_end)

        path_nodes = self._density_astar(backbone, "__start__", "__end__")
        if not path_nodes:
            return None
        route = [backbone.nodes[n]["pos"] for n in path_nodes]
        return [(float(lon), float(lat)) for lon, lat in route]

    def _affine_lonlat_grid(
        self,
        transform,
        height: int,
        width: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cols = np.arange(width)
        rows = np.arange(height)
        C, R = np.meshgrid(cols, rows)
        lon = transform.c + transform.a * C + transform.b * R
        lat = transform.f + transform.d * C + transform.e * R
        return lon, lat

    def _lonlat_to_local_km(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        lon0: float,
        lat0: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        R = 6371.0088
        x = (lon - lon0) * (math.cos(math.radians(lat0)) * (math.pi / 180.0) * R)
        y = (lat - lat0) * ((math.pi / 180.0) * R)
        return x, y

    def _corridor_mask_from_ab(
        self,
        lon2d: np.ndarray,
        lat2d: np.ndarray,
        lon_a: float,
        lat_a: float,
        lon_b: float,
        lat_b: float,
        half_width_km: float,
    ) -> np.ndarray:
        lat0 = 0.5 * (lat_a + lat_b)
        lon0 = 0.5 * (lon_a + lon_b)
        X, Y = self._lonlat_to_local_km(lon2d, lat2d, lon0, lat0)
        Ax, Ay = self._lonlat_to_local_km(np.array(lon_a), np.array(lat_a), lon0, lat0)
        Bx, By = self._lonlat_to_local_km(np.array(lon_b), np.array(lat_b), lon0, lat0)
        ABx, ABy = (Bx - Ax), (By - Ay)
        AB2 = ABx * ABx + ABy * ABy + 1e-12
        APx, APy = (X - Ax), (Y - Ay)
        t = (APx * ABx + APy * ABy) / AB2
        t = np.clip(t, 0.0, 1.0)
        Px, Py = Ax + t * ABx, Ay + t * ABy
        dx, dy = X - Px, Y - Py
        dist_km = np.hypot(dx, dy)
        return dist_km <= half_width_km

    def _pixel_of_lonlat(self, transform, lon: float, lat: float, shape: Tuple[int, int]) -> Tuple[int, int]:
        import rasterio

        row, col = rasterio.transform.rowcol(transform, lon, lat)
        row = int(np.clip(row, 0, shape[0] - 1))
        col = int(np.clip(col, 0, shape[1] - 1))
        return row, col

    def _check_connected(self, binary: np.ndarray, ar: int, ac: int, br: int, bc: int) -> bool:
        from skimage.morphology import dilation, square  # type: ignore
        from skimage import measure  # type: ignore

        if not (0 <= ar < binary.shape[0] and 0 <= ac < binary.shape[1]):
            return False
        if not (0 <= br < binary.shape[0] and 0 <= bc < binary.shape[1]):
            return False
        dilated = dilation(binary, square(3))
        labels = measure.label(dilated)
        labA = labels[ar, ac]
        labB = labels[br, bc]
        return labA != 0 and labA == labB

    def _skeleton_to_graph(self, skel: np.ndarray):
        import networkx as nx  # type: ignore  # pylint: disable=import-outside-toplevel

        H, W = skel.shape
        G = nx.Graph()
        for r in range(H):
            for c in range(W):
                if not skel[r, c]:
                    continue
                node_id = r * W + c
                G.add_node(node_id, pos=(r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < H and 0 <= cc < W and skel[rr, cc]:
                            neighbor_id = rr * W + cc
                            G.add_edge(node_id, neighbor_id)
        return G

    def _resample_polyline(
        self,
        coords: Sequence[Tuple[int, int]],
        step_px: float = 10.0,
    ) -> List[Tuple[float, float]]:
        if len(coords) < 2:
            return list(coords)
        resampled = [coords[0]]
        dist_acc = 0.0
        p_prev = np.array(coords[0], dtype=float)
        for i in range(1, len(coords)):
            p_curr = np.array(coords[i], dtype=float)
            seg_len = np.linalg.norm(p_curr - p_prev)
            if seg_len == 0:
                continue
            while dist_acc + seg_len >= step_px:
                t = (step_px - dist_acc) / seg_len
                new_point = p_prev + t * (p_curr - p_prev)
                resampled.append(tuple(new_point))
                seg_len -= (step_px - dist_acc)
                p_prev = new_point
                dist_acc = 0.0
            dist_acc += seg_len
            p_prev = p_curr
        resampled.append(coords[-1])
        return resampled

    def _pixel_to_lonlat(
        self,
        transform,
        coords: Sequence[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        import rasterio

        lonlats: List[Tuple[float, float]] = []
        for r, c in coords:
            lon, lat = rasterio.transform.xy(transform, r, c)
            lonlats.append((float(lon), float(lat)))
        return lonlats

    def _build_backbone_graph(
        self,
        lines: Sequence[Sequence[Tuple[float, float]]],
    ):
        import networkx as nx  # type: ignore  # pylint: disable=import-outside-toplevel

        G = nx.Graph()
        node_map: Dict[Tuple[float, float], int] = {}
        next_id = 0
        for line in lines:
            prev_node = None
            for point in line:
                key = (_round_coord(point[0]), _round_coord(point[1]))
                if key not in node_map:
                    node_map[key] = next_id
                    G.add_node(next_id, pos=(float(point[0]), float(point[1])))
                    next_id += 1
                current_node = node_map[key]
                if prev_node is not None and not G.has_edge(prev_node, current_node):
                    distance = haversine(
                        *G.nodes[prev_node]["pos"], *G.nodes[current_node]["pos"]
                    )
                    G.add_edge(prev_node, current_node, weight=distance)
                prev_node = current_node
        return G

    def _bridge_nearby_nodes(self, graph, max_gap_km: float):
        import networkx as nx  # type: ignore  # pylint: disable=import-outside-toplevel

        nodes = list(graph.nodes())
        for u, v in combinations(nodes, 2):
            if graph.has_edge(u, v):
                continue
            pos_u = graph.nodes[u]["pos"]
            pos_v = graph.nodes[v]["pos"]
            dist = haversine(*pos_u, *pos_v)
            if dist <= max_gap_km:
                graph.add_edge(u, v, weight=dist)
        return graph

    def _find_nearest_backbone_node(self, graph, lon: float, lat: float) -> Tuple[Optional[int], float]:
        best_node = None
        best_dist = float("inf")
        for node in graph.nodes():
            pos = graph.nodes[node]["pos"]
            dist = haversine(lon, lat, pos[0], pos[1])
            if dist < best_dist:
                best_node = node
                best_dist = dist
        return best_node, best_dist

    def _density_astar(self, graph, start, goal):
        import networkx as nx  # type: ignore  # pylint: disable=import-outside-toplevel

        try:
            return nx.astar_path(
                graph,
                start,
                goal,
                heuristic=lambda u, v: haversine(
                    *graph.nodes[u]["pos"], *graph.nodes[v]["pos"]
                ),
                weight="weight",
            )
        except nx.NetworkXNoPath:
            return None

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _load_land(self) -> gpd.GeoDataFrame:
        if self._land_gdf is None:
            if self.land_shp_path is None:
                raise FileNotFoundError(
                    "Land shapefile path is not configured; cannot build sea graph."
                )
            if not self.land_shp_path.exists():
                raise FileNotFoundError(f"Land shapefile not found: {self.land_shp_path}")
            self._land_gdf = gpd.read_file(self.land_shp_path).to_crs("EPSG:4326")
        return self._land_gdf