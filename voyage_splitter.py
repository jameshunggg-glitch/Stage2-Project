"""
voyage_splitter.py
------------------
負責停泊偵測、航程切分與航程 QA 檢查
"""

import numpy as np
import pandas as pd
from haversine import haversine
from shapely.geometry import LineString
from sklearn.cluster import DBSCAN

# ---------- 小工具：確保丟進 haversine 的是 [-180,180] 經度 ----------
def _to_lon180(lon):
    return ((lon + 180) % 360) - 180

def _safe_haversine(c1, c2):
    """c1,c2 = (lat, lon) ; lon 允許是任意範圍，會自動轉為 [-180,180] 再算距離（km）"""
    return haversine((c1[0], _to_lon180(c1[1])), (c2[0], _to_lon180(c2[1])))

# -----------------------------------
# 停泊判斷
# -----------------------------------
def detect_stops(df, sog_threshold=0.5, max_gap_sec=120,
                 min_stop_sec=1800, max_stop_radius=0.3):
    """
    停泊判斷：根據 SOG 與空間半徑過濾
    需要欄位：Lat, Long, Long_360, Sog, Timestamp
    回傳：停泊區段 (list of range)
    """
    df = df.copy()
    df['Is_Stop'] = df['Sog'] < sog_threshold

    stop_segments = []
    current_start_idx, last_stop_idx = None, None

    for i in range(len(df)):
        if df.loc[i, 'Is_Stop']:
            if current_start_idx is None:
                current_start_idx = i
            last_stop_idx = i
        else:
            if current_start_idx is not None:
                gap = (df.loc[i, 'Timestamp'] - df.loc[last_stop_idx, 'Timestamp']).total_seconds()
                if gap > max_gap_sec:
                    seg = range(current_start_idx, last_stop_idx + 1)
                    stop_segments.append(seg)
                    current_start_idx, last_stop_idx = None, None

    # 收尾
    if current_start_idx is not None:
        seg = range(current_start_idx, last_stop_idx + 1)
        stop_segments.append(seg)

    # 過濾短停泊 + 空間半徑檢查（距離用 Long [-180,180]）
    filtered_segments = []
    for seg in stop_segments:
        start_time = df.loc[seg[0], 'Timestamp']
        end_time = df.loc[seg[-1], 'Timestamp']
        duration = (end_time - start_time).total_seconds()

        lat_mean = df.loc[seg, 'Lat'].median()
        # 半徑判斷請用 Long（不是 Long_360）
        lon_mean = df.loc[seg, 'Long'].median()

        distances = [
            _safe_haversine(
                (lat_mean, lon_mean),
                (df.loc[j, 'Lat'], df.loc[j, 'Long'])
            )
            for j in seg
        ]
        r95 = np.percentile(distances, 95)

        if duration >= min_stop_sec and r95 <= max_stop_radius:
            filtered_segments.append(seg)

    return filtered_segments


# -----------------------------------
# 航程切分 + ETA
# -----------------------------------
def split_voyages(df, stop_segments):
    """
    根據停泊段切分航程，並計算 Real_ETA_sec

    回傳:
        df (pd.DataFrame): 加上 voyage_id 與 Real_ETA_sec
        voyages (pd.DataFrame): 航程摘要
    """
    df = df.copy()
    df['voyage_id'] = np.nan
    df['Real_ETA_sec'] = np.nan

    voyages = []
    voyage_id = 1

    for i in range(len(stop_segments) - 1):
        start_idx = stop_segments[i][-1] + 1
        end_idx = stop_segments[i + 1][0] - 1
        if start_idx > end_idx:
            continue

        df.loc[start_idx:end_idx, 'voyage_id'] = voyage_id

        eta_time = df.loc[stop_segments[i + 1], 'Timestamp'].min()
        df.loc[start_idx:end_idx, 'Real_ETA_sec'] = (
            (eta_time - df.loc[start_idx:end_idx, 'Timestamp']).dt.total_seconds()
        )

        voyages.append({
            "voyage_id": voyage_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "dep_time": df.loc[start_idx, 'Timestamp'],
            "arr_time": df.loc[end_idx, 'Timestamp']
        })
        voyage_id += 1

    return df, pd.DataFrame(voyages)


# -----------------------------------
# QA 輔助函式
# -----------------------------------
def check_large_time_gaps(seg, gap_threshold_hr=2.0):
    """檢查航程中是否有連續兩點時間間隔 > gap_threshold_hr (小時)"""
    if len(seg) <= 1:
        return False
    time_diffs = seg["Timestamp"].diff().dt.total_seconds().dropna()
    return (time_diffs > gap_threshold_hr * 3600).any()


# def check_cross_land(seg, land_gdf):
#     if len(seg) < 2:
#         return False
#     # shapely 要 (lon, lat)；若啟用，建議也改成使用 Long（不是 Long_360）
#     coords = list(zip(seg["Long"], seg["Lat"]))
#     line = LineString(coords)
#     return land_gdf.intersects(line).any()


# -----------------------------------
# 航程 QA 檢查器（含 R0–R6）
# -----------------------------------
def voyage_quality_checker(
    df, voyages,
    min_duration_sec=1800, min_distance_km=2.0,
    hdg_spin_threshold_deg=360, disp_threshold_km=2.0,
    cluster_eps_km=1.0, min_stop_for_cluster=3,
    gap_threshold_hr=2.0, land_gdf=None
):
    """
    對航程進行 QA 檢查（R0–R6）
    - 所有距離計算改用 Long（-180~180），避免 haversine 報錯
    - 停泊聚類(DBSCAN)用 Lat/Long 轉 radians
    """
    results = []

    # Step 1: 停泊 cluster（用 Long 而非 Long_360）
    stop_centers = []
    if "voyage_id" in df.columns:
        stops = df[df["voyage_id"].isna()].copy()
        if not stops.empty:
            coords = np.radians(stops[["Lat", "Long"]].to_numpy())  # <- 關鍵：用 Long
            kms_per_radian = 6371.0088
            db = DBSCAN(
                eps=cluster_eps_km / kms_per_radian,
                min_samples=min_stop_for_cluster,
                metric="haversine"
            )
            labels = db.fit_predict(coords)
            stops["cluster_id"] = labels
            for cid, grp in stops.groupby("cluster_id"):
                if cid == -1:
                    continue
                stop_centers.append({
                    "cluster_id": cid,
                    "lat": grp["Lat"].median(),
                    "lon": grp["Long"].median()
                })

    # Step 2: 航程檢查
    for _, v in voyages.iterrows():
        seg = df[df['voyage_id'] == v['voyage_id']]
        if seg.empty:
            continue

        duration = (seg["Timestamp"].iloc[-1] - seg["Timestamp"].iloc[0]).total_seconds()

        # 位移與路徑長度：全部用 Long
        displacement = _safe_haversine(
            (seg["Lat"].iloc[0], seg["Long"].iloc[0]),
            (seg["Lat"].iloc[-1], seg["Long"].iloc[-1])
        )
        if len(seg) > 1:
            path_len = np.sum([
                _safe_haversine(
                    (seg["Lat"].iloc[i],   seg["Long"].iloc[i]),
                    (seg["Lat"].iloc[i+1], seg["Long"].iloc[i+1])
                )
                for i in range(len(seg)-1)
            ])
        else:
            path_len = 0.0

        # HDG 累積旋轉量（Hdg 可能缺或有 511）
        hdg_series = seg.get("Hdg")
        heading_cum = 0.0
        if hdg_series is not None and not hdg_series.isna().all():
            hdg = hdg_series.replace(511, np.nan).dropna().to_numpy()
            if len(hdg) > 1:
                diffs = np.diff(hdg)
                diffs = np.where(diffs > 180, diffs - 360, diffs)
                diffs = np.where(diffs < -180, diffs + 360, diffs)
                heading_cum = float(np.sum(np.abs(diffs)))

        # 規則檢查
        invalid_reason = None

        # R0/R1 起訖 cluster（用 Long）
        if stop_centers:
            start = (seg["Lat"].iloc[0], seg["Long"].iloc[0])
            end   = (seg["Lat"].iloc[-1], seg["Long"].iloc[-1])
            dists_start = [_safe_haversine(start, (c["lat"], c["lon"])) for c in stop_centers]
            dists_end   = [_safe_haversine(end,   (c["lat"], c["lon"])) for c in stop_centers]
            c_start, c_end = np.argmin(dists_start), np.argmin(dists_end)

            if c_start == c_end:
                invalid_reason = "R0_same_cluster"
            elif min(dists_start) < cluster_eps_km and min(dists_end) < cluster_eps_km:
                if displacement < 1.0:
                    invalid_reason = "R1_nearby_clusters"

        # R2 最小尺度
        if invalid_reason is None:
            if duration < min_duration_sec or displacement < min_distance_km:
                invalid_reason = "R2_too_short"

        # R4 HDG
        if invalid_reason is None and heading_cum > 0:
            if displacement < disp_threshold_km and heading_cum >= hdg_spin_threshold_deg:
                invalid_reason = "R4a_spin_in_place"
            elif path_len > 0 and (heading_cum / path_len) > 150:
                invalid_reason = "R4b_excessive_turns"

        # R5 大時間間隔
        if invalid_reason is None:
            if check_large_time_gaps(seg, gap_threshold_hr):
                invalid_reason = "R5_large_time_gap"

        # R6 陸地相交（若啟用，建議一律用 Long）
        # if invalid_reason is None and land_gdf is not None:
        #     if check_cross_land(seg, land_gdf):
        #         invalid_reason = "R6_cross_land"

        valid = invalid_reason is None
        results.append({
            "voyage_id": v["voyage_id"],
            "duration_hr": duration / 3600,
            "displacement_km": displacement,
            "path_len_km": path_len,
            "HeadingCum_deg": heading_cum,
            "n_points": len(seg),
            "valid_flag": valid,
            "invalid_reason": invalid_reason
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("這是 voyage_splitter 模組，請在 main.py 中 import 使用。")
