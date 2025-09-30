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

# -----------------------------------
# 停泊判斷
# -----------------------------------
def detect_stops(df, sog_threshold=0.5, max_gap_sec=120,
                 min_stop_sec=1800, max_stop_radius=0.3):
    """
    停泊判斷：根據 SOG 與空間半徑過濾

    參數:
        df (pd.DataFrame): AIS 資料，需含 Lat, Long_360, Sog, Timestamp
        sog_threshold (float): 判斷為停泊的速度閾值
        max_gap_sec (int): 停泊內可容忍的最大空檔
        min_stop_sec (int): 最小停泊時間（秒）
        max_stop_radius (float): 停泊區段內的最大半徑（km）

    回傳:
        list: 停泊區段 (每段是一個 index range)
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

    # 過濾短停泊 + 空間半徑檢查
    filtered_segments = []
    for seg in stop_segments:
        start_time = df.loc[seg[0], 'Timestamp']
        end_time = df.loc[seg[-1], 'Timestamp']
        duration = (end_time - start_time).total_seconds()

        lat_mean = df.loc[seg, 'Lat'].median()
        lon_mean = df.loc[seg, 'Long_360'].median()
        distances = [haversine((lat_mean, lon_mean),
                               (df.loc[j, 'Lat'], df.loc[j, 'Long_360']))
                     for j in seg]
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
        df (pd.DataFrame): 加上 voyage_id 與 Real_ETA_sec 的 DataFrame
        voyages (pd.DataFrame): 每條航程的摘要資訊
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

        # 標記 voyage_id
        df.loc[start_idx:end_idx, 'voyage_id'] = voyage_id

        # ETA = 下一停泊起始時間 - 當前時間
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
    time_diffs = seg["Timestamp"].diff().dt.total_seconds().dropna()
    return (time_diffs > gap_threshold_hr * 3600).any()


# def check_cross_land(seg, land_gdf):
#     """檢查航程是否穿越陸地"""
#     if len(seg) < 2:
#         return False
#     coords = list(zip(seg["Long_360"], seg["Lat"]))  # shapely 要 (lon, lat)
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
    對航程進行 QA 檢查，包含 R0–R6 規則
    """
    results = []

    # Step 1: 停泊 cluster
    stop_centers = []
    if "voyage_id" in df:
        stops = df[df["voyage_id"].isna()]
        if not stops.empty:
            coords = np.radians(stops[["Lat", "Long_360"]].to_numpy())
            kms_per_radian = 6371.0088
            db = DBSCAN(eps=cluster_eps_km/kms_per_radian,
                        min_samples=min_stop_for_cluster,
                        metric="haversine")
            labels = db.fit_predict(coords)
            stops["cluster_id"] = labels
            for cid, grp in stops.groupby("cluster_id"):
                if cid == -1: continue
                stop_centers.append({
                    "cluster_id": cid,
                    "lat": grp["Lat"].median(),
                    "lon": grp["Long_360"].median()
                })

    # Step 2: 航程檢查
    for _, v in voyages.iterrows():
        seg = df[df['voyage_id'] == v['voyage_id']]
        if seg.empty:
            continue

        duration = (seg["Timestamp"].iloc[-1] - seg["Timestamp"].iloc[0]).total_seconds()
        displacement = haversine(
            (seg["Lat"].iloc[0], seg["Long_360"].iloc[0]),
            (seg["Lat"].iloc[-1], seg["Long_360"].iloc[-1])
        )
        path_len = np.sum([
            haversine((seg["Lat"].iloc[i], seg["Long_360"].iloc[i]),
                      (seg["Lat"].iloc[i+1], seg["Long_360"].iloc[i+1]))
            for i in range(len(seg)-1)
        ])

        # HDG 累積旋轉量
        hdg = seg["Hdg"].replace(511, np.nan).dropna().to_numpy()
        heading_cum = 0
        if len(hdg) > 1:
            diffs = np.diff(hdg)
            diffs = np.where(diffs > 180, diffs-360, diffs)
            diffs = np.where(diffs < -180, diffs+360, diffs)
            heading_cum = np.sum(np.abs(diffs))

        # 規則檢查
        invalid_reason = None

        # R0/R1 起訖 cluster
        if stop_centers:
            start = (seg["Lat"].iloc[0], seg["Long_360"].iloc[0])
            end   = (seg["Lat"].iloc[-1], seg["Long_360"].iloc[-1])
            dists_start = [haversine(start, (c["lat"], c["lon"])) for c in stop_centers]
            dists_end   = [haversine(end,   (c["lat"], c["lon"])) for c in stop_centers]
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

        # R6 陸地相交 (暫時註解掉)
        # if invalid_reason is None and land_gdf is not None:
        #     if check_cross_land(seg, land_gdf):
        #         invalid_reason = "R6_cross_land"

        valid = invalid_reason is None
        
        results.append({
            "voyage_id": v["voyage_id"],
            "duration_hr": duration/3600,
            "displacement_km": displacement,
            "path_len_km": path_len,
            "HeadingCum_deg": heading_cum,
            "n_points": len(seg),
            "valid_flag": valid,
            "invalid_reason": invalid_reason
        })

    return pd.DataFrame(results)


# 測試用：直接執行時
if __name__ == "__main__":
    print("這是 voyage_splitter 模組，請在 main.py 中 import 使用。")
