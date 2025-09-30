"""
od_marker.py
------------
標註航程的起訖港口
"""

import pandas as pd
from haversine import haversine


def assign_ports(voyages, df, ports_csv, radius_km=8.0, max_dist_km=10.0, debug=False):
    """
    標註每條航程的起訖港口

    參數:
        voyages (pd.DataFrame): 航程表，需含 voyage_id, start_idx, end_idx, valid_flag, invalid_reason
        df (pd.DataFrame): AIS 資料，需含 Lat, Long
        ports_csv (str or Path): 港口清單 CSV 路徑 (需有 name, lat, lon 欄位)
        radius_km (float): 判定為「在港口內」的半徑
        max_dist_km (float): 超過這個距離就視為 unknown
        debug (bool): 若 True，會印出 debug 資訊

    回傳:
        pd.DataFrame: 加上起訖港口資訊的航程表
    """
    ports = pd.read_csv(ports_csv)

    # 統一欄位名稱
    ports = ports.rename(
        columns={
            "name": "port_name",
            "lat": "lat",
            "lon": "lon",
        }
    )

    # 檢查必要欄位
    required_cols = {"port_name", "lat", "lon"}
    if not required_cols.issubset(ports.columns):
        raise ValueError(
            f" 港口清單缺少必要欄位，必須包含: {required_cols}, 目前欄位: {list(ports.columns)}"
        )

    voyages = voyages.copy()

    # 先把起迄點的經緯度補進 voyages
    voyages["dep_lat"] = df.loc[voyages["start_idx"], "Lat"].values
    voyages["dep_lon"] = df.loc[voyages["start_idx"], "Long"].values
    voyages["arr_lat"] = df.loc[voyages["end_idx"], "Lat"].values
    voyages["arr_lon"] = df.loc[voyages["end_idx"], "Long"].values

    origin_ports, dest_ports = [], []

    for _, v in voyages.iterrows():
        dep_point = (v["dep_lat"], v["dep_lon"])
        arr_point = (v["arr_lat"], v["arr_lon"])

        # 計算與所有港口的距離
        ports["dep_dist"] = ports.apply(
            lambda p: haversine(dep_point, (p["lat"], p["lon"])), axis=1
        )
        ports["arr_dist"] = ports.apply(
            lambda p: haversine(arr_point, (p["lat"], p["lon"])), axis=1
        )

        # 找最近的港口
        dep_min_dist = ports["dep_dist"].min()
        arr_min_dist = ports["arr_dist"].min()

        dep_port = (
            ports.loc[ports["dep_dist"].idxmin(), "port_name"]
            if dep_min_dist <= max_dist_km
            else "unknown"
        )
        arr_port = (
            ports.loc[ports["arr_dist"].idxmin(), "port_name"]
            if arr_min_dist <= max_dist_km
            else "unknown"
        )

        # Debug 輸出
        if debug:
            print(
                f"Voyage {v['voyage_id']} 起點最近港口: {dep_port} ({dep_min_dist:.2f} km)"
            )
            print(
                f"Voyage {v['voyage_id']} 終點最近港口: {arr_port} ({arr_min_dist:.2f} km)"
            )

        # 如果 unknown → 標記 invalid，但不要覆蓋原本的 invalid_reason
        if (dep_port == "unknown" or arr_port == "unknown") and pd.isna(
            v["invalid_reason"]
        ):
            voyages.loc[voyages["voyage_id"] == v["voyage_id"], "valid_flag"] = False
            voyages.loc[voyages["voyage_id"] == v["voyage_id"], "invalid_reason"] = (
                "R7_unknown_port"
            )

        origin_ports.append(dep_port)
        dest_ports.append(arr_port)

    voyages["origin_port"] = origin_ports
    voyages["destination_port"] = dest_ports

    return voyages


# 測試用
if __name__ == "__main__":
    print("這是 od_marker 模組，請在 main.py 中 import 使用。")
