"""
visualization.py
----------------
航程視覺化模組 (使用 Folium 地圖)
"""

import folium
import pandas as pd


def visualize_voyages(df: pd.DataFrame, voyages: pd.DataFrame, map_filename: str = "voyages_map.html"):
    """
    將航程資料可視化在地圖上。

    參數:
        df (pd.DataFrame): 帶有 voyage_id 的 AIS dataframe
        voyages (pd.DataFrame): 航程表 (需含 voyage_id, valid_flag, invalid_reason)
        map_filename (str): 輸出 HTML 地圖檔案路徑

    回傳:
        folium.Map: Folium 地圖物件
    """
    if df.empty or voyages.empty:
        print("No voyages to visualize.")
        return None

    # 設定地圖中心點
    center_lat = df["Lat"].mean()
    center_lon = df["Long_360"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap")

    for _, v in voyages.iterrows():
        seg = df[df["voyage_id"] == v["voyage_id"]]
        if seg.empty:
            continue

        coords = list(zip(seg["Lat"], seg["Long_360"]))
        color = "blue" if v.get("valid_flag", True) else "red"

        # 航程線
        folium.PolyLine(
            coords, color=color, weight=3, opacity=0.7,
            popup=(f"Voyage {v['voyage_id']} | "
                   f"Valid={v.get('valid_flag', 'N/A')} | "
                   f"Reason={v.get('invalid_reason', 'N/A')}")
        ).add_to(m)

        # 起點 marker
        folium.Marker(
            location=[seg.iloc[0]["Lat"], seg.iloc[0]["Long_360"]],
            icon=folium.Icon(color="green", icon="play"),
            popup=f"Voyage {v['voyage_id']} Start"
        ).add_to(m)

        # 終點 marker
        folium.Marker(
            location=[seg.iloc[-1]["Lat"], seg.iloc[-1]["Long_360"]],
            icon=folium.Icon(color="orange", icon="stop"),
            popup=f"Voyage {v['voyage_id']} End"
        ).add_to(m)

    # 輸出地圖
    m.save(map_filename)
    print(f"Map saved to {map_filename}")
    return m


def visualize_voyages_with_gaps(df: pd.DataFrame, voyages: pd.DataFrame, map_filename: str = "voyages_map.html"):
    """
    將航程資料可視化在地圖上，並標示 gap 段落。
    gap 顏色根據 gap_type 區分：
        - small_time_gap: 黃色
        - mid_time_gap: 橘色
        - large_time_gap: 紅色
    """
    if df.empty or voyages.empty:
        print("No voyages to visualize.")
        return None

    # 地圖中心
    center_lat = df["Lat"].mean()
    center_lon = df["Long_360"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap")

    for _, v in voyages.iterrows():
        seg = df[df["voyage_id"] == v["voyage_id"]]
        if seg.empty:
            continue

        # 主航線（藍色）
        coords = list(zip(seg["Lat"], seg["Long_360"]))
        folium.PolyLine(
            coords, color="blue", weight=3, opacity=0.5,
            popup=f"Voyage {v['voyage_id']} (Valid={v.get('valid_flag', 'N/A')})"
        ).add_to(m)

        # 起點 / 終點
        folium.Marker(
            location=[seg.iloc[0]["Lat"], seg.iloc[0]["Long_360"]],
            icon=folium.Icon(color="green", icon="play"),
            popup=f"Voyage {v['voyage_id']} Start"
        ).add_to(m)

        folium.Marker(
            location=[seg.iloc[-1]["Lat"], seg.iloc[-1]["Long_360"]],
            icon=folium.Icon(color="orange", icon="stop"),
            popup=f"Voyage {v['voyage_id']} End"
        ).add_to(m)

        # 如果該航程有 gap，就畫出各段 gap 線
        gap_list = v.get("gap_list", [])
        if gap_list:
            for gap in gap_list:
                a = gap["A"]
                b = gap["B"]
                gap_type = gap["gap_type"]
                gap_hr = gap["gap_hr"]

                # 顏色根據 gap_type
                if gap_type == "small_time_gap":
                    gap_color = "yellow"
                elif gap_type == "mid_time_gap":
                    gap_color = "orange"
                elif gap_type == "large_time_gap":
                    gap_color = "red"
                else:
                    gap_color = "gray"

                folium.PolyLine(
                    [(a["lat"], a["lon"]), (b["lat"], b["lon"])],
                    color=gap_color, weight=5, opacity=0.9,
                    popup=f"Gap: {gap_type}<br>{gap_hr:.2f} hr"
                ).add_to(m)

    m.save(map_filename)
    print(f"Map with gaps saved to {map_filename}")
    return m



if __name__ == "__main__":
    print("這是 visualization 模組，請在 main.py 中 import 使用。")
