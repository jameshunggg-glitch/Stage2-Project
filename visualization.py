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


if __name__ == "__main__":
    print("這是 visualization 模組，請在 main.py 中 import 使用。")
