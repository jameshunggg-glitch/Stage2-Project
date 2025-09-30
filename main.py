"""
main.py
-------
專案主程式，串接 data_loader 與 voyage_splitter
"""

from pathlib import Path
import webbrowser
import pandas as pd

# 匯入自訂模組
from data_loader import load_and_preprocess
from voyage_splitter import detect_stops, split_voyages, voyage_quality_checker
from visualization import visualize_voyages 


def main():
    # -----------------------------------
    # Step 1: 載入與前處理
    # -----------------------------------
    csv_path = Path(r"C:\Users\slab\Desktop\Slab Project\Stage1\data\Device_AB00035.csv")
    target_mmsi = 416426000

    print("載入與前處理資料中...")
    df = load_and_preprocess(csv_path, target_mmsi)
    print(f"清理後資料筆數: {len(df)}")

    # -----------------------------------
    # Step 2: 停泊判斷
    # -----------------------------------
    print("執行停泊判斷...")
    stop_segments = detect_stops(df)
    print(f"偵測到停泊區段數量: {len(stop_segments)}")

    # -----------------------------------
    # Step 3: 航程切分
    # -----------------------------------
    print("切分航程...")
    df_with_voyages, voyages = split_voyages(df, stop_segments)
    print(f"切分後航程數量: {len(voyages)}")

    # -----------------------------------
    # Step 4: 航程 QA
    # -----------------------------------
    print("執行航程 QA 檢查...")
    voyages_qc = voyage_quality_checker(df_with_voyages, voyages)
    print(f"QA 完成，有效航程數量: {voyages_qc['valid_flag'].sum()}")

    # -----------------------------------
    # Step 5: 合併 QA 結果
    # -----------------------------------
    voyages = voyages.merge(voyages_qc, on="voyage_id", how="left")

    # 畫圖
    visualize_voyages(df_with_voyages, voyages, "voyages_map.html")
    import webbrowser
    webbrowser.open("voyages_map.html")

    # -----------------------------------
    # Step 6: 輸出結果
    # -----------------------------------
    output_path = Path("output_voyages.csv")
    voyages.to_csv(output_path, index=False)
    print(f"航程摘要輸出完成: {output_path.resolve()}")

    return df_with_voyages, voyages


if __name__ == "__main__":
    df_final, voyages_final = main()
