"""
data_loader.py
---------------
負責讀取與前處理 AIS 原始資料
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd

# 預設參數（可依需要修改）
EARTH_RADIUS_KM = 6371.0088
CSV_PATH = Path(r"C:\Users\slab\Desktop") / "Slab Project" / "Stage1" / "data" / "Device_AB00035.csv"
TARGET_MMSI = 416426000
FIRST_STAGE_EPS = 0.01  # radians; ~63 km
FIRST_STAGE_MIN_SAMPLES = 10
SECOND_STAGE_EPS_KM = 1.0  # merge harbour centers within 1 km
LAND_FILE = Path(r"C:\Users\slab\Desktop\Slab Project\Stage1\data\Land\ne_10m_land.shp")

# 載入陸地 shapefile（有需要的時候才用）
land = gpd.read_file(LAND_FILE)


def load_and_preprocess(csv_path: Path = CSV_PATH, target_mmsi: int = TARGET_MMSI) -> pd.DataFrame:
    """
    載入原始 AIS 資料並進行清理與前處理。

    參數:
        csv_path (Path): AIS CSV 檔案路徑
        target_mmsi (int): 目標船舶的 MMSI

    回傳:
        pd.DataFrame: 前處理後的 AIS 資料
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["MMSI"] == target_mmsi].copy()

    # 基本欄位清理
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Long"] = pd.to_numeric(df["Long"], errors="coerce")
    df["Long_360"] = df["Long"] % 360
    df["Sog"] = pd.to_numeric(df["Sog"], errors="coerce")
    df = df.dropna(subset=["Lat", "Long", "Sog"])

    # 經緯度範圍過濾
    df = df[df["Lat"].between(-90.0, 90.0)]
    df = df[df["Long"].between(-180.0, 180.0)]

    # 時間處理
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"].astype(str),
        format="%Y%m%d%H%M%S",
        errors="coerce"
    )
    df = df.dropna(subset=["Timestamp"])

    # 排序（用穩定排序）
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)

    return df


# 測試用：當直接執行這個檔案時，跑一次看看
if __name__ == "__main__":
    df = load_and_preprocess()
    print("資料筆數:", len(df))
    print(df.head())
