"""
data_loader.py
---------------
負責讀取與前處理 AIS 原始資料
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

# 常數
EARTH_RADIUS_KM = 6371.0088
FIRST_STAGE_EPS = 0.01  # radians; ~63 km
FIRST_STAGE_MIN_SAMPLES = 10
SECOND_STAGE_EPS_KM = 1.0  # merge harbour centers within 1 km
LAND_FILE = Path(r"C:\Users\slab\Desktop\Slab Project\Stage1\data\Land\ne_10m_land.shp")
PORTS_CSV = Path(r"C:\Users\slab\Desktop\Slab Project\Stage1\data\filtered_ports.csv")

# 載入陸地 shapefile（可延遲載入）
land = gpd.read_file(LAND_FILE)


# --------------------------------------------------------
#  原始穩定版時間解析函式
# --------------------------------------------------------
def parse_timestamp(series: pd.Series) -> pd.Series:
    """
    嘗試解析時間欄位，支援:
    - AIS 格式: 20250711081924
    - ISO 格式: 2025-07-11T08:19:24
    """
    # 先全部轉成字串，避免 int 被誤當成 epoch nanoseconds
    series = series.astype(str).str.strip()

    # 先試著用 AIS 格式 (14位數)
    mask_ais = series.str.match(r"^\d{14}$")
    ts = pd.to_datetime(series[mask_ais], format="%Y%m%d%H%M%S", errors="coerce")

    # 其他格式再用一般解析
    ts_other = pd.to_datetime(series[~mask_ais], errors="coerce")

    # 合併結果
    ts = pd.concat([ts, ts_other]).sort_index()

    return ts


# --------------------------------------------------------
#  主流程：載入與前處理
# --------------------------------------------------------
def load_and_preprocess(csv_path: Path, target_mmsi: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    # -----------------------------------
    # 經度欄位名稱處理
    # -----------------------------------
    if "Long" in df.columns:
        pass
    elif "Lng" in df.columns:
        df = df.rename(columns={"Lng": "Long"})
    else:
        print(f" 找不到經度欄位，欄位清單: {list(df.columns)}")
        raise ValueError("請確認經度欄位名稱 (必須是 Long 或 Lng)")

    # -----------------------------------
    # 時間欄位名稱處理
    # -----------------------------------
    if "Timestamp" in df.columns:
        pass
    elif "DataSourceLastTime_UTC" in df.columns:
        df = df.rename(columns={"DataSourceLastTime_UTC": "Timestamp"})
    else:
        print(f" 找不到時間欄位，欄位清單: {list(df.columns)}")
        raise ValueError("請確認時間欄位名稱 (必須是 Timestamp 或 DataSourceLastTime_UTC)")

    # 現在才用 MMSI 篩選
    #df = df[df["MMSI"] == target_mmsi].copy()
    #  強制轉成整數後比對（避免 636012794 vs 636012794.0）
    df = df[df["MMSI"].round().astype("Int64") == int(target_mmsi)].copy()


    # 基本欄位清理
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Long"] = pd.to_numeric(df["Long"], errors="coerce")
    df["Long_360"] = df["Long"] % 360
    df["Sog"] = pd.to_numeric(df["Sog"], errors="coerce")
    df = df.dropna(subset=["Lat", "Long", "Sog"])

    # 經緯度範圍過濾
    df = df[df["Lat"].between(-90.0, 90.0)]
    df = df[df["Long"].between(-180.0, 180.0)]

    # 時間處理 (自動判斷格式)
    df["Timestamp"] = parse_timestamp(df["Timestamp"])
    df = df.dropna(subset=["Timestamp"])

    # 排序
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)

    return df


# --------------------------------------------------------
# 測試用（可單獨執行）
# --------------------------------------------------------
if __name__ == "__main__":
    sample_path = Path(r"C:\Users\slab\Desktop") / "Slab Project" / "Stage1" / "data" / "370359000.csv"
    sample_mmsi = 370359000
    df = load_and_preprocess(sample_path, sample_mmsi)
    print("資料筆數:", len(df))
    print(df.head())
