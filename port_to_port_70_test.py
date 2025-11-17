"""
測試 searoute 對 70 條航線的可行性

需求：
- 使用 searoute 套件，對 port2port_70.xlsx 裡的 70 條「起迄港」做路徑計算
- 港口對應的經緯度存在 filtered_ports.csv
- 找出哪些航線 searoute 會回傳空值（或失敗）

使用前請確認：
1. 已經安裝 searoute：
   pip install searoute pandas openpyxl

2. port2port_70.xlsx 與 filtered_ports.csv 跟本程式在同一個資料夾
   或調整 ROUTES_FILE / PORTS_FILE 的路徑
"""

import pandas as pd
import searoute as sr


# ===================== 使用者可調整區 =====================

# 檔案路徑
PORTS_FILE = r"C:\Users\slab\Desktop\Slab Project\Stage1\data\ports70.csv"
ROUTES_FILE = r"C:\Users\slab\Desktop\Slab Project\Stage1\data\port2port_70.csv"

# 若自動偵測欄位失敗，可以手動指定以下欄位名稱
# （全部設成 None 代表先嘗試自動偵測）
MANUAL_PORT_CODE_COL = "locode"   # 例如 "UNLOCODE" 或 "port_code"
MANUAL_LON_COL = "lon"         # 例如 "lon" 或 "Longitude" 或 "port_lon"
MANUAL_LAT_COL = "lat"       # 例如 "lat" 或 "Latitude" 或 "port_lat"

MANUAL_ORIGIN_COL = "From_code"      # 例如 "origin", "FromPort", "起運港"
MANUAL_DEST_COL = "To_code"        # 例如 "destination", "ToPort", "目的港"

# searoute 參數（依你需求調整）
SEAROUTE_UNITS = "naut"       # 距離單位：'km', 'naut', 'mi' ...
SEAROUTE_SPEED_KNOT = None    # 如果不需要時間估計，可以維持 None


# ===================== 輔助函式 =====================

def detect_col(df: pd.DataFrame, candidates, desc: str) -> str:
    """
    嘗試從 DataFrame 欄位中找到候選名稱之一。
    找不到就拋出錯誤，請你手動去 MANUAL_* 裡指定。
    """
    for c in candidates:
        if c in df.columns:
            print(f"[info] 偵測到 {desc} 欄位：'{c}'")
            return c
    raise ValueError(
        f"在欄位中找不到任何 {desc} 欄位候選：{candidates}\n"
        f"目前欄位：{list(df.columns)}\n"
        f"請到程式上方 MANUAL_{desc.upper()}_COL 手動指定。"
    )


def detect_port_columns(df_ports: pd.DataFrame):
    """自動或手動偵測：港口代碼欄、經度欄、緯度欄"""
    # 港口代碼欄
    if MANUAL_PORT_CODE_COL is not None:
        port_code_col = MANUAL_PORT_CODE_COL
        print(f"[info] 使用手動指定港口代碼欄位：'{port_code_col}'")
    else:
        port_code_col = detect_col(
            df_ports,
            candidates=[
                "port_code", "PortCode", "UNLOCODE", "unlocode", "locode",
                "Port", "port", "港口代碼"
            ],
            desc="port_code"
        )

    # 經度欄
    if MANUAL_LON_COL is not None:
        lon_col = MANUAL_LON_COL
        print(f"[info] 使用手動指定經度欄位：'{lon_col}'")
    else:
        lon_col = detect_col(
            df_ports,
            candidates=[
                "lon", "Lon", "longitude", "Longitude", "LONGITUDE",
                "port_lon", "x", "X", "lng", "Lng"
            ],
            desc="lon"
        )

    # 緯度欄
    if MANUAL_LAT_COL is not None:
        lat_col = MANUAL_LAT_COL
        print(f"[info] 使用手動指定緯度欄位：'{lat_col}'")
    else:
        lat_col = detect_col(
            df_ports,
            candidates=[
                "lat", "Lat", "latitude", "Latitude", "LATITUDE",
                "port_lat", "y", "Y"
            ],
            desc="lat"
        )

    return port_code_col, lon_col, lat_col


def detect_route_columns(df_routes: pd.DataFrame):
    """自動或手動偵測：起港欄、迄港欄"""
    if MANUAL_ORIGIN_COL is not None:
        origin_col = MANUAL_ORIGIN_COL
        print(f"[info] 使用手動指定起港欄位：'{origin_col}'")
    else:
        origin_col = detect_col(
            df_routes,
            candidates=[
                "origin", "Origin", "from", "From",
                "start", "Start",
                "origin_port", "OriginPort", "FromPort",
                "起點", "起運港"
            ],
            desc="origin"
        )

    if MANUAL_DEST_COL is not None:
        dest_col = MANUAL_DEST_COL
        print(f"[info] 使用手動指定迄港欄位：'{dest_col}'")
    else:
        dest_col = detect_col(
            df_routes,
            candidates=[
                "destination", "Destination", "to", "To",
                "end", "End",
                "dest_port", "DestPort", "ToPort",
                "終點", "目的港"
            ],
            desc="dest"
        )

    return origin_col, dest_col


def build_port_lookup(df_ports, port_code_col, lon_col, lat_col):
    """
    建立：port_code -> (lon, lat) 的查詢 dict
    若有重複港口代碼，取第一筆。
    """
    df_ports = df_ports.copy()
    # 去掉空值
    df_ports = df_ports.dropna(subset=[port_code_col, lon_col, lat_col])

    # 保留第一筆
    df_ports = df_ports.drop_duplicates(subset=[port_code_col], keep="first")

    port_to_coord = {}
    for _, row in df_ports.iterrows():
        code = str(row[port_code_col]).strip()
        lon = float(row[lon_col])
        lat = float(row[lat_col])
        port_to_coord[code] = (lon, lat)

    print(f"[info] 建立港口查詢表，共 {len(port_to_coord)} 個港口座標。")
    return port_to_coord


def call_searoute(origin_ll, dest_ll):
    """
    封裝 searoute 呼叫：
    - origin_ll, dest_ll = (lon, lat)
    - 回傳 (success: bool, route, error_msg: str)
    """
    try:
        kwargs = {"units": SEAROUTE_UNITS}
        if SEAROUTE_SPEED_KNOT is not None:
            kwargs["speed_knot"] = SEAROUTE_SPEED_KNOT

        route = sr.searoute(
            list(origin_ll),
            list(dest_ll),
            **kwargs
        )

        # searoute 通常回傳 GeoJSON Feature
        if route is None:
            return False, None, "searoute_returned_None"

        geom = route.get("geometry", None)
        if geom is None:
            return False, route, "missing_geometry"

        coords = geom.get("coordinates", None)
        if not coords:
            return False, route, "empty_coordinates"

        # 成功
        return True, route, ""

    except Exception as e:
        return False, None, f"exception: {repr(e)}"


# ===================== 主程式 =====================

def main():
    print("=== 讀取港口檔案 ===")
    df_ports = pd.read_csv(PORTS_FILE)
    port_code_col, lon_col, lat_col = detect_port_columns(df_ports)
    port_lookup = build_port_lookup(df_ports, port_code_col, lon_col, lat_col)

    print("\n=== 讀取 70 條航線檔案 ===")
    df_routes = pd.read_csv(ROUTES_FILE)
    origin_col, dest_col = detect_route_columns(df_routes)

    results = []

    print("\n=== 開始逐航線測試 searoute ===")
    for idx, row in df_routes.iterrows():
        origin_port = str(row[origin_col]).strip()
        dest_port = str(row[dest_col]).strip()

        key = f"{origin_port}->{dest_port}"

        # 檢查港口是否有座標
        if origin_port not in port_lookup:
            results.append({
                "index": idx,
                "origin_port": origin_port,
                "dest_port": dest_port,
                "status": "fail",
                "reason": "origin_port_not_in_ports_file",
                "error_detail": "",
            })
            print(f"[FAIL] {key} - 起港座標缺失")
            continue

        if dest_port not in port_lookup:
            results.append({
                "index": idx,
                "origin_port": origin_port,
                "dest_port": dest_port,
                "status": "fail",
                "reason": "dest_port_not_in_ports_file",
                "error_detail": "",
            })
            print(f"[FAIL] {key} - 迄港座標缺失")
            continue

        origin_ll = port_lookup[origin_port]  # (lon, lat)
        dest_ll = port_lookup[dest_port]      # (lon, lat)

        success, route, err = call_searoute(origin_ll, dest_ll)

        if success:
            # 路徑長度（如果需要）
            length = route.get("properties", {}).get("length", None)
            results.append({
                "index": idx,
                "origin_port": origin_port,
                "dest_port": dest_port,
                "status": "ok",
                "reason": "",
                "error_detail": "",
                "length": length,
            })
            print(f"[ OK ] {key} - 長度: {length}")
        else:
            results.append({
                "index": idx,
                "origin_port": origin_port,
                "dest_port": dest_port,
                "status": "fail",
                "reason": "searoute_failed",
                "error_detail": err,
            })
            print(f"[FAIL] {key} - {err}")

    print("\n=== 統計結果 ===")
    df_res = pd.DataFrame(results)
    n_total = len(df_res)
    n_ok = (df_res["status"] == "ok").sum()
    n_fail = (df_res["status"] == "fail").sum()

    print(f"總航線數：{n_total}")
    print(f"  成功：{n_ok}")
    print(f"  失敗：{n_fail}")

    # 儲存結果
    out_file = "searoute_70_results.csv"
    df_res.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n結果已輸出到：{out_file}")

    # 額外印出所有失敗的航線列表
    if n_fail > 0:
        print("\n=== searoute 失敗 / 回傳空值的航線 ===")
        df_fail = df_res[df_res["status"] == "fail"]
        for _, r in df_fail.iterrows():
            print(f"- index={r['index']} | {r['origin_port']} -> {r['dest_port']} | reason={r['reason']} | detail={r['error_detail']}")


if __name__ == "__main__":
    main()
