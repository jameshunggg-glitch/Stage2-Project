# voyage_port_pipeline.py
# -------------------------------------------------------------
# 目的：把 AIS 切出的停泊段 enrich → 停泊分類(錨泊/靠泊) → 合併為港口事件
#       → 以港口事件切出航程 → 用起訖代表點雙重驗證 → 產出航程與QA
# 可匯入使用，也可直接在最下方 run 小測試（留白）
# -------------------------------------------------------------

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

# 外部工具：最近港口與聚類
from sklearn.neighbors import BallTree
from haversine import haversine

# -------------------------------------------------------------
# 基礎小工具（與你現有風格一致）
# -------------------------------------------------------------
def _to_lon180(lon: float) -> float:
    return ((lon + 180) % 360) - 180

def _safe_haversine(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    # c1,c2 = (lat, lon in deg)
    return haversine((c1[0], _to_lon180(c1[1])), (c2[0], _to_lon180(c2[1])))

def _build_balltree(ports_df: pd.DataFrame) -> Tuple[BallTree, np.ndarray]:
    """建立港口 BallTree（haversine metric，輸入 radians）"""
    assert {'lat','lon'}.issubset(set(map(str.lower, ports_df.columns.str.lower()))), \
        "ports_df 需包含 lat, lon 欄位（小數度）"
    lat = ports_df[[c for c in ports_df.columns if c.lower()=='lat'][0]].to_numpy()
    lon = ports_df[[c for c in ports_df.columns if c.lower()=='lon'][0]].to_numpy()
    X = np.deg2rad(np.c_[lat, lon])  # (n,2) radians
    tree = BallTree(X, metric='haversine')
    return tree, X

def _nearest_port(lat: float, lon: float, ports_df: pd.DataFrame, tree: BallTree) -> Tuple[str, float, int]:
    """回傳 (port_name/locode優先, 距離km, index)"""
    lat_rad = math.radians(lat); lon_rad = math.radians(_to_lon180(lon))
    dist_rad, idx = tree.query([[lat_rad, lon_rad]], k=1)
    km = dist_rad[0,0] * 6371.0088
    j = int(idx[0,0])
    # 名稱欄位優先順序
    name = None
    for col in ['port_name','name','Name','PORT_NAME','locode','LOCODE']:
        if col in ports_df.columns:
            name = str(ports_df.iloc[j][col])
            break
    if name is None:
        name = f"PORT_{j}"
    return name, km, j

def _segment_time(df: pd.DataFrame, idx_range: range) -> Tuple[pd.Timestamp, pd.Timestamp]:
    s = df.loc[idx_range[0], 'Timestamp']
    e = df.loc[idx_range[-1], 'Timestamp']
    return s, e

def _heading_std(series: pd.Series) -> float:
    """Heading 標準差（deg），處理 0/360 wrap 與 511 缺值"""
    if series is None or series.isna().all():
        return np.nan
    hdg = series.replace(511, np.nan).dropna().to_numpy()
    if len(hdg) < 2:
        return 0.0
    # unwrap
    diffs = np.diff(hdg)
    diffs = np.where(diffs > 180, diffs - 360, diffs)
    diffs = np.where(diffs < -180, diffs + 360, diffs)
    # 重建一條去跳變的序列
    hdg_unwrap = np.r_[hdg[0], hdg[0] + np.cumsum(diffs)]
    return float(np.std(hdg_unwrap))

def _r95_km(df: pd.DataFrame, idx_range: range, lat_med: float, lon_med: float) -> float:
    d = [
        _safe_haversine((lat_med, lon_med), (df.loc[j,'Lat'], df.loc[j,'Long']))
        for j in idx_range
    ]
    return float(np.percentile(d, 95)) if len(d) else 0.0

def _d_seg_km(df: pd.DataFrame, idx_range: range) -> float:
    # 停泊段內最遠兩點距離（近似，取抽樣提高效率）
    idxs = list(idx_range)
    if len(idxs) <= 1:
        return 0.0
    # 抽樣上限，避免 O(n^2)
    S = idxs if len(idxs) <= 200 else np.random.choice(idxs, 200, replace=False)
    coords = [(df.loc[i,'Lat'], df.loc[i,'Long']) for i in S]
    maxd = 0.0
    for i in range(len(coords)):
        for k in range(i+1, len(coords)):
            maxd = max(maxd, _safe_haversine(coords[i], coords[k]))
    return float(maxd)

# -------------------------------------------------------------
# LAYER 2: enrich 停泊段 → stop_events DataFrame
# -------------------------------------------------------------
def build_stop_events(
    df: pd.DataFrame,
    stop_segments: List[range],
    ports_df: pd.DataFrame,
    add_shore_distance: bool=False,
    land_gdf=None
) -> pd.DataFrame:
    """
    將 detect_stops 的 range list 轉為明確的停泊事件表，並計算特徵：
    - start_time, end_time, duration_sec
    - center_lat, center_lon, r95_km, d_seg_km
    - dh_std (Heading 標準差)
    - 最近港口：nearest_port, dist_to_port_km
    - (選) 最近岸線距離：dist_to_shore_km
    """
    if len(stop_segments) == 0:
        return pd.DataFrame(columns=[
            'stop_id','MMSI','start_idx','end_idx','start_time','end_time','duration_sec',
            'center_lat','center_lon','r95_km','d_seg_km','dh_std',
            'nearest_port','dist_to_port_km','dist_to_shore_km','stop_type'
        ])

    df = df.reset_index(drop=False).rename(columns={'index':'orig_index'})
    have_mmsi = 'MMSI' in df.columns
    tree, _ = _build_balltree(ports_df)

    rows = []
    for sid, seg in enumerate(stop_segments, start=1):
        s, e = _segment_time(df, seg)
        duration = (e - s).total_seconds()
        lat_med = float(df.loc[list(seg), 'Lat'].median())
        lon_med = float(df.loc[list(seg), 'Long'].median())
        r95 = _r95_km(df, seg, lat_med, lon_med)
        dseg = _d_seg_km(df, seg)
        dhs = _heading_std(df.loc[list(seg), 'Hdg'] if 'Hdg' in df.columns else pd.Series(dtype=float))
        port_name, dist_km, _j = _nearest_port(lat_med, lon_med, ports_df, tree)

        shore_km = np.nan
        if add_shore_distance and (land_gdf is not None):
            # 使用 (lon,lat) 幾何最近距離（需 EPSG 投影轉米，再轉 km）
            try:
                from shapely.geometry import Point
                import geopandas as gpd
                p = gpd.GeoSeries([Point(_to_lon180(lon_med), lat_med)], crs="EPSG:4326").to_crs(3857)
                land_3857 = land_gdf.to_crs(3857)
                shore_km = float(land_3857.distance(p.iloc[0]).min() / 1000.0)
            except Exception:
                shore_km = np.nan

        rows.append({
            'stop_id': sid,
            'MMSI': df.loc[seg[0], 'MMSI'] if have_mmsi else None,
            'start_idx': int(df.loc[seg[0],'orig_index']),
            'end_idx': int(df.loc[seg[-1],'orig_index']),
            'start_time': s, 'end_time': e, 'duration_sec': duration,
            'center_lat': lat_med, 'center_lon': lon_med,
            'r95_km': r95, 'd_seg_km': dseg, 'dh_std': dhs,
            'nearest_port': port_name, 'dist_to_port_km': dist_km,
            'dist_to_shore_km': shore_km,
            'stop_type': None,  # 後續填
        })
    return pd.DataFrame(rows)

# -------------------------------------------------------------
# LAYER 3: 停泊類型分類（Rule-based，先簡版、再支持自適應）
# -------------------------------------------------------------
@dataclass
class StopTypeThreshold:
    in_port_km: float = 2.0          # ≤ 2 km 視為貼港
    anchorage_km: float = 15.0       # 2~15 km 常見錨區
    r95_small_km: float = 0.10       # 泊位 r95 通常很小
    r95_anchor_km: float = 0.10      # 錨泊 r95 通常 > 0.1 km
    dh_std_moored_deg: float = 10.0  # 泊位 heading 波動小
    dh_std_anchor_deg: float = 15.0  # 錨泊 heading 波動大

def classify_stop_type(
    stops: pd.DataFrame,
    thresholds: StopTypeThreshold = StopTypeThreshold(),
    adaptive_by_port: bool = False
) -> pd.DataFrame:
    """
    依距港距離 + 幾何擴散 + heading 變異做簡單規則分類
    若 adaptive_by_port=True，會針對每個港口用分位數微調距離門檻（資料夠多時）
    """
    s = stops.copy()

    if adaptive_by_port and not s.empty:
        # 每港口的距離分位數決定 in_port / anchorage 門檻（避免全球一把尺）
        adj = (
            s.groupby('nearest_port')['dist_to_port_km']
             .agg(q25=lambda x: np.nanpercentile(x, 25),
                  q75=lambda x: np.nanpercentile(x, 75))
             .reset_index()
        )
        s = s.merge(adj, on='nearest_port', how='left')
        s['in_port_thr'] = np.where(s['q25'].notna(),
                                    np.minimum(s['q25'], thresholds.in_port_km),
                                    thresholds.in_port_km)
        s['anch_thr'] = np.where(s['q75'].notna(),
                                 np.maximum(s['q75'], thresholds.anchorage_km),
                                 thresholds.anchorage_km)
    else:
        s['in_port_thr'] = thresholds.in_port_km
        s['anch_thr'] = thresholds.anchorage_km

    def _rule(row):
        d = row['dist_to_port_km']
        r95 = row['r95_km']
        dhs = row['dh_std'] if not np.isnan(row['dh_std']) else 0.0

        # Moored（靠泊）
        if (d <= row['in_port_thr']) and (r95 < thresholds.r95_small_km) and (dhs < thresholds.dh_std_moored_deg):
            return 'moored'

        # Anchorage（錨泊）
        if (row['in_port_thr'] < d <= row['anch_thr']) and (r95 >= thresholds.r95_anchor_km) and (dhs >= thresholds.dh_std_anchor_deg):
            return 'anchorage'

        # 其他：離岸/漂移/慢速移動
        return 'offshore'

    s['stop_type'] = s.apply(_rule, axis=1)
    return s

# -------------------------------------------------------------
# LAYER 4: 合併停泊段 → 港口事件（port_event）
# -------------------------------------------------------------
def merge_stops_into_port_events(
    stops: pd.DataFrame,
    same_port_time_gap_hr: float = 6.0
) -> pd.DataFrame:
    """
    規則：同一艘船，若相鄰停泊段的 nearest_port 相同，且間隔 < same_port_time_gap_hr，
    則視為同一次進港事件（把錨泊 + 靠泊合併）
    產出欄位：
        port_event_id, port_name, start_time, end_time,
        anchorage_duration, moored_duration, total_duration
    """
    if stops.empty:
        return pd.DataFrame(columns=[
            'port_event_id','MMSI','port_name','start_time','end_time',
            'anchorage_duration','moored_duration','total_duration',
            'stop_ids'
        ])

    s = stops.sort_values(['MMSI','start_time'], na_position='first').copy()
    groups = []
    curr_id = 1

    def _finalize(g):
        if g.empty: return
        port = g['nearest_port'].mode().iloc[0]
        start = g['start_time'].min()
        end = g['end_time'].max()
        anch = float(g.loc[g['stop_type']=='anchorage', 'duration_sec'].sum())
        moor = float(g.loc[g['stop_type']=='moored', 'duration_sec'].sum())
        groups.append({
            'port_event_id': curr_id_map[id(g)],
            'MMSI': g['MMSI'].iloc[0],
            'port_name': port,
            'start_time': start,
            'end_time': end,
            'anchorage_duration': anch,
            'moored_duration': moor,
            'total_duration': anch + moor,
            'stop_ids': list(g['stop_id'])
        })

    # 按 MMSI 分組處理
    port_event_rows = []
    for mmsi, g0 in s.groupby('MMSI', dropna=False):
        g0 = g0.reset_index(drop=True)
        # 逐段串接
        buckets = []
        curr_bucket = [0]  # row indices in g0
        for i in range(1, len(g0)):
            prev = g0.iloc[i-1]; now = g0.iloc[i]
            same_port = (prev['nearest_port'] == now['nearest_port'])
            gap_hr = (now['start_time'] - prev['end_time']).total_seconds()/3600.0
            if same_port and (gap_hr <= same_port_time_gap_hr):
                curr_bucket.append(i)
            else:
                buckets.append(curr_bucket)
                curr_bucket = [i]
        buckets.append(curr_bucket)

        # 生成事件
        for b in buckets:
            sub = g0.iloc[b]
            port_event_rows.append({
                'port_event_id': None,  # 稍後編號
                'MMSI': mmsi,
                'port_name': sub['nearest_port'].mode().iloc[0],
                'start_time': sub['start_time'].min(),
                'end_time': sub['end_time'].max(),
                'anchorage_duration': float(sub.loc[sub['stop_type']=='anchorage','duration_sec'].sum()),
                'moored_duration': float(sub.loc[sub['stop_type']=='moored','duration_sec'].sum()),
                'total_duration': float(sub['duration_sec'].sum()),
                'stop_ids': list(sub['stop_id'])
            })

    pe = pd.DataFrame(port_event_rows)
    if pe.empty:
        return pe
    # 重新連號
    pe = pe.sort_values(['MMSI','start_time']).reset_index(drop=True)
    pe['port_event_id'] = np.arange(1, len(pe)+1, dtype=int)
    return pe

# -------------------------------------------------------------
# LAYER 5: 以港口事件切航程 + 代表點雙重驗證
# -------------------------------------------------------------
def split_voyages_by_port_events(
    df: pd.DataFrame,
    port_events: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    用 port_events[i] → port_events[i+1] 作為一段 voyage
    回傳：
      df_with_vid: 在 df 上加 voyage_id 欄位
      voyages: 摘要表（含 origin/dest port 與時間）
    """
    if port_events.empty:
        d2 = df.copy()
        d2['voyage_id'] = np.nan
        return d2, pd.DataFrame(columns=[
            'voyage_id','MMSI','origin_port','dest_port',
            'dep_time','arr_time','origin_event_id','dest_event_id',
            'origin_idx','dest_idx'
        ])

    have_mmsi = 'MMSI' in df.columns
    d2 = df.copy()
    d2['voyage_id'] = np.nan

    voyages = []
    vid = 1

    for mmsi, ev in port_events.sort_values(['MMSI','start_time']).groupby('MMSI', dropna=False):
        ev = ev.reset_index(drop=True)
        for i in range(len(ev)-1):
            e0 = ev.iloc[i]
            e1 = ev.iloc[i+1]
            # 航程時間窗 = (e0.end_time, e1.start_time)
            mask = (d2['Timestamp'] > e0['end_time']) & (d2['Timestamp'] < e1['start_time'])
            if have_mmsi:
                mask &= (d2['MMSI'] == mmsi)
            seg_idx = d2.index[mask]
            if len(seg_idx) == 0:
                continue
            d2.loc[seg_idx, 'voyage_id'] = vid

            origin_idx = int(seg_idx[0])   # 代表點（可調整）
            dest_idx   = int(seg_idx[-1])

            voyages.append({
                'voyage_id': vid,
                'MMSI': mmsi,
                'origin_port': e0['port_name'],
                'dest_port': e1['port_name'],
                'dep_time': d2.loc[origin_idx, 'Timestamp'],
                'arr_time': d2.loc[dest_idx, 'Timestamp'],
                'origin_event_id': int(e0['port_event_id']),
                'dest_event_id': int(e1['port_event_id']),
                'origin_idx': origin_idx,
                'dest_idx': dest_idx
            })
            vid += 1

    return d2, pd.DataFrame(voyages)

def validate_with_origin_dest_points(
    voyages: pd.DataFrame,
    df: pd.DataFrame,
    ports_df: pd.DataFrame,
    tree: Optional[BallTree]=None,
    mismatch_km_tol: float = 5.0
) -> pd.DataFrame:
    """
    用每條航程的 origin_idx / dest_idx 代表點 → 最近港口，跟 port_event 推論對比
    若不一致且差距明顯 → 標記 R7_port_mismatch
    """
    if voyages.empty:
        return voyages
    if tree is None:
        tree, _ = _build_balltree(ports_df)

    v = voyages.copy()
    nearest_from_points = []
    for _, row in v.iterrows():
        o = (df.loc[row['origin_idx'], 'Lat'], df.loc[row['origin_idx'], 'Long'])
        d = (df.loc[row['dest_idx'],   'Lat'], df.loc[row['dest_idx'],   'Long'])
        op, okm, _ = _nearest_port(o[0], o[1], ports_df, tree)
        dp, dkm, _ = _nearest_port(d[0], d[1], ports_df, tree)
        nearest_from_points.append((op, okm, dp, dkm))

    v[['origin_port_pt','origin_port_pt_km','dest_port_pt','dest_port_pt_km']] = \
        pd.DataFrame(nearest_from_points, index=v.index)

    # R7: 若 dest_port 與 dest_port_pt 不同，且最近點距離仍很近（代表真的在另一港）
    v['R7_port_mismatch'] = (
        (v['dest_port'] != v['dest_port_pt']) &
        (v['dest_port_pt_km'] <= mismatch_km_tol)
    )
    return v

# -------------------------------------------------------------
# 一條龍管線（你可以在 main.py 呼叫）
# -------------------------------------------------------------
@dataclass
class PipelineConfig:
    same_port_time_gap_hr: float = 6.0
    thresholds: StopTypeThreshold = StopTypeThreshold()
    adaptive_by_port: bool = False
    mismatch_km_tol: float = 5.0
    compute_shore_distance: bool = False # 需要 land_gdf 才能運作

def run_port_voyage_pipeline(
    df: pd.DataFrame,
    stop_segments: List[range],
    ports_df: pd.DataFrame,
    land_gdf=None,
    cfg: PipelineConfig = PipelineConfig()
) -> Dict[str, pd.DataFrame]:
    """
    輸入：
      df: 你的 AIS 全表（至少含 Lat, Long, Sog, Timestamp；若含 MMSI/Hdg 更好）
      stop_segments: 由 detect_stops(...) 算出來的停泊range清單（沿用你現有邏輯）
      ports_df: 港口清單（至少 lat, lon，最好還有 port_name/locode）
    回傳：
      {
        'stops': 停泊事件表（含特徵與 stop_type）,
        'port_events': 港口事件表（錨泊+靠泊合併）,
        'df_with_vid': 在 df 上貼上 voyage_id,
        'voyages': 航程摘要（含起迄港與代表點index）,
        'voyages_validated': 加上代表點雙重驗證欄位
      }
    """
    # Layer 2
    stops = build_stop_events(
        df, stop_segments, ports_df,
        add_shore_distance=cfg.compute_shore_distance,
        land_gdf=land_gdf
    )

    # Layer 3
    stops = classify_stop_type(
        stops,
        thresholds=cfg.thresholds,
        adaptive_by_port=cfg.adaptive_by_port
    )

    # Layer 4
    port_events = merge_stops_into_port_events(
        stops, same_port_time_gap_hr=cfg.same_port_time_gap_hr
    )

    # Layer 5 (a): 以港口事件切航程
    df_with_vid, voyages = split_voyages_by_port_events(df, port_events)

    # Layer 5 (b): 代表點雙重驗證
    validated = validate_with_origin_dest_points(
        voyages, df, ports_df, tree=None, mismatch_km_tol=cfg.mismatch_km_tol
    )

    return {
        'stops': stops,
        'port_events': port_events,
        'df_with_vid': df_with_vid,
        'voyages': voyages,
        'voyages_validated': validated
    }

# -------------------------------------------------------------
# (可選) 簡單 smoke test 區塊：留空，避免干擾匯入
# -------------------------------------------------------------
if __name__ == "__main__":
    print("voyage_port_pipeline 模組已載入。請在你的 main.py 中呼叫 run_port_voyage_pipeline。")
