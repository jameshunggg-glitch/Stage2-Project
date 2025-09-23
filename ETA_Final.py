import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
#import movingpandas as mpd
from shapely.geometry import Point
from datetime import datetime, timedelta
import webbrowser
import numpy as np
import os
from haversine import haversine
from math import radians, sin, cos, sqrt, atan2
##讀檔案
file_path = r"C:\Users\slab\Desktop\Slab Project\Stage2 ETA\Raw Data\477769500_VesselHistoryLineInfo.csv"
try:
    df
except NameError:
    df = pd.read_csv(file_path, low_memory=False)
df = pd.read_csv(file_path, low_memory=False)

# 只畫出477769500這艘船的航線圖，先把其他的洗掉
mmsi_target = 477769500
df_filtered = df[df['MMSI'] == mmsi_target].copy()
#print(len(df_filtered))
#print(df_filtered.head())

df_filtered = df_filtered[
    (df_filtered['Lat'] >= -90) & (df_filtered['Lat'] <= 90) &
    (df_filtered['Lng'] >= -180) & (df_filtered['Lng'] <= 180)
].copy()

# 再檢查經緯度是否都合理
#print(df_filtered[['Lat', 'Lng']].describe())

# Data Preprocessing
# 經緯度轉數值型態，時間轉datetime
df_filtered['Lat'] = df_filtered['Lat'].astype(float)
df_filtered['Lng'] = df_filtered['Lng'].astype(float)
df_filtered['CreateTime'] = pd.to_datetime(df_filtered['CreateTime'], errors='coerce')
# 刪除包含NaN的行
df_filtered = df_filtered.dropna(subset=['Lat', 'Lng', 'CreateTime'])
# 為了不讓圖片上線斷掉，將經度[-180, 180] 轉成 [0, 360]
df_filtered['Lng_360'] = df_filtered['Lng'] % 360
# 計算地圖中心
map_center_lat = df_filtered['Lat'].mean()
map_center_lon = df_filtered['Lng_360'].mean()

# 先清理 SOG 異常值
'''''
sog_min, sog_max = 0, 50  # 合理範圍
before_len = len(df_filtered)
df_filtered = df_filtered[(df_filtered['Sog'] >= sog_min) & (df_filtered['Sog'] <= sog_max)].reset_index(drop=True)
after_len = len(df_filtered)
print(f"[INFO] 移除異常 SOG 筆數: {before_len - after_len}, 剩餘: {after_len}")
'''
# 設定時間閾值
threshold = pd.to_datetime("2025-07-25 10:58:10")

# 條件更新 Sog 欄位
df_filtered.loc[df_filtered['CreateTime'] > threshold, 'Sog'] = (
    df_filtered.loc[df_filtered['CreateTime'] > threshold, 'Sog'] / 10
)

df_filtered = df_filtered.sort_values('CreateTime').reset_index(drop=True)

# Trajectory Reconstruction
m = folium.Map(
    location=[map_center_lat, map_center_lon],
    zoom_start=4,
    tiles='OpenStreetMap'
)
coords = df_filtered[['Lat', 'Lng_360']].iloc[::1].values.tolist()  # ::10 可抽樣10倍
folium.PolyLine(
    coords,
    color='blue',
    weight=3,
    opacity=0.7,
    popup=f"航線點數: {len(coords)}"
).add_to(m)

# 起訖點標記

if len(coords) > 0:
    # 起點
    folium.Marker(
        location=coords[0],
        popup=f"起點\n時間: {df_filtered.iloc[0]['CreateTime']}",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)

    # 終點
    folium.Marker(
        location=coords[-1],
        popup=f"終點\n時間: {df_filtered.iloc[-1]['CreateTime']}",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)

html_file = "ship_trajectory_360.html"
m.save(html_file)

# 自動用瀏覽器開啟
webbrowser.open('file://' + os.path.realpath(html_file))

# Feature Engineering

# -----------------------------------
# 確保時間排序
# -----------------------------------
df_filtered = df_filtered.sort_values('CreateTime').reset_index(drop=True)

# -----------------------------------
# 計算經緯度差分
# -----------------------------------
df_filtered['Delta_Lat'] = df_filtered['Lat'].diff().abs()
df_filtered['Delta_Long'] = df_filtered['Lng_360'].diff().abs()

# 計算時間差（秒）
df_filtered['Delta_Time'] = df_filtered['CreateTime'].diff().dt.total_seconds()

# -----------------------------------
# 閾值設定
# -----------------------------------
short_time_thresh = 1.0   # 秒
sog_threshold = 0.5       # knots
rate_threshold = 0.0001   # 經緯度變化閾值

# 修正 delta 經緯度
df_filtered['Adj_Delta_Lat'] = np.where(
    df_filtered['Delta_Time'] > short_time_thresh,
    df_filtered['Delta_Lat'] / df_filtered['Delta_Time'],
    df_filtered['Delta_Lat']
)
df_filtered['Adj_Delta_Long'] = np.where(
    df_filtered['Delta_Time'] > short_time_thresh,
    df_filtered['Delta_Long'] / df_filtered['Delta_Time'],
    df_filtered['Delta_Long']
)

# -----------------------------------
# 停泊判斷
# -----------------------------------
df_filtered['Is_Stop'] = (
    (df_filtered['Sog'] < sog_threshold) &
    (df_filtered['Adj_Delta_Lat'].abs() < rate_threshold) &
    (df_filtered['Adj_Delta_Long'].abs() < rate_threshold)
)

# -----------------------------------
# 找停泊區段（容忍短暫中斷）
# -----------------------------------
max_gap_sec = 120  # 允許 2 分鐘內 False 不切斷停泊

stop_segments = []
current_start_idx = None
last_stop_idx = None

for i in range(len(df_filtered)):
    if df_filtered.loc[i, 'Is_Stop']:
        if current_start_idx is None:
            current_start_idx = i
        last_stop_idx = i
    else:
        if current_start_idx is not None:
            gap = (df_filtered.loc[i, 'CreateTime'] - df_filtered.loc[last_stop_idx, 'CreateTime']).total_seconds()
            if gap > max_gap_sec:
                segment_idx = range(current_start_idx, last_stop_idx + 1)
                stop_segments.append(segment_idx)
                current_start_idx = None
                last_stop_idx = None

# 最後一段停泊若持續到資料尾端
if current_start_idx is not None:
    segment_idx = range(current_start_idx, last_stop_idx + 1)
    stop_segments.append(segment_idx)

# -----------------------------------
# 過濾短暫停泊 (<30分鐘)
# -----------------------------------
min_stop_sec = 1800  # 30 分鐘
filtered_segments = []
for segment in stop_segments:
    start_time = df_filtered.loc[segment[0], 'CreateTime']
    end_time = df_filtered.loc[segment[-1], 'CreateTime']
    duration = (end_time - start_time).total_seconds()
    if duration >= min_stop_sec:
        filtered_segments.append(segment)

# -----------------------------------
# 航程標註與 Real_ETA_sec
# -----------------------------------
df_filtered['voyage_id'] = np.nan
df_filtered['Real_ETA_sec'] = np.nan  # 剩餘到港時間（秒）

voyage_id = 1
for i in range(len(filtered_segments) - 1):
    # 航程 = 當前停泊區段的下一點到下一停泊區段的起點
    start_idx = filtered_segments[i][-1] + 1
    end_idx = filtered_segments[i+1][0] - 1
    if start_idx > end_idx:
        continue  # 沒有航程點就跳過

    df_filtered.loc[start_idx:end_idx, 'voyage_id'] = voyage_id

    # Real_ETA_sec = 下一停泊區段時間均值 - 當前點時間
    eta_segment = filtered_segments[i+1]
    eta_time = df_filtered.loc[eta_segment, 'CreateTime'].min()

    df_filtered.loc[start_idx:end_idx, 'Real_ETA_sec'] = (
        (eta_time - df_filtered.loc[start_idx:end_idx, 'CreateTime']).dt.total_seconds()
    )

    voyage_id += 1

# -----------------------------------
# 新增「距離目的港」與「靠港判定」
# -----------------------------------
df_filtered['dist_to_dest_km'] = np.nan
df_filtered['near_port_flag'] = np.nan

# 建立 voyage_id → 目的港座標 mapping
dest_map = {}
for v in range(1, voyage_id):
    if v < len(filtered_segments):
        seg = filtered_segments[v]  # 下一停泊
        dest_lat = df_filtered.loc[seg, 'Lat'].mean()
        dest_lon360 = df_filtered.loc[seg, 'Lng_360'].mean()
        # 把 0-360 轉 -180~180
        dest_lon = dest_lon360 if dest_lon360 <= 180 else dest_lon360 - 360
        dest_map[v] = (dest_lat, dest_lon)

# 計算距離
for idx, row in df_filtered.iterrows():
    v = row['voyage_id']
    if pd.notna(v):
        v = int(v)
        if v in dest_map:
            lat1 = row['Lat']
            lon1 = row['Lng_360']
            lon1 = lon1 if lon1 <= 180 else lon1 - 360
            lat2, lon2 = dest_map[v]
            df_filtered.at[idx, 'dist_to_dest_km'] = haversine((lat1, lon1), (lat2, lon2))

# 判定靠港 flag
df_filtered.loc[df_filtered['dist_to_dest_km'] < 1.0, 'near_port_flag'] = 1.0
df_filtered.loc[(df_filtered['dist_to_dest_km'] >= 1.0) & (df_filtered['dist_to_dest_km'] <= 5.0), 'near_port_flag'] = 0.5
df_filtered.loc[df_filtered['dist_to_dest_km'] > 5.0, 'near_port_flag'] = 0.0


df_filtered = df_filtered.dropna(subset=['voyage_id', 'Real_ETA_sec']).reset_index(drop=True)


# Random Forest Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# -----------------------------
# 選特徵 (示範可選)
# -----------------------------
numeric_features = [
    'Lat', 'Lng_360', 'Sog', 'dist_to_dest_km', 'near_port_flag'
]

X = df_filtered[numeric_features]
y = df_filtered['Real_ETA_sec']

# -----------------------------
# 切分訓練與測試集
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 訓練模型
# -----------------------------
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,  # 可調整
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# -----------------------------
# 預測
# -----------------------------
y_pred = rf_model.predict(X_test)

# -----------------------------
# 評估
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f} sec")
print(f"RMSE: {rmse:.2f} sec")
print(f"R2: {r2:.3f}")

# -----------------------------
# 預測示例
# -----------------------------
df_filtered.loc[X_test.index, 'Pred_Real_ETA_sec'] = y_pred
#print(df_filtered[['CreateTime','voyage_id','Real_ETA_sec','Pred_Real_ETA_sec']].head(20))

# 結果作圖
# -----------------------------
# 安全填回 Pred_Real_ETA_sec
# -----------------------------
# 確保索引對齊
df_filtered.loc[X_test.index, 'Pred_Real_ETA_sec'] = y_pred

# 檢查 NaN
print("Pred_Real_ETA_sec NaN 數量:", df_filtered['Pred_Real_ETA_sec'].isna().sum())

# -----------------------------
# 畫散點圖：實際 ETA vs 預測 ETA
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(df_filtered.loc[X_test.index, 'Real_ETA_sec'],
            df_filtered.loc[X_test.index, 'Pred_Real_ETA_sec'],
            alpha=0.5, s=10, color='blue')
plt.plot([0, df_filtered['Real_ETA_sec'].max()],
         [0, df_filtered['Real_ETA_sec'].max()],
         color='red', linestyle='--', linewidth=1)  # 理想對角線
plt.xlabel('Actual Real_ETA_sec (s)')
plt.ylabel('Predicted Real_ETA_sec (s)')
plt.title('Actual ETA vs Predicted ETA Scatter Plot')
plt.grid(True)
plt.show()

# 查看特徵重要性
# -----------------------------
# 特徵重要性
# -----------------------------
importances = rf_model.feature_importances_
feature_names = X.columns

# 排序 (可視化時比較好看)
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), feature_names[indices], rotation=45)
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()

# 預測ETA 在這裡改就好
#------------------------------
prelon = 131.22899
prelat = 31.121067
presog = 15.11284071183854
#------------------------------
lat1 = radians(prelat)
lon1 = radians(prelon)
lat2 = radians(34.21535)
lon2 = radians(135.13684)

# 哈弗辛公式
dlat = lat2 - lat1
dlon = lon2 - lon1
a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
c = 2 * atan2(sqrt(a), sqrt(1-a))

# 地球半徑（公里）
R = 6371.0

# 計算距離
distance = R * c
print(f"剩餘距離：{distance:.2f} 公里")

# 假設這是從 API 獲取的即時船舶資料
new_data = {
    'Lat': prelat,
    'Lng_360': prelon,
    'Sog': presog,
    'dist_to_dest_km': distance,
    'near_port_flag': 0.0
}

# 將資料轉換為 DataFrame
new_df = pd.DataFrame([new_data])

# 使用標準化器進行資料標準化


# 使用模型進行預測
predicted_eta = rf_model.predict(new_df)

# 顯示預測結果
print(f"預測的剩餘到港時間：{predicted_eta[0]:.2f} 秒")

# 假設 predicted_eta 是模型的輸出（秒）
predicted_eta = predicted_eta.astype(float)[0]  

# 1. 取得現在時間
now = datetime.now()

# 2. 換成 timedelta（四捨五入到分鐘）
eta_timedelta = timedelta(seconds=round(predicted_eta))

# 3. 加到現在時間
predicted_arrival = now + eta_timedelta

# 4. 輸出結果
print("現在時間:", now.strftime("%Y-%m-%d %H:%M:%S"))
print("預測到港時間:", predicted_arrival.strftime("%Y-%m-%d %H:%M"))