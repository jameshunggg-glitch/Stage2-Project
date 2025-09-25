# 實作流程markdown

## 專案目標
用 python 實作船舶資料的歷史航線比較

# 功能需求如下
# 讀取檔案
- 利用pandas 讀取檔案
- 檔案的路徑為 "C:\Users\slab\Desktop\Slab Project\Stage2 ETA\Raw Data\Device_AB00035.csv"

# 資料前處裡
- 對讀取的檔案篩選，只留下MMSI = 416426000的資料
- 對Lat欄位還有Long欄位將他們資料型態轉成float 型態，並將缺失值欄位dropna
- 只留下資料Lat欄位在範圍-90與90之間的資料
- 只留下資料Long欄位在範圍-180與180之間的資料
- 將資料的Timestamp欄位轉換成datetime型態，format = '%Y%m%d%H%M%S'

# 港口判定
- 資料先過濾出低速點，留下Sog < 0.5 的資料
- 將Lat,Long轉乘numpy，然後再將它們換成弧度
- 對資料的Lat, Long做DBSCAN聚類，metric = haversine，eps = 0.01, min_samples = 10
- 對其中的每個cluster:
    - 算質心作為港口中心點(centroid)
    - 算每個點到質心的距離然後取95%分位數
    - 半徑 = 95%分位數*1.2
    - 得到港口清單: port_id, center_lat, center_lon, radius
- 對所有cluster的港口中心再做一次DBSCAN，如果兩個cluster中心的距離小於1km，合併為同一個港口
- 重新得到港口清單: port_id, center_lat, center_lon, radius

# 這些程式碼幫我寫在 Optimal.py裡面