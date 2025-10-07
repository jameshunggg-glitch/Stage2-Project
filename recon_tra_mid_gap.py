import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure
from skimage.morphology import skeletonize, square, dilation
import networkx as nx
import math
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import webbrowser
import folium

# ---------- 參數：檔案與 A/B ----------
tif_path = r"C:\Users\slab\Desktop\Slab Project\Stage1\data\ShipDensity_Commercial\ShipDensity_Commercial1.tif"
# 你的範例座標
lonA, latA = 103.15, 9.06
lonB, latB = 108.26, 9.48

pad_deg = 2.0                 # bbox 外擴（度）
corridor_half_width_km = 200   # A→B 走廊半寬（km）
min_area_pixels = 50          # 連通元件最小面積（像素）
percentiles = [95, 90, 80, 70, 60, 40, 20]  # 迭代下降門檻
do_plot = True

# ---------- 小工具 ----------
def affine_lonlat_grid(transform, height, width):
    """由 rasterio Affine 生成每個像素中心的 lon/lat 2D 網格"""
    # x = a*col + b*row + c ; y = d*col + e*row + f
    cols = np.arange(width)
    rows = np.arange(height)
    C, R = np.meshgrid(cols, rows)  # shape (H,W)
    lon = transform.c + transform.a*C + transform.b*R
    lat = transform.f + transform.d*C + transform.e*R
    return lon, lat

def lonlat_to_local_km(lon, lat, lon0, lat0):
    """把經緯度轉成局部平面 (km)，以 lat0 做 cos 比例（適合不太大的 bbox）"""
    R = 6371.0088
    x = (lon - lon0) * (np.cos(np.deg2rad(lat0)) * (np.pi/180.0) * R)
    y = (lat - lat0) * ((np.pi/180.0) * R)
    return x, y

def corridor_mask_from_AB(lon2d, lat2d, lonA, latA, lonB, latB, half_width_km):
    """建立 A→B 走廊遮罩（帶狀區域，±half_width_km）"""
    lat0 = 0.5*(latA + latB)
    lon0 = 0.5*(lonA + lonB)
    # 格點 → 局部公里座標
    X, Y = lonlat_to_local_km(lon2d, lat2d, lon0, lat0)
    Ax, Ay = lonlat_to_local_km(np.array(lonA), np.array(latA), lon0, lat0)
    Bx, By = lonlat_to_local_km(np.array(lonB), np.array(latB), lon0, lat0)
    # 向量
    ABx, ABy = (Bx - Ax), (By - Ay)
    AB2 = ABx*ABx + ABy*ABy + 1e-12  # 避免除零
    # 各像素到 AB 的投影參數 t（0~1 表示段內）
    APx, APy = (X - Ax), (Y - Ay)
    t = (APx*ABx + APy*ABy) / AB2
    t = np.clip(t, 0.0, 1.0)
    # 最近點 P = A + t*AB；距離 d = |X - P|
    Px, Py = Ax + t*ABx, Ay + t*ABy
    dx, dy = X - Px, Y - Py
    dist_km = np.hypot(dx, dy)
    mask = dist_km <= half_width_km
    return mask

def pixel_of_lonlat(transform, lon, lat, shape):
    """把 lon/lat 映到 window 內的像素索引 (row,col)，並裁界限"""
    r, c = rasterio.transform.rowcol(transform, lon, lat)
    r = int(np.clip(r, 0, shape[0]-1))
    c = int(np.clip(c, 0, shape[1]-1))
    return r, c

def check_connected(binary, Ar, Ac, Br, Bc):
    """
    檢查 A 像素與 B 像素是否在 binary 的同一連通元件；
    若 A/B 落在 False 上，先對 binary 做輕微膨脹再檢查。
    """
    if not (0 <= Ar < binary.shape[0] and 0 <= Ac < binary.shape[1]): return False
    if not (0 <= Br < binary.shape[0] and 0 <= Bc < binary.shape[1]): return False
    # 先小膨脹 1 次，避免 A/B 剛好踩在邊界 False
    bin_use = dilation(binary, square(3))
    labels = measure.label(bin_use)
    labA = labels[Ar, Ac]
    labB = labels[Br, Bc]
    return (labA != 0) and (labA == labB)

# ---------- 讀取子窗 ----------
bbox = (min(lonA, lonB) - pad_deg,
        min(latA, latB) - pad_deg,
        max(lonA, lonB) + pad_deg,
        max(latA, latB) + pad_deg)

with rasterio.open(tif_path) as src:
    window = rasterio.windows.from_bounds(*bbox, transform=src.transform)
    data = src.read(1, window=window)              # 密度 band
    transform = src.window_transform(window)

H, W = data.shape
lon2d, lat2d = affine_lonlat_grid(transform, H, W)

# ---------- 強度標準化（log） ----------
data_log = np.log1p(data.astype(np.float64))

# ---------- 走廊遮罩（只在 A→B 帶狀內做統計與提取） ----------
mask_corridor = corridor_mask_from_AB(lon2d, lat2d, lonA, latA, lonB, latB, corridor_half_width_km)

# ---------- 是否需要削頂（有尖塔才削） ----------
vals = data_log[(mask_corridor) & (data_log > 0)]
if vals.size == 0:
    raise RuntimeError("走廊內沒有有效密度值，請增大 pad 或檢查座標/資料。")

p90  = np.percentile(vals, 90)
p999 = np.percentile(vals, 99.9)
spike_ratio = (p999 + 1e-6) / (p90 + 1e-6)

if spike_ratio > 2.0:  # 有尖塔（如港口）才削頂
    cap = np.percentile(vals, 99.5)
    work = np.minimum(data_log, cap)
else:
    work = data_log.copy()

# ---------- 將 A/B 轉到像素座標 ----------
Ar, Ac = pixel_of_lonlat(transform, lonA, latA, (H, W))
Br, Bc = pixel_of_lonlat(transform, lonB, latB, (H, W))

# ---------- 動態分位 + 連通檢查 ----------
binary = None
used_q = None
for q in percentiles:
    thr = np.percentile(work[(mask_corridor) & (work > 0)], q)
    cand = (work >= thr) & mask_corridor
    # 形態學清理
    cand = morphology.opening(cand, square(3))
    cand = morphology.closing(cand, square(3))
    # 連通元件過濾（太小的去掉）
    labels = measure.label(cand)
    if labels.max() > 0:
        props = measure.regionprops(labels)
        keep_labs = [p.label for p in props if p.area >= min_area_pixels]
        cand = np.isin(labels, keep_labs)

    if check_connected(cand, Ar, Ac, Br, Bc):
        binary = cand
        used_q = q
        break

if binary is None:
    print("[警告] 在設定的 percentiles 內（{}）仍無法讓 A/B 連通，請加大走廊或放寬門檻。".format(percentiles))
    # 退一步：至少給個骨架在現有最高分位（可能是斷的）
    q = percentiles[-1]
    thr = np.percentile(work[(mask_corridor) & (work > 0)], q)
    binary = (work >= thr) & mask_corridor
    binary = morphology.opening(binary, square(3))
    binary = morphology.closing(binary, square(3))

# ---------- 骨架化 ----------
skeleton = skeletonize(binary)

import networkx as nx

def skeleton_to_graph(skel):
    """將 skeleton mask 轉成 NetworkX graph"""
    G = nx.Graph()
    H, W = skel.shape
    idx = lambda r, c: r*W + c
    for r in range(H):
        for c in range(W):
            if skel[r, c]:
                G.add_node(idx(r, c), pos=(r, c))
                # 8 鄰居
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r+dr, c+dc
                        if 0 <= rr < H and 0 <= cc < W and skel[rr, cc]:
                            G.add_edge(idx(r, c), idx(rr, cc))
    return G

def extract_keypoints(G):
    """抽取端點與分岔點"""
    endpoints = []
    junctions = []
    for n in G.nodes:
        deg = G.degree[n]
        r, c = G.nodes[n]["pos"]
        if deg == 1:
            endpoints.append((r, c))
        elif deg >= 3:
            junctions.append((r, c))
    return endpoints, junctions

def resample_polyline(coords, step_px=10):
    """對骨架 polyline 做固定間隔重抽樣 (像素為單位)"""
    if len(coords) < 2:
        return coords
    resampled = [coords[0]]
    dist_acc = 0.0
    for i in range(1, len(coords)):
        p0 = np.array(coords[i-1])
        p1 = np.array(coords[i])
        seg_len = np.linalg.norm(p1 - p0)
        while dist_acc + seg_len >= step_px:
            t = (step_px - dist_acc) / seg_len
            newp = p0 + t * (p1 - p0)
            resampled.append(tuple(newp))
            seg_len -= (step_px - dist_acc)
            p0 = newp
            dist_acc = 0.0
        dist_acc += seg_len
    resampled.append(coords[-1])
    return resampled

# ---------- 骨架轉 graph ----------
G = skeleton_to_graph(skeleton)
endpoints, junctions = extract_keypoints(G)

# 找連通分量，抽取 polyline
resampled_lines = []
for comp in nx.connected_components(G):
    subG = G.subgraph(comp)
    # 嘗試找一條 path（從端點到端點）
    ep = [n for n in subG.nodes if subG.degree[n] == 1]
    if len(ep) >= 2:
        path = nx.shortest_path(subG, source=ep[0], target=ep[-1])
        coords = [subG.nodes[n]["pos"] for n in path]
        resampled = resample_polyline(coords, step_px=15)
        resampled_lines.append(resampled)


# ---------- 視覺化（加一張圖顯示抽樣節點） ----------

fig, ax = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=True)

ax[0].imshow(work, cmap="hot")
ax[0].contour(mask_corridor, levels=[0.5], colors="cyan", linewidths=1)
ax[0].plot([Ac, Bc], [Ar, Br], "co-", ms=4, lw=1)
ax[0].set_title("log密度（含走廊覆蓋）")

ax[1].imshow(binary, cmap="gray")
ax[1].plot([Ac, Bc], [Ar, Br], "yo-", ms=4, lw=1)
ax[1].set_title(f"保留走廊（二值），使用分位：P{used_q}" if used_q else "保留走廊（二值）")

ax[2].imshow(skeleton, cmap="gray")
ax[2].plot([Ac, Bc], [Ar, Br], "go-", ms=4, lw=1)
ax[2].set_title("Skeleton 航道脊線")

ax[3].imshow(skeleton, cmap="gray")
for line in resampled_lines:
    rs, cs = zip(*line)
    ax[3].plot(cs, rs, "r.-", ms=4, lw=1)
ax[3].set_title("Resampled 海骨節點")

for a in ax: a.set_axis_off()
plt.show()


import networkx as nx
import math
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import webbrowser
import folium



# ---------- 小工具 ----------
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0088
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def pixel_to_lonlat(transform, coords):
    """把 skeleton resampled (row,col) 轉回 (lon,lat)"""
    lonlats = []
    for (r, c) in coords:
        lon, lat = rasterio.transform.xy(transform, r, c)
        lonlats.append((lon, lat))
    return lonlats

# ---------- 建立骨幹圖 ----------
def build_backbone_graph(resampled_lines, transform):
    G = nx.Graph()
    node_id = 0
    node_map = {}  # (lon,lat) -> id

    for line in resampled_lines:
        lonlats = pixel_to_lonlat(transform, line)
        for i in range(len(lonlats)):
            if lonlats[i] not in node_map:
                node_map[lonlats[i]] = node_id
                G.add_node(node_id, pos=lonlats[i])
                node_id += 1
            if i > 0:
                u = node_map[lonlats[i-1]]
                v = node_map[lonlats[i]]
                dist = haversine(*lonlats[i-1], *lonlats[i])
                G.add_edge(u, v, weight=dist)
    return G

# ---- bridge nearby nodes ----
def bridge_nearby_nodes(G, max_gap_km=20):
    nodes = list(G.nodes(data=True))
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            ni, pos_i = nodes[i]
            nj, pos_j = nodes[j]
            if not G.has_edge(ni, nj):
                d = haversine(*pos_i["pos"], *pos_j["pos"])
                if d < max_gap_km:
                    G.add_edge(ni, nj, weight=d)
    return G

# ---------- 找最近節點 ----------
def find_nearest_node(G, lon, lat):
    dmin, best = 1e9, None
    for n in G.nodes:
        nl, la = G.nodes[n]["pos"]
        d = haversine(lon, lat, nl, la)
        if d < dmin:
            dmin, best = d, n
    return best, dmin

# ---------- A* 搜索 ----------
def astar_path(G, A, B):
    try:
        path = nx.astar_path(
            G, A, B,
            heuristic=lambda u,v: haversine(*G.nodes[u]["pos"], *G.nodes[v]["pos"]),
            weight="weight"
        )
        return path
    except nx.NetworkXNoPath:
        print("[警告] A* 找不到路徑")
        return None

# ---------- 主流程 ----------
A_lon, A_lat = lonA, latA
B_lon, B_lat = lonB, latB

# 建立骨幹圖
G = build_backbone_graph(resampled_lines, transform)
G = bridge_nearby_nodes(G, max_gap_km=30)

# 找 A/B 最近的骨幹節點
na, dA = find_nearest_node(G, A_lon, A_lat)
nb, dB = find_nearest_node(G, B_lon, B_lat)

# 永遠掛接駁節點
G.add_node("A_temp", pos=(A_lon, A_lat))
G.add_edge("A_temp", na, weight=dA)
na = "A_temp"

G.add_node("B_temp", pos=(B_lon, B_lat))
G.add_edge("B_temp", nb, weight=dB)
nb = "B_temp"

# A* 尋路
path_nodes = astar_path(G, na, nb)

# 拼接完整路徑：A → 骨幹 → B
route = []
if path_nodes:
    for n in path_nodes:
        route.append(G.nodes[n]["pos"])



# ---------- 畫圖 ----------
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# skeleton → lon/lat 底圖
H, W = skeleton.shape
bounds = rasterio.transform.array_bounds(H, W, transform)
miny, maxy, minx, maxx = bounds  # array_bounds: (ymin, ymax, xmin, xmax)

# A/B 與路徑
if route:
    rlons, rlats = zip(*route)
    ax.plot(rlons, rlats, "g-", lw=2, label="repaired path")

ax.scatter([A_lon], [A_lat], c="r", marker="o", label="A")
ax.scatter([B_lon], [B_lat], c="b", marker="o", label="B")

ax.legend()
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("A* repaired path")
plt.show()

# ---------- 儲存 ----------
df_route = pd.DataFrame(route, columns=["Longitude", "Latitude"])
#df_route.to_csv("repaired_path.csv", index=False)
#print("路徑已儲存到 repaired_path.csv")



# 建立地圖中心
m = folium.Map(location=[(A_lat+B_lat)/2, (A_lon+B_lon)/2], zoom_start=5)

# Folium 要 (lat, lon)，所以要調換順序
route_latlon = [(lat, lon) for lon, lat in route]
# 畫路徑
folium.PolyLine(route_latlon, color="green", weight=3).add_to(m)

# 畫 A/B 點
folium.Marker([A_lat, A_lon], popup="A", icon=folium.Icon(color="red")).add_to(m)
folium.Marker([B_lat, B_lon], popup="B", icon=folium.Icon(color="blue")).add_to(m)

# 輸出地圖
m.save("repaired_path_map.html")
webbrowser.open("repaired_path_map.html")

