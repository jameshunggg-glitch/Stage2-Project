import heapq
from shapely.geometry import Point, LineString
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# ========== 前置：生成節點與邊 ==========
land = gpd.read_file(r"C:\Users\slab\Desktop\Slab Project\Stage1\data\Land\ne_10m_land.shp")

minx, miny, maxx, maxy = 117.5, 23.0, 122.5, 25.5
step = 0.02  # 粗網格

xs = np.arange(minx, maxx+1e-9, step)
ys = np.arange(miny, maxy+1e-9, step)
grid_points = [Point(x,y) for x in xs for y in ys]
gdf_points = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:4326")

land_clip = land.cx[minx:maxx, miny:maxy]
land_union = land_clip.unary_union
sea_points = gdf_points[~gdf_points.within(land_union)].copy()
sea_points.reset_index(drop=True, inplace=True)

def q(x, nd=6): return round(float(x), nd)
point_index = {(q(p.x), q(p.y)): i for i,p in enumerate(sea_points.geometry)}

neighbor_steps = [
    ( step, 0), (-step, 0), (0, step), (0, -step),
    ( step, step), ( step,-step), (-step, step), (-step,-step),
    (2*step,0), (-2*step,0), (0,2*step), (0,-2*step)
]

# ========== A* 搜索 ==========
def haversine(lon1, lat1, lon2, lat2):
    """球面距離 km"""
    import math
    R = 6371
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2-lon1, lat2-lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def find_nearest_node(x, y):
    dmin, idx = 1e9, None
    for i,p in enumerate(sea_points.geometry):
        d = haversine(x,y,p.x,p.y)
        if d < dmin:
            dmin, idx = d, i
    return idx

def astar(start_idx, goal_idx):
    start = sea_points.geometry[start_idx]
    goal = sea_points.geometry[goal_idx]

    open_set = [(0, start_idx)]
    came_from = {}
    gscore = {start_idx: 0}
    fscore = {start_idx: haversine(start.x, start.y, goal.x, goal.y)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_idx:
            # reconstruct
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_idx)
            return path[::-1]

        x0, y0 = q(sea_points.geometry[current].x), q(sea_points.geometry[current].y)
        for dx,dy in neighbor_steps:
            x1, y1 = q(x0+dx), q(y0+dy)
            if (x1,y1) in point_index:
                neighbor = point_index[(x1,y1)]
                tentative_g = gscore[current] + haversine(x0,y0,x1,y1)
                if tentative_g < gscore.get(neighbor, 1e9):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + haversine(x1,y1,goal.x,goal.y)
                    heapq.heappush(open_set, (fscore[neighbor], neighbor))
    return None

# ========== 定義台中 & 廈門 ==========
taichung = (120.3, 24.3)  # 台中港外海
xiamen   = (118.1, 24.5)  # 廈門港外海

start_idx = find_nearest_node(*taichung)
goal_idx  = find_nearest_node(*xiamen)

path_idx = astar(start_idx, goal_idx)
path_points = [sea_points.geometry[i] for i in path_idx]

# ========== 視覺化 ==========
fig, ax = plt.subplots(figsize=(10,10))
land_clip.boundary.plot(ax=ax, color="black", linewidth=1, zorder=1)
sea_points.plot(ax=ax, color="blue", markersize=10, alpha=0.5, zorder=2)
if path_points:
    xs, ys = [p.x for p in path_points], [p.y for p in path_points]
    ax.plot(xs, ys, color="red", linewidth=2, zorder=3, label="A* path")
ax.scatter(*taichung, color="green", s=100, zorder=4, label="Taichung start")
ax.scatter(*xiamen, color="orange", s=100, zorder=4, label="Xiamen goal")
plt.xlim(minx, maxx)
plt.ylim(miny, maxy)
plt.legend()
plt.title("A* 最短路徑：台中港 → 廈門港")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
