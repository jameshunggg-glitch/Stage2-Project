from pathlib import Path
import routing_map
from routing_map import build_aoi, RoutingMapConfig
from routing_map.config import AoiConfig, LandConfig
from routing_map.ring_types import RingBuildConfig
from routing_map import cache_utils

# Sri lanka AOI (< 10 sec)
# 79.52576, 10.01686 
# 82.182, 5.83924 
# Taiwan AOI  (< 10 sec)
# 120.053 , 25.532
# 122.222 , 21.596
# Philippines AOI (5~7 min)
# 118.125, 17.978
# 130.253 , 3.776
# Northen Indonesia AOI (10 sec)
# 108.149, 7.013
# 124.189, -6.358
# Big AOI (Japan to Australia) (24.5 min)
# 107.37256, -45.14933
# 155.14554, 40.056
# Korea AOI (1.5 min)
# 130.09132, 34.09742
# 124.57837, 36.77376
# Big AOI (Asia to EU) (40 min)
# 10,-40
# 150, 45
# world aoi (5hrs)
# (-179.999, -85, 179.999, 85)
cfg = RoutingMapConfig(
    aoi=AoiConfig(
        bbox_ll=((-179.999, -85, 179.999, 85)),
    ),
    land=LandConfig(
        shp_path=Path(r"C:\Users\slab\Desktop\Slab Project\Stage1\data\Land\ne_10m_land.shp"),
        buffer_km=20.0,
        avoid_km= 1,
        collision_safety_km=0.5,
    ),
        rings=RingBuildConfig(
        clearance_m=10000.0,      # 1 km：你 land.avoid_km 的概念
        ring_sample_km=5.0,      # 采樣間距（先保守）
        taut_window_size=80,
        taut_max_tries=8,
        min_ring_length_km=20.0,

        taut_use_clearance_buffer= True,
        taut_collision_buffer_m= 5000,
    ),
    
)


CACHE_DIR = r"C:\Users\slab\Desktop\Slab Project\Stage2 ETA\aoi_cache"
CACHE_TAG = "global"  # 你可以改成 "taiwan" / "world" / "dateline" 都行

out = cache_utils.get_out(cfg, cache_dir=CACHE_DIR, use_cache=True)

### ---------------- ###
# RUN P2P #
### ---------------- ###
from routing_map.pipeline import (
    run_p2p,
    GraphConfig, SnapConfig, SimplifyConfig, RunConfig,
)
from routing_map.repairer import RepairConfig
from routing_map.viz_layers import (
    make_base_map,
    add_sea_layers, add_ring_layers, add_connector_layers,
    add_points_layer, add_path_layer,
    finalize_map,
)
from routing_map.pipeline import run_p2p_multiworld

# 1) 你 build_aoi(cfg) 得到 out 後：
bbox_ll = out["bbox_ll"]
print("bbox_ll:", bbox_ll)

# 2) 設定起終點（自己改）
origin_ll = (114.12681, -3.47298)  
dest_ll   = (110.26443, 2.24094)


# 3) 設定流程 config（你先用預設即可，之後再調參）
graph_cfg = GraphConfig(
    bbox_ll=bbox_ll,
    max_sea_edges=None,
    max_ring_edges=None,
    weight_unit="km",
)

snap_cfg = SnapConfig(
    k_near=30,
    r_max_km=150.0,
    k_inject=4,
)

repair_cfg = RepairConfig(
    debug=True,
)

simplify_cfg = SimplifyConfig(
    enabled=True,
    window_size=80,
    max_tries=300,
    use_prepared_collision=True,
    dateline_unwrap=True,
    wrap_output_lon=True,
)

run_cfg = RunConfig(
    do_repair=True,
    do_simplify=True,
    debug=True,
)

# 4) 跑新 pipeline（內含：建圖 + snap/inject + A* + repair + simplify）

graph_build_args = dict(
    include_sea=graph_cfg.include_sea,
    include_cc=graph_cfg.include_cc,
    include_gateb_sea=graph_cfg.include_gateb_sea,
    include_c_gateb=graph_cfg.include_c_gateb,
    include_rings=graph_cfg.include_rings,
    include_et=graph_cfg.include_et,
    include_tgate_sea=graph_cfg.include_tgate_sea,
    max_sea_edges=graph_cfg.max_sea_edges,
    max_ring_edges=graph_cfg.max_ring_edges,
    weight_unit=graph_cfg.weight_unit,
    bbox_ll=bbox_ll,
)

G_base = cache_utils.get_graph(
    out,
    cfg=cfg,
    graph_build_args=graph_build_args,
    cache_dir=CACHE_DIR,
    use_cache=True,
)


res = run_p2p_multiworld(
    out,
    origin_ll,
    dest_ll,
    graph_cfg=graph_cfg,
    snap_cfg=SnapConfig(R_NEAR_COAST_KM=120.0, S_MAX_SNAP_KM=200.0),
    repair_cfg=repair_cfg,
    simplify_cfg=simplify_cfg,
    run_cfg=RunConfig(debug=True),
    G_in=G_base,
)

print("error:", res.error)
print("final points:", 0 if not res.path_ll_final else len(res.path_ll_final))
print("lengths_km:", res.lengths_km)

# 5) 畫圖（新版 layers）
m = make_base_map(bbox_ll=bbox_ll, zoom_start=5)

add_sea_layers(
    m, out,
    node_sample=5000,
    max_edges=5000,
    show=True,
    bbox_ll=bbox_ll,
)
add_ring_layers(
    m, out,
    e_node_sample=60000,
    t_node_sample=50000,
    max_e_edges=50000,
    max_t_edges=50000,
    show=True,
    bbox_ll=bbox_ll,
)
add_connector_layers(
    m, out,
    max_et=10000,
    max_tgate=5000,
    show=True,
    bbox_ll=bbox_ll,
)

# points
add_points_layer(m, [origin_ll], name="origin_raw", radius=7, show=True, bbox_ll=bbox_ll)
add_points_layer(m, [dest_ll],   name="dest_raw",   radius=7, show=True, bbox_ll=bbox_ll)
if res.start_ll_snap:
    add_points_layer(m, [res.start_ll_snap], name="origin_snap", radius=6, show=True, bbox_ll=bbox_ll)
if res.end_ll_snap:
    add_points_layer(m, [res.end_ll_snap],   name="dest_snap",   radius=6, show=True, bbox_ll=bbox_ll)

# paths（你想看哪條就開哪條）
if res.path_ll_raw:
    add_path_layer(m, res.path_ll_raw, name="path_raw", weight=3, opacity=0.6, show=False)
if res.path_ll_repaired:
    add_path_layer(m, res.path_ll_repaired, name="path_repaired", weight=4, opacity=0.7, show=False)
if res.path_ll_simplified:
    #add_path_layer(m, res.path_ll_simplified, name="path_simplified", weight=5, opacity=0.9, show=True)
    add_path_layer(
        m, res.path_ll_simplified,
        name="path_simplified",
        weight=5, opacity=0.9, show=True,
        geodesic=True,
        geodesic_step_km=20.0,   # 想更平滑就 10；想更快就 30~50
    )
elif res.path_ll_final:
    add_path_layer(m, res.path_ll_final, name="path_ll_final", weight=5, opacity=0.9, show=True)

html = finalize_map(m, html_path="aoi_rings_p2p_debug.html")
from webbrowser import open as wb_open
wb_open("aoi_rings_p2p_debug.html")
print("saved:", html)
