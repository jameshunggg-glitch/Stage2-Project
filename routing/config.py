# routing/config.py
# -*- coding: utf-8 -*-

BUFFER_KM = 5.0                 # 近岸內圈，若在內圈內會 nudge 到外圈
COLLISION_SAFETY_KM = 0.25      # 可視碰撞膨脹（安全距）
PAD_DEG = 6.0                   # bbox padding（度）
STEP_KM_GEODESIC = 3.0          # 可視抽樣步長（公里）
DRAW_STEP_KM = 20.0             # folium 繪線抽樣步長（公里）
AVOID_KM = 15.0                 # O/D 與特徵點的目標外圈
NEIGHBOR_K = 24                 # 每個節點的 KNN 鄰居數（LVS）
LVS_MAX_NODES = 4000            # 最大節點數保護
SIMPLIFY_MAX_PASSES = 8         # 簡化最多迭代次數
