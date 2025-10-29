# routing/scgraph.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Tuple, Dict
from pathlib import Path
import shapely
from shapely.geometry import LineString, Point
import geopandas as gpd
import math

from .geodesy import to_m, to_ll, gc_distance_km

class SCGraph:
    """
    簡化版航道圖結構：
    - nodes_ll: List[(lon,lat)]
    - edges: Dict[int, List[int]]
    - edges_km: Dict[(int,int), float]
    """

    def __init__(self):
        self.nodes_ll: List[Tuple[float,float]] = []
        self.edges: Dict[int, List[int]] = {}
        self.edges_km: Dict[Tuple[int,int], float] = {}

    # === 匯入與建立 ===
    @classmethod
    def from_gpkg(cls, file_path: str | Path, layer_name: str = None) -> "SCGraph":
        gdf = gpd.read_file(file_path, layer=layer_name)
        gdf = gdf.to_crs(epsg=4326)

        g = cls()
        node_index: Dict[Tuple[float,float], int] = {}

        def _add_node(pt: Tuple[float,float]) -> int:
            key = (round(pt[0], 6), round(pt[1], 6))
            if key not in node_index:
                node_index[key] = len(g.nodes_ll)
                g.nodes_ll.append(pt)
            return node_index[key]

        for geom in gdf.geometry:
            if geom.is_empty:
                continue
            if geom.geom_type == "LineString":
                coords = list(geom.coords)
                for a, b in zip(coords[:-1], coords[1:]):
                    u = _add_node((a[0], a[1]))
                    v = _add_node((b[0], b[1]))
                    g.add_edge(u, v)
            elif geom.geom_type == "MultiLineString":
                for ls in geom.geoms:
                    coords = list(ls.coords)
                    for a, b in zip(coords[:-1], coords[1:]):
                        u = _add_node((a[0], a[1]))
                        v = _add_node((b[0], b[1]))
                        g.add_edge(u, v)

        return g

    # === 建立與查詢 ===
    def add_edge(self, u: int, v: int):
        """新增雙向邊"""
        if u == v:
            return
        self.edges.setdefault(u, []).append(v)
        self.edges.setdefault(v, []).append(u)
        d = gc_distance_km(self.nodes_ll[u], self.nodes_ll[v])
        self.edges_km[(u, v)] = d
        self.edges_km[(v, u)] = d

    def find_nearest_nodes(self, pt: Tuple[float,float], k: int = 3) -> List[int]:
        """找到距離最近的 scgraph 節點索引"""
        scored=[]
        for idx, n in enumerate(self.nodes_ll):
            d = gc_distance_km(pt, n)
            scored.append((d, idx))
        scored.sort(key=lambda t: t[0])
        return [idx for _, idx in scored[:k]]

    def to_adj_dict(self) -> Dict[int, List[int]]:
        """輸出鄰接字典"""
        return self.edges

    def node_count(self) -> int:
        return len(self.nodes_ll)
