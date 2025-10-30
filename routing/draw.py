# routing/draw.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Iterable, Dict, Any, Optional
import folium

from shapely.geometry import mapping as shapely_mapping
from shapely.geometry.base import BaseGeometry

from .geodesy import geodesic_sample, normalize_lon_to_pacific_view

LonLat = Tuple[float, float]

# ------------------------------------------------------------
# 基礎工具：經緯度轉 folium 座標（太平洋視角：經度 0~360）
# ------------------------------------------------------------
def _to_latlon_ll(points_ll: Iterable[LonLat], pacific_view: bool = True) -> List[Tuple[float, float]]:
    latlons: List[Tuple[float, float]] = []
    for lon, lat in points_ll:
        lon_plot = normalize_lon_to_pacific_view(lon) if pacific_view else lon
        latlons.append((lat, lon_plot))
    return latlons

# ------------------------------------------------------------
# 1) 連續大圓折線（跨換日線不斷裂）
# ------------------------------------------------------------
def draw_gc_polyline_continuous(
    m: folium.Map | folium.FeatureGroup,
    a_ll: LonLat,
    b_ll: LonLat,
    step_km: float = 20.0,
    weight: int = 5,
    opacity: float = 0.9,
    dash_array: Optional[str] = None,
    color: str = "#1f77b4",
) -> folium.PolyLine:
    """
    以大圓分段密化 (lon,lat)→(lat,lon_plot)，確保跨國際換日線時視覺連續。
    """
    pts = geodesic_sample(a_ll, b_ll, step_km=max(1.0, float(step_km)))
    latlons = _to_latlon_ll(pts, pacific_view=True)
    pl = folium.PolyLine(
        latlons,
        weight=weight,
        opacity=opacity,
        color=color,
        dash_array=dash_array
    )
    pl.add_to(m)
    return pl

# ------------------------------------------------------------
# 2) 形狀資料轉太平洋視角（給 folium.GeoJson 使用）
# ------------------------------------------------------------
def _convert_coords_pacific(obj: Any) -> Any:
    """
    遞迴處理 coordinates：將 (lon,lat) 的 lon 轉為 0~360。
    可處理 GeoJSON-like 結構。
    """
    if isinstance(obj, (list, tuple)):
        if len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj):
            lon, lat = obj
            return [normalize_lon_to_pacific_view(float(lon)), float(lat)]
        else:
            return [ _convert_coords_pacific(x) for x in obj ]
    elif isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "coordinates":
                out[k] = _convert_coords_pacific(v)
            else:
                out[k] = _convert_coords_pacific(v)
        return out
    else:
        return obj

def convert_geom_to_pacific(geom: BaseGeometry | Dict[str, Any]) -> Dict[str, Any]:
    """
    接受 shapely geometry 或 GeoJSON-like dict，回傳 GeoJSON-like dict，
    其 coordinates 的經度已轉為 0~360。
    """
    if isinstance(geom, BaseGeometry):
        gj = shapely_mapping(geom)  # -> {'type': 'Polygon', 'coordinates': [...]}
    elif isinstance(geom, dict):
        gj = geom
    else:
        raise TypeError("convert_geom_to_pacific expects a shapely geometry or GeoJSON-like dict.")
    # GeoJSON Feature/Geometry 都支援
    if "type" in gj:
        if gj["type"] == "Feature":
            feat = dict(gj)
            geom_part = feat.get("geometry")
            if geom_part:
                geom_part = _convert_coords_pacific(geom_part)
                feat["geometry"] = geom_part
            return feat
        else:
            return _convert_coords_pacific(gj)
    else:
        # 非標準格式，嘗試遞迴
        return _convert_coords_pacific(gj)

# ------------------------------------------------------------
# 3) SCGraph 單一路徑圖層（可切換）
# ------------------------------------------------------------
def add_scgraph_layer(
    m: folium.Map | folium.FeatureGroup,
    track_ll: List[LonLat],
    name: str = "SCGraph O→D path",
    show: bool = False,
    step_km: float = 20.0,
    weight: int = 5,
    opacity: float = 0.9,
    dash_array: str = "8,6",
    color: str = "#ff7f0e",
) -> folium.FeatureGroup:
    """
    將「單一路徑」(list[(lon,lat)]) 畫成可切換圖層。
    以大圓分段密化，確保跨日界線連續與不穿陸視覺。
    """
    fg = folium.FeatureGroup(name=name, show=show)

    if not track_ll or len(track_ll) < 2:
        fg.add_to(m)
        return fg

    step_km = max(1.0, float(step_km))
    for a, b in zip(track_ll[:-1], track_ll[1:]):
        densified = geodesic_sample(a, b, step_km=step_km)
        latlons = _to_latlon_ll(densified, pacific_view=True)
        folium.PolyLine(
            latlons,
            weight=weight,
            opacity=opacity,
            dash_array=dash_array,
            color=color
        ).add_to(fg)

    fg.add_to(m)
    return fg

# ------------------------------------------------------------
# 4) SCGraph 子網路邊集合圖層（可切換）
# ------------------------------------------------------------
def add_scgraph_network_layer(
    m: folium.Map | folium.FeatureGroup,
    edges_ll: List[Tuple[LonLat, LonLat]],
    name: str = "SCGraph Network",
    show: bool = False,
    weight: int = 2,
    opacity: float = 0.8,
    dash_array: str = "4,6",
    color: str = "#ff7f0e",
    step_km: float = 12.0,
) -> folium.FeatureGroup:
    """
    將「子網的所有邊」畫成可切換圖層。
    edges_ll: list of ((lon1,lat1),(lon2,lat2))
    """
    fg = folium.FeatureGroup(name=name, show=show)

    if not edges_ll:
        fg.add_to(m)
        return fg

    step_km = max(1.0, float(step_km))
    for (a, b) in edges_ll:
        densified = geodesic_sample(a, b, step_km=step_km)
        latlons = _to_latlon_ll(densified, pacific_view=True)
        folium.PolyLine(
            latlons,
            weight=weight,
            opacity=opacity,
            dash_array=dash_array,
            color=color
        ).add_to(fg)

    fg.add_to(m)
    return fg
