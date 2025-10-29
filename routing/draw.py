# routing/draw.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Optional, Iterable

import folium

from .geodesy import geodesic_sample, normalize_lon_to_pacific_view


def draw_gc_polyline_continuous(
    m_or_fg,
    a: Tuple[float, float],
    b: Tuple[float, float],
    step_km: float = 20.0,
    **style,
):
    """
    在「太平洋視角」下，沿大圓將 a->b 畫成連續折線，
    自動把經度 <0 轉到 [0,360) 以避免越界斷線。
    m_or_fg: folium.Map 或 folium.FeatureGroup
    """
    pts = geodesic_sample(a, b, step_km=step_km)
    folium_coords = []
    for lon, lat in pts:
        lon_pacific = normalize_lon_to_pacific_view(lon)
        folium_coords.append([lat, lon_pacific])
    folium.PolyLine(folium_coords, **style).add_to(m_or_fg)


def convert_geom_to_pacific(geom):
    """
    把 Polygon/MultiPolygon 的經度換成太平洋視角（[0,360)），
    回傳可直接丟給 folium.GeoJson 的 dict。
    """
    from shapely.geometry import mapping
    geom_dict = mapping(geom)

    def convert_coords(coords):
        if isinstance(coords[0], (list, tuple)):
            return [convert_coords(c) for c in coords]
        else:
            lon, lat = coords[0], coords[1]
            lon_pacific = normalize_lon_to_pacific_view(lon)
            return [lon_pacific, lat]

    if geom_dict["type"] == "Polygon":
        geom_dict["coordinates"] = [convert_coords(ring) for ring in geom_dict["coordinates"]]
    elif geom_dict["type"] == "MultiPolygon":
        geom_dict["coordinates"] = [[convert_coords(ring) for ring in poly] for poly in geom_dict["coordinates"]]
    return geom_dict


def add_scgraph_layer(
    m: folium.Map,
    sc_track: List[Tuple[float, float]],
    *,
    name: str = "SCGraph Path",
    show: bool = False,
    step_km: float = 20.0,
    weight: int = 4,
    opacity: float = 0.9,
    dash_array: str = "8,6",
    color: str = "#ff7f0e",
) -> Optional[folium.FeatureGroup]:
    """
    把 scgraph 的 polyline（(lon,lat) 座標序列）畫成可切換的圖層。
    - 會在圖層控制（LayerControl）右上角顯示 `name` 供勾選
    - 預設使用橘色虛線，避免與你主航線（藍/紅）混淆
    回傳建立的 FeatureGroup（若 sc_track 無法畫則回傳 None）
    """
    if not sc_track or len(sc_track) < 2:
        return None

    fg = folium.FeatureGroup(name=name, show=show)
    for a, b in zip(sc_track[:-1], sc_track[1:]):
        draw_gc_polyline_continuous(
            fg, a, b, step_km=step_km,
            color=color, weight=weight, opacity=opacity, dash_array=dash_array
        )
    fg.add_to(m)
    return fg
