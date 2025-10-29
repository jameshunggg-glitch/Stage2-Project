# routing/draw.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Tuple, List, Optional
import folium
import shapely
from shapely.geometry import mapping
from .geodesy import normalize_lon_to_pacific_view, geodesic_sample
from .config import  DRAW_STEP_KM

def convert_geom_to_pacific(geom):
    gdict = mapping(geom)
    def conv(coords):
        if isinstance(coords[0], (list, tuple)):
            return [conv(c) for c in coords]
        lon,lat = coords[0], coords[1]
        return [normalize_lon_to_pacific_view(lon), lat]
    if gdict['type'] == 'Polygon':
        gdict['coordinates'] = [conv(r) for r in gdict['coordinates']]
    elif gdict['type'] == 'MultiPolygon':
        gdict['coordinates'] = [[conv(r) for r in poly] for poly in gdict['coordinates']]
    return gdict

def draw_gc_polyline_continuous(m, a, b, step_km=DRAW_STEP_KM, **style):
    pts = geodesic_sample(a,b, step_km=step_km)
    folium_coords = [[lat, normalize_lon_to_pacific_view(lon)] for lon,lat in pts]
    folium.PolyLine(folium_coords, **style).add_to(m)
