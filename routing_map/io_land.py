from __future__ import annotations
from pathlib import Path
from typing import List

import geopandas as gpd
from shapely.geometry import box
from shapely.errors import GEOSException

try:
    from shapely.validation import make_valid  # shapely 2.x
except Exception:  # pragma: no cover
    make_valid = None

def load_polys_in_bbox(shp_path: str | Path, bbox_ll: tuple[float, float, float, float]):
    """Load land polygons intersecting bbox (lon/lat), robustly.

    Returns: list[shapely geometry] in lon/lat (EPSG:4326).
    """
    shp_path = Path(shp_path)
    (min_lon, min_lat, max_lon, max_lat) = bbox_ll
    bbox = box(min_lon, min_lat, max_lon, max_lat)

    gdf = gpd.read_file(shp_path, bbox=bbox)
    if gdf.empty:
        return []

    # Ensure lon/lat
    if gdf.crs is not None and str(gdf.crs).lower() not in ("epsg:4326", "wgs84"):
        gdf = gdf.to_crs("EPSG:4326")

    # explode multiparts
    try:
        gdf = gdf.explode(index_parts=False, ignore_index=True)
    except TypeError:
        gdf = gdf.explode()

    geoms = []
    for g in gdf.geometry:
        if g is None:
            continue
        try:
            gg = g
            if make_valid is not None:
                gg = make_valid(gg)
            gg = gg.buffer(0)
            if not gg.is_empty:
                geoms.append(gg)
        except GEOSException:
            # last resort: buffer(0) only
            try:
                gg = g.buffer(0)
                if not gg.is_empty:
                    geoms.append(gg)
            except Exception:
                continue
    return geoms
