from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pyproj import CRS, Transformer
from shapely.geometry import Point, box
from shapely.ops import transform as shp_transform


LonLat = Tuple[float, float]
BBoxLL = Tuple[float, float, float, float]
XY = Tuple[float, float]


# ---------------------------
# Backward compatible helpers
# ---------------------------

def make_aoi_bbox(bbox_ll: BBoxLL) -> BBoxLL:
    """
    Normalize bbox_ll into (min_lon, min_lat, max_lon, max_lat).

    Accepts common user input variants:
    - (min_lon, min_lat, max_lon, max_lat)  -> unchanged
    - (min_lon, max_lat, max_lon, min_lat)  -> lat swapped (your example)
    - dateline-cross case is preserved when it's clearly shorter to wrap.
      (i.e., min_lon may be > max_lon to indicate crossing).
    """
    lon1, lat1, lon2, lat2 = map(float, bbox_ll)

    # latitude always sorted
    min_lat, max_lat = (lat1, lat2) if lat1 <= lat2 else (lat2, lat1)

    # longitude: decide whether to preserve dateline crossing
    # direct span vs wrap span
    direct = abs(lon2 - lon1)
    wrap = 360.0 - direct

    # If user gave lon1 > lon2, it *might* mean dateline crossing.
    # Keep crossing only if wrapping is clearly shorter.
    if lon1 > lon2 and wrap < direct:
        min_lon, max_lon = lon1, lon2  # crossing bbox (min_lon > max_lon)
    else:
        min_lon, max_lon = (lon1, lon2) if lon1 <= lon2 else (lon2, lon1)

    return (min_lon, min_lat, max_lon, max_lat)


# ---------------------------
# AOI Projector
# ---------------------------

@dataclass
class AOIProjector:
    """Local metric projection centered at AOI centroid (AEQD)."""

    crs_ll: CRS
    crs_m: CRS
    to_m: Transformer
    to_ll: Transformer

    def ll2m(self, lon: float, lat: float) -> XY:
        x, y = self.to_m.transform(float(lon), float(lat))
        return float(x), float(y)

    def m2ll(self, x: float, y: float) -> LonLat:
        lon, lat = self.to_ll.transform(float(x), float(y))
        return float(lon), float(lat)


def build_projector_from_bbox(bbox_ll: BBoxLL) -> AOIProjector:
    bbox_ll = make_aoi_bbox(bbox_ll)
    min_lon, min_lat, max_lon, max_lat = bbox_ll

    # dateline-safe midpoint
    if min_lon <= max_lon:
        lon0 = (min_lon + max_lon) / 2.0
    else:
        lon0 = (min_lon + (max_lon + 360.0)) / 2.0
        if lon0 > 180.0:
            lon0 -= 360.0
    lat0 = (min_lat + max_lat) / 2.0

    crs_ll = CRS.from_epsg(4326)
    crs_m = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    to_m = Transformer.from_crs(crs_ll, crs_m, always_xy=True)
    to_ll = Transformer.from_crs(crs_m, crs_ll, always_xy=True)
    return AOIProjector(crs_ll=crs_ll, crs_m=crs_m, to_m=to_m, to_ll=to_ll)


def geom_to_m(geom, proj: AOIProjector):
    return shp_transform(lambda x, y, z=None: proj.to_m.transform(x, y), geom)


def geom_to_ll(geom, proj: AOIProjector):
    return shp_transform(lambda x, y, z=None: proj.to_ll.transform(x, y), geom)


def linestring_sample_points(line, ds_m: float):
    """Sample points along a LineString/LinearRing every ds_m meters."""
    if line is None or line.is_empty:
        return []
    L = float(line.length)
    if L <= 0:
        try:
            return [Point(line.coords[0])]
        except Exception:
            return []

    ds_m = float(ds_m)
    if ds_m <= 0:
        return [line.interpolate(0.0), line.interpolate(L)]

    n = int(L // ds_m) + 1
    pts = [line.interpolate(i * ds_m) for i in range(n)]
    try:
        end_pt = line.interpolate(L)
        if pts and pts[-1].distance(end_pt) > 1e-6:
            pts[-1] = end_pt
    except Exception:
        pass
    return pts


def expand_bbox_ll(bbox_ll: BBoxLL, pad_deg: float) -> BBoxLL:
    """Simple bbox expand (no split). OK for most AOIs you use."""
    min_lon, min_lat, max_lon, max_lat = make_aoi_bbox(bbox_ll)
    p = float(pad_deg)
    return (
        max(-180.0, min_lon - p),
        max(-89.9999, min_lat - p),
        min(180.0, max_lon + p),
        min(89.9999, max_lat + p),
    )


# ---------------------------
# B-scheme unified helpers
# ---------------------------

def get_projector(out: Dict[str, Any], bbox_ll: Optional[BBoxLL] = None) -> AOIProjector:
    """
    Resolve AOI projector (B-scheme).
    Preference:
      1) out['proj'] if present (AOIProjector or compatible)
      2) build_projector_from_bbox(bbox_ll)
      3) build_projector_from_bbox(out['bbox_ll'])
    """
    proj = out.get("proj", None)
    if proj is not None:
        if hasattr(proj, "ll2m") and hasattr(proj, "m2ll"):
            return proj  # type: ignore
        if hasattr(proj, "to_m") and hasattr(proj, "to_ll"):
            # wrap transformers into AOIProjector
            crs_ll = getattr(proj, "crs_ll", CRS.from_epsg(4326))
            crs_m = getattr(proj, "crs_m", CRS.from_proj4("+proj=aeqd +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"))
            return AOIProjector(crs_ll=crs_ll, crs_m=crs_m, to_m=proj.to_m, to_ll=proj.to_ll)  # type: ignore

    if bbox_ll is None:
        bbox_ll = out.get("bbox_ll", None)
    if bbox_ll is None:
        raise ValueError("Cannot resolve projector: provide bbox_ll or out['bbox_ll'] or out['proj']")
    return build_projector_from_bbox(bbox_ll)


def get_collision_metric(
    out: Dict[str, Any],
    *,
    prefer_prepared: bool = True,
    key_prepared: str = "COLLISION_PREP_M",
    key_raw: str = "COLLISION_M",
) -> Tuple[Optional[Any], bool]:
    """
    Resolve collision geometry in metric CRS.
    Returns (geom, is_prepared).
    """
    layers = out.get("layers") or {}
    if prefer_prepared and layers.get(key_prepared) is not None:
        return layers[key_prepared], True
    if layers.get(key_raw) is not None:
        return layers[key_raw], False
    if layers.get(key_prepared) is not None:
        return layers[key_prepared], True
    return None, False


def geom_m_to_ll(geom_m, proj: AOIProjector):
    """
    Convert metric geom to lonlat geom with axis-order guard.
    In theory always_xy=True should be stable, but keep guard for safety.
    """
    def inv(x, y, z=None):
        lon, lat = proj.to_ll.transform(x, y)
        lon = float(lon); lat = float(lat)
        # if swapped looks more plausible, swap back
        if abs(lon) <= 90 and abs(lat) > 90:
            lon, lat = lat, lon
        return lon, lat

    return shp_transform(inv, geom_m)


def clip_collision_to_aoi_bbox(
    collision_m,
    bbox_ll: BBoxLL,
    proj: AOIProjector,
    *,
    pad_m: float = 0.0,
    step_deg: float = 0.2,
):
    """
    Clip collision geometry to AOI bbox window (in metric).
    - Supports prepared geometry: uses .context if available.
    - Uses densified bbox boundary to better bound AEQD distortion.
    """
    if collision_m is None:
        return None

    # unwrap prepared geometry if possible
    if hasattr(collision_m, "context"):
        collision_raw = collision_m.context
    else:
        collision_raw = collision_m

    bbox_ll = make_aoi_bbox(bbox_ll)
    min_lon, min_lat, max_lon, max_lat = bbox_ll
    step_deg = max(1e-6, float(step_deg))

    def lin(a: float, b: float) -> List[float]:
        if abs(b - a) < 1e-12:
            return [a]
        n = int(abs(b - a) / step_deg) + 1
        return [a + (b - a) * i / n for i in range(n + 1)]

    xs = lin(min_lon, max_lon)
    ys = lin(min_lat, max_lat)

    pts_ll: List[LonLat] = []
    for x in xs: pts_ll.append((x, min_lat))
    for y in ys: pts_ll.append((max_lon, y))
    for x in reversed(xs): pts_ll.append((x, max_lat))
    for y in reversed(ys): pts_ll.append((min_lon, y))

    pts_m = [proj.ll2m(lon, lat) for lon, lat in pts_ll]
    minx = min(p[0] for p in pts_m) - float(pad_m)
    miny = min(p[1] for p in pts_m) - float(pad_m)
    maxx = max(p[0] for p in pts_m) + float(pad_m)
    maxy = max(p[1] for p in pts_m) + float(pad_m)

    win = box(minx, miny, maxx, maxy)
    try:
        return collision_raw.intersection(win)
    except Exception:
        # conservative fallback: return raw
        return collision_raw
