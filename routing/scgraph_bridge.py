# routing/scgraph_bridge.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple

def _ensure_scgraph():
    try:
        from scgraph.geographs.marnet import marnet_geograph  # noqa: F401
    except Exception as e:
        raise ImportError(
            "scgraph 套件未安裝或不可用。請先執行：pip install scgraph"
        ) from e

def sc_shortest_path_lonlat(
    origin_ll: Tuple[float, float],
    dest_ll: Tuple[float, float],
    *,
    output_units: str = "km",
    node_addition_lat_lon_bound: float = 180.0,
    node_addition_type: str = "quadrant",
    destination_node_addition_type: str = "all",
    node_addition_circuity: float = 10.0,
    cache: bool = False,
) -> Dict:
    """
    使用 scgraph 的海運網路，回傳：
      {
        "track": [(lon,lat), ...],    # 轉成 (lon,lat)
        "length_km": float,           # 統一為公里
      }
    """
    _ensure_scgraph()
    from scgraph.geographs.marnet import marnet_geograph

    out = marnet_geograph.get_shortest_path(
        origin_node={"latitude": origin_ll[1], "longitude": origin_ll[0]},
        destination_node={"latitude": dest_ll[1], "longitude": dest_ll[0]},
        output_units=output_units,
        node_addition_lat_lon_bound=node_addition_lat_lon_bound,
        node_addition_type=node_addition_type,
        destination_node_addition_type=destination_node_addition_type,
        node_addition_circuity=node_addition_circuity,
        cache=cache,
    )

    # scgraph 回傳 [lat, lon]；轉成 (lon, lat)
    latlon_list = out.get("coordinate_path", [])
    track_ll = [(lon, lat) for (lat, lon) in latlon_list]
    length = float(out.get("length", 0.0))

    units = (output_units or "km").lower()
    if units == "km":
        length_km = length
    elif units == "m":
        length_km = length / 1000.0
    elif units == "mi":
        length_km = length * 1.609344
    elif units == "ft":
        length_km = length * 0.0003048
    else:
        length_km = length  # 預設當 km

    return {"track": track_ll, "length_km": length_km}
