from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import folium
import pandas as pd
from folium import MacroElement
from jinja2 import Template

LonLat = Tuple[float, float]
BBoxLL = Tuple[float, float, float, float]

import folium
from folium import MacroElement
from jinja2 import Template


class _SelectAllNoneControl(MacroElement):
    """
    Add two buttons (All / None) that toggle all Overlay layers in folium.LayerControl.
    """
    def __init__(self, position="topright", all_text="All", none_text="None"):
        super().__init__()
        self._name = "SelectAllNoneControl"
        self.position = position
        self.all_text = all_text
        self.none_text = none_text

        self._template = Template(
            """
            {% macro script(this, kwargs) %}
            (function() {
              // map variable name in folium
              var map = {{ this._parent.get_name() }};

              function setAllOverlays(checked) {
                // LayerControl creates checkboxes under:
                // .leaflet-control-layers-overlays input[type=checkbox]
                var overlays = document.querySelectorAll(
                  ".leaflet-control-layers-overlays input[type='checkbox']"
                );
                overlays.forEach(function(cb) {
                  // Click to ensure Leaflet fires handlers properly
                  if (cb.checked !== checked) cb.click();
                });
              }

              var control = L.control({position: "{{ this.position }}"});
              control.onAdd = function() {
                var div = L.DomUtil.create("div", "leaflet-bar leaflet-control");
                div.style.background = "white";
                div.style.padding = "6px";
                div.style.borderRadius = "4px";

                var btnAll = L.DomUtil.create("a", "", div);
                btnAll.href = "#";
                btnAll.innerHTML = "{{ this.all_text }}";
                btnAll.style.display = "block";
                btnAll.style.textAlign = "center";
                btnAll.style.padding = "2px 8px";
                btnAll.style.textDecoration = "none";

                var btnNone = L.DomUtil.create("a", "", div);
                btnNone.href = "#";
                btnNone.innerHTML = "{{ this.none_text }}";
                btnNone.style.display = "block";
                btnNone.style.textAlign = "center";
                btnNone.style.padding = "2px 8px";
                btnNone.style.textDecoration = "none";

                // Prevent map dragging/zooming when clicking the buttons
                L.DomEvent.disableClickPropagation(div);
                L.DomEvent.disableScrollPropagation(div);

                L.DomEvent.on(btnAll, "click", function(e) {
                  L.DomEvent.preventDefault(e);
                  setAllOverlays(true);
                });

                L.DomEvent.on(btnNone, "click", function(e) {
                  L.DomEvent.preventDefault(e);
                  setAllOverlays(false);
                });

                return div;
              };

              control.addTo(map);
            })();
            {% endmacro %}
            """
        )


def add_select_all_none_layer_control(
    m: folium.Map,
    *,
    position: str = "topright",
    all_text: str = "All",
    none_text: str = "None",
) -> None:
    """
    Add 'All/None' buttons to toggle all overlay layers in folium.LayerControl.

    Call this AFTER you've added folium.LayerControl to the map
    (or at least after layers are created).
    """
    _SelectAllNoneControl(position=position, all_text=all_text, none_text=none_text).add_to(m)



def _in_bbox(p: LonLat, bbox_ll: Optional[BBoxLL]) -> bool:
    if bbox_ll is None:
        return True
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_ll)
    lon, lat = float(p[0]), float(p[1])
    return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)


def make_base_map(bbox_ll: BBoxLL, *, zoom_start: int = 5, control_scale: bool = True) -> folium.Map:
    min_lon, min_lat, max_lon, max_lat = map(float, bbox_ll)
    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    m = folium.Map(location=center, zoom_start=zoom_start, control_scale=control_scale)
    folium.Rectangle(bounds=[[min_lat, min_lon], [max_lat, max_lon]], fill=False, weight=3, opacity=0.9).add_to(m)
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    return m


def add_points_layer(
    m: folium.Map,
    pts_ll: Sequence[LonLat],
    *,
    name: str,
    radius: int = 4,
    show: bool = True,
    bbox_ll: Optional[BBoxLL] = None,
) -> None:
    fg = folium.FeatureGroup(name=name, show=show)
    for lon, lat in pts_ll:
        p = (float(lon), float(lat))
        if not _in_bbox(p, bbox_ll):
            continue
        folium.CircleMarker([p[1], p[0]], radius=int(radius)).add_to(fg)
    fg.add_to(m)


def add_path_layer(
    m: folium.Map,
    path_ll: Sequence[LonLat],
    *,
    name: str,
    weight: int = 4,
    opacity: float = 0.95,
    show: bool = True,
) -> None:
    if not path_ll or len(path_ll) < 2:
        return
    fg = folium.FeatureGroup(name=name, show=show)
    folium.PolyLine([[lat, lon] for (lon, lat) in path_ll], color="red", weight=int(weight), opacity=float(opacity)).add_to(fg)
    fg.add_to(m)


def add_sea_layers(
    m: folium.Map,
    out: Dict[str, Any],
    *,
    show_nodes: bool = True,
    show_edges: bool = True,
    node_sample: Optional[int] = None,
    max_edges: int = 6000,
    show: bool = True,
    bbox_ll: Optional[BBoxLL] = None,
) -> None:
    S_nodes = out.get("S_nodes")
    S_edges = out.get("S_edges")
    if isinstance(S_nodes, pd.DataFrame) and len(S_nodes) > 0 and show_nodes:
        df = S_nodes
        if node_sample is not None and len(df) > int(node_sample):
            df = df.sample(int(node_sample), random_state=7)
        fgN = folium.FeatureGroup(name=f"S_nodes ({len(df)}/{len(S_nodes)})", show=show)
        for _, r in df.iterrows():
            p = (float(r["lon"]), float(r["lat"]))
            if not _in_bbox(p, bbox_ll):
                continue
            folium.CircleMarker([p[1], p[0]], color="blue", radius=3).add_to(fgN)
        fgN.add_to(m)

    if isinstance(S_nodes, pd.DataFrame) and S_edges is not None and show_edges:
        # Normalize edge container to an iterable of 2-endpoints.
        # Supported:
        #  - list/tuple of (u,v) where u,v are indices or lonlat
        #  - DataFrame with columns (u,v) or (a,b)
        if isinstance(S_edges, pd.DataFrame):
            if {"u", "v"}.issubset(S_edges.columns):
                take_iter = S_edges[["u", "v"]].head(int(max_edges)).itertuples(index=False, name=None)
                total_edges = len(S_edges)
            elif {"a", "b"}.issubset(S_edges.columns):
                take_iter = S_edges[["a", "b"]].head(int(max_edges)).itertuples(index=False, name=None)
                total_edges = len(S_edges)
            else:
                take_iter = []
                total_edges = len(S_edges)
            take = list(take_iter)
        else:
            total_edges = len(S_edges)
            take = S_edges[:max_edges] if len(S_edges) > int(max_edges) else S_edges
        fgE = folium.FeatureGroup(name=f"S_edges ({len(take)}/{len(S_edges)})", show=show)

        def sea_ll(i: int) -> LonLat:
            s = S_nodes.iloc[int(i)]
            return (float(s["lon"]), float(s["lat"]))

        def parse_ll(obj) -> Optional[LonLat]:
            if isinstance(obj, (list, tuple)) and len(obj) >= 2:
                try:
                    return (float(obj[0]), float(obj[1]))
                except Exception:
                    return None
            if isinstance(obj, str) and "," in obj:
                try:
                    a, b = obj.split(",")
                    return (float(a), float(b))
                except Exception:
                    return None
            return None

        for e in take:
            if not isinstance(e, (list, tuple)) or len(e) < 2:
                continue
            a = b = None
            u, v = e[0], e[1]

            # case1: integer indices into S_nodes
            if isinstance(u, (int,)) and isinstance(v, (int,)):
                try:
                    a = sea_ll(int(u))
                    b = sea_ll(int(v))
                except Exception:
                    a = b = None
            else:
                # case2: endpoints already lon/lat
                a = parse_ll(u)
                b = parse_ll(v)

            if a is None or b is None:
                continue
            if not (_in_bbox(a, bbox_ll) or _in_bbox(b, bbox_ll)):
                continue
            folium.PolyLine([[a[1], a[0]], [b[1], b[0]]], weight=2, opacity=0.8).add_to(fgE)
        fgE.add_to(m)


def add_ring_layers(
    m: folium.Map,
    out: Dict[str, Any],
    *,
    show_e: bool = True,
    show_t: bool = True,
    e_node_sample: Optional[int] = None,
    t_node_sample: Optional[int] = None,
    max_e_edges: int = 4000,
    max_t_edges: int = 4000,
    show: bool = True,
    bbox_ll: Optional[BBoxLL] = None,
) -> None:
    rg = out.get("ring_graph", {}) or {}
    E_nodes = rg.get("E_nodes")
    T_nodes = rg.get("T_nodes")
    E_edges = rg.get("E_edges")
    T_edges = rg.get("T_edges")

    def _nodes_layer(df: pd.DataFrame, title: str, sample: Optional[int], radius: int):
        dd = df
        if sample is not None and len(dd) > int(sample):
            dd = dd.sample(int(sample), random_state=7)
        fg = folium.FeatureGroup(name=f"{title} ({len(dd)}/{len(df)})", show=show)
        for _, r in dd.iterrows():
            p = (float(r.get("lon")), float(r.get("lat")))
            if not _in_bbox(p, bbox_ll):
                continue
            folium.CircleMarker([p[1], p[0]], radius=radius).add_to(fg)
        fg.add_to(m)

    if show_e and isinstance(E_nodes, pd.DataFrame) and len(E_nodes) > 0:
        _nodes_layer(E_nodes, "E_nodes", e_node_sample, radius=2)

    if show_t and isinstance(T_nodes, pd.DataFrame) and len(T_nodes) > 0:
        _nodes_layer(T_nodes, "T_nodes", t_node_sample, radius=3)

    def _edge_layer(edges_df: pd.DataFrame, nodes_df: pd.DataFrame, title: str, max_edges: int):
        if not isinstance(edges_df, pd.DataFrame) or len(edges_df) == 0:
            return
        if not isinstance(nodes_df, pd.DataFrame) or len(nodes_df) == 0:
            return
        if "node_id" not in nodes_df.columns:
            return
        id2ll = {int(r["node_id"]): (float(r.get("lon")), float(r.get("lat"))) for _, r in nodes_df.iterrows()}
        take = edges_df.head(int(max_edges))
        fg = folium.FeatureGroup(name=f"{title} ({len(take)}/{len(edges_df)})", show=show)
        for _, r in take.iterrows():
            try:
                u = int(r.get("u"))
                v = int(r.get("v"))
                a = id2ll.get(u)
                b = id2ll.get(v)
                if a is None or b is None:
                    continue
                if not (_in_bbox(a, bbox_ll) or _in_bbox(b, bbox_ll)):
                    continue
                folium.PolyLine([[a[1], a[0]], [b[1], b[0]]], color="green", weight=2, opacity=0.8).add_to(fg)
            except Exception:
                continue
        fg.add_to(m)

    if show_e and isinstance(E_edges, pd.DataFrame) and isinstance(E_nodes, pd.DataFrame):
        _edge_layer(E_edges, E_nodes, "E_edges", max_e_edges)

    if show_t and isinstance(T_edges, pd.DataFrame) and isinstance(T_nodes, pd.DataFrame):
        _edge_layer(T_edges, T_nodes, "T_edges", max_t_edges)


def add_connector_layers(
    m: folium.Map,
    out: Dict[str, Any],
    *,
    show_et: bool = True,
    show_tgate_sea: bool = True,
    max_et: int = 2000,
    max_tgate: int = 2000,
    show: bool = True,
    bbox_ll: Optional[BBoxLL] = None,
) -> None:
    rg = out.get("ring_graph", {}) or {}
    E_nodes = rg.get("E_nodes")
    T_nodes = rg.get("T_nodes")
    ET = rg.get("ET_edges")
    dfTG = out.get("tgate_sea_connectors")
    S_nodes = out.get("S_nodes")

    # maps
    e_map = {}
    t_map = {}
    if isinstance(E_nodes, pd.DataFrame) and "node_id" in E_nodes.columns:
        e_map = {int(r["node_id"]): (float(r.get("lon")), float(r.get("lat"))) for _, r in E_nodes.iterrows()}
    if isinstance(T_nodes, pd.DataFrame) and "node_id" in T_nodes.columns:
        t_map = {int(r["node_id"]): (float(r.get("lon")), float(r.get("lat"))) for _, r in T_nodes.iterrows()}

    if show_et and isinstance(ET, pd.DataFrame) and e_map and t_map:
        take = ET.head(int(max_et))
        fg = folium.FeatureGroup(name=f"E<->T ({len(take)}/{len(ET)})", show=show)
        for _, r in take.iterrows():
            try:
                u = int(r.get("u"))
                v = int(r.get("v"))
            except Exception:
                continue
            # By design in e_t_transfer_v2:
            #   u = e_node_id, v = t_node_id
            # Using `or` here can explode if node_id ranges overlap between E and T.
            a = e_map.get(u)
            b = t_map.get(v)
            if a is None or b is None:
                # fallback: tolerate swapped columns in older experiments
                a = e_map.get(v)
                b = t_map.get(u)
            if a is None or b is None:
                continue
            if not (_in_bbox(a, bbox_ll) or _in_bbox(b, bbox_ll)):
                continue
            folium.PolyLine([[a[1], a[0]], [b[1], b[0]]], color="orange", weight=2, opacity=0.85).add_to(fg)
        fg.add_to(m)

    if show_tgate_sea and isinstance(dfTG, pd.DataFrame) and isinstance(S_nodes, pd.DataFrame) and t_map:
        sea_col = "sea_idx" if "sea_idx" in dfTG.columns else ("s_idx" if "s_idx" in dfTG.columns else None)
        tcol = "t_node_id" if "t_node_id" in dfTG.columns else ("t_id" if "t_id" in dfTG.columns else None)
        if sea_col and tcol:
            take = dfTG.head(int(max_tgate))
            fg = folium.FeatureGroup(name=f"Tgate->Sea ({len(take)}/{len(dfTG)})", show=show)
            for _, r in take.iterrows():
                try:
                    tid = int(r[tcol])
                    sid = int(r[sea_col])
                    a = t_map.get(tid)
                    s = S_nodes.iloc[sid]
                    b = (float(s["lon"]), float(s["lat"]))
                except Exception:
                    continue
                if a is None:
                    continue
                if not (_in_bbox(a, bbox_ll) or _in_bbox(b, bbox_ll)):
                    continue
                folium.PolyLine([[a[1], a[0]], [b[1], b[0]]], weight=2, opacity=0.85).add_to(fg)
            fg.add_to(m)


def finalize_map(m: folium.Map, *, html_path: str) -> str:
    m.add_child(folium.LayerControl(collapsed=False))

    # add all/none toggle buttons
    add_select_all_none_layer_control(m, position="topright", all_text="ALL", none_text="NONE")

    m.save(html_path)
    return html_path



__all__ = [
    "make_base_map",
    "add_points_layer",
    "add_path_layer",
    "add_sea_layers",
    "add_ring_layers",
    "add_connector_layers",
    "finalize_map",
]
