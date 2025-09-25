from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
CSV_PATH = Path("Raw Data") / "Device_AB00035.csv"
TARGET_MMSI = 416426000
FIRST_STAGE_EPS = 0.01  # radians; ~63 km
FIRST_STAGE_MIN_SAMPLES = 10
SECOND_STAGE_EPS_KM = 1.0  # merge harbour centers within 1 km

# haversine to radius km
def _haversine_km(latitudes: Iterable[float], longitudes: Iterable[float], center_lat: float, center_lon: float) -> np.ndarray:
    """Vectorised haversine distance from points to a single center in kilometres."""
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)

    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    center_lat_rad = np.radians(center_lat)
    center_lon_rad = np.radians(center_lon)

    delta_lat = lat_rad - center_lat_rad
    delta_lon = lon_rad - center_lon_rad

    sin_lat = np.sin(delta_lat / 2.0)
    sin_lon = np.sin(delta_lon / 2.0)

    a = sin_lat ** 2 + np.cos(center_lat_rad) * np.cos(lat_rad) * sin_lon ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c

# haversine to radius radians
def _haversine_rad_vector(points_rad: np.ndarray, center_rad: np.ndarray) -> np.ndarray:
    """Great-circle distance from multiple points to a single center, expressed in radians."""
    delta_lat = points_rad[:, 0] - center_rad[0]
    delta_lon = points_rad[:, 1] - center_rad[1]

    sin_lat = np.sin(delta_lat / 2.0)
    sin_lon = np.sin(delta_lon / 2.0)

    a = sin_lat ** 2 + np.cos(center_rad[0]) * np.cos(points_rad[:, 0]) * sin_lon ** 2
    return 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

# DBSCAN 
def _dbscan_haversine(coords_rad: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Lightweight DBSCAN implementation tailored for haversine distances."""
    n_samples = len(coords_rad)
    if n_samples == 0:
        return np.empty((0,), dtype=int)

    coords_rad = np.asarray(coords_rad, dtype=float)
    labels = np.full(n_samples, -1, dtype=int)
    visited = np.zeros(n_samples, dtype=bool)

    eps_deg = max(np.degrees(eps), 1e-6)
    lat_deg = np.degrees(coords_rad[:, 0])
    lon_deg = np.degrees(coords_rad[:, 1])

    grid: dict[tuple[int, int], list[int]] = {}
    for idx in range(n_samples):
        cell = (int(np.floor(lat_deg[idx] / eps_deg)), int(np.floor(lon_deg[idx] / eps_deg)))
        grid.setdefault(cell, []).append(idx)

    def region_query(point_idx: int) -> np.ndarray:
        base_cell = (int(np.floor(lat_deg[point_idx] / eps_deg)), int(np.floor(lon_deg[point_idx] / eps_deg)))
        candidate_indices: list[int] = []
        for d_lat in (-1, 0, 1):
            for d_lon in (-1, 0, 1):
                candidate_indices.extend(grid.get((base_cell[0] + d_lat, base_cell[1] + d_lon), []))
        if not candidate_indices:
            return np.empty((0,), dtype=int)
        unique_candidates = np.unique(candidate_indices)
        distances = _haversine_rad_vector(coords_rad[unique_candidates], coords_rad[point_idx])
        return unique_candidates[distances <= eps]

    cluster_id = -1
    for point_idx in range(n_samples):
        if visited[point_idx]:
            continue

        visited[point_idx] = True
        neighbors = region_query(point_idx)
        if neighbors.size < min_samples:
            labels[point_idx] = -1
            continue

        cluster_id += 1
        labels[point_idx] = cluster_id
        neighbor_set = set(map(int, neighbors.tolist()))
        neighbor_set.discard(point_idx)
        queue = deque(neighbor_set)

        while queue:
            current_point = queue.pop()
            if not visited[current_point]:
                visited[current_point] = True
                current_neighbors = region_query(current_point)
                if current_neighbors.size >= min_samples:
                    for neighbor_idx in map(int, current_neighbors.tolist()):
                        if neighbor_idx not in neighbor_set:
                            neighbor_set.add(neighbor_idx)
                            queue.append(neighbor_idx)
            labels[current_point] = cluster_id

    return labels

# Data Preprocessing and Filtering
def load_and_preprocess(csv_path: Path = CSV_PATH, target_mmsi: int = TARGET_MMSI) -> pd.DataFrame:
    """Load the raw AIS data and apply the filters specified in the spec."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["MMSI"] == target_mmsi].copy()

    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Long"] = pd.to_numeric(df["Long"], errors="coerce")
    df["Sog"] = pd.to_numeric(df["Sog"], errors="coerce")
    df = df.dropna(subset=["Lat", "Long", "Sog"])

    df = df[df["Lat"].between(-90.0, 90.0)]
    df = df[df["Long"].between(-180.0, 180.0)]

    df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    slow_df = df[df["Sog"] < 0.2].copy()
    return slow_df.reset_index(drop=True)


def _first_stage_clusters(slow_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the first DBSCAN over low-speed points and return cluster summaries and memberships."""
    if slow_df.empty:
        empty_cols = ["cluster_id", "center_lat", "center_lon", "radius_km"]
        return pd.DataFrame(columns=empty_cols), slow_df.assign(cluster_id=pd.Series(dtype=int))

    coords_rad = np.radians(slow_df[["Lat", "Long"]].to_numpy())
    labels = _dbscan_haversine(coords_rad, eps=FIRST_STAGE_EPS, min_samples=FIRST_STAGE_MIN_SAMPLES)

    slow_df = slow_df.copy()
    slow_df["cluster_id"] = labels
    member_df = slow_df[slow_df["cluster_id"] >= 0].copy()

    if member_df.empty:
        empty_cols = ["cluster_id", "center_lat", "center_lon", "radius_km"]
        return pd.DataFrame(columns=empty_cols), member_df

    summaries = []
    for cluster_id, cluster_points in member_df.groupby("cluster_id"):
        center_lat = cluster_points["Lat"].mean()
        center_lon = cluster_points["Long"].mean()
        distances = _haversine_km(cluster_points["Lat"].to_numpy(), cluster_points["Long"].to_numpy(), center_lat, center_lon)
        radius_km = float(np.quantile(distances, 0.95) * 1.2) if len(distances) else 0.0
        summaries.append({
            "cluster_id": cluster_id,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "radius_km": radius_km,
        })

    summary_df = pd.DataFrame(summaries)
    return summary_df, member_df


def _merge_clusters(summary_df: pd.DataFrame, member_df: pd.DataFrame) -> pd.DataFrame:
    """Merge nearby harbour clusters using a secondary DBSCAN over the centroids."""
    if summary_df.empty:
        return pd.DataFrame(columns=["port_id", "center_lat", "center_lon", "radius_km"])

    centers_rad = np.radians(summary_df[["center_lat", "center_lon"]].to_numpy())
    eps_rad = SECOND_STAGE_EPS_KM / EARTH_RADIUS_KM
    merge_labels = _dbscan_haversine(centers_rad, eps=eps_rad, min_samples=1)

    summary_df = summary_df.copy()
    summary_df["merge_id"] = merge_labels

    merged_ports = []
    for port_index, (merge_id, subset) in enumerate(summary_df.groupby("merge_id")):
        member_ids = subset["cluster_id"].tolist()
        merged_points = member_df[member_df["cluster_id"].isin(member_ids)]
        center_lat = merged_points["Lat"].mean()
        center_lon = merged_points["Long"].mean()
        distances = _haversine_km(merged_points["Lat"].to_numpy(), merged_points["Long"].to_numpy(), center_lat, center_lon)
        radius_km = float(np.quantile(distances, 0.95) * 1.2) if len(distances) else 0.0

        merged_ports.append({
            "port_id": port_index,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "radius_km": radius_km,
        })

    merged_df = pd.DataFrame(merged_ports)
    return merged_df.sort_values("port_id").reset_index(drop=True)


def detect_ports(csv_path: Path = CSV_PATH, target_mmsi: int = TARGET_MMSI) -> pd.DataFrame:
    """High-level helper that runs the entire harbour-detection pipeline."""
    slow_df = load_and_preprocess(csv_path, target_mmsi)
    summary_df, member_df = _first_stage_clusters(slow_df)
    return _merge_clusters(summary_df, member_df)


if __name__ == "__main__":
    ports = detect_ports()
    if ports.empty:
        print("No ports detected with the current parameters.")
    else:
        print(ports.to_string(index=False))
