"""Spatial analysis — clustering and anomaly detection.

Uses DBSCAN to identify deforestation fronts: spatial clusters of high-risk
cells that form contiguous areas of concern. Individual cell predictions
become actionable when they form patterns.
"""

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_high_risk_cells(
    lats: np.ndarray,
    lons: np.ndarray,
    risk_probs: np.ndarray,
    risk_threshold: float = 0.5,
    eps_km: float = 15.0,
    min_samples: int = 3,
) -> dict:
    """Cluster high-risk cells into deforestation fronts using DBSCAN.

    Args:
        lats: Array of latitudes
        lons: Array of longitudes
        risk_probs: Array of risk probabilities (0-1)
        risk_threshold: Minimum probability to consider a cell high-risk
        eps_km: Maximum distance between cells in a cluster (km)
        min_samples: Minimum cells to form a cluster

    Returns:
        Dict with clusters, noise points, and summary statistics
    """
    # Filter to high-risk cells only
    high_risk_mask = risk_probs >= risk_threshold
    if high_risk_mask.sum() < min_samples:
        return {
            "total_high_risk": int(high_risk_mask.sum()),
            "n_clusters": 0,
            "clusters": [],
            "noise_points": [],
            "method": f"DBSCAN (eps={eps_km}km, min_samples={min_samples})",
        }

    hr_lats = lats[high_risk_mask]
    hr_lons = lons[high_risk_mask]
    hr_probs = risk_probs[high_risk_mask]

    # Convert lat/lon to approximate km for DBSCAN
    # 1 degree latitude ~ 111 km, 1 degree longitude ~ 111 * cos(lat) km
    mean_lat = np.mean(hr_lats)
    coords_km = np.column_stack([
        hr_lats * 111.0,
        hr_lons * 111.0 * np.cos(np.radians(mean_lat)),
    ])

    # Run DBSCAN
    db = DBSCAN(eps=eps_km, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(coords_km)

    # Build cluster summaries
    clusters = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label

    for cluster_id in sorted(unique_labels):
        mask = labels == cluster_id
        cluster_lats = hr_lats[mask]
        cluster_lons = hr_lons[mask]
        cluster_probs = hr_probs[mask]

        # Compute cluster centroid and extent
        centroid_lat = float(np.mean(cluster_lats))
        centroid_lon = float(np.mean(cluster_lons))

        # Approximate area (bounding box in km2)
        lat_range = float(cluster_lats.max() - cluster_lats.min()) * 111.0
        lon_range = float(cluster_lons.max() - cluster_lons.min()) * 111.0 * np.cos(np.radians(centroid_lat))
        approx_area_km2 = max(lat_range * lon_range, 1.0)

        clusters.append({
            "cluster_id": int(cluster_id),
            "n_cells": int(mask.sum()),
            "centroid": {"lat": round(centroid_lat, 4), "lon": round(centroid_lon, 4)},
            "bounds": {
                "lat_min": round(float(cluster_lats.min()), 4),
                "lat_max": round(float(cluster_lats.max()), 4),
                "lon_min": round(float(cluster_lons.min()), 4),
                "lon_max": round(float(cluster_lons.max()), 4),
            },
            "mean_risk": round(float(cluster_probs.mean()), 4),
            "max_risk": round(float(cluster_probs.max()), 4),
            "approx_area_km2": round(approx_area_km2, 1),
            "severity": "critical" if cluster_probs.mean() > 0.7 else "elevated" if cluster_probs.mean() > 0.4 else "moderate",
        })

    # Sort by severity (highest mean risk first)
    clusters.sort(key=lambda c: -c["mean_risk"])

    # Noise points (high-risk but isolated)
    noise_mask = labels == -1
    noise_points = [
        {"lat": round(float(lat), 4), "lon": round(float(lon), 4), "risk": round(float(p), 4)}
        for lat, lon, p in zip(hr_lats[noise_mask], hr_lons[noise_mask], hr_probs[noise_mask])
    ]

    return {
        "total_high_risk": int(high_risk_mask.sum()),
        "n_clusters": len(clusters),
        "n_noise": int(noise_mask.sum()),
        "clusters": clusters,
        "noise_points": noise_points,
        "method": f"DBSCAN (eps={eps_km}km, min_samples={min_samples})",
    }
