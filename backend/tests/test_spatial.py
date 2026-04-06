import pytest
import numpy as np
from core.ml.spatial import cluster_high_risk_cells


def test_cluster_basic():
    """Two tight groups of high-risk cells should form 2 clusters."""
    # Cluster 1: around (10, 119)
    lats_1 = np.array([10.0, 10.01, 10.02, 9.99])
    lons_1 = np.array([119.0, 119.01, 119.02, 118.99])
    # Cluster 2: around (11, 120) — far from cluster 1
    lats_2 = np.array([11.0, 11.01, 11.02, 10.99])
    lons_2 = np.array([120.0, 120.01, 120.02, 119.99])
    # Low risk scattered
    lats_low = np.array([12.0, 13.0])
    lons_low = np.array([121.0, 122.0])

    lats = np.concatenate([lats_1, lats_2, lats_low])
    lons = np.concatenate([lons_1, lons_2, lons_low])
    probs = np.array([0.8, 0.9, 0.7, 0.85, 0.6, 0.7, 0.65, 0.75, 0.1, 0.2])

    result = cluster_high_risk_cells(lats, lons, probs, risk_threshold=0.5, eps_km=15, min_samples=3)

    assert result["n_clusters"] == 2
    assert result["total_high_risk"] == 8
    assert len(result["clusters"]) == 2
    assert result["clusters"][0]["severity"] in ("critical", "elevated", "moderate")


def test_cluster_no_high_risk():
    """No high-risk cells should return empty clusters."""
    lats = np.array([10.0, 10.1])
    lons = np.array([119.0, 119.1])
    probs = np.array([0.1, 0.2])

    result = cluster_high_risk_cells(lats, lons, probs, risk_threshold=0.5)
    assert result["n_clusters"] == 0
    assert result["total_high_risk"] == 0


def test_cluster_all_noise():
    """Scattered high-risk cells should be noise, not clusters."""
    lats = np.array([10.0, 12.0, 14.0])
    lons = np.array([119.0, 121.0, 123.0])
    probs = np.array([0.8, 0.9, 0.7])

    result = cluster_high_risk_cells(lats, lons, probs, risk_threshold=0.5, eps_km=10, min_samples=3)
    assert result["n_clusters"] == 0
    assert result["n_noise"] == 3
