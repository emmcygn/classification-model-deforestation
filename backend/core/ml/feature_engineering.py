"""Spatial context feature engineering.

Computes features that capture the spatial neighborhood of each grid cell.
Deforestation is contagious — if neighboring cells are losing forest,
the current cell is at higher risk. These features capture that dynamic.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def add_spatial_features(df: pd.DataFrame, radius_km: float = 15.0) -> pd.DataFrame:
    """Add spatial context features using pre-period loss data only.

    Uses 'pre_period_loss' (2001-2019) if available, to avoid leaking
    the prediction target (2020-2022 loss) into features.

    New features:
    - neighbor_loss_count: cells with confirmed pre-period loss within radius
    - neighbor_loss_density: proportion of neighbors with pre-period loss
    - neighbor_mean_tree_cover: average tree cover of neighbors
    - loss_acceleration: local loss intensity relative to regional average

    Args:
        df: DataFrame with lat, lon, tree_cover_2000_pct, annual_loss_rate_pct
        radius_km: radius for neighborhood computation

    Returns:
        DataFrame with additional spatial features
    """
    coords = df[["lat", "lon"]].values

    # Use pre_period_loss if available, else fall back to annual_loss_rate_pct > median
    if "pre_period_loss" in df.columns:
        loss_indicator = df["pre_period_loss"].values
    else:
        # Fallback: use loss rate as proxy (doesn't leak target)
        loss_indicator = (df["annual_loss_rate_pct"] > df["annual_loss_rate_pct"].median()).astype(int).values

    # Convert to approximate km
    mean_lat = coords[:, 0].mean()
    coords_km = np.column_stack([
        coords[:, 0] * 111.0,
        coords[:, 1] * 111.0 * np.cos(np.radians(mean_lat)),
    ])

    # Compute pairwise distances (for datasets < 10k cells, this is fine)
    n = len(df)
    if n > 10000:
        # For large datasets, use chunked computation
        neighbor_loss_count = np.zeros(n)
        neighbor_loss_density = np.zeros(n)
        neighbor_mean_tc = np.zeros(n)

        chunk = 1000
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            dists = cdist(coords_km[i:end], coords_km)
            for j in range(end - i):
                mask = (dists[j] < radius_km) & (dists[j] > 0)
                if mask.any():
                    neighbor_loss_count[i + j] = loss_indicator[mask].sum()
                    neighbor_loss_density[i + j] = loss_indicator[mask].mean()
                    neighbor_mean_tc[i + j] = df["tree_cover_2000_pct"].values[mask].mean()
    else:
        dists = cdist(coords_km, coords_km)

        neighbor_loss_count = np.zeros(n)
        neighbor_loss_density = np.zeros(n)
        neighbor_mean_tc = np.zeros(n)

        for i in range(n):
            mask = (dists[i] < radius_km) & (dists[i] > 0)
            if mask.any():
                neighbor_loss_count[i] = loss_indicator[mask].sum()
                neighbor_loss_density[i] = loss_indicator[mask].mean()
                neighbor_mean_tc[i] = df["tree_cover_2000_pct"].values[mask].mean()

    # Loss acceleration: local loss rate vs regional average
    regional_mean_loss = df["annual_loss_rate_pct"].mean()
    loss_acceleration = df["annual_loss_rate_pct"].values / max(regional_mean_loss, 0.001) - 1.0

    # Fire hotspot proxy: count of high-loss-rate neighbors as a spatial
    # surrogate for fire activity. Uses pre-period loss rate (2001-2019 only)
    # instead of the live FIRMS 7-day feed, which would introduce present-day
    # data into a model that claims pre-2020 features.
    high_loss_threshold = df["annual_loss_rate_pct"].quantile(0.75)
    high_loss_indicator = (df["annual_loss_rate_pct"] > high_loss_threshold).astype(int).values
    fire_density = np.zeros(n)
    if n <= 10000:
        for i in range(n):
            mask = (dists[i] < radius_km) & (dists[i] > 0)
            if mask.any():
                fire_density[i] = high_loss_indicator[mask].sum()
    else:
        for i in range(0, n, 1000):
            end = min(i + 1000, n)
            chunk_dists = cdist(coords_km[i:end], coords_km)
            for j in range(end - i):
                mask = (chunk_dists[j] < radius_km) & (chunk_dists[j] > 0)
                if mask.any():
                    fire_density[i + j] = high_loss_indicator[mask].sum()

    df = df.copy()
    df["neighbor_loss_count"] = np.round(neighbor_loss_count, 0).astype(int)
    df["neighbor_loss_density"] = np.round(neighbor_loss_density, 4)
    df["neighbor_mean_tree_cover"] = np.round(neighbor_mean_tc, 1)
    df["loss_acceleration"] = np.round(loss_acceleration, 4)
    df["fire_hotspot_density"] = fire_density.astype(int)

    return df
