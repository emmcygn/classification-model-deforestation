"""Generate a synthetic but realistic deforestation risk dataset.

Covers a grid over Rondônia, Brazil (~10.5°S to ~13.5°S, ~60°W to ~63.5°W).
Each row is a 0.05° grid cell (~5.5km) with features derived from
known distributions of Hansen/GFW statistics.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

# Rondônia bounding box
LAT_MIN, LAT_MAX = -13.5, -10.5
LON_MIN, LON_MAX = -63.5, -60.0
STEP = 0.05  # ~5.5km grid cells


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    lats = np.arange(LAT_MIN, LAT_MAX, STEP)
    lons = np.arange(LON_MIN, LON_MAX, STEP)
    grid = np.array(np.meshgrid(lats, lons)).T.reshape(-1, 2)

    n = len(grid)

    # --- Features ---
    tree_cover_2000 = rng.beta(5, 2, n) * 100  # mostly forested
    elevation_m = rng.normal(200, 80, n).clip(50, 600)
    slope_deg = rng.exponential(3, n).clip(0, 30)
    dist_to_road_km = rng.exponential(15, n).clip(0.1, 200)
    dist_to_deforestation_frontier_km = rng.exponential(20, n).clip(0.1, 300)
    protected_area = rng.choice([0, 1], n, p=[0.7, 0.3])
    population_density = rng.exponential(10, n).clip(0, 200)

    # Historical annual loss rate (%/yr) — higher near roads, lower in protected
    base_loss = rng.exponential(1.5, n)
    road_effect = np.exp(-dist_to_road_km / 10) * 3
    protection_effect = protected_area * (-1.5)
    annual_loss_rate = (base_loss + road_effect + protection_effect).clip(0, 15)

    # --- Target: did significant loss occur in most recent 3 years? ---
    logit = (
        -2.0
        + 0.3 * annual_loss_rate
        - 0.01 * dist_to_road_km
        + 0.005 * population_density
        - 0.003 * elevation_m
        - 0.8 * protected_area
        - 0.01 * dist_to_deforestation_frontier_km
        + 0.02 * slope_deg
    )
    prob = 1 / (1 + np.exp(-logit))
    high_risk = rng.binomial(1, prob)

    df = pd.DataFrame({
        "lat": grid[:, 0],
        "lon": grid[:, 1],
        "tree_cover_2000_pct": np.round(tree_cover_2000, 1),
        "elevation_m": np.round(elevation_m, 1),
        "slope_deg": np.round(slope_deg, 1),
        "dist_to_road_km": np.round(dist_to_road_km, 2),
        "dist_to_deforestation_frontier_km": np.round(dist_to_deforestation_frontier_km, 2),
        "protected_area": protected_area,
        "population_density_per_km2": np.round(population_density, 1),
        "annual_loss_rate_pct": np.round(annual_loss_rate, 2),
        "high_risk": high_risk,
    })

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    out_path = OUTPUT_DIR / "rondonia_grid.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} grid cells -> {out_path}")
    print(f"High risk: {df['high_risk'].sum()} ({df['high_risk'].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
