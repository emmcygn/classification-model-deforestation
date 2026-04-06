"""One-shot data pipeline for Philippine deforestation data.

Usage: python -m core.data.fetch_philippines
"""

import numpy as np
import pandas as pd
import math
import time
from pathlib import Path

from core.data.sources.hansen import fetch_all_hansen
from core.data.sources.srtm_source import fetch_elevation_and_slope
from core.data.sources.osm import (
    fetch_roads_for_region,
    compute_distance_to_nearest_road,
    fetch_protected_areas_for_region,
    is_protected,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

REGIONS = {
    "palawan": {
        "lat_min": 8.5, "lat_max": 12.3,
        "lon_min": 117.2, "lon_max": 120.3,
        "step": 0.02,
    },
    "sierra_madre": {
        "lat_min": 15.5, "lat_max": 17.5,
        "lon_min": 121.0, "lon_max": 122.5,
        "step": 0.02,
    },
    "mindanao_bukidnon": {
        "lat_min": 7.5, "lat_max": 8.8,
        "lon_min": 124.5, "lon_max": 125.5,
        "step": 0.02,
    },
}

PROVENANCE = {
    "tree_cover_2000_pct": {"source": "Hansen/UMD GFW v1.7", "url": "https://glad.umd.edu/dataset/gfw"},
    "elevation_m": {"source": "SRTM 30m (USGS/NASA)", "url": "https://www.usgs.gov/centers/eros"},
    "slope_deg": {"source": "SRTM 30m (USGS/NASA)", "url": "https://www.usgs.gov/centers/eros"},
    "dist_to_road_km": {"source": "OpenStreetMap", "url": "https://www.openstreetmap.org"},
    "dist_to_deforestation_frontier_km": {"source": "Derived from Hansen GFW loss_year", "url": "https://glad.umd.edu/dataset/gfw"},
    "protected_area": {"source": "OpenStreetMap (boundary=protected_area)", "url": "https://www.openstreetmap.org"},
    "population_density_per_km2": {"source": "WorldPop/WOPR API (2019)", "url": "https://www.worldpop.org"},
    "annual_loss_rate_pct": {"source": "Hansen/UMD GFW v1.7 loss_year", "url": "https://glad.umd.edu/dataset/gfw"},
    "exg_change_2018_2019": {"source": "Sentinel-2 cloudless composites (EOX), ExG vegetation index (pre-target period)", "url": "https://tiles.maps.eox.at"},
    "neighbor_loss_count": {"source": "Computed from Hansen GFW loss data (15km radius)", "url": "https://glad.umd.edu/dataset/gfw"},
    "neighbor_loss_density": {"source": "Computed from Hansen GFW loss data (15km radius)", "url": "https://glad.umd.edu/dataset/gfw"},
    "neighbor_mean_tree_cover": {"source": "Computed from Hansen GFW tree cover (15km radius)", "url": "https://glad.umd.edu/dataset/gfw"},
    "loss_acceleration": {"source": "Derived from annual loss rate vs regional average", "url": "https://glad.umd.edu/dataset/gfw"},
    "fire_hotspot_density": {"source": "Derived: high-loss-rate neighbor count (pre-2020 spatial proxy)", "url": "https://glad.umd.edu/dataset/gfw"},
}


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def fetch_region(region_name: str) -> pd.DataFrame:
    cfg = REGIONS[region_name]
    lats = np.arange(cfg["lat_min"], cfg["lat_max"], cfg["step"])
    lons = np.arange(cfg["lon_min"], cfg["lon_max"], cfg["step"])
    grid = np.array(np.meshgrid(lats, lons)).T.reshape(-1, 2)

    print(f"[{region_name}] Grid: {len(grid)} total cells")

    # Temporal split: features from 2001-2019, target from 2020-2022
    # loss_year_raw: 1=2001, ..., 19=2019, 20=2020, 21=2021, 22=2022
    FEATURE_CUTOFF = 19  # Features use loss years 1-19 (2001-2019)
    TARGET_START = 20     # Target: loss in years 20-22 (2020-2022)

    # Phase 1: Hansen (tree cover + loss) — single call per cell
    print(f"[{region_name}] Phase 1: Hansen GFW tiles...", flush=True)
    hansen_data = []
    for i, (lat, lon) in enumerate(grid):
        h = fetch_all_hansen(lat, lon)
        ly = h["loss_year"]
        # Target: loss in prediction period (2020-2022) only
        high_risk = 1 if (ly >= TARGET_START) else 0
        # Track pre-period loss for features (2001-2019)
        pre_period_loss = 1 if (0 < ly <= FEATURE_CUTOFF) else 0
        hansen_data.append({
            "tree_cover_2000_pct": h["tree_cover_2000_pct"],
            "loss_year_raw": ly,
            "has_loss": h["has_loss"],
            "pre_period_loss": pre_period_loss,
            "high_risk": high_risk,
            "is_land": h["is_land"],
        })
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(grid)}", flush=True)

    # Filter ocean cells
    land_mask = [h["is_land"] for h in hansen_data]
    land_grid = grid[land_mask]
    land_hansen = [h for h, m in zip(hansen_data, land_mask) if m]
    print(f"[{region_name}] Land cells: {len(land_grid)} (filtered {len(grid)-len(land_grid)} ocean)")

    if len(land_grid) == 0:
        return pd.DataFrame()

    # Phase 2: SRTM
    print(f"[{region_name}] Phase 2: SRTM elevation...")
    elev_data = []
    for i, (lat, lon) in enumerate(land_grid):
        try:
            elev_data.append(fetch_elevation_and_slope(lat, lon))
        except Exception:
            elev_data.append({"elevation_m": 0.0, "slope_deg": 0.0})
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(land_grid)}")

    # Phase 3: OSM roads
    print(f"[{region_name}] Phase 3: OSM roads...")
    lat_min_r, lat_max_r = float(land_grid[:,0].min()), float(land_grid[:,0].max())
    lon_min_r, lon_max_r = float(land_grid[:,1].min()), float(land_grid[:,1].max())

    road_nodes = []
    chunk = 1.0
    for lat_s in np.arange(lat_min_r, lat_max_r, chunk):
        for lon_s in np.arange(lon_min_r, lon_max_r, chunk):
            try:
                roads = fetch_roads_for_region(
                    lat_s, min(lat_s+chunk, lat_max_r),
                    lon_s, min(lon_s+chunk, lon_max_r),
                )
                road_nodes.extend(roads)
                time.sleep(2)
            except Exception as e:
                print(f"  Road chunk failed ({lat_s:.1f},{lon_s:.1f}): {e}")
    print(f"[{region_name}] Road nodes: {len(road_nodes)}")

    # Subsample for speed
    if len(road_nodes) > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(road_nodes), 10000, replace=False)
        road_sample = [road_nodes[i] for i in idx]
    else:
        road_sample = road_nodes

    road_dists = []
    for i, (lat, lon) in enumerate(land_grid):
        road_dists.append(compute_distance_to_nearest_road(lat, lon, road_sample))
        if (i+1) % 500 == 0:
            print(f"  Road dist: {i+1}/{len(land_grid)}")

    # Phase 4: Protected areas
    print(f"[{region_name}] Phase 4: Protected areas...")
    try:
        protected = fetch_protected_areas_for_region(lat_min_r, lat_max_r, lon_min_r, lon_max_r)
        print(f"  Polygons: {len(protected)}")
    except Exception as e:
        print(f"  Protected areas failed: {e}")
        protected = []
    pa_status = [is_protected(lat, lon, protected) for lat, lon in land_grid]

    # Phase 5: Derived features (using PRE-PERIOD loss only to avoid target leakage)
    print(f"[{region_name}] Phase 5: Derived features...")
    # Frontier distance from PRE-PERIOD loss cells only (2001-2019)
    pre_loss_cells = [(lat, lon) for (lat, lon), h in zip(land_grid, land_hansen) if h["pre_period_loss"]]
    frontier_dists = []
    loss_sample = pre_loss_cells[:2000]  # Limit for speed
    for lat, lon in land_grid:
        if not loss_sample:
            frontier_dists.append(999.0)
        else:
            min_d = min(_haversine(lat, lon, lc[0], lc[1]) for lc in loss_sample)
            frontier_dists.append(round(min_d, 2))

    # Annual loss rate from PRE-PERIOD only (2001-2019)
    pre_loss_count = sum(1 for h in land_hansen if h["pre_period_loss"])
    pre_period_years = 19  # 2001-2019
    base_rate = (pre_loss_count / max(len(land_hansen), 1)) * 100 / pre_period_years
    annual_rates = [round(base_rate * (2 if h["pre_period_loss"] else 0.5), 2) for h in land_hansen]
    # Phase 5.5: Population density from WorldPop API
    print(f"[{region_name}] Phase 5.5: WorldPop population density...")
    try:
        from core.data.sources.worldpop import fetch_population_bulk
        pop_dens = fetch_population_bulk(
            land_grid[:, 0], land_grid[:, 1],
            fallback_road_dists=road_dists,
        )
    except Exception as e:
        print(f"  WorldPop failed, using road-distance fallback: {e}")
        pop_dens = [round(max(0, 50 * np.exp(-d/10)), 1) for d in road_dists]

    # Phase 6: Satellite vegetation change (ExG from Sentinel-2)
    print(f"[{region_name}] Phase 6: Satellite vegetation index change...")
    from core.cv.change_detection import compute_exg, _fetch_s2_tile, _tile_coords, ANALYSIS_ZOOM

    exg_changes = []
    for i, (lat, lon) in enumerate(land_grid):
        try:
            tx, ty = _tile_coords(float(lat), float(lon), ANALYSIS_ZOOM)
            before = _fetch_s2_tile(2018, ANALYSIS_ZOOM, tx, ty)
            after = _fetch_s2_tile(2019, ANALYSIS_ZOOM, tx, ty)

            if before is not None and after is not None:
                exg_b = compute_exg(before)
                exg_a = compute_exg(after)

                # Get pixel position within tile
                n = 2 ** ANALYSIS_ZOOM
                x_float = (float(lon) + 180) / 360 * n
                y_float = (1 - math.log(math.tan(math.radians(float(lat))) + 1 / math.cos(math.radians(float(lat)))) / math.pi) / 2 * n
                px = min(int((x_float - int(x_float)) * 256), 255)
                py = min(int((y_float - int(y_float)) * 256), 255)

                change = float(exg_a[py, px] - exg_b[py, px])
                exg_changes.append(round(change, 4))
            else:
                exg_changes.append(0.0)
        except Exception:
            exg_changes.append(0.0)

        if (i + 1) % 500 == 0:
            print(f"  ExG: {i+1}/{len(land_grid)}")

    df = pd.DataFrame({
        "lat": land_grid[:, 0],
        "lon": land_grid[:, 1],
        "tree_cover_2000_pct": [h["tree_cover_2000_pct"] for h in land_hansen],
        "elevation_m": [e["elevation_m"] for e in elev_data],
        "slope_deg": [e["slope_deg"] for e in elev_data],
        "dist_to_road_km": road_dists,
        "dist_to_deforestation_frontier_km": frontier_dists,
        "protected_area": pa_status,
        "population_density_per_km2": pop_dens,
        "annual_loss_rate_pct": annual_rates,
        "exg_change_2018_2019": exg_changes,
        "pre_period_loss": [h["pre_period_loss"] for h in land_hansen],
        "high_risk": [h["high_risk"] for h in land_hansen],
        "loss_year": [h["loss_year_raw"] for h in land_hansen],
    })
    return df


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in REGIONS:
        print(f"\n{'='*60}\nFetching {name}...\n{'='*60}")
        df = fetch_region(name)
        if len(df) == 0:
            print(f"Skipping {name} — no land cells")
            continue
        path = OUTPUT_DIR / f"{name}_grid.csv"
        df.to_csv(path, index=False)
        print(f"\n[{name}] Saved {len(df)} cells -> {path}")
        print(f"  High risk: {df['high_risk'].sum()} ({df['high_risk'].mean()*100:.1f}%)")
        print(f"  Tree cover mean: {df['tree_cover_2000_pct'].mean():.1f}%")


if __name__ == "__main__":
    main()
