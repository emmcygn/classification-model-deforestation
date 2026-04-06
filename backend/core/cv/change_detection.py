"""Satellite image change detection using RGB vegetation analysis.

Computes Excess Green Index (ExG) from Sentinel-2 RGB tiles to detect
vegetation change between two years. ExG = 2*G - R - B, normalized.

This is a real computer vision pipeline operating on satellite imagery:
1. Download tiles for a bounding box at two time points
2. Compute per-pixel vegetation index
3. Difference the indices to detect loss/gain
4. Classify changes above threshold as deforestation events
"""

import math
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Sentinel-2 cloudless yearly composites (EOX, free, no auth)
S2_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-{year}_3857/default/g/{z}/{y}/{x}.jpg"
TILE_SIZE = 256
ANALYSIS_ZOOM = 12  # ~38m per pixel at equator


def _tile_coords(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lon to tile x/y at given zoom."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    return x, y


def _tiles_for_bounds(lat_min: float, lat_max: float, lon_min: float, lon_max: float, zoom: int) -> list[tuple[int, int]]:
    """Get all tile coordinates covering a bounding box."""
    x_min, y_max = _tile_coords(lat_min, lon_min, zoom)  # Note: y is inverted
    x_max, y_min = _tile_coords(lat_max, lon_max, zoom)
    tiles = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles.append((x, y))
    return tiles


@lru_cache(maxsize=256)
def _fetch_s2_tile(year: int, z: int, x: int, y: int) -> np.ndarray | None:
    """Fetch a Sentinel-2 tile and return as numpy RGB array."""
    url = S2_URL.format(year=year, z=z, x=x, y=y)
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return np.array(img, dtype=np.float32)
    except Exception as e:
        logger.warning("Failed to fetch S2 tile z=%d x=%d y=%d year=%d: %s", z, x, y, year, e)
        return None


def compute_exg(rgb: np.ndarray) -> np.ndarray:
    """Compute Excess Green Index from RGB image.

    ExG = 2*g - r - b where r,g,b are normalized channel values.
    Range: [-1, 1], higher = more vegetation.
    """
    total = rgb.sum(axis=2, keepdims=True).clip(1)  # Avoid division by zero
    normalized = rgb / total
    r, g, b = normalized[:, :, 0], normalized[:, :, 1], normalized[:, :, 2]
    return 2 * g - r - b


def detect_change(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    year_before: int = 2018, year_after: int = 2023,
    loss_threshold: float = -0.08,
    gain_threshold: float = 0.08,
) -> dict:
    """Detect vegetation change between two years for a bounding box.

    Returns:
        dict with change statistics, per-tile results, and classification counts.
    """
    zoom = ANALYSIS_ZOOM
    tiles = _tiles_for_bounds(lat_min, lat_max, lon_min, lon_max, zoom)

    if len(tiles) > 100:
        # Limit to avoid excessive API calls
        logger.warning("Too many tiles (%d), limiting to 100", len(tiles))
        tiles = tiles[:100]

    total_pixels = 0
    loss_pixels = 0
    gain_pixels = 0
    no_change_pixels = 0
    tile_results = []

    for tx, ty in tiles:
        before = _fetch_s2_tile(year_before, zoom, tx, ty)
        after = _fetch_s2_tile(year_after, zoom, tx, ty)

        if before is None or after is None:
            continue

        exg_before = compute_exg(before)
        exg_after = compute_exg(after)
        diff = exg_after - exg_before

        n_pixels = diff.size
        n_loss = int((diff < loss_threshold).sum())
        n_gain = int((diff > gain_threshold).sum())
        n_stable = n_pixels - n_loss - n_gain

        total_pixels += n_pixels
        loss_pixels += n_loss
        gain_pixels += n_gain
        no_change_pixels += n_stable

        # Compute tile center for spatial reference
        n = 2 ** zoom
        lon_center = (tx + 0.5) / n * 360 - 180
        lat_center = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 0.5) / n))))

        tile_results.append({
            "tile_x": tx,
            "tile_y": ty,
            "lat": round(lat_center, 4),
            "lon": round(lon_center, 4),
            "total_pixels": n_pixels,
            "loss_pixels": n_loss,
            "gain_pixels": n_gain,
            "loss_pct": round(n_loss / n_pixels * 100, 1),
            "gain_pct": round(n_gain / n_pixels * 100, 1),
            "mean_exg_before": round(float(exg_before.mean()), 4),
            "mean_exg_after": round(float(exg_after.mean()), 4),
            "mean_change": round(float(diff.mean()), 4),
        })

    # Sort tiles by loss severity
    tile_results.sort(key=lambda t: -t["loss_pct"])

    result = {
        "year_before": year_before,
        "year_after": year_after,
        "bounds": {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max},
        "zoom": zoom,
        "tiles_analyzed": len(tile_results),
        "total_pixels": total_pixels,
        "loss_pixels": loss_pixels,
        "gain_pixels": gain_pixels,
        "no_change_pixels": no_change_pixels,
        "loss_pct": round(loss_pixels / max(total_pixels, 1) * 100, 2),
        "gain_pct": round(gain_pixels / max(total_pixels, 1) * 100, 2),
        "estimated_loss_hectares": round(loss_pixels * 0.0038 * 0.0038 * 100, 1),
        "hotspot_tiles": tile_results[:10],  # Top 10 tiles by loss
        "method": "Excess Green Index (ExG) differencing on Sentinel-2 RGB composites. Area estimate is approximate — assumes ~38m/pixel at z12 equator; actual ground size varies with latitude. Based on visualization tiles, not analysis-ready bands.",
    }

    return result
