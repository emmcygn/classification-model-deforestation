"""SRTM elevation data source.

Uses the Open-Meteo Elevation API as a fallback since the `srtm` Python package
requires Python 3.12+. The API returns SRTM-derived elevation data.
Results are cached to minimize API calls.
"""

import math
import requests
from functools import lru_cache


@lru_cache(maxsize=10000)
def fetch_elevation(lat: float, lon: float) -> float:
    """Get elevation in meters using Open-Meteo Elevation API."""
    # Round to 3 decimal places for caching (roughly 100m precision)
    lat_r = round(lat, 3)
    lon_r = round(lon, 3)
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_r}&longitude={lon_r}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            elev = data.get("elevation", [0.0])
            if isinstance(elev, list) and len(elev) > 0:
                return float(elev[0])
            return float(elev)
    except Exception:
        pass
    return 0.0


def fetch_elevation_batch(lats: list[float], lons: list[float]) -> list[float]:
    """Fetch elevation for multiple points in one API call (up to 100)."""
    lat_str = ",".join(str(round(la, 3)) for la in lats)
    lon_str = ",".join(str(round(lo, 3)) for lo in lons)
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            elev = data.get("elevation", [])
            if isinstance(elev, list):
                return [float(e) for e in elev]
    except Exception:
        pass
    return [0.0] * len(lats)


def fetch_elevation_and_slope(lat: float, lon: float, delta: float = 0.001) -> dict:
    """Get elevation and approximate slope using a 5-point stencil."""
    lats = [lat, lat + delta, lat - delta, lat, lat]
    lons = [lon, lon, lon, lon + delta, lon - delta]
    elevs = fetch_elevation_batch(lats, lons)

    elev = elevs[0]
    elev_n = elevs[1]
    elev_s = elevs[2]
    elev_e = elevs[3]
    elev_w = elevs[4]

    dy = delta * 111320
    dx = delta * 111320 * math.cos(math.radians(lat))

    dz_dx = (elev_e - elev_w) / (2 * dx) if dx > 0 else 0
    dz_dy = (elev_n - elev_s) / (2 * dy) if dy > 0 else 0
    slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = math.degrees(slope_rad)

    return {
        "elevation_m": round(elev, 1),
        "slope_deg": round(slope_deg, 2),
    }
