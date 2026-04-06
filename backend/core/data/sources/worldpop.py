"""WorldPop population density data source.

Uses a combination of approaches:
1. WorldPop API for individual point queries (slow but accurate)
2. Cached results for repeated queries
3. Fallback to OSM-based estimation if API is unavailable
"""

import math
import requests
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

WORLDPOP_API = "https://api.worldpop.org/v1/wopr/pointquery"


@lru_cache(maxsize=10000)
def fetch_population_density(lat: float, lon: float, year: int = 2019) -> float:
    """Fetch population density for a point from WorldPop API.

    Returns population count per ~100m grid cell, converted to per km².
    Falls back to 0.0 if API is unavailable.
    """
    # Round to 0.01 for cache efficiency (WorldPop resolution is ~100m)
    lat_r = round(lat, 2)
    lon_r = round(lon, 2)

    try:
        resp = requests.get(
            WORLDPOP_API,
            params={"year": year, "lat": lat_r, "lon": lon_r, "dataset": "wpgppop"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data and data["data"]:
                pop = data["data"].get("pop_value", 0)
                if pop and isinstance(pop, (int, float)):
                    # WorldPop returns people per ~100m cell
                    # Convert to per km²: multiply by 100 (100m cell → km²)
                    return round(float(pop) * 100, 1)
    except Exception as e:
        logger.debug("WorldPop API failed for (%s, %s): %s", lat, lon, e)

    return 0.0


def fetch_population_bulk(lats, lons, fallback_road_dists=None) -> list[float]:
    """Fetch population density for multiple points.

    Uses WorldPop API with fallback to road-distance estimation.
    """
    import numpy as np
    results = []
    api_successes = 0

    for i, (lat, lon) in enumerate(zip(lats, lons)):
        pop = fetch_population_density(float(lat), float(lon))
        if pop > 0:
            results.append(pop)
            api_successes += 1
        elif fallback_road_dists is not None:
            # Fallback: estimate from road distance
            d = fallback_road_dists[i] if i < len(fallback_road_dists) else 999
            results.append(round(max(0, 50 * np.exp(-d / 10)), 1))
        else:
            results.append(0.0)

        if (i + 1) % 100 == 0:
            logger.info("WorldPop: %d/%d (API hits: %d)", i + 1, len(lats), api_successes)

    logger.info("WorldPop complete: %d/%d from API, %d from fallback",
                api_successes, len(lats), len(lats) - api_successes)
    return results
