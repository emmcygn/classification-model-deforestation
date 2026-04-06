"""NASA FIRMS active fire data source.

Uses VIIRS (Visible Infrared Imaging Radiometer Suite) fire hotspot data
from NASA's Fire Information for Resource Management System.

The 7-day Southeast Asia feed is free and requires no authentication.
For historical data, a FIRMS API key is needed (free registration).
"""

import math
import requests
import pandas as pd
import numpy as np
from io import StringIO
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

FIRMS_7D_URL = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_SouthEast_Asia_7d.csv"

# Philippines bounding box
PH_LAT_MIN, PH_LAT_MAX = 5.0, 20.0
PH_LON_MIN, PH_LON_MAX = 117.0, 127.0


@lru_cache(maxsize=1)
def _fetch_fire_data() -> pd.DataFrame:
    """Fetch recent fire hotspot data for Southeast Asia."""
    try:
        resp = requests.get(FIRMS_7D_URL, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        # Filter to Philippines
        mask = (
            (df["latitude"] >= PH_LAT_MIN) & (df["latitude"] <= PH_LAT_MAX) &
            (df["longitude"] >= PH_LON_MIN) & (df["longitude"] <= PH_LON_MAX)
        )
        ph = df[mask][["latitude", "longitude", "confidence", "frp"]].copy()
        logger.info("FIRMS: %d fire hotspots in Philippines (7-day)", len(ph))
        return ph
    except Exception as e:
        logger.warning("FIRMS data fetch failed: %s", e)
        return pd.DataFrame(columns=["latitude", "longitude", "confidence", "frp"])


def compute_fire_density(
    lats: np.ndarray,
    lons: np.ndarray,
    radius_km: float = 10.0,
) -> np.ndarray:
    """Compute fire hotspot density within radius of each grid cell.

    Returns array of fire counts within radius_km of each cell.
    """
    fires = _fetch_fire_data()
    if len(fires) == 0:
        return np.zeros(len(lats))

    fire_lats = fires["latitude"].values
    fire_lons = fires["longitude"].values

    # Convert to approximate km
    mean_lat = np.mean(lats)
    cos_lat = math.cos(math.radians(mean_lat))

    densities = np.zeros(len(lats))
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        dlat = (fire_lats - lat) * 111.0
        dlon = (fire_lons - lon) * 111.0 * cos_lat
        dists = np.sqrt(dlat**2 + dlon**2)
        densities[i] = (dists < radius_km).sum()

    return densities
