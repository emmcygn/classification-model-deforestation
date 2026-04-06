"""Live data pipeline — stub for on-demand fetching."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from core.data.fetch_philippines import fetch_region, REGIONS, PROVENANCE


async def fetch_region_live(lat_min, lat_max, lon_min, lon_max, step=0.02):
    temp = "_live"
    REGIONS[temp] = {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max, "step": step}
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(ThreadPoolExecutor(1), fetch_region, temp)
    finally:
        REGIONS.pop(temp, None)


def get_provenance():
    return PROVENANCE
