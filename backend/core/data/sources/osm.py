"""OpenStreetMap data source via Overpass API."""

import math
import requests
from shapely.geometry import Point, Polygon

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _overpass_query(query: str) -> dict:
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def fetch_roads_for_region(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> list[dict]:
    """Fetch road nodes in a bounding box."""
    query = f"""
    [out:json][timeout:120];
    way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|track)$"]
      ({lat_min},{lon_min},{lat_max},{lon_max});
    (._;>;);
    out body;
    """
    data = _overpass_query(query)
    nodes = []
    for el in data.get("elements", []):
        if el["type"] == "node" and "lat" in el and "lon" in el:
            nodes.append({"lat": el["lat"], "lon": el["lon"]})
    return nodes


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def compute_distance_to_nearest_road(lat, lon, road_nodes):
    if not road_nodes:
        return 999.0
    min_dist = float("inf")
    for node in road_nodes:
        d = _haversine(lat, lon, node["lat"], node["lon"])
        if d < min_dist:
            min_dist = d
    return round(min_dist, 2)


def fetch_protected_areas_for_region(lat_min, lat_max, lon_min, lon_max):
    query = f"""
    [out:json][timeout:120];
    (
      way["boundary"="protected_area"]({lat_min},{lon_min},{lat_max},{lon_max});
      relation["boundary"="protected_area"]({lat_min},{lon_min},{lat_max},{lon_max});
      way["leisure"="nature_reserve"]({lat_min},{lon_min},{lat_max},{lon_max});
      relation["leisure"="nature_reserve"]({lat_min},{lon_min},{lat_max},{lon_max});
    );
    (._;>;);
    out body;
    """
    data = _overpass_query(query)
    nodes = {}
    for el in data.get("elements", []):
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    polygons = []
    for el in data.get("elements", []):
        if el["type"] == "way" and "nodes" in el:
            coords = [nodes[nid] for nid in el["nodes"] if nid in nodes]
            if len(coords) >= 4:
                try:
                    polygons.append(Polygon(coords))
                except Exception:
                    pass
    return polygons


def is_protected(lat, lon, protected_areas):
    point = Point(lon, lat)
    for poly in protected_areas:
        try:
            if poly.contains(point):
                return 1
        except Exception:
            pass
    return 0
