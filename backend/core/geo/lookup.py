"""Geocoding and region resolution."""

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable


_geolocator = Nominatim(user_agent="deforestai-prototype", timeout=5)


def geocode(query: str) -> dict | None:
    """Geocode a search query to lat/lon/bounds.

    Returns None if not found, raises descriptive errors for service failures.
    """
    try:
        location = _geolocator.geocode(query, exactly_one=True, viewbox=None)
    except GeocoderTimedOut:
        raise RuntimeError("Geocoding service timed out — try again")
    except (GeocoderServiceError, GeocoderUnavailable) as e:
        raise RuntimeError(f"Geocoding service unavailable: {e}")
    except Exception as e:
        raise RuntimeError(f"Geocoding failed: {e}")
    if not location:
        return None
    return {
        "lat": location.latitude,
        "lon": location.longitude,
        "display_name": location.address,
    }
