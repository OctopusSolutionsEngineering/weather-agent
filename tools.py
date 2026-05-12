"""Custom tools the agent can call, with caching."""
import logging
import requests
from langchain_core.tools import tool

from cache import get_cache, make_cache_key
from config import get_settings

logger = logging.getLogger(__name__)


def _cached_tool_call(namespace: str, ttl: int, fn, *args, **kwargs):
    """Wrap a function call with cache lookup."""
    settings = get_settings()
    if not settings.feature_tool_cache:
        return fn(*args, **kwargs)
    
    cache = get_cache()
    key = make_cache_key(namespace, *args, **kwargs)
    
    cached = cache.get(key)
    if cached is not None:
        logger.info(f"🎯 Cache HIT: {namespace}")
        return cached
    
    logger.info(f"💨 Cache MISS: {namespace}")
    result = fn(*args, **kwargs)
    
    # Only cache successful results
    if isinstance(result, dict) and "error" not in result:
        cache.set(key, result, ttl)
    
    return result


# ===== Inner (uncached) implementations =====

def _fetch_coordinates(city: str) -> dict:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data.get("results"):
            return {"error": f"City '{city}' not found"}
        result = data["results"][0]
        return {
            "city": result["name"],
            "country": result.get("country", "Unknown"),
            "latitude": result["latitude"],
            "longitude": result["longitude"],
            "timezone": result.get("timezone", "UTC"),
        }
    except Exception as e:
        return {"error": f"Failed to get coordinates: {str(e)}"}


def _fetch_current_weather(latitude: float, longitude: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                   "is_day,precipitation,weather_code,wind_speed_10m",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data["current"]
        weather_descriptions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail",
        }
        return {
            "temperature_celsius": current["temperature_2m"],
            "feels_like_celsius": current["apparent_temperature"],
            "humidity_percent": current["relative_humidity_2m"],
            "precipitation_mm": current["precipitation"],
            "wind_speed_kmh": current["wind_speed_10m"],
            "is_day": bool(current["is_day"]),
            "conditions": weather_descriptions.get(current["weather_code"], "Unknown"),
        }
    except Exception as e:
        return {"error": f"Failed to get weather: {str(e)}"}


def _fetch_forecast(latitude: float, longitude: float, days: int = 3) -> dict:
    days = max(1, min(days, 7))
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
        "forecast_days": days,
        "temperature_unit": "celsius",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        daily = data["daily"]
        forecast = []
        for i, date in enumerate(daily["time"]):
            forecast.append({
                "date": date,
                "max_temp_c": daily["temperature_2m_max"][i],
                "min_temp_c": daily["temperature_2m_min"][i],
                "precipitation_mm": daily["precipitation_sum"][i],
                "weather_code": daily["weather_code"][i],
            })
        return {"forecast": forecast}
    except Exception as e:
        return {"error": f"Failed to get forecast: {str(e)}"}


# ===== Cached tool wrappers (exposed to the agent) =====

@tool
def get_coordinates(city: str) -> dict:
    """Get latitude and longitude for a given city name.
    
    Args:
        city: The name of the city (e.g., 'London', 'Tokyo').
    """
    ttl = get_settings().cache_ttl_geocoding
    # Normalize city name for better cache hits
    normalized = city.strip().lower()
    return _cached_tool_call("geocoding", ttl, _fetch_coordinates, normalized)


@tool
def get_current_weather(latitude: float, longitude: float) -> dict:
    """Get current weather conditions for given coordinates.
    
    Args:
        latitude: The latitude of the location.
        longitude: The longitude of the location.
    """
    ttl = get_settings().cache_ttl_current_weather
    # Round coords to ~1km precision for cache reuse across similar lookups
    lat = round(latitude, 2)
    lon = round(longitude, 2)
    return _cached_tool_call("weather", ttl, _fetch_current_weather, lat, lon)


@tool
def get_forecast(latitude: float, longitude: float, days: int = 3) -> dict:
    """Get a multi-day weather forecast.
    
    Args:
        latitude: The latitude of the location.
        longitude: The longitude of the location.
        days: Number of forecast days (1-7). Default is 3.
    """
    ttl = get_settings().cache_ttl_forecast
    lat = round(latitude, 2)
    lon = round(longitude, 2)
    return _cached_tool_call("forecast", ttl, _fetch_forecast, lat, lon, days)


WEATHER_TOOLS = [get_coordinates, get_current_weather, get_forecast]
