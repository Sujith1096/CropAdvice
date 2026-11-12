from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import requests
import time
import logging
import auth, models, schemas, database

router = APIRouter(
    prefix="/api/weather",
    tags=["Weather"]
)

_WEATHER_CACHE_TTL_SECONDS = 900
_weather_cache: dict[tuple[float, float], tuple[float, dict]] = {}


def fetch_weather(lat: float, lon: float):
    # Using Open-Meteo for free weather data with a 15-minute TTL cache
    cache_key = (round(lat, 3), round(lon, 3))
    now = time.monotonic()
    cached = _weather_cache.get(cache_key)
    if cached and (now - cached[0]) < _WEATHER_CACHE_TTL_SECONDS:
        return cached[1]

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=temperature_2m,relativehumidity_2m,precipitation_probability"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        result = {
            "temp": data["current_weather"]["temperature"],
            "humidity": data["hourly"]["relativehumidity_2m"][0],
            "windSpeed": data["current_weather"]["windspeed"],
            "condition": "Clear" if data["current_weather"]["weathercode"] == 0 else "Cloudy/Rain",
        }
        _weather_cache[cache_key] = (now, result)
        return result
    except Exception as e:
        logging.error(f"Failed to fetch weather: {e}")
        return {"temp": 25, "humidity": 60, "windSpeed": 12, "condition": "Unknown"}

@router.get("/")
def get_weather(current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(database.get_db)):
    profile = db.query(models.FarmerProfile).filter(models.FarmerProfile.user_id == current_user.id).first()
    
    # Default to center of India if completely missing
    lat, lon = 20.5937, 78.9629
    
    if profile:
        if profile.latitude and profile.longitude:
            # Exact coordinates provided by profile!
            lat, lon = profile.latitude, profile.longitude
        elif profile.location:
            # Fallback to reverse geocoding if only location string is provided
            try:
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={profile.location}&count=1"
                res = requests.get(geo_url)
                if res.status_code == 200:
                    data = res.json()
                    if "results" in data and len(data["results"]) > 0:
                        lat = data["results"][0]["latitude"]
                        lon = data["results"][0]["longitude"]
            except Exception as e:
                logging.error(f"Geocoding failed: {e}")

    return fetch_weather(lat, lon)

@router.get("/history")
def get_weather_history(current_user: models.User = Depends(auth.get_current_user)):
    return [
        {"date": "2023-01-01", "temperature": 22, "humidity": 55, "description": "Clear"},
        {"date": "2023-01-02", "temperature": 24, "humidity": 60, "description": "Cloudy"},
    ]
