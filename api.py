"""FastAPI service with live config refresh."""
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import get_agent
from cache import get_cache
from config import get_settings, refresh_settings, get_app_config_loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting weather-agent (model={settings.openai_model})")
    logger.info(f"App Config enabled: {settings.use_app_configuration}")
    logger.info(f"Feature flags: streaming={settings.feature_streaming}, "
                f"response_cache={settings.feature_response_cache}")
    
    get_cache()
    get_agent()
    logger.info("Agent ready ✓")
    yield


app = FastAPI(title="Weather Agent API", version="1.3.0", lifespan=lifespan)
logger = logging.getLogger(__name__)


class WeatherQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    bypass_cache: bool = False


class WeatherResponse(BaseModel):
    answer: str
    latency_ms: int
    cached: bool = False
    model_used: str


@app.get("/")
def root():
    return {"service": "weather-agent", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/ready")
def ready():
    try:
        settings = get_settings()
        return {
            "status": "ready",
            "model": settings.openai_model,
            "cache_backend": settings.cache_backend,
            "app_config": settings.use_app_configuration,
            "features": {
                "response_cache": settings.feature_response_cache,
                "tool_cache": settings.feature_tool_cache,
                "streaming": settings.feature_streaming,
                "strict_mode": settings.feature_strict_mode,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ask", response_model=WeatherResponse)
def ask_weather(payload: WeatherQuery):
    start = time.time()
    settings = get_settings()  # ← refreshes from App Config if needed
    cache = get_cache()
    
    # Feature flag check
    use_response_cache = (
        settings.feature_response_cache and not payload.bypass_cache
    )
    
    cache_key = None
    if use_response_cache:
        from cache import make_cache_key
        normalized = payload.query.strip().lower()
        cache_key = make_cache_key("response", normalized, settings.openai_model)
        cached = cache.get(cache_key)
        if cached:
            return WeatherResponse(
                answer=cached,
                latency_ms=int((time.time() - start) * 1000),
                cached=True,
                model_used=settings.openai_model,
            )
    
    try:
        agent = get_agent()
        result = agent.invoke({"input": payload.query})
        answer = result["output"]
        latency = int((time.time() - start) * 1000)
        
        if cache_key:
            cache.set(cache_key, answer, settings.cache_ttl_agent_response)
        
        return WeatherResponse(
            answer=answer,
            latency_ms=latency,
            cached=False,
            model_used=settings.openai_model,
        )
    except Exception as e:
        logger.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Config management =====

@app.get("/config")
def get_config():
    """Show the current effective config (secrets redacted)."""
    settings = get_settings()
    data = settings.model_dump()
    # Redact secrets
    for key in list(data.keys()):
        if "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
            if data[key]:
                data[key] = f"***{str(data[key])[-4:]}"
    return data


@app.post("/config/refresh")
def trigger_refresh():
    """Force-refresh config from Azure App Configuration."""
    try:
        settings = refresh_settings()
        return {
            "status": "refreshed",
            "model": settings.openai_model,
            "log_level": settings.log_level,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/keys")
def list_config_keys():
    """List all keys loaded from App Configuration (debug)."""
    loader = get_app_config_loader()
    if not loader:
        return {"app_config_enabled": False}
    try:
        return {
            "app_config_enabled": True,
            "endpoint": loader.endpoint,
            "label": loader.label,
            "keys": loader.all_keys(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== Cache management =====

@app.get("/cache/stats")
def cache_stats():
    return get_cache().stats()


@app.post("/cache/clear")
def cache_clear():
    get_cache().clear()
    return {"status": "cleared"}
