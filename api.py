"""FastAPI service wrapping the weather agent."""
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import get_agent
from cache import get_cache, make_cache_key
from config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting weather-agent (model={settings.openai_model})")
    logger.info(f"Cache backend: {settings.cache_backend}")
    
    get_cache()  # init cache
    get_agent()  # warm up agent
    logger.info("Agent ready ✓")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Weather Agent API", version="1.2.0", lifespan=lifespan)
logger = logging.getLogger(__name__)


class WeatherQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    bypass_cache: bool = Field(default=False, description="Skip response cache")


class WeatherResponse(BaseModel):
    answer: str
    latency_ms: int
    cached: bool = False


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
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ask", response_model=WeatherResponse)
def ask_weather(payload: WeatherQuery):
    start = time.time()
    settings = get_settings()
    cache = get_cache()
    
    # ===== Layer 1: Response cache =====
    response_cache_key = None
    if settings.enable_response_cache and not payload.bypass_cache:
        # Normalize for better hit rate (lowercase + strip)
        normalized = payload.query.strip().lower()
        response_cache_key = make_cache_key("response", normalized)
        cached_response = cache.get(response_cache_key)
        if cached_response:
            latency = int((time.time() - start) * 1000)
            logger.info(f"🎯 Response cache HIT | {latency}ms")
            return WeatherResponse(
                answer=cached_response,
                latency_ms=latency,
                cached=True,
            )
    
    # ===== Invoke agent (tool + LLM caches still apply) =====
    try:
        agent = get_agent()
        result = agent.invoke({"input": payload.query})
        answer = result["output"]
        latency = int((time.time() - start) * 1000)
        
        # Populate response cache
        if response_cache_key:
            cache.set(response_cache_key, answer, settings.cache_ttl_agent_response)
        
        logger.info(f"💨 Agent invoked | {latency}ms")
        return WeatherResponse(answer=answer, latency_ms=latency, cached=False)
    
    except Exception as e:
        logger.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Cache management endpoints =====

@app.get("/cache/stats")
def cache_stats():
    """Get cache statistics."""
    return get_cache().stats()


@app.post("/cache/clear")
def cache_clear():
    """Clear the entire cache (admin use)."""
    get_cache().clear()
    return {"status": "cleared"}
