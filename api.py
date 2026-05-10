"""FastAPI service wrapping the weather agent."""
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import get_agent
from config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config and warm up the agent at startup."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting weather-agent (model={settings.openai_model})")
    
    # Pre-build agent to fail fast if config is bad
    get_agent()
    logger.info("Agent ready ✓")
    
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Weather Agent API",
    version="1.1.0",
    description="An AI agent that answers weather questions.",
    lifespan=lifespan,
)
logger = logging.getLogger(__name__)


class WeatherQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


class WeatherResponse(BaseModel):
    answer: str
    latency_ms: int


@app.get("/")
def root():
    return {"service": "weather-agent", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/ready")
def ready():
    """Readiness check — verifies config is loaded."""
    try:
        settings = get_settings()
        return {
            "status": "ready",
            "model": settings.openai_model,
            "key_vault_enabled": settings.use_key_vault,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {e}")


@app.post("/ask", response_model=WeatherResponse)
def ask_weather(payload: WeatherQuery):
    start = time.time()
    try:
        agent = get_agent()
        result = agent.invoke({"input": payload.query})
        latency_ms = int((time.time() - start) * 1000)
        logger.info(f"Query: {payload.query[:50]}... | Latency: {latency_ms}ms")
        return WeatherResponse(answer=result["output"], latency_ms=latency_ms)
    except Exception as e:
        logger.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail=str(e))