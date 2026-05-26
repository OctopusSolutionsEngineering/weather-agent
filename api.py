"""FastAPI service with live config refresh."""
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from config import get_settings, verify_azure_auth, get_auth_report
from cache import get_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ===== Phase 1: Logging =====
    logging.basicConfig(
        level="INFO",  # Will be overridden once config is loaded
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("🌤️  Weather Agent starting up")
    logger.info("=" * 60)
    
    # ===== Phase 2: Verify Azure auth (fail fast) =====
    try:
        report = verify_azure_auth(strict=True)
        if not report.overall_success:
            # This branch only hits when strict=False; with strict=True
            # an exception will have been raised.
            logger.error("Azure auth verification FAILED")
            sys.exit(1)
    except RuntimeError as e:
        logger.error(f"❌ Startup failed: {e}")
        # Exit with non-zero so K8s restarts the pod (likely with backoff)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"❌ Unexpected startup error: {e}")
        sys.exit(1)
    
    # ===== Phase 3: Load config =====
    try:
        settings = get_settings()
        logging.getLogger().setLevel(settings.log_level)
        logger.info(f"Config loaded (model={settings.openai_model})")
    except Exception as e:
        logger.exception(f"❌ Config load failed: {e}")
        sys.exit(1)
    
    # ===== Phase 4: Warm up dependencies =====
    try:
        get_cache()
        from agent import get_agent
        get_agent()
        logger.info("✅ Agent ready — accepting traffic")
    except Exception as e:
        logger.exception(f"❌ Agent warm-up failed: {e}")
        sys.exit(1)
    
    yield  # ← App is running
    
    logger.info("👋 Shutting down")


app = FastAPI(title="Weather Agent API", version="1.4.0", lifespan=lifespan)
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
    """Readiness probe — pod can serve traffic.
    
    Returns 503 if Azure auth checks haven't passed.
    """
    report = get_auth_report()
    if report is None or not report.overall_success:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": "Azure authentication not verified",
                "failed_checks": [
                    {"name": c.name, "error": c.error}
                    for c in (report.failed_checks if report else [])
                ],
            },
        )
    
    settings = get_settings()
    return {
        "status": "ready",
        "model": settings.openai_model,
        "azure": {
            "app_config": settings.use_app_configuration,
            "key_vault": settings.use_key_vault,
            "identity": report.identity_info,
        },
    }

@app.get("/auth/status")
def auth_status():
    """Show the Azure authentication verification report."""
    report = get_auth_report()
    if report is None:
        return {"status": "not_yet_verified"}
    return {
        "overall_success": report.overall_success,
        "identity": report.identity_info,
        "checks": [
            {
                "name": c.name,
                "success": c.success,
                "duration_ms": c.duration_ms,
                "detail": c.detail,
                "error": c.error,
            }
            for c in report.checks
        ],
    }

@app.post("/auth/verify")
def reverify_auth():
    """Manually re-run the auth verification (useful after credential rotation)."""
    try:
        report = verify_azure_auth(strict=False)
        return {
            "overall_success": report.overall_success,
            "checks_passed": len([c for c in report.checks if c.success]),
            "checks_total": len(report.checks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
