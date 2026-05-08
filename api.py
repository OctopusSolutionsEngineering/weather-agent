"""FastAPI service wrapping the weather agent."""
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import get_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Weather Agent API",
    version="1.0.0",
    description="An AI agent that answers weather questions.",
)


class WeatherQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500,
                       examples=["What's the weather in Paris?"])


class WeatherResponse(BaseModel):
    answer: str
    latency_ms: int


@app.get("/")
def root():
    return {"service": "weather-agent", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask", response_model=WeatherResponse)
def ask_weather(payload: WeatherQuery):
    """Ask the weather agent a question."""
    start = time.time()
    try:
        agent = get_agent()
        result = agent.invoke({"input": payload.query})
        latency_ms = int((time.time() - start) * 1000)
        
        logger.info(f"Query: {payload.query[:50]}... | Latency: {latency_ms}ms")
        
        return WeatherResponse(
            answer=result["output"],
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
