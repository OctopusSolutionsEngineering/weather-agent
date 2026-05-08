# Weather Agent 🌤️

An AI agent that answers weather questions using OpenAI + Open-Meteo.

## Quick Start

### Local

```bash
cp .env.example .env  # add your OPENAI_API_KEY
pip install -r requirements.txt
uvicorn api:app --reload
```

### Docker

```bash
docker build -t weather-agent .
docker run -p 8000:8000 --env-file .env weather-agent
```

## Usage

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Tokyo right now?"}'
```

Visit http://localhost:8000/docs for interactive Swagger UI.
