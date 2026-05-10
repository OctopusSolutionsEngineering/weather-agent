"""Weather agent using LangChain + OpenAI."""
import logging
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache as LCInMemoryCache
from langchain_community.cache import RedisCache as LCRedisCache

from tools import WEATHER_TOOLS
from config import get_settings

logger = logging.getLogger(__name__)


def _configure_llm_cache() -> None:
    """Enable LangChain's built-in LLM-level cache."""
    settings = get_settings()
    
    if settings.cache_backend == "redis" and settings.redis_url:
        try:
            import redis
            client = redis.Redis.from_url(settings.redis_url)
            set_llm_cache(LCRedisCache(redis_=client))
            logger.info("LangChain LLM cache → Redis")
            return
        except Exception as e:
            logger.warning(f"Redis LLM cache failed: {e}; using in-memory")
    
    set_llm_cache(LCInMemoryCache())
    logger.info("LangChain LLM cache → in-memory")


SYSTEM_PROMPT = """You are a helpful weather assistant.

You have access to tools that can:
1. Look up coordinates for any city
2. Get the current weather for given coordinates
3. Get a multi-day forecast for given coordinates

When a user asks about weather:
- First, get coordinates using `get_coordinates`
- Then call the appropriate weather tool
- Respond in friendly, natural language
- Include both Celsius and Fahrenheit when reporting temperatures
- Mention key conditions, wind, and any notable weather

Be concise but informative."""


def build_agent() -> AgentExecutor:
    settings = get_settings()
    _configure_llm_cache()
    
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
        cache=True,  # Enable LLM caching
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, WEATHER_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=WEATHER_TOOLS,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )


_agent_executor = None

def get_agent() -> AgentExecutor:
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = build_agent()
    return _agent_executor
