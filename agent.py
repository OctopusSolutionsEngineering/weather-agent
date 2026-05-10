"""Weather agent using LangChain + OpenAI."""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from tools import WEATHER_TOOLS
from config import get_settings

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
    """Construct and return the weather agent executor."""
    settings = get_settings()
    
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
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