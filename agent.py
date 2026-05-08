"""Weather agent using LangChain + OpenAI."""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from tools import WEATHER_TOOLS

load_dotenv()

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
  (F = C * 9/5 + 32)
- Mention key conditions, wind, and any notable weather

If a city can't be found, ask the user to clarify.
Be concise but informative."""


def build_agent() -> AgentExecutor:
    """Construct and return the weather agent executor."""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
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


# Singleton instance
_agent_executor = None

def get_agent() -> AgentExecutor:
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = build_agent()
    return _agent_executor


if __name__ == "__main__":
    # Quick CLI test
    agent = get_agent()
    while True:
        query = input("\n🌤️  Ask about the weather (or 'quit'): ")
        if query.lower() in {"quit", "exit"}:
            break
        result = agent.invoke({"input": query})
        print(f"\n💬 {result['output']}")
