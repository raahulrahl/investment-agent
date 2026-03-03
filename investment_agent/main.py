"""Investment Agent - AI investment analysis agent."""

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from textwrap import dedent
from typing import Any

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.openrouter import OpenRouter
from agno.tools.yfinance import YFinanceTools
from agno.os import AgentOS
from bindu.penguin.bindufy import bindufy
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global agent instance
agent: Agent | None = None
_initialized = False
_init_lock = asyncio.Lock()


def load_config() -> dict:
    """Load agent configuration from project root."""
    # Try multiple possible locations for agent_config.json
    possible_paths = [
        Path(__file__).parent.parent / "agent_config.json",  # Project root
        Path(__file__).parent / "agent_config.json",  # Same directory as main.py
        Path.cwd() / "agent_config.json",  # Current working directory
    ]

    for config_path in possible_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except (PermissionError, json.JSONDecodeError) as e:
                print(f"⚠️  Error reading {config_path}: {type(e).__name__}")
                continue
            except Exception as e:
                print(f"⚠️  Unexpected error reading {config_path}: {type(e).__name__}")
                continue

    # If no config found or readable, create a minimal default
    print("⚠️  No agent_config.json found, using default configuration")
    return {
        "name": "investment-agent",
        "description": "AI investment agent for stock analysis and investment research",
        "version": "1.0.0",
        "deployment": {
            "url": "http://127.0.0.1:3773",
            "expose": True,
            "protocol_version": "1.0.0",
            "proxy_urls": ["127.0.0.1"],
            "cors_origins": ["*"],
        },
        "environment_variables": [
            {"key": "OPENAI_API_KEY", "description": "OpenAI API key for LLM calls", "required": False},
            {"key": "OPENROUTER_API_KEY", "description": "OpenRouter API key for LLM calls", "required": False},
        ],
    }


async def initialize_agent() -> None:
    """Initialize the investment agent with proper model and tools."""
    global agent

    # Get API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("MODEL_NAME", "openai/gpt-4o")

    # Model selection logic (supports both OpenAI and OpenRouter)
    if openai_api_key:
        model = OpenAIChat(id="gpt-5.2-2025-12-11", api_key=openai_api_key)
        print("✅ Using OpenAI GPT-5.2-2025-12-11")
    elif openrouter_api_key:
        model = OpenRouter(
            id=model_name,
            api_key=openrouter_api_key,
            cache_response=True,
            supports_native_structured_outputs=True,
        )
        print(f"✅ Using OpenRouter model: {model_name}")
    else:
        # Define error message separately to avoid TRY003
        error_msg = (
            "No API key provided. Set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable.\n"
            "For OpenRouter: https://openrouter.ai/keys\n"
            "For OpenAI: https://platform.openai.com/api-keys"
        )
        raise ValueError(error_msg)

    # Initialize tools
    yfinance_tools = YFinanceTools()

    # Create the investment agent
    agent = Agent(
        name="AI Investment Agent",
        model=model,
        tools=[yfinance_tools],
        description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
        instructions=[
            "Format your response using markdown and use tables to display data where possible.",
            "When comparing stocks, provide detailed analysis including price trends, fundamentals, and analyst recommendations.",
            "Always provide actionable insights for investors.",
        ],
        debug_mode=True,
        markdown=True,
    )
    print("✅ Investment Agent initialized")


async def run_agent(messages: list[dict[str, str]]) -> Any:
    """Run the agent with the given messages."""
    global agent
    if not agent:
        # Define error message separately to avoid TRY003
        error_msg = "Agent not initialized"
        raise RuntimeError(error_msg)

    # Run the agent and get response
    return await agent.arun(messages)


async def handler(messages: list[dict[str, str]]) -> Any:
    """Handle incoming agent messages with lazy initialization."""
    global _initialized

    # Lazy initialization on first call
    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing Investment Agent...")
            await initialize_agent()
            _initialized = True

    # Run the async agent
    result = await run_agent(messages)
    return result


async def cleanup() -> None:
    """Clean up any resources."""
    print("🧹 Cleaning up Investment Agent resources...")


def main():
    """Run the main entry point for the Investment Agent."""
    parser = argparse.ArgumentParser(description="Bindu Investment Agent")
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (env: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "openai/gpt-4o"),
        help="Model ID for OpenRouter (env: MODEL_NAME)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to agent_config.json (optional)",
    )
    parser.add_argument(
        "--use-agentos",
        action="store_true",
        help="Use AgentOS UI instead of Bindu server",
    )
    args = parser.parse_args()

    # Set environment variables if provided via CLI
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    print("🤖 Investment Agent - AI Investment Analysis")
    print("📈 Capabilities: Stock analysis, financial research, investment recommendations")

    # Load configuration
    config = load_config()

    try:
        if args.use_agentos:
            # Use AgentOS UI
            print("🚀 Starting Investment Agent with AgentOS UI...")
            
            # Initialize agent synchronously for AgentOS
            asyncio.run(initialize_agent())
            
            # Create AgentOS instance
            if agent is not None:
                agent_os = AgentOS(agents=[agent])
                app = agent_os.get_app()
                
                # Serve with AgentOS
                agent_os.serve(app="investment_agent:app", reload=True)
            else:
                print("❌ Agent not initialized for AgentOS")
                sys.exit(1)
        else:
            # Use Bindu server
            print("🚀 Starting Bindu Investment Agent server...")
            print(f"🌐 Server will run on: {config.get('deployment', {}).get('url', 'http://127.0.0.1:3773')}")
            bindufy(config, handler)
    except KeyboardInterrupt:
        print("\n🛑 Investment Agent stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup on exit
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()
