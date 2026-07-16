"""
Headless simulation driver — no LiveKit server or room required.

Uses AgentSession.run(user_input=...) to drive scripted conversation turns and
produce a verifiable transcript. Each turn is logged to stdout and to
transcripts/sample_run.log.

Run:
    python run_simulation.py
"""
import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv
from livekit.agents import AgentSession, RunResult
from livekit.agents.llm import ChatMessage, FunctionCall, FunctionCallOutput
from livekit.plugins import google

from agent import AirlineAgent
from stubs import SttStub, TtsStub

load_dotenv()

LOG_FILE = os.path.join(os.path.dirname(__file__), "transcripts", "sample_run.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("simulation")


def _log_run_events(result: RunResult, *, label: str) -> None:
    """Print the structured events from a completed run turn."""
    logger.info("--- %s events ---", label)
    for event in result.events:
        item = event.item
        if isinstance(item, ChatMessage):
            role = getattr(item, "role", "?")
            content = item.text_content if hasattr(item, "text_content") else str(item)
            logger.info("[%s] %s", role.upper(), content)
        elif isinstance(item, FunctionCall):
            args = item.arguments
            if isinstance(args, str):
                args = json.loads(args) if args else {}
            logger.info(
                "[LLM->TOOL] %s(%s)",
                item.name,
                ", ".join(f"{k}={v!r}" for k, v in args.items()),
            )
        elif isinstance(item, FunctionCallOutput):
            logger.info("[TOOL->LLM] %s", item.output)
        else:
            logger.info("[EVENT] %s: %s", type(item).__name__, item)


async def main() -> None:
    logger.info("=== Simulation start ===")

    # gemini-3.1-flash-lite has a separate quota from gemini-2.0-flash (30 RPM free tier)
    session = AgentSession(
        stt=SttStub(),
        llm=google.LLM(model="gemini-3.1-flash-lite"),
        tts=TtsStub(),
    )

    _TURN_DELAY = 5  # seconds between turns — spreads token usage across the quota window

    async with session:
        await session.start(agent=AirlineAgent())

        # Turn 1: greeting — triggers on_enter
        logger.info("--- Turn 1: greeting ---")
        result = await session.run(user_input="Hello")
        _log_run_events(result, label="Turn 1")

        await asyncio.sleep(_TURN_DELAY)

        # Turn 2: flight status — triggers get_flight_status tool
        logger.info("--- Turn 2: flight status query (tool call expected) ---")
        result = await session.run(user_input="What is the status of flight BA123?")
        _log_run_events(result, label="Turn 2")

        await asyncio.sleep(_TURN_DELAY)

        # Turn 3: unknown flight — exercises the not-found branch
        logger.info("--- Turn 3: unknown flight ---")
        result = await session.run(user_input="Can you check flight ZZ999?")
        _log_run_events(result, label="Turn 3")

    logger.info("=== Simulation end ===")
    logger.info("Full transcript written to: %s", LOG_FILE)


if __name__ == "__main__":
    asyncio.run(main())
