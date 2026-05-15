import os
import sys

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext
from livekit.plugins import anthropic, openai, silero

REQUIRED_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")

LLM_MODEL = "claude-haiku-4-5"

# Placeholder system prompt — the real persona lives in a separate Linear issue.
PLACEHOLDER_INSTRUCTIONS = (
    "You are a helpful voice assistant. Keep responses short, natural, and "
    "free of formatting that doesn't read aloud well."
)


def _require_env(*keys: str) -> None:
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Set them in your .env (see .env.example)."
        )


load_dotenv()
_require_env(*REQUIRED_ENV_VARS)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=PLACEHOLDER_INSTRUCTIONS)


def build_session() -> AgentSession:
    return AgentSession(
        stt=openai.STT(),
        llm=anthropic.LLM(model=LLM_MODEL),
        tts=openai.TTS(),
        vad=silero.VAD.load(),
    )


server = AgentServer()


@server.rtc_session(agent_name="personal-assistant")
async def entrypoint(ctx: JobContext) -> None:
    session = build_session()
    await session.start(room=ctx.room, agent=Assistant())


def main() -> None:
    agents.cli.run_app(server)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
