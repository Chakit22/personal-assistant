import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, TurnHandlingOptions
from livekit.agents import vad as vad_api
from livekit.plugins import anthropic, openai, silero

REQUIRED_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")

LLM_MODEL = "claude-haiku-4-5"
TTS_SPEED = 1.15
PROFILE_PATH = Path(__file__).parent / "data" / "profile.md"
PROJECTS_PATH = Path(__file__).parent / "data" / "projects.json"
LOG_PATH = Path(__file__).parent / "logs" / "agent.log"
logger = logging.getLogger("personal_assistant.vad")

LIVEKIT_LOG_ALLOWLIST = (
    "received user transcript",
    "using preemptive generation",
    "failed to recognize speech",
    "end_of_turn",
    "turn_completed_cb",
    "llm_ttft",
    "tts_ttfb",
    "e2e",
)

PORTFOLIO_INSTRUCTIONS = (
    "You are Chakit's portfolio voice assistant. Help recruiters and engineers "
    "understand Chakit's projects, technical skills, and work style. Keep spoken "
    "answers short, natural, and easy to follow. Keep responses to 1-3 sentences. "
    "Prefer concrete examples from the profile. When asked about a project, explain "
    "the problem, stack, architecture, technical challenge, and outcome. "
    "Never read raw URLs aloud; refer to links by name, like LinkedIn, GitHub, "
    "resume, portfolio, demo, or source code."
)


def load_profile() -> str:
    if not PROFILE_PATH.exists():
        raise RuntimeError(f"Missing profile file: {PROFILE_PATH}")
    return PROFILE_PATH.read_text(encoding="utf-8")


def load_projects() -> str:
    if not PROJECTS_PATH.exists():
        raise RuntimeError(f"Missing projects file: {PROJECTS_PATH}")
    return PROJECTS_PATH.read_text(encoding="utf-8")


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
        instructions = (
            f"{PORTFOLIO_INSTRUCTIONS}\n\n"
            "Use the following profile as your source of truth:\n\n"
            f"{load_profile()}\n\n"
            "Use the following project data as the source of truth for project walkthroughs:\n\n"
            f"{load_projects()}"
        )
        super().__init__(instructions=instructions)


class LoggedVAD(vad_api.VAD):
    def __init__(self, inner: vad_api.VAD) -> None:
        super().__init__(capabilities=inner.capabilities)
        self.inner = inner
        self._logged_events: set[tuple[str, int]] = set()

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def provider(self) -> str:
        return self.inner.provider

    def stream(self) -> "LoggedVADStream":
        return LoggedVADStream(self, self.inner.stream())


class LoggedVADStream:
    def __init__(self, owner: LoggedVAD, inner: vad_api.VADStream) -> None:
        self.owner = owner
        self.inner = inner

    def push_frame(self, frame) -> None:
        self.inner.push_frame(frame)

    def flush(self) -> None:
        self.inner.flush()

    def end_input(self) -> None:
        self.inner.end_input()

    async def aclose(self) -> None:
        await self.inner.aclose()

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self.inner.__anext__()
        if event.type in (
            vad_api.VADEventType.START_OF_SPEECH,
            vad_api.VADEventType.END_OF_SPEECH,
        ):
            event_key = (event.type.value, event.samples_index)
            if event_key in self.owner._logged_events:
                return event

            self.owner._logged_events.add(event_key)
            if len(self.owner._logged_events) > 100:
                self.owner._logged_events.clear()

            message = (
                "VAD end_of_turn detected"
                if event.type == vad_api.VADEventType.END_OF_SPEECH
                else "VAD start_of_speech detected"
            )
            logger.info(
                "%s | speech=%.3fs silence=%.3fs sample=%s",
                message,
                event.speech_duration,
                event.silence_duration,
                event.samples_index,
            )
        return event


class LiveKitObservationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True

        message = record.getMessage()
        return any(allowed in message for allowed in LIVEKIT_LOG_ALLOWLIST)


def configure_observation_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger = logging.getLogger()
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == LOG_PATH
        for handler in root_logger.handlers
    ):
        root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.DEBUG)

    logging.getLogger("personal_assistant").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.ERROR)

    livekit_logger = logging.getLogger("livekit.agents")
    livekit_logger.setLevel(logging.DEBUG)
    livekit_logger.filters.clear()
    livekit_logger.addFilter(LiveKitObservationFilter())


def build_session() -> AgentSession:
    return AgentSession(
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
        ),
        stt=openai.STT(),
        llm=anthropic.LLM(model=LLM_MODEL),
        tts=openai.TTS(speed=TTS_SPEED),
        vad=LoggedVAD(silero.VAD.load()),
    )


server = AgentServer()


@server.rtc_session(agent_name="personal-assistant")
async def entrypoint(ctx: JobContext) -> None:
    session = build_session()
    await session.start(room=ctx.room, agent=Assistant())


def main() -> None:
    configure_observation_logging()
    agents.cli.run_app(server)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
