import os
import sys
import logging
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    function_tool,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents import vad as vad_api
from livekit.plugins import anthropic, openai, silero

REQUIRED_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")

LLM_MODEL = "claude-haiku-4-5"
TTS_SPEED = 1.4
PROFILE_PATH = Path(__file__).parent / "data" / "profile.md"
PROJECTS_PATH = Path(__file__).parent / "data" / "projects.json"
LEADS_PATH = Path(__file__).parent / "data" / "leads.jsonl"
CONVERSATION_SUMMARIES_PATH = Path(__file__).parent / "data" / "conversation_summaries.jsonl"
TRANSCRIPTS_PATH = Path(__file__).parent / "transcripts"
LOG_PATH = Path(__file__).parent / "logs" / "agent.log"
logger = logging.getLogger("personal_assistant.vad")
tool_logger = logging.getLogger("personal_assistant.tools")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_PATTERN = re.compile(r"^\+?[0-9][0-9\s().-]{6,}$")

LIVEKIT_LOG_ALLOWLIST = (
    "received user transcript",
    "using preemptive generation",
    "failed to recognize speech",
    "end_of_turn",
    "turn_completed_cb",
    "llm_ttft",
    "tts_ttfb",
    "e2e",
    "end_call",
)

PORTFOLIO_INSTRUCTIONS = (
    "<role>"
    "You are Chakit's Personal Assistant. Help recruiters and engineers "
    "understand Chakit's projects, technical skills, and work style. Keep spoken "
    "answers short, natural, and easy to follow. Keep responses to 1-3 sentences. "
    "</role>"
    "<project_answers>"
    "Prefer concrete examples from the profile. When asked about any project, "
    "answer in exactly three short parts: one sentence giving an overview, one "
    "sentence listing the main features, and the exact final sentence: Would "
    "you like to know more about it? Do not explain the problem, stack, "
    "architecture, technical challenge, or outcome unless the user specifically "
    "asks for those details. "
    "Never read raw URLs aloud; refer to links by name, like LinkedIn, GitHub, "
    "resume, portfolio, demo, or source code. "
    "</project_answers>"
    "<early_contact_capture>"
    "At the start of the conversation, "
    "after greeting the visitor, softly ask: Before we go further, may I get "
    "your name and an email or phone number in case Chakit wants to follow up? "
    "If you would rather not share it, that is totally fine. If the visitor "
    "shares their name and contact, confirm the contact if needed, then call "
    "capture_lead with unknown fields set to Not provided and message set to "
    "Initial visitor contact. If the visitor declines, continue the conversation "
    "normally and do not push. "
    "</early_contact_capture>"
    "<opportunity_capture>"
    "If the visitor later mentions hiring, recruiting, "
    "collaboration, project work, an opportunity, or anything Chakit may "
    "reasonably want to follow up on, collect lead details one question at a "
    "time in this order: name, contact, company, role or hiring context, and "
    "message. Contact is required; say you need at least an email or another "
    "contact method so Chakit can reach them. Only confirm email addresses and "
    "phone numbers; do not confirm name, company, role, or message unless they "
    "sound unclear. "
    "</opportunity_capture>"
    "<contact_confirmation>"
    "When confirming an email or phone number, spell it character by character: "
    "say at for @, dot for periods, and read phone digits one by one. If an "
    "email or phone number sounds unclear, ask the visitor to repeat or spell it "
    "out. "
    "</contact_confirmation>"
    "<tool_call_phrasing>"
    "Before calling capture_lead, explicitly say: Let me save those details. "
    "One moment please. Before calling any other tool that performs an external "
    "action, use the same pattern: Let me <action>. One moment please. "
    "</tool_call_phrasing>"
    "<capture_lead_policy>"
    "Call capture_lead whenever there is useful contact or follow-up context "
    "for Chakit to review. It is okay to call capture_lead more than once if "
    "the first call saved basic contact details and a later part of the "
    "conversation adds hiring, collaboration, project, or opportunity context. "
    "</capture_lead_policy>"
    "<end_call_policy>"
    "Call end_call when any closure condition is met. Closing language used: "
    "if your response includes a phrase like Thanks for reaching out, Chakit "
    "will follow up, Hope that helps, Have a great day, or Take care, call "
    "end_call instead of continuing to listen. Examples: Chakit will follow up "
    "with you soon; Thanks for reaching out; Hope that helps. No pending "
    "question: if you do not need any more information from the user, call "
    "end_call after the final response. Examples: all required details are "
    "saved; the requested project explanation is complete; the user asked for "
    "one specific answer and you gave it. User gave final intent: if the user "
    "says that's all, thank you, bye, no more questions, sure, or okay thanks, "
    "call end_call. Examples: No, that's all; Thank you, bye; Okay thanks. "
    "</end_call_policy>"
)


def load_profile() -> str:
    if not PROFILE_PATH.exists():
        raise RuntimeError(f"Missing profile file: {PROFILE_PATH}")
    return PROFILE_PATH.read_text(encoding="utf-8")


def load_projects() -> str:
    if not PROJECTS_PATH.exists():
        raise RuntimeError(f"Missing projects file: {PROJECTS_PATH}")
    return PROJECTS_PATH.read_text(encoding="utf-8")


def is_valid_contact(contact: str) -> bool:
    normalized_contact = contact.strip()
    if not normalized_contact:
        return False
    if "@" in normalized_contact:
        return bool(EMAIL_PATTERN.fullmatch(normalized_contact))
    if PHONE_PATTERN.match(normalized_contact):
        return True
    return any(
        marker in normalized_contact.lower()
        for marker in ("linkedin", "linked in", "http://", "https://")
    )


def format_transcript_text(value: object) -> str:
    return str(value).replace("\n", " ").strip()


class TranscriptRecorder:
    def __init__(self) -> None:
        self.started_at = datetime.now(timezone.utc)
        self.items: list[str] = []
        self.turns: list[tuple[str, str]] = []
        self.transcript_saved = False
        self.summary_saved = False

    def record(self, event: object) -> None:
        item = getattr(event, "item", None)
        role = getattr(item, "role", None)
        text_content = getattr(item, "text_content", None)
        if not role or not text_content:
            return

        formatted_text = format_transcript_text(text_content)
        timestamp = datetime.fromtimestamp(
            getattr(item, "created_at", time.time()), tz=timezone.utc
        ).isoformat()
        self.items.append(f"[{timestamp}] {role}: {formatted_text}")
        self.turns.append((role, formatted_text))

    def save_transcript(self) -> None:
        if self.transcript_saved or not self.items:
            return
        self.transcript_saved = True

        TRANSCRIPTS_PATH.mkdir(parents=True, exist_ok=True)
        filename = self.started_at.strftime("conversation_%Y%m%d_%H%M%S.md")
        transcript_path = TRANSCRIPTS_PATH / filename
        transcript_path.write_text("\n".join(self.items) + "\n", encoding="utf-8")

    async def save_summary(self) -> None:
        if self.summary_saved or not self.turns:
            return
        self.summary_saved = True

        transcript = "\n".join(f"{role}: {text}" for role, text in self.turns)
        try:
            summary = await summarize_conversation(transcript)
        except Exception:
            tool_logger.exception("failed to generate conversation summary")
            summary = "Summary generation failed. See transcript for conversation details."
        CONVERSATION_SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
        }
        with CONVERSATION_SUMMARIES_PATH.open("a", encoding="utf-8") as summaries_file:
            summaries_file.write(json.dumps(row) + "\n")


async def summarize_conversation(transcript: str) -> str:
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = await client.messages.create(
        model=LLM_MODEL,
        max_tokens=120,
        temperature=0,
        system=(
            "Summarize this voice assistant conversation for Chakit in 1-3 "
            "sentences. Include visitor identity, contact details, hiring or "
            "project context, and follow-up needs if present. Do not include a "
            "transcript or bullet points."
        ),
        messages=[
            {
                "role": "user",
                "content": transcript[-12000:],
            }
        ],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()


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


async def log_end_call_tool_called(_: object) -> None:
    tool_logger.info("tool_call end_call called")


async def log_end_call_tool_completed(event: object) -> None:
    output = getattr(event, "output", None)
    tool_logger.info("tool_call end_call completed | output=%s", output)


class Assistant(Agent):
    def __init__(self) -> None:
        instructions = (
            f"{PORTFOLIO_INSTRUCTIONS}\n\n"
            "Use the following profile as your source of truth:\n\n"
            f"{load_profile()}\n\n"
            "Use the following project data as the source of truth for project walkthroughs:\n\n"
            f"{load_projects()}"
        )
        super().__init__(
            instructions=instructions,
            tools=[
                EndCallTool(
                    extra_description=(
                        "For this portfolio assistant, also call when the visitor says "
                        "they are finished, says thanks after their details are saved, "
                        "or says there is nothing else they need. Also call when your "
                        "own next response would be a natural final wrap-up, closing "
                        "line, or goodbye, even if the user has not explicitly said bye."
                    ),
                    end_instructions="Give a short, warm goodbye in one sentence.",
                    on_tool_called=log_end_call_tool_called,
                    on_tool_completed=log_end_call_tool_completed,
                )
            ],
        )

    @function_tool
    async def capture_lead(
        self,
        name: str,
        contact: str,
        company: str,
        role: str,
        message: str,
    ) -> str:
        """Save a visitor's follow-up details after collecting them step by step."""
        if not contact.strip():
            return "A contact method is required before saving the lead."
        if not is_valid_contact(contact):
            return (
                "The contact method looks unclear or invalid. Ask the visitor "
                "to repeat or spell it out, then confirm it before saving."
            )

        LEADS_PATH.parent.mkdir(parents=True, exist_ok=True)
        lead = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": name.strip(),
            "contact": contact.strip(),
            "company": company.strip(),
            "role": role.strip(),
            "message": message.strip(),
        }

        with LEADS_PATH.open("a", encoding="utf-8") as leads_file:
            leads_file.write(json.dumps(lead) + "\n")

        return "Lead captured for Chakit."

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
        llm=anthropic.LLM(model=LLM_MODEL, max_tokens=120),
        tts=openai.TTS(speed=TTS_SPEED),
        vad=LoggedVAD(silero.VAD.load()),
    )


server = AgentServer()


@server.rtc_session(agent_name="personal-assistant")
async def entrypoint(ctx: JobContext) -> None:
    session = build_session()
    recorder = TranscriptRecorder()
    session.on("conversation_item_added", recorder.record)
    session.on("close", lambda _: recorder.save_transcript())
    ctx.add_shutdown_callback(recorder.save_summary)
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
