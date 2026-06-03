# Personal Assistant

LiveKit voice agent for Chakit's personal portfolio. It uses OpenAI STT/TTS, Claude Haiku for reasoning, and Silero VAD for turn detection.

## Setup

```bash
uv sync
cp .env.example .env
```

Fill in `.env`:

```env
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

## Run

```bash
uv run python agent.py console
```

Useful console options:

```bash
uv run python agent.py console --text
uv run python agent.py console --list-devices
uv run python agent.py console --input-device "MacBook" --output-device "MacBook"
```

Exit with `Ctrl+C`.

## What It Does

- Answers questions about Chakit's profile and projects.
- Softly asks for name and contact early in the call.
- Saves useful contact or hiring context to `data/leads.jsonl`.
- Writes full conversation transcripts to `transcripts/`.
- Writes detailed runtime logs to `logs/agent.log`.
- Ends the call with LiveKit's `EndCallTool` when the conversation naturally closes.

## Voice Smoke Test

1. Say: "Hi, how are you?"
2. Confirm it asks for name and email/phone, while allowing refusal.
3. Mention hiring or an opportunity later.
4. Provide name, contact, company, and role/context.
5. Confirm it says: "Let me save those details. One moment please."
6. Confirm the lead is saved and the call ends after the closing response.

Provider API keys must stay server-side and should never be exposed to a frontend.
