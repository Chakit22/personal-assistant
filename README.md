# personal-assistant

A local voice assistant built with LiveKit Agents and Claude.

## Setup

```bash
uv sync
cp .env.example .env
```

Fill in:

```env
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

## Run Locally

Start the assistant in console voice mode:

```bash
uv run python agent.py console
```

Use text mode if you want to test without microphone/audio setup:

```bash
uv run python agent.py console --text
```

List available audio devices:

```bash
uv run python agent.py console --list-devices
```

Pick a specific microphone or speaker:

```bash
uv run python agent.py console --input-device "MacBook" --output-device "MacBook"
```

## Test

```bash
uv run pytest
```

## Notes

- The current agent runs locally through LiveKit Agents console mode.
- Browser/portfolio usage will need LiveKit room token handling later.
- Provider API keys must stay server-side and should never be exposed to a frontend.
