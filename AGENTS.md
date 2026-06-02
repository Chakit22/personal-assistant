# Personal Assistant — Project Guidelines

LiveKit voice agent built with `livekit-agents`. Uses Anthropic and OpenAI plugins.

## Linear Workspace

- **Team**: Personal (key: `PER`)
- **Project**: Personal Assistant (id: `9e7d3f9d-8e79-4175-8f55-8e1a08cd7f06`)
- **GitHub repo**: `Chakit22/personal-assistant`
- **Branch prefix**: `chakitbhandari22/`
- **Issue prefix**: `PER-`
- **Default base branch**: `master`

## Tech Stack

- Python 3.13+ (managed via `uv`)
- `livekit-agents` with Anthropic + OpenAI + Silero plugins
- `python-dotenv` for env loading

## LiveKit Documentation

- When answering questions about LiveKit, `livekit-agents`, LiveKit plugins, turn detection, deployment, or SDK behavior, always use the installed `livekit-docs` MCP server first so answers are based on the latest LiveKit documentation.
- If the MCP server is unavailable, fall back to official LiveKit docs only and mention that fallback.

## Conventions

- Load `.env` explicitly with `dotenv` — never assume frameworks auto-load
- Validate required env vars at startup with clear missing-key errors
- 2-space indent (where applicable), conventional commit messages (`feat:`, `fix:`, `docs:`, `chore:`)
- One PR per Linear issue; PR title format: `[PER-N] <issue title>`
- Branch naming: `chakitbhandari22/per-N-<slug>` (matches Linear's `gitBranchName`)

## Local Setup

```bash
uv sync
uv run python agent.py
```

## Required Env Vars

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
