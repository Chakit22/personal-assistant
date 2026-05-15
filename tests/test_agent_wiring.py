"""Tests for the AgentSession wiring in `agent.py`.

These are pure structural tests: they confirm the agent factory wires up the
correct provider plugins (OpenAI STT/TTS, Anthropic Claude Haiku, Silero VAD)
without touching the network. The actual conversational behavior is verified
by running `uv run agent.py console`, which is out of scope for unit tests.
"""

from __future__ import annotations

import os

import pytest

# Provide dummy credentials so the module-level env validation doesn't trip.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from livekit.agents import Agent, AgentSession  # noqa: E402
from livekit.plugins import anthropic, openai, silero  # noqa: E402

import agent as agent_module  # noqa: E402


def test_required_env_vars_includes_anthropic_and_openai() -> None:
    assert "ANTHROPIC_API_KEY" in agent_module.REQUIRED_ENV_VARS
    assert "OPENAI_API_KEY" in agent_module.REQUIRED_ENV_VARS


def test_assistant_has_placeholder_instructions() -> None:
    assistant = agent_module.Assistant()
    assert isinstance(assistant, Agent)
    instructions = assistant.instructions
    assert isinstance(instructions, str)
    assert instructions.strip(), "Assistant must have non-empty placeholder instructions"


def test_build_session_uses_correct_providers() -> None:
    session = agent_module.build_session()
    try:
        assert isinstance(session, AgentSession)
        assert isinstance(session.stt, openai.STT)
        assert isinstance(session.llm, anthropic.LLM)
        assert session.llm.model == "claude-haiku-4-5"
        assert isinstance(session.tts, openai.TTS)
        assert isinstance(session.vad, silero.VAD)
    finally:
        # AgentSession holds plugin resources; nothing to close synchronously
        # since we never started it. Drop the reference.
        del session


def test_server_exposes_run_callable() -> None:
    """The module must expose a `server` registered for `cli.run_app(server)`
    so that `uv run agent.py console` works."""
    server = agent_module.server
    assert hasattr(server, "rtc_session")
    assert callable(server.rtc_session)
