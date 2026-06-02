# LiveKit Architecture Notes

## Current Assistant Flow

1. `agent.py` loads `.env` and validates the required API keys.
2. `AgentServer()` creates the LiveKit agent server.
3. `@server.rtc_session(agent_name="personal-assistant")` registers the room entrypoint.
4. When LiveKit starts a job, `entrypoint(ctx)` receives the room context.
5. `build_session()` creates the voice pipeline.
6. `session.start(room=ctx.room, agent=Assistant())` connects the assistant to the room.

## Voice Pipeline

`build_session()` wires the realtime conversation pieces:

- `openai.STT()` turns user speech into text.
- `anthropic.LLM(model="claude-haiku-4-5")` decides the assistant response.
- `openai.TTS()` turns the response text back into speech.
- `silero.VAD.load()` detects when the user is speaking.

## Interview Explanation

The browser or console joins a LiveKit room. The Python agent also joins that
room as a participant. LiveKit moves audio between participants, while the
agent session handles speech-to-text, LLM reasoning, text-to-speech, and turn
detection.

Secrets stay on the server side. The client should only receive a scoped
LiveKit room token, not provider API keys.
