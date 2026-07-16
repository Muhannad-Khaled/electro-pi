# Section 1 — Write-up Notes

## a. Barge-in / Interruption Handling

`AgentSession` ships with `allow_interruptions=True` and handles barge-in at the
pipeline level, requiring no changes to `agent.py`. To enable it, supply a VAD
model as a constructor argument:

```python
from livekit.plugins import silero
session = AgentSession(..., vad=silero.VAD.load())
```

With VAD attached, the session monitors microphone input continuously while TTS
audio is playing. When the VAD detects speech onset, the session calls
`interrupt()`, which cancels the active `ChunkedStream` mid-playback and
immediately routes the new utterance into the STT pipeline. Crucially, the
partial agent turn that was cut off is preserved in the chat context, so the LLM
receives an accurate picture of what was said before the interruption and can
respond naturally rather than restarting from scratch. Two knobs control
sensitivity: `min_interruption_duration` sets the minimum speech duration before
an interruption is treated as real (filtering out coughs and brief noise), and
`resume_false_interruption=True` restores playback if the user falls silent
immediately after the VAD fires, avoiding the disruption of a spurious trigger.


## b. Adding a Second Tool Safely

The `@function_tool` decorator exposes a Python method to the LLM as a callable
schema, and the LLM's only signal for when and how to invoke it is the method's
type hints and docstring. A vague or missing `Args:` block causes mistimed calls
or incorrect argument values, so the docstring should state precisely which user
intent the tool serves and what each parameter must look like — for example,
specifying that `booking_ref` is a six-character alphanumeric code rather than
just "a reference number". Input should be validated against that format before
touching any external API: reject malformed values immediately with a clear,
actionable message so the user can correct themselves without spending a network
round-trip.

The most important safety rule is that a `@function_tool` must never raise an
unhandled exception. If an exception escapes the tool body, the tool-call chain
fails silently from the user's perspective — the LLM receives no output and
cannot recover gracefully. Instead, wrap the external call in a `try/except` and
return a structured error string:

```python
try:
    result = _lookup_baggage(booking_ref)
except Exception as exc:
    logger.warning("baggage lookup failed: %s", exc)
    return f"Unable to retrieve baggage status for {booking_ref}. Please try again later."
```

The LLM will relay that string to the user naturally. Finally, distinguish
read-only tools — which are safe to call any number of times — from write tools
such as cancellations or rebooking, which should require explicit user
confirmation before executing and should be idempotent where the backend allows.


## c. Provider Decoupling — Bonus Task 1.2

The STT, LLM, and TTS are all passed as constructor arguments to `AgentSession`.
Swapping any one provider is a **single line change**.

### Swap TTS stub → Google Cloud TTS

```python
# Before (stub — no audio, just logs):
from stubs import TtsStub
tts = TtsStub()

# After (Google Cloud TTS — real audio synthesis):
from livekit.plugins import google
tts = google.TTS(voice="en-US-Neural2-F")
```

Requires `GOOGLE_APPLICATION_CREDENTIALS` set to a service-account JSON with
Cloud Text-to-Speech API enabled. Everything else — `agent.py`, the tool, the
persona instructions — stays identical.

### Swap LLM → OpenAI GPT-4o

```python
# Before (Gemini Flash):
from livekit.plugins import google
llm = google.LLM(model="gemini-2.0-flash")

# After (OpenAI GPT-4o):
from livekit.plugins import openai
llm = openai.LLM(model="gpt-4o")
```

Requires `OPENAI_API_KEY` in `.env`. The `@function_tool` decorator and the
`Args:` docstring are provider-agnostic — both Gemini and GPT-4o parse the tool
schema from the same Python type hints.

### Swap STT stub → Google Cloud Speech-to-Text

```python
# Before (stub — text passthrough):
from stubs import SttStub
stt = SttStub(transcript="What is the status of flight BA123?")

# After (Google Cloud STT — real microphone input):
from livekit.plugins import google
stt = google.STT(model="long", spoken_punctuation=False)
```

Requires `GOOGLE_APPLICATION_CREDENTIALS` with Cloud Speech API enabled and a
microphone input connected via `room_io` or `session.input.audio`.

**Key point:** because `SttStub`, `TtsStub`, and any real plugin all implement
the same base-class interfaces (`stt.STT`, `tts.TTS`), `AgentSession` is
completely unaware of which implementation it holds. The decoupling is structural,
not just by convention.
