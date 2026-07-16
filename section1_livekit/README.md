# Section 1 — LiveKit Agents: Real-Time Voice AI

Airline booking support voice agent demonstrating a full **STT → LLM → TTS**
pipeline using the LiveKit Agents SDK, with verified tool-calling via Gemini Flash.

---

## What We Built

We implemented a headless voice agent that simulates a real-time airline support
assistant. The agent accepts scripted user utterances, passes them through an
STT → LLM → TTS pipeline, and logs every stage — including tool invocations — to
a transcript file. No microphone, speakers, or LiveKit server are required; the
simulation driver runs entirely in-process.

The three pipeline stages are:

1. **STT (Speech-to-Text)** — `SttStub` passes the user's text directly to the
   session as a `FINAL_TRANSCRIPT` speech event. It implements the official
   `livekit.agents.stt.STT` base class (`_recognize_impl`), so the pipeline shape
   is architecturally identical to a production STT plugin. Swapping in
   `google.STT()` or `deepgram.STT()` requires changing one constructor argument.

2. **LLM** — `google.LLM(model="gemini-2.0-flash-lite")` from
   `livekit-plugins-google`. The plugin handles tool-schema generation from Python
   type hints, sends the schema to Gemini on every turn, and routes any
   `function_call` response back through the agent's `@function_tool` methods
   before the final reply is generated.

3. **TTS (Text-to-Speech)** — `TtsStub` logs the text it would speak and emits
   100 ms of silent PCM audio via `output_emitter.push()` so the pipeline
   completes its audio stage cleanly. It implements `livekit.agents.tts.TTS`
   (`synthesize` → `ChunkedStream._run`). Swapping in `google.TTS()` or
   `cartesia.TTS()` is again a single constructor argument.

---

## Agent

`AirlineAgent` subclasses `livekit.agents.Agent` and carries the airline support
persona in its `instructions` string. It exposes one tool:

```python
@function_tool
async def get_flight_status(self, context: RunContext, flight_id: str) -> str:
    """Look up the current operational status of a flight.

    Use this tool whenever a passenger asks about a specific flight's status,
    delay, gate number, or departure and arrival times.

    Args:
        flight_id: The IATA flight identifier, e.g. 'BA123' or 'AA456'.
    """
```

The `@function_tool` decorator auto-generates a JSON schema from the type hints
and docstring. The LLM reads this schema to decide when to call the tool and how
to populate `flight_id`. The tool looks up the flight ID in an in-memory dict and
returns a plain string; the LLM incorporates that string into its reply.

---

## Pipeline Flow (per turn)

```
user_input (str)
    │
    ▼
SttStub._recognize_impl()
    └─ returns SpeechEvent(FINAL_TRANSCRIPT, text=user_input)
    │
    ▼
google.LLM  ←──────────────────────────────────────────┐
    │  generates reply or function_call               │
    ├─ function_call detected                         │
    │       │                                         │
    │       ▼                                         │
    │  AirlineAgent.get_flight_status(flight_id)      │
    │       └─ logs [TOOL CALL] / [TOOL RESULT]       │
    │       └─ returns status string ─────────────────┘
    │
    ▼  (final assistant message)
TtsStub.synthesize(text)
    └─ logs [TTS] Synthesizing speech ← "..."
    └─ emits silent PCM frame via output_emitter
    │
    ▼
RunResult  →  _log_run_events()  →  sample_run.log
```

---

## Simulation Driver

`run_simulation.py` drives three scripted turns through `AgentSession.run()`:

| Turn | User input | Expected behaviour |
|------|------------|-------------------|
| 1 | "Hello" | Greeting — `on_enter` fires, LLM responds |
| 2 | "What is the status of flight BA123?" | Tool called, result injected, LLM replies with live data |
| 3 | "Can you check flight ZZ999?" | Tool called, not-found branch returned, LLM relays gracefully |

A 5-second delay between turns stays within the Gemini free-tier rate limit
(30 RPM for `gemini-2.0-flash-lite`).

---

## Verified Transcript

The actual output of `transcripts/sample_run.log` from the completed run:

```
2026-07-16 23:00:21  simulation   === Simulation start ===
2026-07-16 23:00:22  simulation   --- Turn 1: greeting ---
2026-07-16 23:00:23  simulation   [ASSISTANT] Hello! How can I help you with your travel plans today?

2026-07-16 23:00:28  simulation   --- Turn 2: flight status query (tool call expected) ---
2026-07-16 23:00:29  airline_agent  [TOOL CALL] get_flight_status(flight_id='BA123')
2026-07-16 23:00:29  airline_agent  [TOOL RESULT] 'On time — scheduled departure 14:30 UTC from LHR, arrival 17:45 UTC at JFK.'
2026-07-16 23:00:29  simulation   [LLM->TOOL] get_flight_status(flight_id='BA123')
2026-07-16 23:00:29  simulation   [TOOL->LLM] On time — scheduled departure 14:30 UTC from LHR, arrival 17:45 UTC at JFK.
2026-07-16 23:00:29  simulation   [ASSISTANT] Flight BA123 is currently on time. It is scheduled to depart from LHR at 14:30 UTC and arrive at JFK at 17:45 UTC.

2026-07-16 23:00:34  simulation   --- Turn 3: unknown flight ---
2026-07-16 23:00:35  airline_agent  [TOOL CALL] get_flight_status(flight_id='ZZ999')
2026-07-16 23:00:35  airline_agent  [TOOL RESULT] 'Flight ZZ999 was not found in our system. Please verify the flight number and try again.'
2026-07-16 23:00:35  simulation   [ASSISTANT] I'm sorry, but I couldn't find a flight with the number ZZ999 in our system. Please double-check the flight number.

2026-07-16 23:00:35  simulation   === Simulation end ===
```

The `[TOOL CALL]` and `[TOOL RESULT]` lines are emitted from inside
`get_flight_status()` in `agent.py`, not reconstructed after the fact. They prove
the LLM invoked the tool mid-turn rather than hallucinating an answer.

---

## Setup

```powershell
cd section1_livekit
py -m venv venv
.\venv\Scripts\pip install -r requirements.txt
copy .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_key_here
```

## Run

```powershell
.\venv\Scripts\python run_simulation.py
```

Transcript is written to `transcripts/sample_run.log`.

---

## Files

| File | Purpose |
|------|---------|
| `agent.py` | `AirlineAgent` — persona, `on_enter`, `@function_tool get_flight_status` |
| `stubs/stt_stub.py` | Text-passthrough STT implementing `stt.STT` |
| `stubs/tts_stub.py` | Silent-audio TTS implementing `tts.TTS` / `ChunkedStream` |
| `run_simulation.py` | Headless 3-turn driver — no LiveKit server needed |
| `transcripts/sample_run.log` | Saved proof of tool invocation |
| `NOTES.md` | Write-up: barge-in handling, second tool safety, provider swapping |
