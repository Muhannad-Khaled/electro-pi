# Section 1 — LiveKit Agents (Real-time Voice AI)

> **Status:** Proposed plan for review before writing any code.
> **Goal:** Build a real voice agent with a full STT → LLM → TTS pipeline, with tool calling actually demonstrated.
> **Points:** 20 (+5 bonus).

---

## 1. Requirements (from the test)

### Task 1.1 (20 points)
Build a minimal voice agent using the `livekit-agents` Python SDK:

- Run an `AgentSession` pipeline: **STT → LLM → TTS** (any providers, or stub the STT/TTS with text I/O as long as the LLM + tool-calling logic is real).
- Define an `Agent` subclass with a **system persona** (e.g. a support assistant for a food delivery app).
- Expose at least one `@function_tool`-decorated method the LLM can call mid-conversation — e.g. `get_order_status(order_id: str) -> str` returning a mocked lookup result.
- **Demonstrate**, via a transcript/log or short screen recording, the LLM actually invoking the tool call.

**Write-up (half page):** how to extend this to support barge-in / interruption handling, and how to add a second tool safely (schema, error handling if the tool call fails).

### Task 1.2 — bonus (5 points)
Show the same agent working with a different STT or TTS provider than the first implementation (or explain precisely, with code snippets, what would change if providers were swapped). This tests whether the design is decoupled from any single vendor.

---

## 2. Architectural Decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| **LLM** | Gemini (`gemini-2.0-flash`) via `livekit-plugins-google` | Official support, reliable tool calling, generous free tier |
| **STT** | Stubbed (text input) | Allowed by the test; focus is on pipeline + tool calling |
| **TTS** | Stubbed (text output / log) | Same reason |
| **Architecture** | Explicit STT→LLM→TTS pipeline (NOT Gemini Realtime) | The test asks for an explicit pipeline, not a speech-to-speech multimodal model |
| **Persona** | Support assistant for a food delivery app | The example suggested in the test itself |
| **Tool** | `get_order_status(order_id) -> str` (mocked lookup) | The example suggested in the test |

### Why stub STT/TTS instead of Gemini Realtime?
The test is explicit: it wants an **STT → LLM → TTS pipeline**. Gemini Realtime (speech-to-speech) hides the three components inside a single model, which answers a different question than the one asked. A stub that preserves the pipeline shape proves we understand the architecture.

### Why Gemini instead of Qwen2.5-1.5B?
The 1.5B model is relatively weak at tool calling, and the core of this section is that the tool call actually fires. Gemini Flash guarantees this. (Qwen is reserved for Sections 3 & 4, where the goal is quantization/deployment, not tool-calling reliability.)

---

## 3. Task Breakdown

- [ ] **T1 — Environment setup:** requirements (`livekit-agents`, `livekit-plugins-google`), `.env` for `GOOGLE_API_KEY`.
- [ ] **T2 — Stub STT:** class implementing the LiveKit STT interface, taking text and returning it as a transcript (mimics a real STT's shape).
- [ ] **T3 — Stub TTS:** class implementing the TTS interface, taking text and logging/printing it instead of speaking.
- [ ] **T4 — Agent subclass:** with the system persona (food delivery support assistant).
- [ ] **T5 — Function tool:** `get_order_status` with `@function_tool`, mocked lookup (dict with a few orders).
- [ ] **T6 — AgentSession wiring:** connect STT (stub) → Gemini LLM → TTS (stub).
- [ ] **T7 — Driver script:** run a simulated session with text input that triggers the tool call (e.g. "Where is my order 12345?").
- [ ] **T8 — Logging/transcript:** prove the tool actually fired — print/log the tool invocation + arguments + result.
- [ ] **T9 — Write-up:** barge-in + adding a second tool safely.
- [ ] **T10 — (bonus 1.2):** document swapping the TTS stub for Google TTS with a snippet.

---

## 4. Proposed File Structure

```
section1_livekit/
├── README.md              # setup + run + explanation of tool-call proof
├── requirements.txt
├── .env.example           # GOOGLE_API_KEY=
├── agent.py               # Agent subclass + function_tool + persona
├── stubs/
│   ├── __init__.py
│   ├── stt_stub.py        # Stub STT (text → transcript)
│   └── tts_stub.py        # Stub TTS (text → log)
├── run_simulation.py      # driver that runs a session and triggers the tool call
├── transcripts/
│   └── sample_run.log     # output of a real run proving the tool invocation
└── NOTES.md               # write-up (barge-in, second tool, decoupling)
```

---

## 5. Tool-Call Proof Plan

The most important point in this section. The plan:
1. The driver sends a message that triggers the tool (a question about order status).
2. Add logging on entry to `get_order_status` (the arguments coming from the LLM).
3. Log the result returned to the LLM.
4. Log the LLM's final reply after using the result.
5. Save the full log to `transcripts/sample_run.log` as evidence.

Expected shape in the log:
```
[USER]  Where is my order 12345?
[LLM->TOOL] get_order_status(order_id="12345")
[TOOL->LLM] {"order_id": "12345", "status": "Out for delivery", "eta": "15 min"}
[LLM->USER] Your order 12345 is out for delivery and should arrive in about 15 minutes.
```

---

## 6. Write-up Plan (NOTES.md)

### a. Barge-in / interruption handling
- LiveKit supports interruption via VAD (voice activity detection) + endpointing.
- The idea: when the user speaks while the agent is talking, the TTS stops (`interrupt`) and the pipeline re-routes to the new input.
- Explain the role of `AgentSession` in managing turn-taking and `allow_interruptions`.

### b. Adding a second tool safely
- **Clear schema:** type hints + a precise docstring so the LLM knows when to use it.
- **Error handling:** try/except inside the tool, returning a structured error message to the LLM instead of crashing (e.g. order not found → clear message, not an exception).
- **Validation:** check the arguments before executing (order_id format).
- **Idempotency / side effects:** distinguish read tools (safe) from write tools (need confirmation).

### c. Decoupling (for bonus 1.2)
- STT/TTS/LLM are all plugins behind interfaces. Swapping any one = changing a single instantiation line.
- Snippet: swap the TTS stub for `google.TTS()` or `openai.TTS()`.

---

## 7. Open Questions Before Coding

1. **Persona:** go with "food delivery support assistant" (the test's example), or a different domain?
2. **Stub level:** prefer the stub to implement the official LiveKit interfaces exactly (cleaner, proves deeper understanding), or a simpler pipeline simulation (faster, but less "official")?
3. **Bonus 1.2:** implement it for real (swap TTS for actual Google TTS), or settle for a code snippet + explanation (as the test allows)?
