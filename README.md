# StudyMate — a multi-agent study & research assistant

StudyMate is a small multi-agent system that runs a **local LLM through Ollama** (tool calling) with a **hand-written agent loop** (no agent framework). An **Orchestrator** agent reads the student's request and delegates to two specialists:

| Agent | Role | Tools |
|---|---|---|
| **Orchestrator** | Plans, delegates, combines results | `ask_researcher`, `ask_study_coach` |
| **Researcher** | Finds and verifies information | `web_search`, `fetch_webpage`, `wikipedia_summary` |
| **Study Coach** | Notes, quizzes, schedules, maths | `save_note`, `list_notes`, `read_note`, `calculator`, `get_current_date`, `days_until`, `build_schedule` |

Every tool call by every agent is printed live (indented by agent depth) and logged to `logs/trace-*.jsonl`.

**Demo video:** [`docs/demo_video.mp4`](docs/demo_video.mp4) (about 4.5 minutes: architecture, a single-delegation task, and a multi-step task with the tool-call trace visible).
**Technical report:** [`docs/Technical_Report.pdf`](docs/Technical_Report.pdf).

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com). No API key is needed: the model runs on your own machine.

```bash
# 1. Install Ollama from https://ollama.com (Windows: winget install Ollama.Ollama), then pull the model
ollama pull qwen3:4b-instruct            # ~2.5 GB; runs on a 4 GB GPU or on CPU

# 2. Python environment
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt          # runtime only; use requirements-dev.txt to also run the tests

# 3. Optional: change the model or the Ollama address
cp .env.example .env        # Windows: copy .env.example .env
```

Make sure Ollama is running (it starts automatically after installation on Windows and macOS; otherwise run `ollama serve`).

## Run

**Web app / REST API (FastAPI):**

```bash
uvicorn studymate.api:app --reload      # then open http://127.0.0.1:8000
```

The page at `/` is a chat UI with a **live tool-call trace** (streamed while the agents work, colour-coded per agent) and rendered markdown answers. Interactive Swagger docs are at `/docs`. The trace is also printed in the server terminal.

| Endpoint | Purpose |
|---|---|
| `POST /chat` | `{"message": "...", "session_id": null}` → `{session_id, answer, trace[]}`. Reuse `session_id` to continue a conversation. |
| `POST /chat/stream` | Same request, but streams the trace live as Server-Sent Events (`step`, `tool_call`, `tool_result`, `final`), ending with a `done` event holding the answer (or an `error` event). Used by the web UI. |
| `DELETE /sessions/{id}` | Forget a conversation |
| `GET /notes`, `GET /notes/{title}` | List / read saved notes |
| `GET /health` | Status and model name |

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
     -d '{"message": "Explain gradient descent and give two sources"}'
```

**Command line (same agents, no server):**

```bash
python main.py                          # interactive chat (keeps conversation history)
python main.py "your request here"      # one-shot
```

**Tests** (53, all offline, no API key needed):

```bash
pip install -r requirements-dev.txt
pytest -q
```

Saved notes go to `notes/` and the tool-call trace to `logs/`; both folders are created on first use.

## Three example prompts

1. **Single-agent task (research)**
   `Explain what gradient descent is and give me two reliable sources.`
2. **Multi-step task (research → notes → quiz)**
   `Research how the transformer attention mechanism works, save a short study note called "Attention", then write a 5-question quiz with an answer key based on it.`
3. **Multi-step task (planning with tools)**
   `My exam is on 2026-11-20. I need to cover backpropagation, CNNs, RNNs, transformers and optimisation at 2 hours a day. How many days do I have, and what is my timetable? Save it as a note.`

## Project layout

```
main.py                    CLI entry point
studymate/
  api.py                   FastAPI app (/chat, /chat/stream, /notes, chat UI)
  static/index.html        Chat UI: live trace, markdown answers, notes panel
  agent.py                 The agent loop (model -> tool calls -> results -> model)
  agents.py                Prompts + orchestrator/delegation wiring
  llm.py                   Ollama wrapper (message format, tool calls, clear errors)
  trace.py                 Console + JSONL tool-call trace (with live event hook)
  config.py                Settings (model, context size, step limit, paths)
  tools/research.py        Web search, page fetch (SSRF-guarded), Wikipedia
  tools/study.py           Notes, safe calculator, dates, schedule builder
tests/test_core.py         Offline tests: agent loop, model wrapper, API, streaming
tests/test_tools.py        Offline tests: tools, safety checks, prompts, delegation
tests/helpers.py, conftest.py   Scripted fake model and shared fixtures
docs/                      Technical report (PDF, Markdown source, build script) and demo video
```

## Configuration

Set in `.env` (see `.env.example`): `OLLAMA_MODEL` (default `qwen3:4b-instruct`), `OLLAMA_HOST` (default `http://localhost:11434`) and `OLLAMA_NUM_CTX` (context window, default 4096). Any Ollama model that supports tool calling works, for example `llama3.2:3b` on a weaker laptop. The step limit per agent run is `MAX_STEPS` in `studymate/config.py`. `GET /health` reports whether Ollama is reachable and the model has been pulled.

## Limitations

See the technical report. Briefly: a 4B local model is slower and less reliable than a large cloud model, web search quality depends on DuckDuckGo, sub-agents are stateless between delegations, and notes are stored as plain files.

## Credits

Built with open-source libraries: [Ollama](https://ollama.com) and its Python client, [FastAPI](https://fastapi.tiangolo.com) and Uvicorn, [ddgs](https://pypi.org/project/ddgs/) (DuckDuckGo search), Beautiful Soup, Requests, python-dotenv, Rich, and pytest. Background lookups use the public Wikipedia REST API. No tutorial or other group's project was copied. We built this with the help of an AI coding assistant Claude.