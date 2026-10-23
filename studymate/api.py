"""FastAPI interface. Run:  uvicorn studymate.api:app --reload   then open http://127.0.0.1:8000"""
import json
import queue
import threading
import uuid
from functools import lru_cache
from pathlib import Path

import ollama
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config
from .agents import build_orchestrator
from .llm import OllamaLLM
from .tools import study
from .trace import Tracer

DESCRIPTION = """
Multi-agent study & research assistant running a local model through Ollama (function calling).

An **Orchestrator** agent delegates to a **Researcher** (web search, page fetch, Wikipedia) and a **Study Coach**
(notes, calculator, dates, timetable). `POST /chat` returns the answer **and the full tool-call trace**:
each `tool_call` / `tool_result` event has an `agent` and a `depth` (0 = orchestrator, 1 = specialist).

Try it: open **POST /chat -> Try it out**, pick an example, **Execute**, and read `trace` in the response.
"""

app = FastAPI(
    title="StudyMate",
    description=DESCRIPTION,
    version="1.0.0",
    openapi_tags=[
        {"name": "chat", "description": "Talk to the agent system"},
        {"name": "notes", "description": "Study notes saved by the Study Coach"},
        {"name": "system", "description": "Health check"},
    ],
)

_sessions: dict[str, list] = {}  # session_id -> orchestrator conversation history
_lock = threading.Lock()


@lru_cache
def get_llm() -> OllamaLLM:
    return OllamaLLM()  # created on first request so the app can start (and be tested) without Ollama running


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, description="Omit to start a new conversation")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "Explain what gradient descent is and give me two reliable sources."},
                {
                    "message": 'Research how the transformer attention mechanism works, save a short study note called "Attention", '
                    "then write a 5-question quiz with an answer key based on it."
                },
                {
                    "message": "My exam is on 2026-11-20. I need to cover backpropagation, CNNs, RNNs, transformers and "
                    "optimisation at 2 hours a day. How many days do I have, and what is my timetable? Save it as a note."
                },
            ]
        }
    }


class ChatResponse(BaseModel):
    session_id: str = Field(description="Send this back to continue the conversation")
    answer: str
    trace: list[dict] = Field(description="Ordered agent steps, tool calls and tool results")


@app.get("/health", tags=["system"], summary="Health check")
def health() -> dict:
    try:  # a short-timeout probe: is Ollama up, and has the model been pulled?
        names = [m.model for m in ollama.Client(host=config.OLLAMA_HOST, timeout=2).list().models]
        state = "ready" if any(n == config.MODEL or n.split(":")[0] == config.MODEL for n in names) else "model not pulled"
    except Exception:
        state = "not reachable"
    return {"status": "ok", "model": config.MODEL, "ollama": state}


@app.post("/chat", response_model=ChatResponse, tags=["chat"], summary="Send a request to the agents")
def chat(req: ChatRequest, llm=Depends(get_llm)) -> ChatResponse:
    # Plain `def` endpoint: FastAPI runs it in a worker thread, so the blocking agent loop doesn't stall the server.
    session_id = req.session_id or uuid.uuid4().hex
    with _lock:
        history = list(_sessions.get(session_id, []))
    tracer = Tracer()
    try:
        answer, history = build_orchestrator(llm, tracer).run(req.message, history)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent run failed: {type(e).__name__}: {e}")
    with _lock:
        _sessions[session_id] = history
    return ChatResponse(session_id=session_id, answer=answer, trace=tracer.events)


@app.post(
    "/chat/stream",
    tags=["chat"],
    summary="Same as /chat, but streams the trace live (Server-Sent Events)",
    response_description="`text/event-stream`: one JSON event per agent step / tool call / tool result, then a final "
    "`done` event with the answer and session_id (or an `error` event).",
)
def chat_stream(req: ChatRequest, llm=Depends(get_llm)) -> StreamingResponse:
    session_id = req.session_id or uuid.uuid4().hex
    with _lock:
        history = list(_sessions.get(session_id, []))
    events: queue.Queue = queue.Queue()
    tracer = Tracer(on_event=events.put)

    def work() -> None:  # the agent loop is blocking, so it runs in its own thread and feeds the queue
        try:
            answer, new_history = build_orchestrator(llm, tracer).run(req.message, history)
            with _lock:
                _sessions[session_id] = new_history
            events.put({"type": "done", "session_id": session_id, "answer": answer})
        except Exception as e:
            events.put({"type": "error", "detail": f"{type(e).__name__}: {e}"})
        events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (event := events.get()) is not None:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.delete("/sessions/{session_id}", tags=["chat"], summary="Forget a conversation")
def reset_session(session_id: str) -> dict:
    with _lock:
        existed = _sessions.pop(session_id, None) is not None
    return {"deleted": existed}


@app.get("/notes", tags=["notes"], summary="List saved notes")
def get_notes() -> dict:
    names = sorted(p.stem for p in config.NOTES_DIR.glob("*.md")) if config.NOTES_DIR.exists() else []
    return {"notes": names}


@app.get("/notes/{title}", tags=["notes"], summary="Read a note")
def get_note(title: str) -> dict:
    try:
        path = study.note_path(title)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid note title")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Note not found")
    return {"title": title, "content": path.read_text(encoding="utf-8")}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")
