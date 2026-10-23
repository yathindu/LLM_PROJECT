"""Tests for the engine and interface: agent loop, model wrapper (retry/fallback), FastAPI layer, live trace streaming."""
import pytest
from helpers import FakeLLM, call, say

from studymate import config
from studymate.agent import Agent
from studymate.agents import build_orchestrator
from studymate.tools import Tool, study
from studymate.trace import Tracer


# ---- agent loop ----------------------------------------------------------


def make_agent(llm, tools, tmp_path):
    return Agent("T", "SYS-T", tools, llm, Tracer(tmp_path, quiet=True))


def test_loop_executes_tool_then_answers(tmp_path):
    llm = FakeLLM({"SYS-T": [call("calculator", expression="6*7"), say("It is 42.")]})
    agent = make_agent(llm, [t for t in study.STUDY_TOOLS if t.name == "calculator"], tmp_path)
    answer, history = agent.run("what is 6*7")
    assert answer == "It is 42."
    assert llm.seen_tool_results == [("calculator", "6*7 = 42", True)]
    assert len(history) == 4  # user, model call, tool result, model answer


def test_loop_feeds_tool_errors_back_to_model(tmp_path):
    llm = FakeLLM({"SYS-T": [call("nope"), call("calculator", expression="import os"), say("Recovered.")]})
    agent = make_agent(llm, [t for t in study.STUDY_TOOLS if t.name == "calculator"], tmp_path)
    answer, _ = agent.run("x")
    assert answer == "Recovered."
    assert [ok for _, _, ok in llm.seen_tool_results] == [False, False]


def test_loop_reports_bad_arguments(tmp_path):
    llm = FakeLLM({"SYS-T": [call("calculator", wrong="1"), say("done")]})
    agent = make_agent(llm, [t for t in study.STUDY_TOOLS if t.name == "calculator"], tmp_path)
    agent.run("x")
    assert llm.seen_tool_results[0][2] is False and "Bad arguments" in llm.seen_tool_results[0][1]


def test_agent_that_must_use_a_tool_is_pushed_back_once_when_it_answers_from_memory(tmp_path):
    from studymate.agent import NUDGE

    llm = FakeLLM({"SYS-T": [say("from memory"), call("calculator", expression="2+2"), say("grounded answer")]})
    calc = [t for t in study.STUDY_TOOLS if t.name == "calculator"]
    agent = Agent("T", "SYS-T", calc, llm, Tracer(tmp_path, quiet=True), require_tool=True)
    answer, history = agent.run("what is 2+2")
    assert answer == "grounded answer"
    assert ("user", NUDGE) in history and llm.seen_tool_results == [("calculator", "2+2 = 4", True)]


def test_reminder_is_sent_only_once_even_if_the_model_keeps_answering_without_tools(tmp_path):
    llm = FakeLLM({"SYS-T": [say("no"), say("still no"), say("never reached")]})
    agent = Agent("T", "SYS-T", [], llm, Tracer(tmp_path, quiet=True), require_tool=True)
    answer, history = agent.run("hi")
    assert answer == "still no" and len(history) == 4 and len(llm.scripts["SYS-T"]) == 1


def test_guard_can_be_a_function_that_decides_per_task(tmp_path):
    llm = FakeLLM({"SYS-T": [say("plain reply"), say("2+2 from memory"), call("calculator", expression="2+2"), say("4")]})
    calc = [t for t in study.STUDY_TOOLS if t.name == "calculator"]
    agent = Agent("T", "SYS-T", calc, llm, Tracer(tmp_path, quiet=True), require_tool=lambda task: "2+2" in task)
    assert agent.run("write a poem")[0] == "plain reply"  # task not accepted by the function: no reminder, no extra call
    assert agent.run("what is 2+2")[0] == "4"  # accepted: the tool-free answer is pushed back


def test_tool_filter_offers_only_matching_tools_and_a_hidden_tool_cannot_be_called(tmp_path):
    offered_to_model = []

    class Spy(FakeLLM):
        def generate(self, system, history, tools):
            offered_to_model.append([t.name for t in tools])
            return super().generate(system, history, tools)

    llm = Spy({"SYS-T": [call("get_current_date"), say("done")]})
    tools = [t for t in study.STUDY_TOOLS if t.name in ("calculator", "get_current_date")]
    agent = Agent("T", "SYS-T", tools, llm, Tracer(tmp_path, quiet=True), tool_filter=lambda task, name: name == "calculator")
    agent.run("anything")
    assert offered_to_model[0] == ["calculator"]
    assert llm.seen_tool_results[0][2] is False and "Unknown tool" in llm.seen_tool_results[0][1]  # it tried a tool it was not offered


def test_agent_without_the_guard_may_answer_directly(tmp_path):
    llm = FakeLLM({"SYS-T": [say("hello")]})
    answer, history = make_agent(llm, [], tmp_path).run("hi")
    assert answer == "hello" and len(history) == 2


def test_loop_stops_at_max_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_STEPS", 3)
    llm = FakeLLM({"SYS-T": [call("get_current_date")] * 3})
    agent = make_agent(llm, [t for t in study.STUDY_TOOLS if t.name == "get_current_date"], tmp_path)
    answer, _ = agent.run("loop forever")
    assert "stopped after 3 steps" in answer


def test_long_tool_output_is_truncated(tmp_path):
    big = Tool("big", "x", {"type": "object", "properties": {}}, lambda: "x" * 20000)
    llm = FakeLLM({"SYS-T": [call("big"), say("ok")]})
    make_agent(llm, [big], tmp_path).run("go")
    assert len(llm.seen_tool_results[0][1]) < config.MAX_TOOL_OUTPUT + 50


# ---- FastAPI layer -----------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from studymate import api
    from studymate.agents import ORCHESTRATOR_PROMPT

    llm = FakeLLM({ORCHESTRATOR_PROMPT[:40]: [say("Hello!"), say("Hello!"), say("Second turn."), say("Second turn.")]})
    api.app.dependency_overrides[api.get_llm] = lambda: llm
    api._sessions.clear()
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_api_health_and_index(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/").text
    assert "StudyMate" in page and "notesBtn" in page and "/chat/stream" in page


def test_api_chat_returns_answer_trace_and_keeps_session(client):
    r1 = client.post("/chat", json={"message": "hi"}).json()
    assert r1["answer"] == "Hello!" and r1["trace"][-1]["type"] == "final"
    r2 = client.post("/chat", json={"message": "again", "session_id": r1["session_id"]}).json()
    assert r2["session_id"] == r1["session_id"] and r2["answer"] == "Second turn."
    from studymate import api

    assert len(api._sessions[r1["session_id"]]) == 8  # per turn: user, answer, routing reminder, repeated answer


def test_api_rejects_empty_message(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_api_notes_endpoints(client):
    study.save_note("Demo Note", "body text")
    assert client.get("/notes").json() == {"notes": ["demo-note"]}
    assert "body text" in client.get("/notes/Demo Note").json()["content"]
    assert client.get("/notes/missing").status_code == 404


# ---- model wrapper (Ollama) ----------------------------------------------------


class StubOllama:
    """Stands in for ollama.Client: returns or raises whatever the test queued, and records each request."""

    def __init__(self, *outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def chat(self, **kwargs):
        self.requests.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def make_llm(*outcomes):
    from studymate.llm import OllamaLLM

    llm = OllamaLLM(model="test-model", host="http://localhost:1")
    llm.client = StubOllama(*outcomes)
    return llm


def reply(content="", calls=()):
    import ollama

    tool_calls = [ollama.Message.ToolCall(function=ollama.Message.ToolCall.Function(name=n, arguments=a)) for n, a in calls]
    return ollama.ChatResponse(model="test-model", message=ollama.Message(role="assistant", content=content, tool_calls=tool_calls or None))


def test_generate_parses_tool_calls_and_sends_system_prompt_tools_and_context_size():
    from studymate.llm import ToolCall

    llm = make_llm(reply(calls=[("calculator", {"expression": "1+1"})]))
    r = llm.generate("SYS", [llm.user_message("hi")], [t for t in study.STUDY_TOOLS if t.name == "calculator"])
    assert r.calls == [ToolCall("calculator", {"expression": "1+1"})]
    assert r.raw["tool_calls"][0]["function"] == {"name": "calculator", "arguments": {"expression": "1+1"}}
    sent = llm.client.requests[0]
    assert sent["model"] == "test-model" and sent["messages"][0] == {"role": "system", "content": "SYS"}
    assert sent["messages"][1] == {"role": "user", "content": "hi"}
    assert sent["tools"][0]["function"]["name"] == "calculator"
    assert sent["options"]["num_ctx"] == config.NUM_CTX


def test_generate_plain_answer_has_no_calls():
    r = make_llm(reply("It is 42.")).generate("S", [{"role": "user", "content": "q"}], [])
    assert r.text == "It is 42." and r.calls == [] and r.raw == {"role": "assistant", "content": "It is 42."}


def test_tool_results_expand_to_one_tool_message_each_and_errors_are_marked():
    from studymate.llm import OllamaLLM

    llm = make_llm()
    history = [llm.user_message("hi"), {"role": "assistant", "content": ""}, llm.tool_results_message([("a", "out", True), ("b", "boom", False)])]
    messages = OllamaLLM._messages("S", history)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "tool"]
    assert messages[3] == {"role": "tool", "tool_name": "a", "content": "out"}
    assert messages[4]["content"] == "ERROR: boom"


def test_oldest_long_tool_outputs_are_shortened_to_fit_the_context_window():
    from studymate.llm import OllamaLLM

    msgs = [
        {"role": "system", "content": "S"},
        {"role": "tool", "tool_name": "a", "content": "A" * 3000},
        {"role": "tool", "tool_name": "b", "content": "B" * 3000},
        {"role": "user", "content": "question"},
    ]
    fitted = OllamaLLM._fit([dict(m) for m in msgs], budget_chars=3500)
    assert len(fitted[1]["content"]) < 400 and "shortened" in fitted[1]["content"]  # oldest shrinks first
    assert fitted[2]["content"] == "B" * 3000  # stops as soon as it fits
    assert fitted[3]["content"] == "question"
    untouched = OllamaLLM._fit([dict(m) for m in msgs], budget_chars=10_000)
    assert untouched[1]["content"] == "A" * 3000


def test_fitting_never_edits_the_real_history():
    from studymate.llm import OllamaLLM

    llm = make_llm()
    big = {"role": "tool_results", "results": [("fetch_webpage", "X" * 3000, True)]}
    history = [llm.user_message("hi"), big]
    OllamaLLM._fit(OllamaLLM._messages("S", history), budget_chars=100)
    assert big["results"][0][1] == "X" * 3000


def test_reply_length_is_capped_and_timeouts_are_reported_as_timeouts():
    import httpx

    from studymate.llm import LLMError

    llm = make_llm(reply("ok"))
    llm.generate("S", [llm.user_message("hi")], [])
    assert llm.client.requests[0]["options"]["num_predict"] == config.MAX_NEW_TOKENS
    with pytest.raises(LLMError, match="did not answer within"):
        make_llm(httpx.ReadTimeout("slow")).generate("S", [], [])


def test_unreachable_ollama_becomes_llmerror_with_install_hint():
    from studymate.llm import LLMError

    with pytest.raises(LLMError, match="ollama serve"):
        make_llm(ConnectionError("refused")).generate("S", [], [])


def test_missing_model_becomes_llmerror_with_pull_hint():
    import ollama

    from studymate.llm import LLMError

    with pytest.raises(LLMError, match="ollama pull test-model"):
        make_llm(ollama.ResponseError("model not found", 404)).generate("S", [], [])


def test_subagent_llm_failure_stops_whole_run_instead_of_being_retried(tmp_path):
    from studymate.agents import ORCHESTRATOR_PROMPT, RESEARCHER_PROMPT
    from studymate.llm import LLMError

    class DyingResearcherLLM(FakeLLM):
        def generate(self, system, history, tools):
            if RESEARCHER_PROMPT[:40] in system:
                raise LLMError("quota")
            return super().generate(system, history, tools)

    llm = DyingResearcherLLM({ORCHESTRATOR_PROMPT[:40]: [call("ask_researcher", task="x"), call("ask_researcher", task="x")]})
    with pytest.raises(LLMError):
        build_orchestrator(llm, Tracer(tmp_path, quiet=True)).run("research something")
    assert len(llm.scripts[ORCHESTRATOR_PROMPT[:40]]) == 1  # the second delegation was never attempted


# ---- live trace streaming (SSE) ------------------------------------------------


def read_sse(client, body):
    import json as _json

    with client.stream("POST", "/chat/stream", json=body) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        return [_json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]


def test_stream_sends_trace_events_then_done(client):
    events = read_sse(client, {"message": "hi"})
    assert [e["type"] for e in events] == ["step", "step", "final", "done"]  # two steps: the answer, then the reminder
    assert events[-1]["answer"] == "Hello!" and events[-1]["session_id"]


def test_stream_keeps_session_history(client):
    first = read_sse(client, {"message": "hi"})[-1]
    second = read_sse(client, {"message": "again", "session_id": first["session_id"]})[-1]
    assert second["session_id"] == first["session_id"] and second["answer"] == "Second turn."


def test_stream_reports_failures_as_error_event(client):
    from studymate import api

    class Boom:
        def user_message(self, t):
            return t

        def generate(self, *a):
            raise RuntimeError("model down")

    api.app.dependency_overrides[api.get_llm] = lambda: Boom()
    events = read_sse(client, {"message": "hi"})
    assert events[-1]["type"] == "error" and "model down" in events[-1]["detail"]


def test_tracer_on_event_callback_sees_every_event(tmp_path):
    seen = []
    t = Tracer(tmp_path, quiet=True, on_event=seen.append)
    t.step("A", 0, 1)
    t.tool_call("A", 0, "x", {"a": 1})
    assert [e["type"] for e in seen] == ["step", "tool_call"] == [e["type"] for e in t.events]
