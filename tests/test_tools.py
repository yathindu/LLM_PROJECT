"""Tests for the tools and agent prompts: calculator, notes, schedules, safe fetching, delegation."""
from datetime import date, timedelta

import pytest
from helpers import FakeLLM, call, say

from studymate import config
from studymate.agents import build_orchestrator
from studymate.tools import research, study
from studymate.trace import Tracer


# ---- tools ---------------------------------------------------------------


def test_calculator_basic_and_functions():
    assert study.calculator("2 + 3 * 4") == "2 + 3 * 4 = 14"
    assert study.calculator("sqrt(16) + 2 ** 3").endswith("= 12.0")


@pytest.mark.parametrize("bad", ["__import__('os').system('echo hi')", "open('x')", "a + 1"])
def test_calculator_rejects_unsafe_input(bad):
    with pytest.raises(Exception):
        study.calculator(bad)


def test_notes_roundtrip():
    assert "No notes" in study.list_notes()
    study.save_note("Linear Algebra: Eigenvalues!", "Av = λv")
    assert "linear-algebra-eigenvalues" in study.list_notes()
    assert "Av = λv" in study.read_note("Linear Algebra: Eigenvalues!")
    assert "No note called" in study.read_note("missing")


def test_note_title_cannot_escape_notes_dir():
    path = study.note_path("../../etc/passwd")
    assert path.parent == config.NOTES_DIR


def test_days_until():
    assert study.days_until((date.today() + timedelta(days=5)).isoformat()) == f"5 days until {(date.today() + timedelta(days=5)).isoformat()}"


def test_build_schedule_covers_all_topics_and_reserves_revision():
    out = study.build_schedule(["A", "B", "C", "D", "E"], "2026-10-26", "2026-10-30", 3)
    lines = out.splitlines()
    assert len(lines) == 5
    assert "Revision" in lines[-1]
    for t in "ABCDE":
        assert any(t in line for line in lines[:-1])


def test_build_schedule_rejects_reversed_dates():
    with pytest.raises(ValueError):
        study.build_schedule(["A"], "2026-10-30", "2026-10-26")


def test_fetch_webpage_blocks_local_and_non_http():
    with pytest.raises(ValueError):
        research.fetch_webpage("http://localhost:8000/admin")
    with pytest.raises(ValueError):
        research.fetch_webpage("http://127.0.0.1/")
    with pytest.raises(ValueError):
        research.fetch_webpage("file:///etc/passwd")


# ---- multi-agent delegation ---------------------------------------------


def test_orchestrator_chains_researcher_then_coach_and_saves_note(tmp_path, monkeypatch):
    from studymate.agents import COACH_PROMPT, ORCHESTRATOR_PROMPT, RESEARCHER_PROMPT

    llm = FakeLLM(
        {
            ORCHESTRATOR_PROMPT[:40]: [
                call("ask_researcher", task="Explain transformers"),
                call("ask_study_coach", task="Make notes from: Transformers use attention. Save as 'Transformers'."),
                say("Researched and saved note 'Transformers'."),
            ],
            RESEARCHER_PROMPT[:40]: [call("wikipedia_summary", topic="x"), say("Transformers use attention. Sources: wiki")],
            COACH_PROMPT[:40]: [call("save_note", title="Transformers", content="Attention."), say("Saved 'Transformers'.")],
        }
    )
    # keep the researcher offline
    monkeypatch.setattr(research.RESEARCH_TOOLS[2], "func", lambda topic: "stub summary")
    tracer = Tracer(tmp_path, quiet=True)
    answer, _ = build_orchestrator(llm, tracer).run("Research transformers and save notes")
    assert "### Research" in answer and "Transformers use attention. Sources: wiki" in answer  # the researcher's own words
    assert "### Study material" in answer and "Saved 'Transformers'." in answer
    assert "Attention." in study.read_note("Transformers")
    calls = [(e["agent"], e["tool"]) for e in tracer.events if e["type"] == "tool_call"]
    assert calls == [
        ("Orchestrator", "ask_researcher"),
        ("Researcher", "wikipedia_summary"),
        ("Orchestrator", "ask_study_coach"),
        ("StudyCoach", "save_note"),
    ]


# ---- tool schemas (no network) ---------------------------------------------------


def test_all_tool_schemas_are_valid_for_ollama():
    import ollama

    for t in research.RESEARCH_TOOLS + study.STUDY_TOOLS:
        declared = ollama.Tool(type="function", function=ollama.Tool.Function(name=t.name, description=t.description, parameters=t.parameters))
        assert declared.function.name == t.name


def test_save_note_does_not_duplicate_title_heading():
    study.save_note("Attention", "# Attention\n\nQ, K, V.")
    text = study.read_note("Attention")
    assert text.count("# Attention") == 1 and "Q, K, V." in text


def test_every_agent_is_told_the_real_date(tmp_path):
    from datetime import date

    seen = []

    class Spy(FakeLLM):
        def generate(self, system, history, tools):
            seen.append(system)
            return say("hi")

    build_orchestrator(Spy({}), Tracer(tmp_path, quiet=True)).run("hello")
    assert f"{date.today():%Y-%m-%d}" in seen[0]


def test_plain_math_converts_latex_to_plain_text_but_leaves_prices_alone():
    src = r"Score = \[ \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V \], where $Q$ and \(d_{k}\) matter."
    out = study.plain_math(src)
    assert "\\" not in out and "$" not in out
    assert "`softmax((Q K^T)/(sqrt(d_k))) V`" in out and "where Q and d_k matter." in out
    assert study.plain_math("it costs $5 and $10") == "it costs $5 and $10"


def test_save_note_stores_plain_text_maths():
    study.save_note("Attention", r"Formula: \(\sqrt{d_k}\)")
    assert "sqrt(d_k)" in study.read_note("Attention") and "\\" not in study.read_note("Attention")


def test_student_sees_the_specialists_reply_verbatim_not_the_models_rewrite(tmp_path):
    from studymate.agents import COACH_PROMPT, ORCHESTRATOR_PROMPT

    reply = "| Day | Topic |\n| Fri | CNNs |\n28 days until the exam"
    llm = FakeLLM(
        {
            ORCHESTRATOR_PROMPT[:40]: [call("ask_study_coach", task="timetable"), say("Done")],
            COACH_PROMPT[:40]: [say(reply), say(reply)],  # first reply is tool-free, so the coach is reminded once
        }
    )
    answer, _ = build_orchestrator(llm, Tracer(tmp_path, quiet=True)).run("make me a timetable")
    assert answer == reply  # the model's own "Done" is discarded; nothing can be dropped or invented


def test_consecutive_replies_from_one_specialist_share_a_single_heading(tmp_path):
    from studymate.agents import COACH_PROMPT, ORCHESTRATOR_PROMPT, RESEARCHER_PROMPT

    llm = FakeLLM(
        {
            ORCHESTRATOR_PROMPT[:40]: [call("ask_researcher", task="r"), call("ask_study_coach", task="note"), call("ask_study_coach", task="quiz"), say("Done")],
            RESEARCHER_PROMPT[:40]: [call("wikipedia_summary", topic="x"), say("findings")],
            COACH_PROMPT[:40]: [say("note saved"), say("note saved"), say("the quiz")],
        }
    )
    answer, _ = build_orchestrator(llm, Tracer(tmp_path, quiet=True)).run("research, save a note, quiz me")
    assert answer.count("### Study material") == 1 and answer.index("note saved") < answer.index("the quiz")
    assert answer.startswith("### Research\n\nfindings")


def test_without_delegation_the_models_own_reply_is_used_and_nothing_leaks_between_requests(tmp_path):
    from studymate.agents import COACH_PROMPT, ORCHESTRATOR_PROMPT

    llm = FakeLLM(
        {
            ORCHESTRATOR_PROMPT[:40]: [call("ask_study_coach", task="x"), say("Done"), say("Hello! I can research and plan."), say("Hello! I can research and plan.")],
            COACH_PROMPT[:40]: [say("coach reply")],
        }
    )
    orchestrator = build_orchestrator(llm, Tracer(tmp_path, quiet=True))
    assert orchestrator.run("do something")[0] == "coach reply"
    assert orchestrator.run("hi")[0] == "Hello! I can research and plan."


def test_orchestrator_is_reminded_to_route_arithmetic_instead_of_calculating_itself(tmp_path):
    from studymate.agents import COACH_PROMPT, ORCHESTRATOR_PROMPT

    llm = FakeLLM(
        {
            ORCHESTRATOR_PROMPT[:40]: [say("1234 * 5678 = 6,999,952"), call("ask_study_coach", task="calculate 1234*5678"), say("Done")],
            COACH_PROMPT[:40]: [call("calculator", expression="1234*5678"), say("1234*5678 = 7006652")],
        }
    )
    answer, history = build_orchestrator(llm, Tracer(tmp_path, quiet=True)).run("What is 1234 * 5678?")
    reminders = [m[1] for m in history if m[0] == "user" and m[1].startswith("System reminder")]
    assert len(reminders) == 1 and reminders[0].endswith("What is 1234 * 5678?")  # the reminder restates the student's message
    assert answer == "1234*5678 = 7006652"  # the calculator's result, not the model's wrong mental arithmetic


def test_specialists_also_receive_the_students_exact_words_so_numbers_cannot_drift(tmp_path):
    from studymate.agents import COACH_PROMPT, ORCHESTRATOR_PROMPT

    seen = []

    class Spy(FakeLLM):
        def generate(self, system, history, tools):
            if COACH_PROMPT[:40] in system:
                seen.append(history[0][1])
            return super().generate(system, history, tools)

    llm = Spy(
        {
            ORCHESTRATOR_PROMPT[:40]: [call("ask_study_coach", task="Divide 7006652 by -2."), say("Done")],  # the model changed the sign
            COACH_PROMPT[:40]: [say("3503326"), say("3503326")],
        }
    )
    build_orchestrator(llm, Tracer(tmp_path, quiet=True)).run("Now divide that result by 2.")
    assert seen[0].startswith("Divide 7006652 by -2.")
    assert 'the student\'s own words were "Now divide that result by 2."' in seen[0] and "do it once" in seen[0]


def test_coach_guard_only_applies_to_tasks_that_need_a_tool():
    from studymate.agents import needs_coach_tool

    for needs in ("What is 1234 * 5678?", "Divide 7006652 by 2.", "Build a timetable for my exam", "Save a note called Attention", "how many days until Friday"):
        assert needs_coach_tool(needs), needs
    for plain in ("Write a 5-question quiz about attention.", "Make flashcards from this text about CNNs."):
        assert not needs_coach_tool(plain), plain  # ordinary writing does not pay for an extra model call


def test_coach_is_offered_only_the_tools_its_task_needs():
    from studymate.agents import coach_tool_allowed

    def offered(task):
        return sorted(t.name for t in study.STUDY_TOOLS if coach_tool_allowed(task, t.name))

    quiz = "Create a 5-question quiz with an answer key based on the research output."
    assert offered(quiz) == []  # the bug this prevents: a quiz task saving over the note with the same title
    reference = "\n\n[Reference only, this is the SAME request, so do it once: the student's own words were 'research, save a note, quiz me'.]"
    assert offered(quiz + reference) == []  # the appended student wording must not switch tools on
    assert offered('Save a short study note titled "Attention" based on the explanation.') == ["list_notes", "read_note", "save_note"]
    assert offered("Calculate 1234 * 5678") == ["calculator"]
    assert {"build_schedule", "days_until", "get_current_date", "save_note"} <= set(offered("Create a study timetable until 2026-11-20 and save it as a note"))


def test_coach_that_does_arithmetic_in_its_head_is_pushed_to_use_the_calculator(tmp_path):
    from studymate.agents import COACH_PROMPT, ORCHESTRATOR_PROMPT

    llm = FakeLLM(
        {
            ORCHESTRATOR_PROMPT[:40]: [call("ask_study_coach", task="Calculate 1234 * 5678."), say("Done")],
            COACH_PROMPT[:40]: [say("1234 * 5678 = 7,034,352"), call("calculator", expression="1234 * 5678"), say("1234 * 5678 = 7006652")],
        }
    )
    answer, _ = build_orchestrator(llm, Tracer(tmp_path, quiet=True)).run("What is 1234 * 5678?")
    assert answer == "1234 * 5678 = 7006652" and ("calculator", "1234 * 5678 = 7006652", True) in llm.seen_tool_results


def test_trailing_done_is_removed_from_a_direct_reply(tmp_path):
    from studymate.agents import ORCHESTRATOR_PROMPT

    llm = FakeLLM({ORCHESTRATOR_PROMPT[:40]: [say("Hi there!\n\nDone"), say("Hi there!\n\nDone")]})
    assert build_orchestrator(llm, Tracer(tmp_path, quiet=True)).run("hello")[0] == "Hi there!"


def test_prompts_forbid_latex_because_the_chat_page_cannot_render_it():
    from studymate.agents import COACH_PROMPT

    assert "Never use LaTeX" in COACH_PROMPT  # the writer of notes and quizzes; plain_math() is the code-level backstop
