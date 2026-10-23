"""Agent definitions and the orchestrator that delegates to them."""
import re
from datetime import date

from .agent import Agent
from .tools import Tool
from .tools.research import RESEARCH_TOOLS
from .tools.study import STUDY_TOOLS, plain_math

RESEARCHER_PROMPT = """You are the Researcher, a specialist in finding and verifying information.
You have no knowledge of your own to rely on: every answer must come from tool results.
Tools: web_search, fetch_webpage, wikipedia_summary.
RULES (follow them exactly):
1. Start by calling wikipedia_summary (definitions, background) and/or web_search (current or specific facts). Never answer before calling a tool.
2. For detailed claims, read one promising result with fetch_webpage. Use at most three tool calls in total.
3. Use only what the tools returned. Never invent facts, titles or URLs, and cite only URLs that appeared in tool results.
4. If the tools return little or conflicting information, say so.
5. Return a concise, structured answer and end with a 'Sources:' line listing the URLs you used."""

COACH_PROMPT = """You are the Study Coach, a specialist in turning material into learning aids and plans.
Your tools: save_note, list_notes, read_note, calculator, get_current_date, days_until, build_schedule.
You are offered only the tools the current task needs; with no tools offered, just write the answer.
Rules:
- You may be given research text to work from; base summaries, flashcards and quizzes only on it plus well-established knowledge.
- For any timetable, call get_current_date/days_until first, then build_schedule; do not do date maths in your head.
- If a date or day count written in your task disagrees with what your date tools return, the tools are right.
- Your task ends with the student's exact message. If a number in it differs from the task text, the student's number is right.
- In a timetable reply, state the number of days exactly as days_until returned it.
- For arithmetic, use calculator.
- Call only the tools the task needs: build_schedule only when asked for a timetable, get_current_date and days_until only for dates.
- To create or save a note, call save_note directly (never read or list notes first), then confirm the exact note title you used.
- Quizzes: give questions first, then an answer key. Be concrete and brief.
- Your reply must contain the full content you produced (the whole timetable, quiz, or note text), not just a confirmation that you saved it.
- Write mathematics in plain text (for example softmax(Q K^T / sqrt(d_k))). Never use LaTeX or dollar signs."""

ORCHESTRATOR_PROMPT = """You are StudyMate, an assistant that helps university students research topics and study effectively.
You have no knowledge of your own to rely on: you only route work to two specialist tools and report what they return.
ROUTING RULES (follow them exactly):
1. The student asks to explain, look up, research, compare, or find sources: call ask_researcher FIRST. Never answer these from your own memory.
2. The student asks for a note, summary, flashcards, quiz, timetable, schedule, calculation, or to read/list notes: call ask_study_coach.
   It cannot browse, so put every piece of text it needs inside the task.
3. Multi-step requests: do the steps in order, one specialist call per step, and pass each result into the next task
   (for example ask_researcher first, then ask_study_coach with the research text). Specialists cannot see each other or this conversation.
   When you write a task, copy every number from the student's message exactly (same value, same sign: "by 2" never becomes "by -2").
4. Only greetings, thanks, and questions about what you can do are answered without calling a tool.
   Follow-up questions about results already in this conversation can be answered from it.
5. Do only what was asked: never save notes, make quizzes or build schedules unless the student requested them.
6. When the specialists have finished, reply with the single word "Done". The system shows their results to the student
   itself, so never repeat, summarise or rewrite them."""


ORCHESTRATOR_NUDGE = (
    "System reminder (never mention it): requests to explain, look up, research, calculate, make a note, a quiz or a timetable "
    "must be routed with ask_researcher or ask_study_coach; never calculate or answer them yourself. "
    "Greetings, thanks and questions about earlier results in this conversation may be answered directly. "
    "Now answer the student's last message again: {task}"
)


COACH_NUDGE = (
    "You answered without using a tool. If the task involves arithmetic, dates or a timetable, or saving, reading or listing a note, "
    "you must call the matching tool now (calculator, get_current_date, days_until, build_schedule, save_note, read_note, list_notes) "
    "and answer from its result; never calculate in your head or claim to have saved something. Task again: {task}"
)
# The Study Coach is offered a tool only when its task calls for it. A small model that is shown every tool will use some
# of them at random (saving a quiz over a note, building a schedule while writing one), and a tool it is not shown
# cannot be called at all.
_NOTES = re.compile(r"\bsave\b|\bnotes?\b")
_DATES = re.compile(r"\bdates?\b|\bdays?\b|timetable|schedule|deadline|exam|\buntil\b|\bweeks?\b")
_MATHS = re.compile(r"calculat|multipl|divid|\btimes\b|\bsum\b|percent|average|\d\s*[-+*/x×÷^]\s*\d")
COACH_TOOL_GROUPS = {
    "save_note": _NOTES, "read_note": _NOTES, "list_notes": _NOTES,
    "get_current_date": _DATES, "days_until": _DATES, "build_schedule": _DATES,
    "calculator": _MATHS,
}


def _own_wording(task: str) -> str:
    return task.split("\n\n[Reference only", 1)[0].lower()  # not the student's message that is appended for reference


def coach_tool_allowed(task: str, tool_name: str) -> bool:
    return bool(COACH_TOOL_GROUPS[tool_name].search(_own_wording(task)))


def needs_coach_tool(task: str) -> bool:
    """True when the Study Coach's task involves something a tool must do (arithmetic, dates, notes)."""
    return any(group.search(_own_wording(task)) for group in COACH_TOOL_GROUPS.values())


class OrchestratorAgent(Agent):
    """Routes the work. What the student sees is assembled by code from the specialists' own replies, because a small
    local model cannot be trusted to copy long results faithfully (it drops content or invents numbers)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs: list[tuple[str, str]] = []  # (specialist, reply) for the current request, in call order
        self.request: list[str] = [""]  # the student's current message; shared with the delegate tools

    def run(self, task: str, history: list | None = None) -> tuple[str, list]:
        self.outputs.clear()
        self.request[0] = task
        text, history = super().run(task, history)
        if not self.outputs:  # nothing was delegated (greeting, follow-up): the model's own reply is the answer
            return plain_math(re.sub(r"\s*\bDone\.?\s*$", "", text)), history
        sections: list[tuple[str, list[str]]] = []  # consecutive replies from the same specialist share one heading
        for label, reply in self.outputs:
            if sections and sections[-1][0] == label:
                sections[-1][1].append(reply)
            else:
                sections.append((label, [reply]))
        if len(sections) == 1:
            return plain_math("\n\n".join(sections[0][1])), history
        return plain_math("\n\n".join(f"### {label}\n\n" + "\n\n".join(replies) for label, replies in sections)), history


def _delegate_tool(name: str, description: str, agent: Agent, label: str, sink: list, request: list) -> Tool:
    def run(task: str) -> str:
        # A small model paraphrases while delegating and can change numbers ("by 2" became "by -2"), so the specialist
        # also gets the student's exact words.
        task = (
            f"{task}\n\n[Reference only, this is the SAME request, so do it once: the student's own words were \"{request[0]}\". "
            "If a number in those words differs from the task above, use the student's number.]"
        )
        answer, _ = agent.run(task)  # sub-agents are stateless: each delegation starts fresh
        sink.append((label, answer))
        return answer

    return Tool(
        name,
        description,
        {
            "type": "object",
            "properties": {"task": {"type": "string", "description": "Complete, self-contained instructions for the specialist"}},
            "required": ["task"],
        },
        run,
    )


def _with_date(prompt: str) -> str:
    # Models do not know today's date and will invent one (and pass it on to other agents), so state it.
    return f"{prompt}\nToday's date is {date.today():%A %Y-%m-%d}. Never assume or invent any other 'today'."


def build_orchestrator(llm, tracer) -> Agent:
    researcher = Agent("Researcher", RESEARCHER_PROMPT, RESEARCH_TOOLS, llm, tracer, depth=1, require_tool=True)
    coach = Agent("StudyCoach", COACH_PROMPT, STUDY_TOOLS, llm, tracer, depth=1, require_tool=needs_coach_tool, nudge=COACH_NUDGE, tool_filter=coach_tool_allowed)
    outputs: list[tuple[str, str]] = []
    request = [""]
    delegates = [
        _delegate_tool(
            "ask_researcher",
            "Delegate a research question to the Researcher agent (web search, page reading, Wikipedia). Returns findings with sources.",
            researcher,
            "Research",
            outputs,
            request,
        ),
        _delegate_tool(
            "ask_study_coach",
            "Delegate study-material work to the Study Coach agent: summaries, flashcards, quizzes, schedules, calculations, saving/reading notes.",
            coach,
            "Study material",
            outputs,
            request,
        ),
    ]
    orchestrator = OrchestratorAgent(
        "Orchestrator", _with_date(ORCHESTRATOR_PROMPT), delegates, llm, tracer, depth=0, require_tool=True, nudge=ORCHESTRATOR_NUDGE
    )
    orchestrator.outputs, orchestrator.request = outputs, request  # the delegate tools share these same lists
    return orchestrator
