"""Tools for the Study Coach agent: notes, calculator, dates and scheduling."""
import ast
import math
import operator as op
import re
from datetime import date, datetime, timedelta

from .. import config
from . import Tool

def plain_math(text: str) -> str:
    """Turn LaTeX that a small model sometimes writes into plain text, because the chat page and notes do not render it."""
    text = re.sub(r"\\\[(.*?)\\\]", lambda m: "`" + m.group(1).strip() + "`", text, flags=re.S)  # display maths -> inline code
    text = re.sub(r"\\\((.*?)\\\)", r"\1", text, flags=re.S)
    text = re.sub(r"\$([^$\n]*[\\^_{}][^$\n]*|[A-Za-z0-9_]{1,10})\$", r"\1", text)  # $...$ around maths-like text only
    for _ in range(3):  # nested \text{\mathbf{...}}
        text = re.sub(r"\\(?:text|textbf|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", text)  # before \frac, which may contain it
    text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", text)
    text = re.sub(r"([_^])\{([^{}]*)\}", r"\1\2", text)
    for latex, plain in ((r"\cdot", "*"), (r"\times", "x"), (r"\left", ""), (r"\right", ""), (r"\,", " ")):
        text = text.replace(latex, plain)
    return re.sub(r"\\([A-Za-z]+)", r"\1", text)  # leftover commands such as \alpha -> alpha


# ---- notes ---------------------------------------------------------------


def note_path(title: str):
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if not slug:
        raise ValueError("Note title must contain letters or digits.")
    return config.NOTES_DIR / f"{slug}.md"


def save_note(title: str, content: str) -> str:
    config.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    path = note_path(title)
    body = plain_math(content.strip())
    if body.lower().startswith(f"# {title}".lower()):  # the model often repeats the title as a heading
        body = body.split("\n", 1)[1].strip() if "\n" in body else ""
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return f"Saved note '{title}' ({len(content)} characters) to {path.name}."


def list_notes() -> str:
    if not config.NOTES_DIR.exists():
        return "No notes saved yet."
    names = sorted(p.stem for p in config.NOTES_DIR.glob("*.md"))
    return "Saved notes: " + ", ".join(names) if names else "No notes saved yet."


def read_note(title: str) -> str:
    path = note_path(title)
    if not path.exists():
        return f"No note called '{title}'. Use list_notes to see what exists."
    return path.read_text(encoding="utf-8")


# ---- calculator ----------------------------------------------------------

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log10": math.log10, "sin": math.sin, "cos": math.cos, "abs": abs, "round": round}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return -_eval(node.operand) if isinstance(node.op, ast.USub) else _eval(node.operand)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in ("pi", "e"):
        return getattr(math, node.id)
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    """Safe arithmetic: parsed with `ast`, never `eval`."""
    result = _eval(ast.parse(expression, mode="eval").body)
    return f"{expression} = {result}"


# ---- dates and scheduling ------------------------------------------------


def get_current_date() -> str:
    return datetime.now().strftime("%A, %Y-%m-%d")


def days_until(target_date: str) -> str:
    target = date.fromisoformat(target_date)
    delta = (target - date.today()).days
    return f"{delta} days until {target_date}" if delta >= 0 else f"{target_date} was {-delta} days ago"


def build_schedule(topics: list, start_date: str, end_date: str, hours_per_day: float = 2.0) -> str:
    """Spread topics evenly across the available days; the final day is kept for revision."""
    topics = [str(t) for t in topics]
    if not topics:
        raise ValueError("Provide at least one topic.")
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    n_days = (end - start).days + 1
    if n_days < 1:
        raise ValueError("end_date must not be before start_date.")
    study_days = max(1, n_days - 1) if n_days > 1 else 1
    rows = []
    for i in range(n_days):
        day = start + timedelta(days=i)
        if n_days > 1 and i == n_days - 1:
            plan = "Revision of all topics + practice questions"
        else:
            lo = i * len(topics) // study_days
            hi = max(lo + 1, (i + 1) * len(topics) // study_days)
            plan = ", ".join(topics[lo:hi])
        rows.append(f"{day:%a %Y-%m-%d} ({hours_per_day:g}h): {plan}")
    return "\n".join(rows)


STUDY_TOOLS = [
    Tool(
        "save_note",
        "Save a study note (markdown) under a title. Overwrites any note with the same title.",
        {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Note title"},
                "content": {"type": "string", "description": "Markdown body of the note"},
            },
            "required": ["title", "content"],
        },
        save_note,
    ),
    Tool("list_notes", "List the titles of all saved notes.", {"type": "object", "properties": {}}, list_notes),
    Tool(
        "read_note",
        "Read a saved note by title.",
        {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
        read_note,
    ),
    Tool(
        "calculator",
        "Evaluate an arithmetic expression. Supports + - * / ** %, sqrt, log, log10, sin, cos, abs, round, pi, e.",
        {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        calculator,
    ),
    Tool("get_current_date", "Get today's date and weekday.", {"type": "object", "properties": {}}, get_current_date),
    Tool(
        "days_until",
        "Number of days from today until a date.",
        {
            "type": "object",
            "properties": {"target_date": {"type": "string", "description": "ISO date YYYY-MM-DD"}},
            "required": ["target_date"],
        },
        days_until,
    ),
    Tool(
        "build_schedule",
        "Build a day-by-day study timetable that spreads topics across a date range and reserves the last day for revision.",
        {
            "type": "object",
            "properties": {
                "topics": {"type": "array", "items": {"type": "string"}, "description": "Topics to cover, in order"},
                "start_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "ISO date YYYY-MM-DD (e.g. the exam date)"},
                "hours_per_day": {"type": "number", "description": "Study hours per day (default 2)"},
            },
            "required": ["topics", "start_date", "end_date"],
        },
        build_schedule,
    ),
]
