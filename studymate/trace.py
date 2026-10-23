"""Tool-call trace: pretty console output (for the demo) plus a JSONL log file."""
import json
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console

from . import config

console = Console()


class Tracer:
    def __init__(self, log_dir: Path | None = None, quiet: bool = False, on_event=None):
        self.quiet = quiet
        self.on_event = on_event  # called with every event as it happens (used for live streaming)
        log_dir = Path(log_dir or config.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"trace-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
        self.events: list[dict] = []

    def _write(self, event: dict) -> None:
        event["t"] = round(time.time(), 3)
        self.events.append(event)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        if self.on_event:
            self.on_event(event)

    def step(self, agent: str, depth: int, n: int) -> None:
        self._write({"type": "step", "agent": agent, "depth": depth, "n": n})
        if not self.quiet:
            console.print(f"{'    ' * depth}[bold cyan]● {agent}[/] [dim]reasoning step {n}[/]")

    def tool_call(self, agent: str, depth: int, name: str, args: dict) -> None:
        self._write({"type": "tool_call", "agent": agent, "depth": depth, "tool": name, "args": args})
        if not self.quiet:
            shown = json.dumps(args, ensure_ascii=False)
            if len(shown) > 160:
                shown = shown[:157] + "..."
            console.print(f"{'    ' * depth}  [yellow]→ CALL[/] [bold]{name}[/]({shown})", highlight=False)

    def tool_result(self, agent: str, depth: int, name: str, result: str, ok: bool) -> None:
        self._write({"type": "tool_result", "agent": agent, "depth": depth, "tool": name, "ok": ok, "result": result})
        if not self.quiet:
            flat = " ".join(result.split())
            if len(flat) > 140:
                flat = flat[:137] + "..."
            colour = "green" if ok else "red"
            console.print(f"{'    ' * depth}  [{colour}]← RESULT[/] [dim]{flat}[/]", highlight=False)

    def final(self, agent: str, depth: int, text: str) -> None:
        self._write({"type": "final", "agent": agent, "depth": depth, "text": text})
