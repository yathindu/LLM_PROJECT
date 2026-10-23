"""StudyMate CLI.  Usage:  python main.py            (interactive)
                         python main.py "your request"   (one-shot)"""
import sys

from rich.markdown import Markdown

from studymate.agents import build_orchestrator
from studymate.llm import LLMError, OllamaLLM
from studymate.trace import Tracer, console


def main() -> None:
    tracer = Tracer()
    orchestrator = build_orchestrator(OllamaLLM(), tracer)
    console.print(f"[dim]Trace log: {tracer.path}[/]")

    if len(sys.argv) > 1:
        try:
            answer, _ = orchestrator.run(" ".join(sys.argv[1:]))
        except LLMError as e:
            console.print(f"[red]Error:[/] {e}")
            sys.exit(1)
        console.print("\n[bold green]StudyMate:[/]")
        console.print(Markdown(answer))
        return

    console.print("[bold]StudyMate[/] - type a request, or 'quit' to exit.\n")
    history: list = []
    while True:
        try:
            task = console.input("[bold blue]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if task.lower() in ("quit", "exit", "q"):
            break
        if not task:
            continue
        try:
            answer, history = orchestrator.run(task, history)
        except Exception as e:  # keep the session alive on API errors
            console.print(f"[red]Error:[/] {type(e).__name__}: {e}")
            continue
        console.print("\n[bold green]StudyMate:[/]")
        console.print(Markdown(answer))
        console.print()


if __name__ == "__main__":
    main()
