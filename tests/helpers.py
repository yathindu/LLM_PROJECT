"""Shared test helper: a scripted fake model, so agent loops can be tested without the network."""
from studymate.llm import LLMResponse, ToolCall


class FakeLLM:
    """Replays scripted responses per agent system prompt, so loops can be tested without the network."""

    def __init__(self, scripts: dict[str, list[LLMResponse]]):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.seen_tool_results: list = []

    def user_message(self, text):
        return ("user", text)

    def tool_results_message(self, results):
        self.seen_tool_results.extend(results)
        return ("tool", results)

    def generate(self, system, history, tools):
        for key, queue in self.scripts.items():
            if key in system:
                return queue.pop(0)
        raise AssertionError("no script for system prompt")


def call(name, **args):
    return LLMResponse(text="", calls=[ToolCall(name, args)], raw="model-call")


def say(text):
    return LLMResponse(text=text, raw="model-text")
