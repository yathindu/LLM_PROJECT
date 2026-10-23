"""Thin wrapper around a local Ollama model, so the agent loop never touches the client library directly."""
import json
from dataclasses import dataclass, field

import httpx
import ollama

from . import config
from .tools import Tool


class LLMError(RuntimeError):
    """The model could not be reached or refused the request. Stops the run instead of being fed back to the agent."""


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    raw: object = None  # the assistant message, appended to the history verbatim


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or config.MODEL
        self.host = host or config.OLLAMA_HOST
        self.client = ollama.Client(host=self.host, timeout=config.LLM_TIMEOUT)

    # --- history helpers -------------------------------------------------
    def user_message(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def tool_results_message(self, results: list[tuple[str, str, bool]]) -> dict:
        # One history entry per agent step; it is expanded into one `tool` message per result when sent (see _messages).
        return {"role": "tool_results", "results": results}

    @staticmethod
    def _messages(system: str, history: list) -> list[dict]:
        messages = [{"role": "system", "content": system}]
        for item in history:
            if item["role"] == "tool_results":
                for name, output, ok in item["results"]:
                    messages.append({"role": "tool", "tool_name": name, "content": output if ok else f"ERROR: {output}"})
            else:
                messages.append(dict(item))  # copied, so shortening below never edits the real history
        return messages

    @staticmethod
    def _fit(messages: list[dict], budget_chars: int) -> list[dict]:
        """Shorten the oldest long tool outputs until the prompt fits the context window.
        Otherwise Ollama silently drops the start of the prompt and a small model can go off the rails."""
        total = sum(len(m.get("content") or "") for m in messages)
        for m in messages:  # oldest first
            if total <= budget_chars:
                break
            if m["role"] == "tool" and len(m["content"]) > 300:
                short = m["content"][:300] + " ...[older output shortened to fit the context window]"
                total -= len(m["content"]) - len(short)
                m["content"] = short
        return messages

    # --- generation ------------------------------------------------------
    def generate(self, system: str, history: list, tools: list[Tool]) -> LLMResponse:
        declared = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}} for t in tools]
        # roughly 3 characters per token; keep room for the tool declarations and the reply
        budget = (config.NUM_CTX - config.MAX_NEW_TOKENS) * 3 - len(json.dumps(declared))
        try:
            resp = self.client.chat(
                model=self.model,
                messages=self._fit(self._messages(system, history), budget),
                tools=declared or None,
                think=config.THINK,
                options={"temperature": 0.3, "num_ctx": config.NUM_CTX, "num_predict": config.MAX_NEW_TOKENS},
            )
        except httpx.TimeoutException as e:
            raise LLMError(f"Ollama did not answer within {config.LLM_TIMEOUT} s (the model may be too slow for this machine or stuck).") from e
        except (ConnectionError, httpx.TransportError) as e:
            raise LLMError(
                f"Cannot reach Ollama at {self.host} ({type(e).__name__}). Install it from https://ollama.com, "
                f"make sure it is running (`ollama serve`), and pull the model with `ollama pull {self.model}`."
            ) from e
        except ollama.ResponseError as e:
            hint = f" Pull it first with `ollama pull {self.model}`." if e.status_code == 404 else ""
            raise LLMError(f"Ollama request failed ({e.status_code}): {e.error}.{hint}") from e

        msg = resp.message
        calls = [ToolCall(c.function.name, dict(c.function.arguments or {})) for c in (msg.tool_calls or [])]
        raw = {"role": "assistant", "content": msg.content or ""}
        if calls:
            raw["tool_calls"] = [{"function": {"name": c.name, "arguments": c.args}} for c in calls]
        return LLMResponse(text=msg.content or "", calls=calls, raw=raw)
