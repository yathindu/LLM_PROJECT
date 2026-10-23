"""The agent loop: model -> tool calls -> tool results -> model, until the model answers in plain text."""
from . import config
from .llm import LLMError
from .tools import Tool
from .trace import Tracer


NUDGE = "You answered without calling a tool. Call one of your tools now to get real information, then answer only from its results."


class Agent:
    def __init__(self, name: str, system: str, tools: list[Tool], llm, tracer: Tracer, depth: int = 0, require_tool=False, nudge: str = NUDGE, tool_filter=None):
        # grounding guard: the first answer must follow at least one tool call. True = always; a function = only for
        # tasks it accepts (e.g. those involving arithmetic), so ordinary replies do not pay for an extra model call.
        self.require_tool = require_tool
        self.nudge = nudge
        # optional function(task, tool_name) -> bool: offer the model only the tools this task needs. A tool that is not
        # offered cannot be called, which is stronger than asking a small model in the prompt not to use it.
        self.tool_filter = tool_filter
        self.name = name
        self.system = system
        self.tools = {t.name: t for t in tools}
        self.llm = llm
        self.tracer = tracer
        self.depth = depth

    def run(self, task: str, history: list | None = None) -> tuple[str, list]:
        """Run one task. Pass `history` to continue a conversation; returns (answer, updated history)."""
        history = list(history or [])
        history.append(self.llm.user_message(task))
        used_tool = nudged = False
        must_use_tool = self.require_tool(task) if callable(self.require_tool) else self.require_tool
        offered = {n: tool for n, tool in self.tools.items() if self.tool_filter is None or self.tool_filter(task, n)}

        for step in range(1, config.MAX_STEPS + 1):
            self.tracer.step(self.name, self.depth, step)
            resp = self.llm.generate(self.system, history, list(offered.values()))
            history.append(resp.raw)

            if not resp.calls:  # plain-text answer: the loop is finished
                if must_use_tool and not used_tool and not nudged and step < config.MAX_STEPS:
                    nudged = True  # small models sometimes answer from memory; push back once, then accept the reply
                    history.append(self.llm.user_message(self.nudge.replace("{task}", task)))
                    continue
                self.tracer.final(self.name, self.depth, resp.text)
                return resp.text, history
            used_tool = True

            results = []
            for call in resp.calls:
                self.tracer.tool_call(self.name, self.depth, call.name, call.args)
                output, ok = self._execute(call.name, call.args, offered)
                self.tracer.tool_result(self.name, self.depth, call.name, output, ok)
                results.append((call.name, output, ok))
            history.append(self.llm.tool_results_message(results))

        msg = f"[{self.name}] stopped after {config.MAX_STEPS} steps without a final answer."
        self.tracer.final(self.name, self.depth, msg)
        return msg, history

    def _execute(self, name: str, args: dict, offered: dict) -> tuple[str, bool]:
        tool = offered.get(name)
        if tool is None:
            return f"Unknown tool '{name}'. Available: {', '.join(offered) or 'none'}", False
        try:
            out = tool.run(**args)
        except TypeError as e:  # wrong/missing arguments from the model
            return f"Bad arguments for {name}: {e}", False
        except LLMError:  # a sub-agent could not reach the model: stop the whole run, retrying the delegation would not help
            raise
        except Exception as e:  # tool failures go back to the model so it can recover
            return f"{type(e).__name__}: {e}", False
        if len(out) > config.MAX_TOOL_OUTPUT:
            out = out[: config.MAX_TOOL_OUTPUT] + "\n...[truncated]"
        return out, True
