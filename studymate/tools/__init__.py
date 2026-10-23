"""Tool definition shared by all agents."""
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments object
    func: Callable[..., str]

    def run(self, **kwargs) -> str:
        return str(self.func(**kwargs))
