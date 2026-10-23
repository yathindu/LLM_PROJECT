import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))  # fits a 4 GB GPU; raise it if you have more VRAM
THINK = False  # we use a non-thinking (instruct) model; this also switches reasoning off on hybrid models
LLM_TIMEOUT = 180  # seconds for one model call
MAX_NEW_TOKENS = 1200  # cap on one reply, so a rambling or looping model cannot run for minutes
NOTES_DIR = Path(os.getenv("STUDYMATE_NOTES_DIR", ROOT / "notes"))
LOG_DIR = Path(os.getenv("STUDYMATE_LOG_DIR", ROOT / "logs"))
MAX_STEPS = 8  # hard cap on loop iterations per agent run
MAX_TOOL_OUTPUT = 2500  # characters returned to the model from any single tool call (small local context window)
