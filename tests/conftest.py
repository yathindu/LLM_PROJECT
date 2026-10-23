import pytest

from studymate import config


@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    """Every test writes notes and logs to its own temporary folder."""
    monkeypatch.setattr(config, "NOTES_DIR", tmp_path / "notes")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
