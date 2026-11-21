"""Tests for AI assistant configuration loading."""

import pytest

from backend import assistant


def test_settings_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)

    key, primary, fallback = assistant._settings()

    assert key == "test-key"
    assert primary == "gemini-3.7-flash"
    assert fallback == "gemini-3.5-flash-lite"


def test_settings_uses_model_overrides(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "model-a")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "model-b")

    _, primary, fallback = assistant._settings()

    assert (primary, fallback) == ("model-a", "model-b")


def test_settings_fails_clearly_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match=r"\.env"):
        assistant._settings()


def test_dotenv_values_are_read_from_file(tmp_path):
    from dotenv import dotenv_values

    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from-file\nGEMINI_MODEL=model-x\n")

    values = dotenv_values(env_file)

    assert values["GEMINI_API_KEY"] == "from-file"
    assert values["GEMINI_MODEL"] == "model-x"


def test_shell_variables_take_priority_over_dotenv(tmp_path, monkeypatch):
    import os
    from dotenv import load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_MODEL=from-file\n")

    # Both values are set beforehand, so loading the file changes nothing.
    monkeypatch.setenv("GEMINI_MODEL", "from-shell")
    monkeypatch.setenv("GEMINI_API_KEY", "shell-key")

    load_dotenv(env_file, override=False)

    assert os.environ["GEMINI_MODEL"] == "from-shell"
    assert os.environ["GEMINI_API_KEY"] == "shell-key"
