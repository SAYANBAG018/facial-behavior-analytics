"""Checks that Gemini requests have a timeout and no hidden retries."""

import pytest

from backend import assistant


def test_client_is_created_with_timeout_and_no_hidden_retries(monkeypatch):
    captured = {}

    class FakeModels:
        def generate_content(self, model, contents):
            raise assistant.genai.errors.ServerError(
                503, {"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}}, None
            )

    class FakeClient:
        def __init__(self, api_key, http_options):
            captured["api_key"] = api_key
            captured["http_options"] = http_options
            self.models = FakeModels()

        def close(self):
            pass

    monkeypatch.setattr(assistant.genai, "Client", FakeClient)

    with pytest.raises(Exception):
        assistant._generate_response("test-key", "test-model", "hello")

    options = captured["http_options"]
    assert captured["api_key"] == "test-key"
    assert options.timeout == assistant.REQUEST_TIMEOUT_MS
    assert options.retry_options.attempts == 1
