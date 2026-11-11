from __future__ import annotations

import json

import pytest

from core.llm import LLMBackend


def test_ollama_backend_only_targets_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[tuple[str, int, float]] = []

    class FakeResponse:
        status = 200

        def read(self) -> bytes:
            return json.dumps(
                {"message": {"content": json.dumps({"ok": True})}}
            ).encode()

    class FakeConnection:
        def __init__(self, host: str, port: int, timeout: float) -> None:
            connections.append((host, port, timeout))

        def request(self, *_args: object, **_kwargs: object) -> None:
            pass

        def getresponse(self) -> FakeResponse:
            return FakeResponse()

        def close(self) -> None:
            pass

    monkeypatch.setattr("core.llm.http.client.HTTPConnection", FakeConnection)

    assert LLMBackend(timeout=5).generate_json("system", "user") == {"ok": True}
    assert connections == [("127.0.0.1", 1234, 5)]
