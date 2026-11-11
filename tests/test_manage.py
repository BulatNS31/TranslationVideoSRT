from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import manage


def test_core_command_runs_default_pipeline(monkeypatch) -> None:
    class Result:
        video_path = "source/clip.mp4"

    calls: list[manage.PipelineConfig] = []

    def fake_pipeline(config: manage.PipelineConfig) -> Result:
        calls.append(config)
        return Result()

    monkeypatch.setattr(manage, "run_pipeline", fake_pipeline)

    assert manage.main(["core"]) == 0
    assert len(calls) == 1
    assert calls[0].audio_path == Path("source")
    assert calls[0].output_dir == Path("source")


def test_voice_command_delegates_to_script(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class Completed:
        returncode = 7

    entrypoint = tmp_path / "voice.py"
    entrypoint.touch()
    monkeypatch.setattr(manage, "VOICE_ENTRYPOINT", entrypoint)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> Completed:
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr(manage.subprocess, "run", fake_run)

    assert manage.main(["voice"]) == 7
    assert calls[0][0] == [sys.executable, str(entrypoint)]
    assert calls[0][1]["cwd"] == manage.PROJECT_ROOT


def test_tests_command_runs_pytest(monkeypatch) -> None:
    class Completed:
        returncode = 0

    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> Completed:
        calls.append(command)
        return Completed()

    monkeypatch.setattr(manage.subprocess, "run", fake_run)

    assert manage.main(["tests"]) == 0
    assert calls == [[sys.executable, "-m", "pytest", "tests"]]


def test_voice_command_reports_missing_script(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(manage, "VOICE_ENTRYPOINT", tmp_path / "missing.py")

    assert manage.main(["voice"]) == 2
    assert "Не найден файл приложения" in capsys.readouterr().err
