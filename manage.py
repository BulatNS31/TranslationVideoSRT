"""Командный диспетчер приложений репозитория."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from core import PipelineConfig, run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent
VOICE_ENTRYPOINT = PROJECT_ROOT / "voice" / "core.py"


def _run_voice() -> int:
	"""Запускает legacy-приложение voice отдельным Python-процессом."""
	if not VOICE_ENTRYPOINT.is_file():
		print(f"Не найден файл приложения: {VOICE_ENTRYPOINT}", file=sys.stderr)
		return 2
	completed = subprocess.run(
		[sys.executable, str(VOICE_ENTRYPOINT)],
		cwd=PROJECT_ROOT,
		check=False,
	)
	return completed.returncode


def _run_tests() -> int:
	"""Запускает pytest для тестов репозитория."""
	completed = subprocess.run(
		[sys.executable, "-m", "pytest", "tests"],
		cwd=PROJECT_ROOT,
		check=False,
	)
	return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
	"""Разбирает команду и запускает выбранную часть проекта."""
	parser = argparse.ArgumentParser(description="Запуск частей проекта")
	parser.add_argument(
		"command",
		choices=("core", "voice", "tests"),
		help="core — обработать MP3 в source; voice — запустить voice/core.py; "
		"tests — запустить pytest",
	)
	args = parser.parse_args(argv)

	if args.command == "core":
		result = run_pipeline(PipelineConfig())
		print(f"Видео сохранено: {result.video_path}")
		return 0
	if args.command == "voice":
		return _run_voice()
	return _run_tests()


if __name__ == "__main__":
	raise SystemExit(main())
