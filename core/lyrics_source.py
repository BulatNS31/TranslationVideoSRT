"""Получение строк из пользовательских субтитров или локальной Whisper-модели."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .errors import LyricsNotFoundError
from .models import LyricLine

logger = logging.getLogger(__name__)


def _subtitle_seconds(subrip_time: Any) -> float:
	"""Преобразует время pysrt в секунды."""
	return float(subrip_time.ordinal) / 1000.0


def parse_srt(srt_path: Path) -> list[LyricLine]:
	"""Читает пользовательский SRT-файл и возвращает строки с таймингами."""
	import pysrt

	path = Path(srt_path)
	if not path.is_file():
		raise LyricsNotFoundError(f"Файл субтитров не найден: {path}")
	subtitles = pysrt.open(str(path), encoding="utf-8")
	lines = [
		LyricLine(
			start=_subtitle_seconds(item.start),
			end=_subtitle_seconds(item.end),
			text=item.text.replace("\r", "").strip(),
		)
		for item in subtitles
		if item.text.strip()
	]
	if not lines:
		raise LyricsNotFoundError(f"В файле субтитров нет строк: {path}")
	return lines


@lru_cache(maxsize=3)
def _load_whisper_model(model_name: str) -> Any:
	"""Лениво загружает и кеширует указанную локальную модель Whisper."""
	import whisper

	logger.info("Загрузка модели Whisper %s", model_name)
	return whisper.load_model(model_name)


def transcribe_audio(
	audio_path: Path,
	model_name: str = "small",
	*,
	model_loader: Callable[[str], Any] | None = None,
) -> tuple[list[LyricLine], str | None]:
	"""Транскрибирует аудио локальной Whisper-моделью с таймингами сегментов."""
	path = Path(audio_path)
	if not path.is_file():
		raise LyricsNotFoundError(f"Аудиофайл не найден: {path}")
	model = (model_loader or _load_whisper_model)(model_name)
	result = model.transcribe(str(path), fp16=False)
	lines = [
		LyricLine(
			start=float(segment["start"]),
			end=float(segment["end"]),
			text=str(segment["text"]).strip(),
		)
		for segment in result.get("segments", ())
		if str(segment.get("text", "")).strip()
	]
	if not lines and str(result.get("text", "")).strip():
		lines.append(LyricLine(start=0.0, end=0.0, text=str(result["text"]).strip()))
	if not lines:
		raise LyricsNotFoundError(f"Whisper не распознал текст: {path}")
	language = result.get("language")
	return lines, str(language) if language else None


def get_lyrics(
	audio_path: Path,
	srt_path: Path | None = None,
	model_name: str = "small",
	*,
	model_loader: Callable[[str], Any] | None = None,
) -> tuple[list[LyricLine], str | None]:
	"""Получает текст из SRT, если он задан, иначе запускает локальную транскрипцию."""
	if srt_path is not None:
		return parse_srt(srt_path), None
	return transcribe_audio(audio_path, model_name, model_loader=model_loader)
