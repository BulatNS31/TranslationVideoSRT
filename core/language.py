"""Определение и нормализация языковых кодов."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .errors import TranslationError

LANGUAGE_ALIASES = {"zh": "zh-cn", "he": "iw"}


def normalize_language_code(code: str | None) -> str | None:
	"""Нормализует языковые коды, используемые определителем и переводчиком."""
	if code is None:
		return None
	normalized = code.strip().lower().replace("_", "-")
	if not normalized:
		return None
	return LANGUAGE_ALIASES.get(normalized, normalized)


async def detect_language(
	text: str,
	whisper_language: str | None = None,
	*,
	detector: Any | None = None,
) -> str:
	"""Определяет язык по Whisper либо офлайн-библиотекой lingua."""
	if whisper_language:
		normalized = normalize_language_code(whisper_language)
		if normalized:
			return normalized
	sample = text.strip()
	if not sample:
		raise TranslationError("Невозможно определить язык пустого текста")
	try:
		language_detector: Any = detector
		if language_detector is None:
			lingua = import_module("lingua")
			language_detector = lingua.LanguageDetectorBuilder.from_all_languages().build()
		detected = language_detector.detect_language_of(sample)
		if detected is None:
			raise TranslationError("Lingua не смогла определить язык текста")
		code = detected.iso_code_639_1
		if code is None:
			code = detected.iso_code_639_3
		normalized = normalize_language_code(code.name.lower())
	except TranslationError:
		raise
	except Exception as error:
		raise TranslationError(f"Не удалось определить язык текста: {error}") from error
	if not normalized:
		raise TranslationError("Lingua не вернула языковой код")
	return normalized
