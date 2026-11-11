"""Исключения, которые может возвращать пайплайн."""


class LyricsNotFoundError(Exception):
	"""Текст песни отсутствует или не содержит распознаваемых строк."""


class TranslationError(Exception):
	"""Не удалось определить язык или перевести текст."""


class RenderError(Exception):
	"""Не удалось собрать видео с помощью FFmpeg."""


class ConfigError(Exception):
	"""Конфигурация или входные файлы пайплайна некорректны."""
