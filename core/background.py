"""Выбор категории и файла-фона."""

from __future__ import annotations

from pathlib import Path

from .errors import ConfigError
from .models import BackgroundChoice

BACKGROUND_CATEGORIES = ("night_city", "rain", "sea", "car", "nature", "default")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def category_for_tempo(tempo: float) -> str:
	"""Выбирает категорию по темпу, если LLM не предложила свою."""
	if tempo > 130:
		return "car"
	if tempo < 80:
		return "rain"
	return "night_city"


def choose_background(
	category: str,
	background_dir: Path,
	custom_image: Path | None = None,
) -> BackgroundChoice:
	"""Выбирает пользовательское изображение или файл с категорией в имени."""
	if custom_image is not None:
		path = Path(custom_image)
		if not path.is_file():
			raise ConfigError(f"Фоновое изображение не найдено: {path}")
		return BackgroundChoice(category, path)

	directory = Path(background_dir)
	if not directory.is_dir():
		raise ConfigError(f"Папка с фонами не найдена: {directory}")
	images = sorted(
		path
		for path in directory.iterdir()
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
	)
	if category not in BACKGROUND_CATEGORIES:
		category = "default"
	for candidate_category in (category, "default"):
		matching = [
			path
			for path in images
			if path.name.lower().startswith(candidate_category)
		]
		if matching:
			return BackgroundChoice(candidate_category, matching[0])
	if images:
		return BackgroundChoice("default", images[0])
	raise ConfigError(f"В папке нет изображений-фонов: {directory}")
