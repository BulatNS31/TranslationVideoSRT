"""Связующий код полного конвейера анализа песни и рендеринга видео."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Coroutine, TypeVar

from .audio_analysis import analyze_audio, relative_line_energy
from .background import BACKGROUND_CATEGORIES, category_for_tempo, choose_background
from .errors import ConfigError, LyricsNotFoundError, TranslationError
from .language import detect_language, normalize_language_code
from .line_selector import select_interesting_lines
from .lyrics_source import get_lyrics
from .metadata import generate_metadata
from .models import LyricLine, PipelineConfig, PipelineResult
from .subtitles import create_srt
from .translation import translate_lines
from .video_renderer import render_video

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


def _run_async(coroutine: Coroutine[object, object, _T]) -> _T:
	"""Выполняет async-этапы из синхронного API, в том числе при активном loop."""
	try:
		asyncio.get_running_loop()
	except RuntimeError:
		return asyncio.run(coroutine)

	result: list[_T] = []
	failure: list[BaseException] = []

	def run_in_thread() -> None:
		try:
			result.append(asyncio.run(coroutine))
		except BaseException as error:
			failure.append(error)

	thread = threading.Thread(target=run_in_thread, daemon=True)
	thread.start()
	thread.join()
	if failure:
		raise failure[0]
	return result[0]


def _progress(config: PipelineConfig, step: str, fraction: float) -> None:
	"""Передаёт прогресс вызывающему коду, если callback настроен."""
	if config.on_progress is not None:
		config.on_progress(step, fraction)


def _clip_window(
	selection: tuple[LyricLine, ...],
	track_duration: float,
	requested_duration: float,
) -> tuple[float, float]:
	"""Подбирает окно клипа, включающее выбранные строки насколько возможно."""
	if not selection:
		return 0.0, min(track_duration, requested_duration)
	duration = min(track_duration, requested_duration)
	if duration <= 0:
		raise ConfigError("Длительность аудиофайла должна быть больше нуля")
	first_start = max(0.0, min(line.start for line in selection))
	latest_start = max(0.0, track_duration - duration)
	start = min(first_start, latest_start)
	return start, duration


def _jsonable_lines(lines: tuple[LyricLine, ...]) -> list[dict[str, object]]:
	"""Преобразует выбранные строки в сериализуемые словари."""
	return [asdict(line) for line in lines]


def _resolve_audio_path(audio_path: Path) -> Path:
	"""Разрешает каталог source в единственный MP3-файл для обработки."""
	path = Path(audio_path)
	if path.is_dir():
		audio_files = sorted(
			item
			for item in path.iterdir()
			if item.is_file() and item.suffix.lower() == ".mp3"
		)
		if not audio_files:
			raise ConfigError(f"В каталоге нет MP3-файлов: {path}")
		if len(audio_files) > 1:
			raise ConfigError(
				f"В каталоге несколько MP3-файлов; укажите audio_path явно: {path}"
			)
		return audio_files[0]
	if not path.is_file():
		raise ConfigError(f"Аудиофайл не найден: {path}")
	return path


def run_pipeline(config: PipelineConfig) -> PipelineResult:
	"""Анализирует песню, выбирает строки и создаёт видео и метаданные."""
	audio_path = _resolve_audio_path(config.audio_path)
	srt_path = config.srt_path
	if srt_path is None:
		matching_srt = audio_path.with_suffix(".srt")
		if matching_srt.is_file():
			srt_path = matching_srt
	if srt_path is not None and not srt_path.is_file():
		raise LyricsNotFoundError(f"Файл субтитров не найден: {srt_path}")
	if config.clip_duration < 2:
		raise ConfigError("Длительность клипа должна быть не менее двух секунд")

	config.output_dir.mkdir(parents=True, exist_ok=True)
	_progress(config, "audio_analysis", 0.05)
	track, features = analyze_audio(audio_path)

	_progress(config, "lyrics", 0.2)
	lines, whisper_language = get_lyrics(
		audio_path,
		srt_path,
		config.whisper_model,
	)
	if not lines:
		raise LyricsNotFoundError("Не найдено ни одной строки текста")
	for line in lines:
		line.energy = relative_line_energy(features, line.start, line.end)

	_progress(config, "language", 0.35)
	language = normalize_language_code(config.language or whisper_language)
	if language is None:
		language = _run_async(detect_language(" ".join(line.text for line in lines)))

	_progress(config, "translation", 0.48)
	translated = _run_async(
		translate_lines(
			lines,
			language,
			fallback_to_llm=config.translation_fallback_to_llm,
		)
	)

	_progress(config, "selection", 0.6)
	translatable_lines = [line for line in translated if line.ru.strip()]
	if not translatable_lines:
		raise TranslationError("Не удалось перевести ни одной строки для субтитров")
	selection = select_interesting_lines(translatable_lines)
	if not selection.lines:
		raise TranslationError("Не удалось выбрать переведённые строки для субтитров")

	_progress(config, "background", 0.7)
	requested_category = selection.background_category
	if requested_category not in BACKGROUND_CATEGORIES:
		requested_category = category_for_tempo(track.tempo)
	background = choose_background(
		requested_category,
		config.background_dir,
		config.background_image,
	)

	clip_start, clip_duration = _clip_window(
		selection.lines,
		features.duration,
		config.clip_duration,
	)
	subtitle_path = config.output_dir / "subs.srt"
	video_path = config.output_dir / "clip.mp4"
	post_path = config.output_dir / "post.txt"
	metadata_path = config.output_dir / "meta.json"

	create_srt(selection.lines, subtitle_path, clip_start)
	_progress(config, "render", 0.82)
	render_video(
		audio_path,
		background.path,
		subtitle_path,
		video_path,
		clip_start,
		clip_duration,
		config.font_name,
	)

	_progress(config, "metadata", 0.93)
	metadata = generate_metadata(track)
	post_path.write_text(
		f"{metadata.title}\n\n{metadata.description}\n\n"
		f"{' '.join(metadata.hashtags)}\n",
		encoding="utf-8",
	)
	metadata_path.write_text(
		json.dumps(
			{
				"track": asdict(track),
				"background": {
					"category": background.category,
					"path": str(background.path),
				},
				"selected_lines": _jsonable_lines(selection.lines),
				"post": asdict(metadata),
			},
			ensure_ascii=False,
			indent=2,
		)
		+ "\n",
		encoding="utf-8",
	)
	_progress(config, "complete", 1.0)
	return PipelineResult(
		track=track,
		audio_features=features,
		lines=tuple(translated),
		selection=selection,
		background=background,
		metadata=metadata,
		video_path=video_path,
		subtitle_path=subtitle_path,
		post_path=post_path,
		metadata_path=metadata_path,
	)
