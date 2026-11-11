"""Построение и запуск команды FFmpeg для вертикального ролика."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from .errors import RenderError

logger = logging.getLogger(__name__)


def _escape_filter_value(value: str) -> str:
	"""Экранирует спецсимволы значений внутри выражения фильтра FFmpeg."""
	escaped = value.replace("\\", "\\\\")
	for character in (":", "'", ",", "[", "]"):
		escaped = escaped.replace(character, f"\\{character}")
	return escaped


def build_ffmpeg_command(
	audio_path: Path,
	background_path: Path,
	subtitle_path: Path,
	output_path: Path,
	clip_start: float,
	duration: float,
	font_name: str = "DejaVu Sans",
	*,
	ffmpeg_binary: str = "ffmpeg",
) -> list[str]:
	"""Формирует список аргументов FFmpeg без запуска процесса."""
	subtitle_file = Path(subtitle_path).name
	font = _escape_filter_value(font_name)
	filter_graph = (
		"scale=1080:1920:force_original_aspect_ratio=increase,"
		"crop=1080:1920,"
		f"subtitles={subtitle_file}:force_style="
		f"'FontName={font},FontSize=42,PrimaryColour=&H00FFFFFF&,"
		"OutlineColour=&H80000000&,Outline=2,Shadow=1,Alignment=2,MarginV=100'"
	)
	fade_duration = min(1.0, duration / 2)
	fade_out_start = max(0.0, duration - fade_duration)
	audio_filter = (
		f"afade=t=in:st=0:d={fade_duration:.3f},"
		f"afade=t=out:st={fade_out_start:.3f}:d={fade_duration:.3f}"
	)
	return [
		ffmpeg_binary,
		"-y",
		"-loop",
		"1",
		"-framerate",
		"25",
		"-i",
		str(Path(background_path).resolve()),
		"-ss",
		f"{clip_start:.3f}",
		"-t",
		f"{duration:.3f}",
		"-i",
		str(Path(audio_path).resolve()),
		"-vf",
		filter_graph,
		"-af",
		audio_filter,
		"-t",
		f"{duration:.3f}",
		"-c:v",
		"libx264",
		"-tune",
		"stillimage",
		"-pix_fmt",
		"yuv420p",
		"-c:a",
		"aac",
		"-shortest",
		str(Path(output_path).resolve()),
	]


def render_video(
	audio_path: Path,
	background_path: Path,
	subtitle_path: Path,
	output_path: Path,
	clip_start: float,
	duration: float,
	font_name: str = "DejaVu Sans",
	*,
	runner: Callable[..., Any] = subprocess.run,
	which: Callable[[str], str | None] = shutil.which,
) -> Path:
	"""Проверяет FFmpeg в PATH и запускает рендер рядом с файлом субтитров."""
	binary = which("ffmpeg")
	if binary is None:
		raise RenderError("FFmpeg не найден в PATH")
	subtitle_file = Path(subtitle_path).resolve()
	output_file = Path(output_path).resolve()
	output_file.parent.mkdir(parents=True, exist_ok=True)
	command = build_ffmpeg_command(
		audio_path,
		background_path,
		subtitle_file,
		output_file,
		clip_start,
		duration,
		font_name,
		ffmpeg_binary=binary,
	)
	try:
		completed = runner(
			command,
			cwd=subtitle_file.parent,
			capture_output=True,
			text=True,
			check=False,
		)
	except OSError as error:
		raise RenderError(f"Не удалось запустить FFmpeg: {error}") from error
	if completed.returncode != 0:
		stderr = completed.stderr or ""
		raise RenderError(f"FFmpeg завершился с кодом {completed.returncode}: {stderr}")
	logger.info("Видео сохранено: %s", output_file)
	return output_file
