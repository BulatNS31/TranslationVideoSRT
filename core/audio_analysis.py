"""Загрузка аудио и расчёт характеристик композиции."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from .models import AudioFeatures, TrackInfo

logger = logging.getLogger(__name__)

SAMPLE_RATE = 22050
HOP_LENGTH = 512
RMS_FRAME_LENGTH = 2048


def _read_audio(audio_path: Path) -> tuple[Any, int | float]:
	"""Загружает аудио в mono с частотой дискретизации 22050 Гц."""
	import librosa

	return librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)


def _track_identity(audio_path: Path) -> tuple[str, str]:
	"""Извлекает исполнителя и название из имени вида «Исполнитель - Трек»."""
	stem = audio_path.stem.strip()
	if " - " in stem:
		artist, title = (part.strip() for part in stem.split(" - ", maxsplit=1))
		if artist and title:
			return title, artist
	return stem or "Неизвестный трек", "Неизвестный исполнитель"


def analyze_audio(
	audio_path: Path,
	*,
	audio_loader: Callable[[Path], tuple[Any, int]] | None = None,
) -> tuple[TrackInfo, AudioFeatures]:
	"""Измеряет длительность, темп и RMS-характеристики аудиофайла."""
	import librosa
	import numpy as np

	path = Path(audio_path)
	if not path.is_file():
		raise FileNotFoundError(f"Аудиофайл не найден: {path}")
	loader = audio_loader or _read_audio
	samples, sample_rate = loader(path)
	if sample_rate <= 0 or len(samples) == 0:
		raise ValueError(f"Аудиофайл пуст или имеет неверную частоту дискретизации: {path}")

	duration = float(len(samples) / sample_rate)
	tempo_value, _ = librosa.beat.beat_track(y=samples, sr=sample_rate)
	tempo_array = np.asarray(tempo_value, dtype=float).reshape(-1)
	tempo = float(tempo_array[0]) if tempo_array.size else 0.0
	rms_frames = librosa.feature.rms(
		y=samples,
		frame_length=RMS_FRAME_LENGTH,
		hop_length=HOP_LENGTH,
	).reshape(-1)
	rms = tuple(float(value) for value in rms_frames)
	frame_rate = sample_rate / HOP_LENGTH
	title, artist = _track_identity(path)
	logger.info("Проанализирован трек %s: %.1f секунд, %.1f BPM", title, duration, tempo)

	features = AudioFeatures(
		duration=duration,
		tempo=tempo,
		rms=rms,
		fps=frame_rate,
	)
	return TrackInfo(title, artist, duration, tempo), features


def relative_line_energy(
	features: AudioFeatures,
	start: float,
	end: float,
) -> float:
	"""Возвращает среднюю RMS-энергию интервала относительно всего трека."""
	if not features.rms:
		return 1.0
	if end <= start or features.fps <= 0:
		return 0.0

	first = max(0, min(len(features.rms), int(start * features.fps)))
	last = max(first, min(len(features.rms), int(end * features.fps + 0.999999)))
	if first == last:
		return 0.0
	track_mean = sum(features.rms) / len(features.rms)
	if track_mean <= 0:
		return 0.0
	interval_mean = sum(features.rms[first:last]) / (last - first)
	return interval_mean / track_mean
