"""Типизированные модели данных пайплайна."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


ProgressCallback = Callable[[str, float], None]


@dataclass(slots=True)
class TrackInfo:
	"""Основные сведения о композиции."""

	title: str
	artist: str
	duration: float
	tempo: float


@dataclass(slots=True)
class AudioFeatures:
	"""Измеренные характеристики аудио; rms хранит кадры огибающей громкости."""

	duration: float
	tempo: float
	rms: tuple[float, ...]
	fps: float


@dataclass(slots=True)
class LyricLine:
	"""Одна строка текста с таймингом и переводом."""

	start: float
	end: float
	text: str
	ru: str = ""
	energy: float = 1.0


@dataclass(slots=True)
class LineSelection:
	"""Выбранные строки и, если определена, категория фона."""

	indices: tuple[int, ...]
	lines: tuple[LyricLine, ...]
	background_category: str | None = None


@dataclass(slots=True)
class BackgroundChoice:
	"""Выбранная категория и соответствующее изображение."""

	category: str
	path: Path


@dataclass(slots=True)
class PostMetadata:
	"""Текст публикации и набор хештегов."""

	title: str
	description: str
	hashtags: tuple[str, ...]


@dataclass(slots=True)
class PipelineConfig:
	"""Настройки входных файлов, оформления и размещения результатов."""

	audio_path: Path = Path("source")
	output_dir: Path = Path("source")
	srt_path: Path | None = None
	background_image: Path | None = None
	background_dir: Path = Path("backgrounds")
	whisper_model: str = "small"
	clip_duration: float = 30.0
	font_name: str = "DejaVu Sans"
	language: str | None = None
	translation_fallback_to_llm: bool = True
	on_progress: ProgressCallback | None = field(default=None, repr=False)

	def __post_init__(self) -> None:
		"""Приводит переданные пути к pathlib.Path."""
		self.audio_path = Path(self.audio_path)
		self.output_dir = Path(self.output_dir)
		self.background_dir = Path(self.background_dir)
		if self.srt_path is not None:
			self.srt_path = Path(self.srt_path)
		if self.background_image is not None:
			self.background_image = Path(self.background_image)


@dataclass(slots=True)
class PipelineResult:
	"""Результаты анализа и пути к созданным файлам."""

	track: TrackInfo
	audio_features: AudioFeatures
	lines: tuple[LyricLine, ...]
	selection: LineSelection
	background: BackgroundChoice
	metadata: PostMetadata
	video_path: Path
	subtitle_path: Path
	post_path: Path
	metadata_path: Path
