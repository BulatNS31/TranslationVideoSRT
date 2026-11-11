from __future__ import annotations

from pathlib import Path
import asyncio

import pytest

from core.audio_analysis import relative_line_energy
from core.background import choose_background
from core import pipeline
from core.language import detect_language, normalize_language_code
from core.line_selector import validate_line_selection
from core.lyrics_source import parse_srt
from core.metadata import generate_metadata
from core.models import (
	AudioFeatures,
	LineSelection,
	LyricLine,
	PipelineConfig,
	PostMetadata,
	TrackInfo,
)
from core.subtitles import create_srt
from core.video_renderer import build_ffmpeg_command


def test_parse_srt_returns_timed_lyric_lines(tmp_path: Path) -> None:
	srt_path = tmp_path / "input.srt"
	srt_path.write_text(
		"1\n00:00:01,250 --> 00:00:03,500\nHello world\n\n"
		"2\n00:00:04,000 --> 00:00:05,000\nSecond line\n",
		encoding="utf-8",
	)

	lines = parse_srt(srt_path)

	assert lines == [
		LyricLine(1.25, 3.5, "Hello world"),
		LyricLine(4.0, 5.0, "Second line"),
	]


def test_relative_line_energy_is_normalized_to_track_average() -> None:
	features = AudioFeatures(
		duration=4.0,
		tempo=100.0,
		rms=(1.0, 1.0, 3.0, 3.0),
		fps=1.0,
	)

	assert relative_line_energy(features, 2.0, 4.0) == 1.5
	assert relative_line_energy(features, 0.0, 2.0) == 0.5


@pytest.mark.parametrize(
	("source", "expected"),
	[("zh", "zh-cn"), ("zh_TW", "zh-tw"), ("he", "iw")],
)
def test_normalize_language_code(source: str, expected: str) -> None:
	assert normalize_language_code(source) == expected


def test_language_detection_uses_injected_offline_lingua_detector() -> None:
	class LanguageCode:
		name = "ZH"

	class DetectedLanguage:
		iso_code_639_1 = LanguageCode()
		iso_code_639_3 = None

	class Detector:
		def detect_language_of(self, text: str) -> DetectedLanguage:
			assert text == "你好"
			return DetectedLanguage()

	assert asyncio.run(detect_language("你好", detector=Detector())) == "zh-cn"


@pytest.mark.parametrize(
	"response",
	[
		{"indices": [-1]},
		{"indices": [2]},
		{"indices": [0, 1, 2]},
		{"indices": [0, 0]},
	],
)
def test_validate_line_selection_rejects_bad_indices(
	response: dict[str, object],
) -> None:
	lines = [LyricLine(0, 1, "One line"), LyricLine(1, 2, "Two lines")]

	with pytest.raises(ValueError):
		validate_line_selection(response, lines)


def test_choose_background_uses_requested_category(tmp_path: Path) -> None:
	image = tmp_path / "rain_1.jpg"
	image.touch()

	choice = choose_background("rain", tmp_path)

	assert choice.category == "rain"
	assert choice.path == image


def test_choose_background_falls_back_to_default_then_any_image(
	tmp_path: Path,
) -> None:
	default_image = tmp_path / "default.png"
	default_image.touch()
	assert choose_background("sea", tmp_path).path == default_image

	default_image.unlink()
	any_image = tmp_path / "city.webp"
	any_image.touch()
	choice = choose_background("sea", tmp_path)
	assert choice.category == "default"
	assert choice.path == any_image


def test_create_srt_shifts_time_and_keeps_original_and_translation(
	tmp_path: Path,
) -> None:
	output = tmp_path / "subs.srt"

	create_srt(
		[LyricLine(12.0, 12.5, "Original words", "Русский перевод")],
		output,
		clip_start=10.0,
	)

	subtitle = parse_srt(output)[0]
	assert subtitle.start == 2.0
	assert subtitle.end == 4.0
	assert subtitle.text == "Original words\nРусский перевод"


def test_build_ffmpeg_command_keeps_paths_as_separate_arguments(
	tmp_path: Path,
) -> None:
	audio = tmp_path / "song with spaces.mp3"
	background = tmp_path / "night city.jpg"
	subtitles = tmp_path / "subs.srt"
	output = tmp_path / "clip output.mp4"

	command = build_ffmpeg_command(
		audio,
		background,
		subtitles,
		output,
		clip_start=4.5,
		duration=12.0,
		ffmpeg_binary="/usr/bin/ffmpeg",
	)

	assert command[0] == "/usr/bin/ffmpeg"
	assert command[command.index("-i") + 1] == str(background.resolve())
	second_input = command.index("-i", command.index("-i") + 1)
	assert command[second_input + 1] == str(audio.resolve())
	assert command[command.index("-ss") + 1] == "4.500"
	assert "subtitles=subs.srt:force_style=" in command[command.index("-vf") + 1]
	assert command[-1] == str(output.resolve())
	assert "libx264" in command
	assert "yuv420p" in command
	assert "-loop" in command
	assert "-af" in command
	assert "DejaVu Sans" in command[command.index("-vf") + 1]


def test_metadata_falls_back_when_local_llm_returns_invalid_output() -> None:
	class InvalidBackend:
		def generate_json(
			self,
			_system_prompt: str,
			_user_prompt: str,
		) -> dict[str, object]:
			return {"title": "Short title", "description": "Only one sentence.", "hashtags": []}

	result = generate_metadata(
		TrackInfo("Song", "Artist", 20.0, 100.0),
		backend=InvalidBackend(),
	)

	assert len(result.title) <= 70
	assert len(result.hashtags) == 6
	assert result.description.count(".") == 2


def test_run_pipeline_writes_outputs_without_external_services(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	source = tmp_path / "source"
	source.mkdir()
	audio = source / "Artist - Song.mp3"
	audio.touch()
	(audio.with_suffix(".srt")).write_text(
		"1\n00:00:02,000 --> 00:00:04,000\nOne very interesting lyric\n",
		encoding="utf-8",
	)
	backgrounds = tmp_path / "backgrounds"
	backgrounds.mkdir()
	(backgrounds / "default.jpg").touch()
	progress: list[tuple[str, float]] = []
	lyric = LyricLine(2.0, 4.0, "One very interesting lyric", "Очень интересная строка")
	track = TrackInfo("Song", "Artist", 20.0, 100.0)
	features = AudioFeatures(20.0, 100.0, (1.0, 2.0, 1.0), 10.0)

	monkeypatch.setattr(
		pipeline,
		"analyze_audio",
		lambda path: (track, features) if path == audio else pytest.fail(str(path)),
	)
	def read_lyrics(
		audio_path: Path,
		srt_path: Path | None,
		_model_name: str,
	) -> tuple[list[LyricLine], str]:
		assert audio_path == audio
		assert srt_path == audio.with_suffix(".srt")
		return [lyric], "en"

	monkeypatch.setattr(pipeline, "get_lyrics", read_lyrics)

	untranslated = LyricLine(4.0, 6.0, "This translation failed", "")

	async def translate(
		lines: list[LyricLine],
		_language: str,
		**_options: object,
	) -> list[LyricLine]:
		return [*lines, untranslated]

	monkeypatch.setattr(pipeline, "translate_lines", translate)

	def select_translated(lines: list[LyricLine]) -> LineSelection:
		assert all(line.ru for line in lines)
		return LineSelection((0,), (lines[0],))

	monkeypatch.setattr(
		pipeline,
		"select_interesting_lines",
		select_translated,
	)

	def render(*args: object, **_kwargs: object) -> Path:
		output = Path(str(args[3]))
		output.write_bytes(b"mock video")
		return output

	monkeypatch.setattr(pipeline, "render_video", render)
	monkeypatch.setattr(
		pipeline,
		"generate_metadata",
		lambda _track: PostMetadata(
			"A short title",
			"A first sentence. A second sentence.",
			("#music", "#song", "#lyrics", "#shorts", "#video", "#translation"),
		),
	)
	monkeypatch.chdir(tmp_path)
	config = PipelineConfig(
		audio_path=source,
		background_dir=backgrounds,
		on_progress=lambda step, fraction: progress.append((step, fraction)),
	)

	result = pipeline.run_pipeline(config)

	assert config.output_dir == Path("source")
	assert result.subtitle_path.resolve().parent == source
	assert result.video_path.resolve().parent == source
	assert result.post_path.resolve().parent == source
	assert result.metadata_path.resolve().parent == source
	assert result.video_path.read_bytes() == b"mock video"
	assert result.subtitle_path.is_file()
	assert result.post_path.is_file()
	assert result.metadata_path.is_file()
	assert progress[-1] == ("complete", 1.0)


def test_source_directory_requires_exactly_one_mp3(tmp_path: Path) -> None:
	source = tmp_path / "source"
	source.mkdir()

	with pytest.raises(pipeline.ConfigError, match="нет MP3-файлов"):
		pipeline._resolve_audio_path(source)

	(source / "one.mp3").touch()
	assert pipeline._resolve_audio_path(source) == source / "one.mp3"

	(source / "two.mp3").touch()
	with pytest.raises(pipeline.ConfigError, match="несколько MP3-файлов"):
		pipeline._resolve_audio_path(source)
