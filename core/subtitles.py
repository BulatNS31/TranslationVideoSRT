"""Создание SRT с оригиналом и русским переводом."""

from __future__ import annotations

from pathlib import Path

from .models import LyricLine


def create_srt(
	lines: tuple[LyricLine, ...] | list[LyricLine],
	output_path: Path,
	clip_start: float = 0.0,
) -> Path:
	"""Сохраняет строки SRT, сдвинутые к началу клипа, минимум на две секунды."""
	import pysrt

	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	subtitles = pysrt.SubRipFile()
	for index, line in enumerate(lines, start=1):
		start_ms = max(0, round((line.start - clip_start) * 1000))
		end_ms = max(start_ms + 2000, round((line.end - clip_start) * 1000))
		text = line.text.strip()
		if line.ru.strip():
			text = f"{text}\n{line.ru.strip()}"
		subtitles.append(
			pysrt.SubRipItem(
				index=index,
				start=pysrt.SubRipTime(milliseconds=start_ms),
				end=pysrt.SubRipTime(milliseconds=end_ms),
				text=text,
			)
		)
	subtitles.save(str(path), encoding="utf-8")
	return path
