# Automatic Subtitle Generation for Audio with Translation

This project generates subtitle files (SRT) for audio files (e.g., songs or speech recordings) based on the original text and its translation. Subtitles are synchronized with the audio using sound energy (RMS) analysis and peak detection to estimate timing for each translated line.

---

## Features

- Audio analysis (MP3, WAV, etc.) using `librosa`
- Automatic timing estimation for each translated subtitle line
- Generation of SRT subtitle files
- Supports original text and translation (optionally outputs only translation)
- Configurable minimum and maximum subtitle durations

---

## Requirements

- Python 3.6+
- Poetry for dependency management

---

## Installation

Clone the repository and install dependencies with Poetry:

```bash
poetry install
```

This will install the required packages:

- `librosa`
- `numpy`
- `pysrt`

---

## Usage

1. Prepare three files:
   - `orig.txt` — original text, one phrase or sentence per line
   - `translate.txt` — translation, one line per corresponding original phrase
   - `song.mp3` — audio file with the recording

2. Run the script:

```bash
poetry run python core.py
```

3. The script will generate `output.srt` with subtitles containing only the translated text, synchronized approximately with the audio.

---

## Example function call

```python
create_srt("orig.txt", "translate.txt", "song.mp3", "output.srt")
```

---

## Notes

- If the number of detected RMS peaks is less than the number of subtitle lines, subtitles are distributed evenly over the audio duration.
- Minimum subtitle duration is 1.5 seconds, maximum is 8 seconds.
- Only the translation text is saved in the subtitles (no original text).

---

## Possible Improvements

- Use forced alignment for precise text-to-audio synchronization
- Apply Voice Activity Detection (VAD) to better detect vocal segments
- Optionally output both original and translated text in subtitles
- Add a graphical user interface for easier use

---

## License

Apache License
