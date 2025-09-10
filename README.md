# Automatic Subtitle Generation for Audio with Translation

This project generates subtitle files (SRT) for audio files (e.g., songs or speech recordings) based on the original text and its translation. Subtitles are synchronized with the audio using either **local RMS analysis** (`librosa`) or **OpenAI Whisper API** for automatic transcription, and optionally translated to Russian.

---

## Features

- Audio analysis (MP3, WAV, etc.) using `librosa` **or** OpenAI Whisper API
- Automatic timing estimation for each translated subtitle line
- Generation of SRT subtitle files
- Supports original text and translation (optionally outputs only translation)
- Configurable minimum and maximum subtitle durations

---

## Requirements

- Python 3.6+
- Poetry for dependency management
- (Optional) OpenAI API key for server-based transcription

---

## Installation

Clone the repository and install dependencies with Poetry:

```bash
poetry install
```

## Usage
1. Local RMS-based subtitles (core.py)
Prepare three files:

   - orig.txt — original text, one phrase or sentence per line
   - translate.txt — translation, one line per corresponding original phrase
   - song.mp3 — audio file with the recording

Run the local RMS script:
```bash
poetry run python core.py
```
2. The script will generate output.srt with subtitles containing only the translated text, synchronized approximately with the audio.

3. Server-based transcription with OpenAI Whisper (srt_core.py)
Set your OpenAI API key in srt_core.py:

```bash
openai.api_key = "YOUR_API_KEY"
```
Run the script:

```bash
poetry run python srt_core.py
```

The script will:
- Upload audio to OpenAI Whisper API
- Receive transcription
- Translate lines to Russian
- Generate srt_output.srt with subtitles containing translation and optionally original text

## Notes

- For local RMS analysis, if the number of detected peaks is less than the number of subtitle lines, subtitles are distributed evenly.
- Minimum subtitle duration (local) is 1.5–2 seconds, maximum is configurable.
- Server transcription currently generates approximate timestamps; for precise alignment, further processing is needed.
- Subtitles include translated text; original text can optionally be included.

## Possible Improvements

- Use forced alignment for precise text-to-audio synchronization
- Apply Voice Activity Detection (VAD) for better segment detection
- Merge local and server approaches for hybrid accuracy
- Add a graphical user interface for easier usage

## License
Apache License