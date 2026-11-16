"""
mp3 -> (язык: авто или вручную) -> оригинальный SRT -> русский SRT.

Запуск:
	python srt_pipeline.py song.mp3 --out-dir .
	python srt_pipeline.py song.mp3 --out-dir . --lang sk --model large-v3

Результат распознавания кэшируется в <имя>.cache.json: если перевод оборвался
(лимиты Google), повторный запуск НЕ гонит Whisper заново, а продолжает перевод.
Чтобы распознать заново, добавьте --force.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import pysrt
import whisper
from deep_translator import GoogleTranslator
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

SENTENCE_END = re.compile(r"[.!?…。！？]+[\"'»”)\]]*$")
GAP_SPLIT_SEC = 0.6           # пауза между словами => новая строка
MAX_LINE_CHARS = 90           # страховка от слишком длинных строк
MIN_WORDS_BEFORE_CAPITAL = 3  # заглавная буква режет строку, если в ней уже >= N слов
MIN_DURATION_SEC = 0.6

REQUEST_DELAY_SEC = 0.7       # пауза между запросами к Google (лимит ~5/сек)
RETRY_WAITS = (5, 20, 45)     # паузы между повторами при ошибке, сек
MAX_CONSECUTIVE_FAILS = 3     # столько строк подряд без перевода -> остановка

LMSTUDIO_URL = "http://localhost:1234"
LMSTUDIO_MODEL = "huihui-qwen3.5-9b-abliterated"   # id модели из LM Studio (/v1/models)
LM_CHUNK = 20              # строк за один запрос (меньше порция -> меньше контекста и ожидания)
LM_ATTEMPTS = 3

LANG_MAP = {"zh": "zh-CN", "he": "iw", "jw": "jv"}   # Whisper -> Google Translate
LANG_ALIASES = {"cz": "cs"}                           # частая путаница


def log(step: str, msg: str) -> None:
	print(f"[{step}] {msg}", flush=True)


@dataclass
class Line:
	start: float
	end: float
	text: str


def normalize_lang(code: str | None) -> str | None:
	if code is None:
		return None
	c = LANG_ALIASES.get(code.strip().lower(), code.strip().lower())
	if c in LANGUAGES:
		return c
	if c in TO_LANGUAGE_CODE:  # можно писать и названием: "slovak"
		return TO_LANGUAGE_CODE[c]
	raise SystemExit(
		f"Неизвестный язык '{code}'. Примеры: sk (словацкий), cs (чешский), pl, uk, en, es, de, fr"
	)


# ---------- 1. Распознавание ----------
def transcribe(audio_path: Path, model_name: str, language: str | None) -> tuple[str, list[dict[str, Any]]]:
	"""language=None -> Whisper определяет язык сам."""
	log("1/4", f"Загрузка модели Whisper '{model_name}' (при первом запуске скачивается)...")
	t0 = time.time()
	model = whisper.load_model(model_name)
	log("1/4", f"Модель загружена за {time.time() - t0:.1f} с")
	log("1/4", "Распознавание аудио (это самый долгий этап, ниже идёт прогресс)...")
	t0 = time.time()
	result = model.transcribe(
		str(audio_path),
		task="transcribe",
		language=language,
		word_timestamps=True,
		condition_on_previous_text=False,
		temperature=0.0,
		verbose=False,
	)
	log("1/4", f"Распознавание завершено за {time.time() - t0:.1f} с")
	return cast(str, result["language"]), cast(list[dict[str, Any]], result["segments"])


# ---------- 2. Разбиение на строки ----------
def build_lines(segments: list[dict[str, Any]], split_on_capital: bool = True) -> list[Line]:
	words = [w for seg in segments for w in seg.get("words", [])]
	lines: list[Line] = []
	buf: list[dict[str, Any]] = []

	def flush() -> None:
		if buf:
			text = "".join(w["word"] for w in buf).strip()
			if text:
				lines.append(Line(buf[0]["start"], buf[-1]["end"], text))
			buf.clear()

	for w in words:
		token = w["word"].strip()
		if buf:
			gap = w["start"] - buf[-1]["end"]
			prev = buf[-1]["word"].strip()
			cur_len = len("".join(x["word"] for x in buf).strip())
			if (
				gap > GAP_SPLIT_SEC
				or SENTENCE_END.search(prev)
				or cur_len >= MAX_LINE_CHARS
				or (split_on_capital and token[:1].isupper() and len(buf) >= MIN_WORDS_BEFORE_CAPITAL)
			):
				flush()
		buf.append(w)
	flush()

	for i, ln in enumerate(lines):
		ln.end = max(ln.end, ln.start + MIN_DURATION_SEC)
		if i + 1 < len(lines):
			ln.end = min(ln.end, lines[i + 1].start)
	return lines


# ---------- 3. Перевод (строго 1 строка -> 1 строка) ----------
def translate_lines(texts: list[str], src: str, memo: dict[str, str], dest: str = "ru") -> list[str]:
	"""memo: уже переведённые строки (текст -> перевод). Дополняется по ходу работы."""
	if src == dest:
		return list(texts)
	src = LANG_MAP.get(src, src)
	try:
		tr = GoogleTranslator(source=src, target=dest)
	except Exception as e:
		print(f"Язык '{src}' не поддержан переводчиком ({e}), использую auto")
		tr = GoogleTranslator(source="auto", target=dest)

	total = len(texts)
	fails_in_row = 0
	for n, text in enumerate(texts, start=1):
		print(f"\r[3/4] Перевод строк: {n}/{total}", end="", flush=True)
		if text in memo:
			continue

		res, last_err = "", None
		for wait in (*RETRY_WAITS, None):
			try:
				time.sleep(REQUEST_DELAY_SEC)
				res = (tr.translate(text) or "").strip()
				if res:
					break
			except Exception as e:
				last_err = e
			if wait is not None:
				print(f"\n[3/4] Ошибка, жду {wait} с и повторяю...", flush=True)
				time.sleep(wait)

		if res:
			memo[text] = res
			fails_in_row = 0
		else:
			fails_in_row += 1
			if fails_in_row >= MAX_CONSECUTIVE_FAILS:
				print()
				raise RuntimeError(
					f"Google блокирует запросы ({last_err}). Уже переведено {len(memo)}/{total} строк "
					f"и сохранено в кэш. Подождите 15-60 минут (или смените сеть/VPN) "
					f"и запустите ту же команду ещё раз: распознавание повторяться не будет."
				)
	print()

	missing = [t for t in texts if t not in memo]
	if missing:
		print(f"ВНИМАНИЕ: не переведено строк: {len(missing)}/{total} (оставлен оригинал)")
	return [memo.get(t, t) for t in texts]


# ---------- 3b. Перевод через LM Studio (OpenAI-совместимый API) ----------
_LM_SCHEMA = {
	"type": "object",
	"properties": {
		"lines": {
			"type": "array",
			"items": {
				"type": "object",
				"properties": {"id": {"type": "integer"}, "ru": {"type": "string"}},
				"required": ["id", "ru"],
			},
		}
	},
	"required": ["lines"],
}
_LM_RESPONSE_FORMAT = {
	"type": "json_schema",
	"json_schema": {"name": "translation", "strict": True, "schema": _LM_SCHEMA},
}


def _lm_request(url: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 1800) -> Any:
	data = json.dumps(payload).encode("utf-8") if payload is not None else None
	req = urllib.request.Request(
		f"{url.rstrip('/')}{path}", data=data, headers={"Content-Type": "application/json"}
	)
	try:
		with urllib.request.urlopen(req, timeout=timeout) as resp:
			return json.load(resp)
	except urllib.error.HTTPError as e:
		body = e.read().decode("utf-8", errors="replace")[:500]
		raise RuntimeError(f"LM Studio вернул HTTP {e.code}: {body}") from e
	except urllib.error.URLError as e:
		raise RuntimeError(
			f"Не удалось подключиться к LM Studio ({url}): {e}. Сервер запущен (вкладка Developer)? "
			f"Если вы в WSL, а LM Studio на Windows, включите там 'Serve on Local Network' "
			f"и укажите --lm-url http://<IP Windows>:1234"
		) from e


def lm_resolve_model(url: str, model: str) -> str:
	"""Возвращает точный id модели из LM Studio или останавливается с подсказкой."""
	data = _lm_request(url, "/v1/models", timeout=30)
	ids = [m["id"] for m in data.get("data", [])]
	if model in ids:
		return model
	candidates = [i for i in ids if model.lower() in i.lower()]
	if len(candidates) == 1:
		return candidates[0]
	raise SystemExit(
		f"Модель '{model}' не найдена в LM Studio. Доступные id: {', '.join(ids) or 'нет'}.\n"
		f"Укажите точный id: --lm-model <id>"
	)


def _parse_lm_json(text: str) -> dict[int, str]:
	text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
	try:
		obj = json.loads(text)
	except json.JSONDecodeError:
		m = re.search(r"\{.*\}", text, flags=re.S)
		if not m:
			return {}
		try:
			obj = json.loads(m.group(0))
		except json.JSONDecodeError:
			return {}
	out: dict[int, str] = {}
	for item in obj.get("lines", []) if isinstance(obj, dict) else []:
		try:
			ru = str(item["ru"]).strip()
			if ru:
				out[int(item["id"])] = ru
		except (KeyError, TypeError, ValueError):
			continue
	return out


def _lm_chat_stream(url: str, payload: dict[str, Any], timeout: int = 600) -> tuple[str, int, str | None]:
	"""Потоковый запрос: показывает прогресс и не висит молча.
	Возвращает (текст ответа, символов 'размышлений', finish_reason)."""
	body = json.dumps({**payload, "stream": True}).encode("utf-8")
	req = urllib.request.Request(
		f"{url.rstrip('/')}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
	)
	parts: list[str] = []
	reasoning_chars = 0
	finish: str | None = None
	last_print = 0.0
	try:
		with urllib.request.urlopen(req, timeout=timeout) as resp:
			for raw in resp:
				line = raw.decode("utf-8", errors="replace").strip()
				if not line.startswith("data:"):
					continue
				data = line[5:].strip()
				if data == "[DONE]":
					break
				try:
					chunk = json.loads(data)
				except json.JSONDecodeError:
					continue
				choice = (chunk.get("choices") or [{}])[0]
				delta = choice.get("delta") or {}
				if delta.get("content"):
					parts.append(delta["content"])
				reasoning_chars += len(delta.get("reasoning_content") or delta.get("reasoning") or "")
				finish = choice.get("finish_reason") or finish
				now = time.time()
				if now - last_print > 0.5:
					print(f"\r[3/4]   размышления: {reasoning_chars} симв. | ответ: {sum(map(len, parts))} симв.",
						  end="", flush=True)
					last_print = now
	except urllib.error.HTTPError as e:
		body_txt = e.read().decode("utf-8", errors="replace")[:500]
		raise RuntimeError(f"LM Studio вернул HTTP {e.code}: {body_txt}") from e
	except urllib.error.URLError as e:
		raise RuntimeError(f"Не удалось подключиться к LM Studio ({url}): {e}") from e
	except OSError as e:  # включая TimeoutError
		raise RuntimeError(
			f"LM Studio не присылала данные {timeout} с ({type(e).__name__}). "
			f"Слишком медленная генерация или модель зависла."
		) from e
	print(f"\r[3/4]   размышления: {reasoning_chars} симв. | ответ: {sum(map(len, parts))} симв.   ")
	return "".join(parts), reasoning_chars, finish


def translate_lmstudio(texts: list[str], src_lang: str, memo: dict[str, str],
					   url: str = LMSTUDIO_URL, model: str = LMSTUDIO_MODEL) -> list[str]:
	"""Перевод песни порциями с контекстом; ответ привязан к номерам строк."""
	model = lm_resolve_model(url, model)
	log("3/4", f"LM Studio: модель {model}")
	lang_name = LANGUAGES.get(src_lang, src_lang)
	system = (
		f"You are a professional translator of song lyrics from {lang_name} to Russian. "
		"Translate each numbered line faithfully: keep meaning, tone, slang and profanity "
		"(do not censor or soften). The text comes from speech recognition and may contain "
		"small errors: translate the most plausible meaning. Never merge, split, skip or add "
		"lines: return exactly one Russian line per input id. Do not think at length: answer "
		"immediately with the JSON only."
	)

	# необязательные параметры: если сервер вернёт 400, они отбрасываются по одному
	optional: dict[str, Any] = {
		"chat_template_kwargs": {"enable_thinking": False},
		"response_format": _LM_RESPONSE_FORMAT,
	}
	pending = [t for t in dict.fromkeys(texts) if t not in memo]
	total = len(pending)
	done = 0
	for start in range(0, total, LM_CHUNK):
		chunk = pending[start:start + LM_CHUNK]
		todo = {i: t for i, t in enumerate(chunk, start=1)}
		attempt = 0
		while todo and attempt < LM_ATTEMPTS:
			payload_lines = [{"id": i, "text": t} for i, t in todo.items()]
			user = (
				f"Lyrics ({lang_name}) as JSON, translate to Russian.\n"
				'Reply as {"lines": [{"id": <id>, "ru": "<translation>"}, ...]}.\n\n'
				+ json.dumps(payload_lines, ensure_ascii=False)
			)
			log("3/4", f"LM Studio: строки {start + 1}-{start + len(chunk)} из {total}, попытка {attempt + 1}...")
			payload: dict[str, Any] = {
				"model": model,
				"temperature": 0.2,
				"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
				**optional,
			}
			try:
				content, reasoning, finish = _lm_chat_stream(url, payload)
			except RuntimeError as e:
				if "HTTP 400" in str(e) and optional:
					dropped = next(iter(optional))
					optional.pop(dropped)
					print(f"[3/4] LM Studio не принял параметр '{dropped}', пробую без него")
					continue
				if attempt + 1 < LM_ATTEMPTS and "не присылала данные" in str(e):
					attempt += 1
					print(f"[3/4] {e} Повторяю...")
					continue
				raise
			attempt += 1
			got = _parse_lm_json(content)
			for i in list(todo):
				if i in got:
					memo[todo.pop(i)] = got[i]
					done += 1
			if todo and not content.strip():
				print(
					f"[3/4] Пустой ответ (finish_reason={finish}, размышлений: {reasoning} симв.). "
					f"Модель тратит ответ на 'thinking' или не хватает контекста: отключите "
					f"размышления и поставьте Context Length >= 16384 в настройках модели LM Studio."
				)
		if todo:
			print(f"ВНИМАНИЕ: модель не вернула {len(todo)} строк, они останутся на оригинале")
	log("3/4", f"Переведено новых строк: {done}/{total}")
	return [memo.get(t, t) for t in texts]


# ---------- 4. Запись SRT ----------
def write_srt(lines: list[Line], texts: list[str], path: Path) -> None:
	assert len(lines) == len(texts)
	subs = pysrt.SubRipFile()
	for i, (ln, text) in enumerate(zip(lines, texts), start=1):
		subs.append(
			pysrt.SubRipItem(
				index=i,
				start=pysrt.SubRipTime.from_ordinal(int(ln.start * 1000)),
				end=pysrt.SubRipTime.from_ordinal(int(ln.end * 1000)),
				text=text,
			)
		)
	subs.save(str(path), encoding="utf-8")


# ---------- кэш ----------
def save_cache(path: Path, lang: str, lines: list[Line], memo: dict[str, str]) -> None:
	data = {"lang": lang, "lines": [asdict(ln) for ln in lines], "translations": memo}
	path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------- main ----------
def run(audio_path: Path, out_dir: Path, model_name: str = "medium",
		language: str | None = None, split_on_capital: bool = True,
		force: bool = False, translator: str = "lmstudio",
		lm_model: str = LMSTUDIO_MODEL, lm_url: str = LMSTUDIO_URL) -> tuple[Path, Path]:
	language = normalize_lang(language)
	out_dir.mkdir(parents=True, exist_ok=True)
	log("0/4", f"Файл: {audio_path.name}")

	cache_path = out_dir / f"{audio_path.stem}.cache.json"
	cache: dict[str, Any] = {}
	if cache_path.exists() and not force:
		cache = json.loads(cache_path.read_text(encoding="utf-8"))
		if language and cache.get("lang") != language:
			cache = {}  # язык изменили -> кэш не подходит

	if cache:
		lang = cache["lang"]
		lines = [Line(**d) for d in cache["lines"]]
		memo: dict[str, str] = cache.get("translations", {})
		log("1/4", f"Найден кэш {cache_path.name}: распознавание пропущено")
		log("2/4", f"Язык: {lang}, строк: {len(lines)}, уже переведено: {len(memo)}")
	else:
		lang, segments = transcribe(audio_path, model_name, language)
		log("2/4", f"Язык: {lang} ({'задан вручную' if language else 'определён автоматически'})")
		log("2/4", "Разбиение текста на строки...")
		lines = build_lines(segments, split_on_capital)
		if not lines:
			raise RuntimeError("Речь/вокал не распознан")
		log("2/4", f"Получено строк: {len(lines)}")
		memo = {}
		save_cache(cache_path, lang, lines, memo)

	originals = [ln.text for ln in lines]
	orig_path = out_dir / f"{audio_path.stem}.{lang}.srt"
	ru_path = out_dir / f"{audio_path.stem}.ru.srt"

	# оригинал пишем сразу, не дожидаясь перевода
	write_srt(lines, originals, orig_path)
	log("2/4", f"Оригинал сохранён: {orig_path.name}")

	log("3/4", f"Перевод {lang} -> ru ({translator})...")
	try:
		if translator == "lmstudio":
			russian = translate_lmstudio(originals, lang, memo, lm_url, lm_model)
		else:
			russian = translate_lines(originals, lang, memo, "ru")
	finally:
		save_cache(cache_path, lang, lines, memo)  # прогресс не теряется
	log("3/4", "Перевод завершён")

	log("4/4", "Запись русского SRT...")
	write_srt(lines, russian, ru_path)
	assert len(pysrt.open(str(orig_path))) == len(pysrt.open(str(ru_path)))
	log("4/4", f"Готово: {orig_path.name}, {ru_path.name} ({len(lines)} строк)")
	return orig_path, ru_path


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("audio", type=Path)
	ap.add_argument("--out-dir", type=Path, default=Path("source"))
	ap.add_argument("--model", default="medium",
					help="tiny/base/small/medium/large-v3/turbo (для песен лучше medium и выше)")
	ap.add_argument("--lang", default=None,
					help="код языка песни (sk, cs, en, es...). Не указан -> определяется автоматически")
	ap.add_argument("--no-capital-split", action="store_true",
					help="не делить строки по заглавным буквам")
	ap.add_argument("--force", action="store_true",
					help="игнорировать кэш и распознать песню заново")
	ap.add_argument("--translator", choices=["lmstudio", "google"], default="lmstudio",
					help="чем переводить (по умолчанию локальный LM Studio)")
	ap.add_argument("--lm-model", default=LMSTUDIO_MODEL,
					help="id модели в LM Studio (см. http://localhost:1234/v1/models)")
	ap.add_argument("--lm-url", default=LMSTUDIO_URL,
					help="адрес сервера LM Studio (по умолчанию http://localhost:1234)")
	a = ap.parse_args()
	run(a.audio, a.out_dir, a.model, a.lang, not a.no_capital_split, a.force,
		a.translator, a.lm_model, a.lm_url)
