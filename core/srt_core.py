import openai
import pysrt
from googletrans import Translator


openai.api_key = ""


def create_translated_srt(audio_file, orig_text_file, output_srt_file):
	print("🚀 Начало работы функции create_translated_srt")

	# 1. Загружаем оригинальные строки
	print("📄 Загружаем оригинальный текст")
	with open(orig_text_file, "r", encoding="utf-8") as f:
		orig_lines = [line.strip() for line in f if line.strip()]

	# 2. Распознаём аудио через OpenAI Whisper API
	print("🎤 Отправляем аудио на сервер OpenAI для распознавания...")
	with open(audio_file, "rb") as f:
		transcript = openai.audio.transcriptions.create(
			model="whisper-1",
			file=f
		)

	# Текст полностью
	full_text = transcript.text
	print("✅ Распознавание завершено")

	# 3. Разбиваем текст на сегменты (простейший способ — по строкам оригинала)
	segments = []
	lines = full_text.split("\n")
	for i, line in enumerate(lines):
		segments.append({
			"id": i,
			"start": 0,   # Пока нет точных тайм-кодов с API
			"end": 0,     # Можно добавить позже через сторонние инструменты
			"text": line
		})

	# 4. Переводим текст на русский
	print("🌐 Переводим текст на русский язык...")
	translator = Translator()
	translated_lines = [
		translator.translate(seg["text"], src="auto", dest="ru").text
		for seg in segments
	]
	print("✅ Перевод завершен")

	# 5. Создаём SRT (вставляем перевод + оригинал)
	print("📝 Формируем субтитры")
	subs = pysrt.SubRipFile()
	for i, seg in enumerate(segments):
		# В данном варианте без точных тайм-кодов
		start = pysrt.SubRipTime(seconds=0)
		end = pysrt.SubRipTime(seconds=5 + i * 5)  # простая генерация времени
		text = f"{translated_lines[i]}\n({orig_lines[i] if i < len(orig_lines) else seg['text']})"
		sub = pysrt.SubRipItem(index=i+1, start=start, end=end, text=text)
		subs.append(sub)

	# 6. Сохраняем
	subs.save(output_srt_file, encoding="utf-8")
	print(f"✅ Субтитры сохранены в {output_srt_file}")
	print("🏁 Работа функции завершена")


if __name__ == "__main__":
	print("🔥 Запуск скрипта")
	create_translated_srt("song.mp3", "orig.txt", "srt_output.srt")
