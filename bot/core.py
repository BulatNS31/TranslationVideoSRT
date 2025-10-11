import pysrt
import whisper
from googletrans import Translator


def create_translated_srt(audio_file, orig_text_file, output_srt_file, model_size="small"):
	print("🚀 Начало работы функции create_translated_srt")

	# 1. Загружаем оригинальные строки
	print("📄 Загружаем оригинальный текст")
	with open(orig_text_file, "r", encoding="utf-8") as f:
		orig_lines = [line.strip() for line in f if line.strip()]

	# 2. Распознаём аудио через Whisper локально
	print(f"🎤 Загружаем модель Whisper ({model_size}) и распознаём аудио...")
	model = whisper.load_model(model_size)
	result = model.transcribe(audio_file)
	print(f"✅ Распознавание завершено, найдено {len(result['segments'])} сегментов")

	segments = result["segments"]

	# 3. Переводим текст на русский
	print("🌐 Переводим текст на русский язык...")
	translator = Translator()
	translated_lines = [
		translator.translate(seg["text"], src="auto", dest="ru").text
		for seg in segments
	]
	print("✅ Перевод завершен")

	# 4. Создаём SRT
	print("📝 Формируем субтитры")
	subs = pysrt.SubRipFile()
	min_duration = 1.5  # минимальная длительность субтитра
	for i, seg in enumerate(segments):
		start = pysrt.SubRipTime(seconds=seg["start"])
		end = pysrt.SubRipTime(seconds=seg["end"])

		# Проверяем минимальную длительность
		if end.ordinal - start.ordinal < min_duration * 1000:
			end = pysrt.SubRipTime(seconds=seg["start"] + min_duration)

		text = f"{translated_lines[i]}\n({orig_lines[i] if i < len(orig_lines) else seg['text']})"

		sub = pysrt.SubRipItem(index=i+1, start=start, end=end, text=text)
		subs.append(sub)

	# 5. Сохраняем
	subs.save(output_srt_file, encoding="utf-8")
	print(f"✅ Субтитры сохранены в {output_srt_file}")
	print("🏁 Работа функции завершена")


if __name__ == "__main__":
	print("🔥 Запуск скрипта")
	create_translated_srt("song.mp3", "orig.txt", "srt_output.srt")
