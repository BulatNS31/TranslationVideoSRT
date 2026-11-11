import pysrt
import librosa
import numpy as np

def create_srt(orig_text_file, translate_text_file, audio_file, output_srt_file):
	with open(orig_text_file, 'r', encoding='utf-8') as f:
		original_lines = [line.strip() for line in f if line.strip()]
	with open(translate_text_file, 'r', encoding='utf-8') as f:
		translated_lines = [line.strip() for line in f if line.strip()]

	if len(original_lines) != len(translated_lines):
		raise ValueError("Количество строк в оригинальном и переведенном текстах не совпадает.")

	y, sr = librosa.load(audio_file)
	duration = librosa.get_duration(y=y, sr=sr)

	# RMS энергия
	rms = librosa.feature.rms(y=y)[0]
	rms_times = librosa.times_like(rms, sr=sr)

	# Находим пики RMS (вокальные моменты)
	peaks = librosa.util.peak_pick(rms, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.02, wait=10)  # ↑ wait=10

	# Если пиков меньше, чем строк, равномерно распределяем
	if len(peaks) < len(original_lines):
		time_points = np.linspace(0, duration, len(original_lines)+1)
	else:
		# Берем первые N пиков для начала субтитров
		time_points = rms_times[peaks[:len(original_lines)]]
		# Добавляем конечную точку
		time_points = np.append(time_points, duration)

	subs = pysrt.SubRipFile()

	min_duration = 1.5  # Минимум 1.5 секунды
	max_duration = 6.0  # Максимум 6 секунд
	gap_between_subtitles = 0.1  # Зазор между субтитрами

	last_end_time = 0  # Следим за окончанием предыдущего субтитра

	for i in range(len(original_lines)):
		start_time = max(time_points[i], last_end_time + gap_between_subtitles)
		
		# Базовый расчет конечного времени
		if i < len(original_lines) - 1:
			end_time = time_points[i+1]
		else:
			end_time = duration
		
		# Гарантируем минимальную длительность
		if end_time - start_time < min_duration:
			end_time = start_time + min_duration
		
		# Гарантируем максимальную длительность
		if end_time - start_time > max_duration:
			end_time = start_time + max_duration
		
		# Не превышаем общую длительность аудио
		end_time = min(end_time, duration)
		
		# Проверяем, что начало не позже конца
		if start_time >= end_time:
			start_time = end_time - min_duration
		
		# Обновляем время окончания для следующего субтитра
		last_end_time = end_time

		text = f"{translated_lines[i]}"

		sub = pysrt.SubRipItem(
			index=i+1,
			start=pysrt.SubRipTime(seconds=start_time),
			end=pysrt.SubRipTime(seconds=end_time),
			text=text
		)
		subs.append(sub)

	subs.save(output_srt_file, encoding='utf-8')
	print(f"Субтитры сохранены в {output_srt_file}")

if __name__ == "__main__":
	create_srt("orig.txt", "translate.txt", "song.mp3", "output.srt")