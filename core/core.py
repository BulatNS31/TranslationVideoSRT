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
	peaks = librosa.util.peak_pick(rms, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.02, wait=5)

	# Если пиков меньше, чем строк, равномерно распределяем
	if len(peaks) < len(original_lines):
		time_points = np.linspace(0, duration, len(original_lines)+1)
	else:
		# Берем первые N пиков для начала субтитров
		time_points = rms_times[peaks[:len(original_lines)]]
		# Добавляем конечную точку
		time_points = np.append(time_points, duration)

	subs = pysrt.SubRipFile()

	min_duration = 2.0
	max_duration = 8.0

	for i in range(len(original_lines)):
		start_time = time_points[i]
		end_time = time_points[i+1] if i < len(original_lines) - 1 else duration

		# Корректируем длительность
		dur = end_time - start_time
		if dur < min_duration:
			end_time = start_time + min_duration
			if i < len(original_lines) - 1 and end_time > time_points[i+1]:
				end_time = time_points[i+1] - 0.1
		elif dur > max_duration:
			end_time = start_time + max_duration

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
