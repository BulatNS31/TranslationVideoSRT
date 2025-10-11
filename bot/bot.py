import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from tempfile import NamedTemporaryFile

from core import create_translated_srt

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Команды бота
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправь файл orig.txt и аудио (mp3, wav), и я сгенерирую субтитры на русском."
    )

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("orig_lines"):
        await update.message.reply_text("Сначала отправь файл orig.txt с оригинальными строками.")
        return

    file = await update.message.audio.get_file()
    with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
        await file.download_to_drive(tmp_audio.name)
        output_srt = tmp_audio.name.replace(".mp3", ".srt")
        create_translated_srt(tmp_audio.name, context.user_data["orig_lines"], output_srt)
        await update.message.reply_document(document=open(output_srt, "rb"))

    os.remove(tmp_audio.name)
    os.remove(output_srt)

async def handle_txt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.document.get_file()
    with NamedTemporaryFile(delete=False, suffix=".txt") as tmp_txt:
        await file.download_to_drive(tmp_txt.name)
        with open(tmp_txt.name, "r", encoding="utf-8") as f:
            context.user_data["orig_lines"] = [line.strip() for line in f if line.strip()]
    await update.message.reply_text("Файл orig.txt получен! Теперь отправь аудио.")

def main():
    TOKEN = "8457197497:AAE2OsRAlJHBKISqMpYkgvixDT2ElTH8Sss"
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.FileExtension("txt"), handle_txt))
    app.add_handler(MessageHandler(filters.AUDIO, handle_audio))

    print("🤖 Бот запущен")
    app.run_polling()

if __name__ == "__main__":
    main()
