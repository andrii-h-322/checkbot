"""
Главный входной файл для запуска Telegram-бота и веб-панели управления.
Поддерживает одновременную работу обоих сервисов в едином асинхронном цикле.
"""

import sys
import asyncio
import logging
import argparse
import uvicorn

from config import settings
from database import Database
from bot import create_bot_application
from web.server import create_web_app

# Настройка форматирования логов
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("checkbot")


async def run_services(run_bot: bool = True, run_web: bool = True) -> None:
    """Запуск базы данных, телеграм-бота и веб-интерфейса."""
    db = Database(settings.database_path)
    await db.init_db()

    tasks = []
    bot_app = None
    web_server = None

    if run_web:
        web_app = create_web_app(db)
        config = uvicorn.Config(
            app=web_app,
            host=settings.web_host,
            port=settings.web_port,
            log_level="warning",
            access_log=False
        )
        web_server = uvicorn.Server(config)
        logger.info("Веб-панель доступна по адресу: http://localhost:%s (логин: %s)", settings.web_port, settings.admin_username)

    if run_bot:
        if not settings.telegram_bot_token:
            logger.warning("=" * 60)
            logger.warning("ВНИМАНИЕ: TELEGRAM_BOT_TOKEN не указан в файле .env!")
            logger.warning("Бот не сможет запуститься без токена.")
            logger.warning("Заполните .env и перезапустите приложение.")
            logger.warning("=" * 60)
            if not run_web:
                return
        else:
            bot_app = create_bot_application(db)

    # Запуск
    stop_event = asyncio.Event()

    try:
        if bot_app:
            await bot_app.initialize()
            await bot_app.start()
            await bot_app.updater.start_polling(drop_pending_updates=True)
            logger.info("Telegram-бот успешно запущен и ожидает сообщений!")

        if web_server:
            web_task = asyncio.create_task(web_server.serve())
            tasks.append(web_task)

        logger.info("Сервисы запущены. Для остановки нажмите Ctrl+C.")

        if tasks:
            # Ждем завершения веб-сервера или прерывания
            await asyncio.gather(*tasks)
        elif bot_app:
            # Если только бот без веб-сервера
            while True:
                await asyncio.sleep(3600)

    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Получен сигнал завершения работы...")
    finally:
        if web_server:
            web_server.should_exit = True

        if bot_app:
            logger.info("Остановка Telegram-бота...")
            if bot_app.updater and bot_app.updater.running:
                await bot_app.updater.stop()
            if bot_app.running:
                await bot_app.stop()
            await bot_app.shutdown()
            logger.info("Telegram-бот остановлен.")

        logger.info("Все сервисы корректно остановлены.")


def main():
    parser = argparse.ArgumentParser(description="CheckBot - AI Verification Telegram Bot")
    parser.add_argument("--bot-only", action="store_true", help="Запустить только Telegram-бота")
    parser.add_argument("--web-only", action="store_true", help="Запустить только веб-панель управления")
    args = parser.parse_args()

    run_bot = not args.web_only
    run_web = not args.bot_only

    try:
        asyncio.run(run_services(run_bot=run_bot, run_web=run_web))
    except KeyboardInterrupt:
        logger.info("Программа завершена пользователем.")


if __name__ == "__main__":
    main()
