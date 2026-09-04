"""
Модуль инициализации и настройки Telegram-бота (python-telegram-bot v20+).
"""

import html
import json
import logging
import traceback
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
from telegram.constants import ParseMode

from config import settings
from database import Database
from handlers.start import start_handler, help_handler, status_handler, support_handler, contact_admin_callback
from handlers.photo_handler import photo_message_handler
from telegram.ext import CallbackQueryHandler
from telegram.request import HTTPXRequest

logger = logging.getLogger(__name__)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Глобальный обработчик ошибок в Telegram-боте."""
    logger.error("Исключение при обработке обновления:", exc_info=context.error)

    tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
    tb_string = "".join(tb_list)

    update_str = update.to_dict() if isinstance(update, Update) else str(update)
    logger.debug("Update вызвавшее ошибку: %s", json.dumps(update_str, indent=2, ensure_ascii=False))

    # Если есть возможность уведомить пользователя
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "⚠️ Произошла внутренняя ошибка при обработке запроса. "
                "Администраторы уже уведомлены.",
                parse_mode=ParseMode.HTML
            )
        except Exception:
            pass


def create_bot_application(db: Database) -> Application:
    """Создает и настраивает экземпляр Application для python-telegram-bot."""
    token = settings.telegram_bot_token
    if not token:
        logger.warning("TELEGRAM_BOT_TOKEN не задан! Бот не сможет подключиться к Telegram API.")

    # Увеличиваем таймауты для надежного скачивания медиа-файлов
    request_config = HTTPXRequest(
        connection_pool_size=8,
        connect_timeout=30.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=60.0,
        media_write_timeout=60.0
    )

    app = ApplicationBuilder().token(token or "DUMMY_TOKEN").request(request_config).build()

    # Сохраняем БД в bot_data для доступа из любых хэндлеров
    app.bot_data["db"] = db

    # Регистрация команд
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("status", status_handler))
    app.add_handler(CommandHandler("support", support_handler))

    # Регистрация обработчика фото
    app.add_handler(MessageHandler(filters.PHOTO, photo_message_handler))

    # Обработчик нажатия инлайн-кнопки связи с администратором
    app.add_handler(CallbackQueryHandler(contact_admin_callback, pattern="^contact_admin$"))

    # Глобальный обработчик ошибок
    app.add_error_handler(error_handler)

    return app
