"""
Обработчики информационных команд /start, /help, /status, /support и коллбэков связи.
"""

import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
from database import Database
from config import settings

logger = logging.getLogger(__name__)


def get_admin_contact_markup() -> InlineKeyboardMarkup:
    """Возвращает inline-кнопку для связи с администратором."""
    if settings.admin_contact_url:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("💬 Связаться с администратором", url=settings.admin_contact_url)]
        ])
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💬 Связаться с администратором", callback_data="contact_admin")]
    ])


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /start."""
    user = update.effective_user
    if not user or not update.effective_message:
        return

    db: Database = context.bot_data["db"]
    user_data = await db.upsert_user(
        user_id=user.id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name
    )

    # Если пользователь уже прошел верификацию
    if user_data.get("status") == "verified" and user_data.get("invite_link"):
        await update.effective_message.reply_text(
            f"🎉 <b>Здравствуйте, {user.first_name}!</b>\n\n"
            "<i>(Это тестовый бот-пример для проверки работоспособности AI-верификации)</i>\n\n"
            "Вы уже успешно прошли демонстрационную проверку!\n"
            f"Ваша тестовая ссылка для входа в канал:\n👉 {user_data['invite_link']}\n\n"
            "Для повторной проверки работоспособности вы можете в любой момент отправить новый скриншот.",
            parse_mode=ParseMode.HTML,
            reply_markup=get_admin_contact_markup()
        )
        return

    if settings.ai_provider.lower() in ("grok", "xai"):
        ai_name = "xAI Grok"
    elif settings.ai_provider.lower() == "gemini":
        ai_name = "Google Gemini"
    else:
        ai_name = "GPT-4o"

    # Приветствие и правила демо-бота
    rules_text = (
        f"👋 <b>Добро пожаловать, {user.first_name}!</b>\n\n"
        "🤖 <b>Этот бот — демонстрационный пример для проверки работоспособности AI-верификации.</b>\n\n"
        "Бот демонстрирует, как нейросеть в реальном времени анализирует скриншоты пользователей "
        "и автоматически генерирует персональную ссылку для входа в закрытый канал.\n\n"
        "📋 <b>Тестовое задание для проверки:</b>\n"
        "Для проверки работоспособности отправьте <b>всего 1 скриншот</b> "
        "(подходит <b>абсолютно любой скриншот или фото</b> для демонстрации работы системы).\n\n"
        f"⚡ <b>Как работает демонстрация ({ai_name}):</b>\n"
        "• AI моментально проверит скриншот за считанные секунды.\n"
        "• Бот сразу одобрит проверку и создаст рабочую тестовую ссылку в канал!\n"
        "• По любым вопросам тестирования и настройки вы можете связаться с администратором.\n\n"
        "📸 <i>Отправьте любой скриншот прямо сейчас для проверки работоспособности:</i>"
    )

    await update.effective_message.reply_text(
        rules_text,
        parse_mode=ParseMode.HTML,
        reply_markup=get_admin_contact_markup()
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /help."""
    if not update.effective_message:
        return

    help_text = (
        "ℹ️ <b>Справка (Бот-пример для проверки работоспособности):</b>\n\n"
        "Данный бот является <b>примером для тестирования</b> возможностей автоматической AI-проверки скриншотов.\n\n"
        "1. <b>Сколько скриншотов требуется:</b>\n"
        "   • Для демонстрации нужен <b>только 1 скриншот</b>.\n"
        "   • Для проверки подходит <b>абсолютно любой скриншот или фотография</b>.\n\n"
        "2. <b>Как отправить:</b>\n"
        "   • Просто отправьте 1 изображение в чат с ботом.\n"
        "   • Нейросеть мгновенно одобрит проверку и выдаст ссылку для входа.\n\n"
        "3. <b>Назначение бота:</b>\n"
        "   • Проверка точности распознавания AI, скорости ответа и тестирования автоматической выдачи инвайт-ссылок.\n\n"
        "<b>Команды:</b>\n"
        "/start — описание демо-бота и запуск проверки\n"
        "/status — проверить статус демо-заявки\n"
        "/support — поддержка и связь с администратором"
    )
    await update.effective_message.reply_text(
        help_text,
        parse_mode=ParseMode.HTML,
        reply_markup=get_admin_contact_markup()
    )


async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Проверка текущего статуса пользователя."""
    user = update.effective_user
    if not user or not update.effective_message:
        return

    db: Database = context.bot_data["db"]
    user_data = await db.get_user(user.id)

    if not user_data:
        await update.effective_message.reply_text(
            "ℹ️ <i>(Этот бот — пример для проверки работоспособности)</i>\n\n"
            "Вы еще не отправляли скриншот на проверку. Нажмите /start и пришлите 1 скриншот для теста.",
            reply_markup=get_admin_contact_markup()
        )
        return

    status = user_data.get("status")
    attempts = user_data.get("attempts_count", 0)

    if status == "verified":
        link = user_data.get("invite_link", "Ссылка отсутствует")
        msg = (
            "✅ <b>Демо-статус: Проверка успешно пройдена!</b>\n"
            "<i>(Пример проверки работоспособности)</i>\n\n"
            f"Тестовых попыток: {attempts}\n"
            f"Ссылка на канал: 👉 {link}\n\n"
            "Вы можете отправить новый скриншот в любое время для повторного тестирования."
        )
    elif status == "rejected":
        reason = user_data.get("rejection_reason", "Причина не указана")
        msg = (
            "❌ <b>Демо-статус: Тест не пройден</b>\n"
            "<i>(Пример проверки работоспособности: обнаружено несоответствие)</i>\n\n"
            f"Тестовых попыток: {attempts}\n\n"
            f"<b>Замечание AI:</b>\n{reason}\n\n"
            "Вы можете отправить другой скриншот для повторной проверки работоспособности."
        )
    else:
        msg = (
            "⏳ <b>Демо-статус: В ожидании</b>\n"
            "<i>(Пример проверки работоспособности)</i>\n\n"
            "Пожалуйста, пришлите 1 скриншот для проверки работоспособности."
        )

    await update.effective_message.reply_text(
        msg,
        parse_mode=ParseMode.HTML,
        reply_markup=get_admin_contact_markup()
    )


async def support_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /support для связи с администрацией."""
    if not update.effective_message:
        return

    contact = settings.admin_contact_url or settings.admin_telegram_username or "не указан (обратитесь к владельцу канала)"
    msg = (
        "👩‍💻 <b>Поддержка и связь по демо-боту:</b>\n\n"
        "Этот бот — <b>демонстрационный пример для проверки работоспособности</b> автоматической AI-верификации скриншотов.\n\n"
        "Если у вас возникли вопросы по функционалу, работе нейросети или интеграции — "
        f"вы можете напрямую написать: <b>{contact}</b>.\n\n"
        "Нажмите кнопку ниже для быстрого перехода:"
    )
    await update.effective_message.reply_text(
        msg,
        parse_mode=ParseMode.HTML,
        reply_markup=get_admin_contact_markup()
    )


async def contact_admin_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Коллбэк, если URL администратора не был указан напрямую как ссылка."""
    query = update.callback_query
    if not query:
        return

    await query.answer()
    contact = settings.admin_contact_url or settings.admin_telegram_username
    if contact:
        await query.message.reply_text(
            f"💬 Для связи с администратором напишите сюда: {contact}"
        )
    else:
        await query.message.reply_text(
            "⚠️ Контакт администратора пока не настроен в конфигурации бота. "
            "Пожалуйста, обратитесь к создателю канала."
        )
