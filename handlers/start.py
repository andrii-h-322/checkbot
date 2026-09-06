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

    # Приветствие и условия входа в канал
    rules_text = (
        f"👋 <b>Добро пожаловать, {user.first_name}!</b>\n\n"
        "Чтобы получить доступ в наш закрытый Telegram-канал, необходимо выполнить простое условие:\n\n"
        "📋 <b>Что нужно сделать:</b>\n"
        "1. Найдите подобные/тематические видео в <b>TikTok</b>.\n"
        "2. Напишите <b>два комментария</b> под разными видео с текстом:\n"
        "   👉 <code>Мне скинул @fulyashki</code> <i>(или похожим текстом)</i>.\n"
        "3. Сделайте <b>2 скриншота</b> из TikTok, на которых четко видны ваши опубликованные комментарии.\n"
        "4. Отправьте эти <b>2 скриншота</b> сюда в чат (можно одним альбомом или по очереди).\n\n"
        f"⚡ <b>Как работает проверка ({ai_name}):</b>\n"
        "• Искусственный интеллект моментально проверит ваши скриншоты.\n"
        "• После подтверждения комментариев бот сразу выдаст персональную ссылку для входа в канал!\n"
        "• При возникновении любых вопросов вы всегда можете написать администратору.\n\n"
        "📸 <i>Отправьте 2 скриншота из TikTok прямо сейчас для получения ссылки:</i>"
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
        "ℹ️ <b>Инструкция по получению доступа в канал:</b>\n\n"
        "1. <b>Условия задания:</b>\n"
        "   • Напишите <b>два комментария</b> под подобными видео в TikTok.\n"
        "   • Текст комментария: <code>Мне скинул @fulyashki</code> (или аналогичный по смыслу).\n\n"
        "2. <b>Как отправить подтверждение:</b>\n"
        "   • Сделайте 2 скриншота из TikTok с вашими комментариями.\n"
        "   • Пришлите оба скриншота в чат: одним альбомом или по очереди с интервалом до 45 секунд.\n\n"
        "3. <b>Проверка и выдача ссылки:</b>\n"
        "   • Нейросеть автоматически проверит подлинность скриншотов и текст комментариев.\n"
        "   • При успешной проверке бот мгновенно отправит ссылку для входа.\n\n"
        "<b>Команды бота:</b>\n"
        "/start — описание условий и отправка скриншотов\n"
        "/status — проверить текущий статус заявки\n"
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
            "ℹ️ Вы еще не отправляли скриншоты на проверку.\n\n"
            "Напишите 2 комментария в TikTok с текстом «Мне скинул @fulyashki», "
            "нажмите /start и пришлите 2 скриншота для получения доступа.",
            reply_markup=get_admin_contact_markup()
        )
        return

    status = user_data.get("status")
    attempts = user_data.get("attempts_count", 0)

    if status == "verified":
        link = user_data.get("invite_link", "Ссылка отсутствует")
        msg = (
            "✅ <b>Статус: Проверка успешно пройдена!</b>\n\n"
            f"Попыток: {attempts}\n"
            f"Ваша ссылка на канал: 👉 {link}\n\n"
            "Вы уже получили доступ. Добро пожаловать в канал!"
        )
    elif status == "rejected":
        reason = user_data.get("rejection_reason", "Причина не указана")
        msg = (
            "❌ <b>Статус: Проверка не пройдена</b>\n\n"
            f"Попыток: {attempts}\n\n"
            f"<b>Замечание AI:</b>\n{reason}\n\n"
            "Вы можете исправить замечание и отправить 2 скриншота из TikTok заново."
        )
    else:
        msg = (
            "⏳ <b>Статус: В ожидании</b>\n\n"
            "Пожалуйста, пришлите 2 скриншота с комментариями из TikTok для проверки."
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
        "👩‍💻 <b>Служба поддержки и администрация:</b>\n\n"
        "Если у вас возникли сложности с проверкой скриншотов, получением ссылки в канал или появились вопросы — "
        f"вы можете напрямую написать администратору: <b>{contact}</b>.\n\n"
        "Нажмите кнопку ниже для связи:"
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
