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
            f"Вы уже успешно прошли верификацию.\n"
            f"Ваша ссылка для входа в канал:\n👉 {user_data['invite_link']}\n\n"
            "Приятного общения в канале!",
            parse_mode=ParseMode.HTML,
            reply_markup=get_admin_contact_markup()
        )
        return

    ai_name = "Google Gemini" if settings.ai_provider.lower() == "gemini" else "GPT-4o"

    # Приветствие и правила
    rules_text = (
        f"👋 <b>Добро пожаловать, {user.first_name}!</b>\n\n"
        "Этот бот автоматически проверяет выполнение условий и выдает доступ в закрытый канал.\n\n"
        "📋 <b>Условие для получения доступа:</b>\n"
        "Оставьте <b>2 комментария в TikTok со словами 'Мне скинула @qwwext' или подобное, мы любим креативность)</b>Самое главное это упомянуть ник и отправьте боту подтверждающие скриншоты (одним альбомом или по очереди):\n"
        " 1️⃣ <b>Скриншот 1:</b> Первый оставленный комментарий в TikTok\n"
        " 2️⃣ <b>Скриншот 2:</b> Второй оставленный комментарий в TikTok\n\n"
        f"🤖 <b>Проверка через {ai_name}:</b>\n"
        "• AI проверит скриншоты за считанные секунды.\n"
        "• Если всё верно — вы сразу получите персональную ссылку в закрытый канал!\n"
        "• Если возникнут вопросы — вы всегда можете связаться с администратором по кнопке ниже.\n\n"
        "📸 <i>Отправьте 2 скриншота прямо сейчас:</i>"
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
        "ℹ️ <b>Помощь и рекомендации:</b>\n\n"
        "1. <b>Какие скриншоты нужны:</b>\n"
        "   • Ровно 2 скриншота, где видно, что вы оставили комментарии в TikTok.\n"
        "   • На фото должна быть четко видна секция комментариев TikTok с вашим комментарием.\n\n"
        "2. <b>Как отправить 2 скриншота:</b>\n"
        "   • Выберите сразу 2 фото и отправьте их <i>одним альбомом</i>.\n"
        "   • Либо отправьте 1 фото, а затем второе в течение 45 секунд.\n\n"
        "3. <b>Что если возникли вопросы?</b>\n"
        "   • Воспользуйтесь кнопкой «Связаться с администратором» ниже.\n\n"
        "<b>Команды:</b>\n"
        "/start — правила и перезапуск\n"
        "/status — проверить статус заявки\n"
        "/support — поддержка и связь"
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
            "Вы еще не отправляли заявку. Нажмите /start, чтобы ознакомиться с условиями.",
            reply_markup=get_admin_contact_markup()
        )
        return

    status = user_data.get("status")
    attempts = user_data.get("attempts_count", 0)

    if status == "verified":
        link = user_data.get("invite_link", "Ссылка отсутствует")
        msg = (
            f"✅ <b>Ваш статус: Доступ одобрен!</b>\n"
            f"Использовано попыток: {attempts}\n"
            f"Ссылка на канал: 👉 {link}"
        )
    elif status == "rejected":
        reason = user_data.get("rejection_reason", "Причина не указана")
        msg = (
            f"❌ <b>Ваш статус: Отклонено</b>\n"
            f"Использовано попыток: {attempts}\n\n"
            f"<b>Причина:</b>\n{reason}\n\n"
            "Вы можете отправить новые скриншоты или связаться с администратором."
        )
    else:
        msg = (
            f"⏳ <b>Ваш статус: В ожидании</b>\n"
            f"Пожалуйста, пришлите 2 скриншота для проверки условий."
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
        "Если у вас возникли сложности с проверкой скриншотов, получением доступа или бот дал ложный отказ — "
        f"вы можете напрямую написать администратору: <b>{contact}</b>.\n\n"
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
