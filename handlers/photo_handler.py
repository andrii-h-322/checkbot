"""
Обработчик полученных фотографий и скриншотов.
Интегрирует MediaCollector для альбомов и поштучной отправки,
AI-верификатор (Gemini / OpenAI) для проверки, Telegram API для генерации ссылок
и кнопку оперативной связи с администратором при возникновении вопросов или отказе.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Any
from telegram import Update, Message, PhotoSize, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.constants import ParseMode, ChatAction
from telegram.error import TelegramError

from config import settings
from database import Database
from ai_verifier import ai_verifier
from media_collector import media_collector

logger = logging.getLogger(__name__)


def get_admin_contact_keyboard() -> Optional[InlineKeyboardMarkup]:
    """Возвращает клавиатуру с кнопкой связи с администратором."""
    if settings.admin_contact_url:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("💬 Связаться с администратором", url=settings.admin_contact_url)]
        ])
    else:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("💬 Связаться с администратором", callback_data="contact_admin")]
        ])


def get_success_keyboard(invite_link: Optional[str] = None) -> InlineKeyboardMarkup:
    """Возвращает клавиатуру для успешной проверки со ссылкой перехода в чат/канал и кнопкой связи."""
    buttons = []
    if invite_link:
        buttons.append([InlineKeyboardButton("🚀 Вступить в канал / чат", url=invite_link)])

    if settings.admin_contact_url:
        buttons.append([InlineKeyboardButton("💬 Связаться с администратором", url=settings.admin_contact_url)])
    else:
        buttons.append([InlineKeyboardButton("💬 Связаться с администратором", callback_data="contact_admin")])

    return InlineKeyboardMarkup(buttons)


async def generate_channel_invite_link(
    bot,
    channel_chat_id: int | str,
    user: Any
) -> tuple[Optional[str], str, Optional[str]]:
    """
    Надежная многоуровневая генерация пригласительной ссылки:
    1. Попытка создать персонализированную ссылку (create_chat_invite_link) с тайм-аутом и лимитом
    2. Попытка создать базовую ссылку без дополнительных ограничений
    3. Попытка экспорта постоянной ссылки (export_chat_invite_link)
    4. Получение ссылки или username из get_chat
    5. Резервная ссылка INVITE_LINK_FALLBACK из конфигурации
    """
    link_note = ""
    last_err: Optional[str] = None

    # Если channel_chat_id не указан
    if not channel_chat_id:
        fallback = (settings.invite_link_fallback or "").strip()
        if fallback:
            logger.info("CHANNEL_ID не задан, использована резервная ссылка INVITE_LINK_FALLBACK: %s", fallback)
            return fallback, "", None
        return (
            None,
            "\n\n⚠️ <i>CHANNEL_ID не указан в файле .env или переменных окружения сервера. "
            "Укажите ID канала/чата (например, -100...) в переменной CHANNEL_ID.</i>",
            "CHANNEL_ID не задан в конфигурации"
        )

    # Расчет срока действия и лимита
    expire_date = None
    if settings.invite_link_expire_hours > 0:
        expire_date = datetime.now(timezone.utc) + timedelta(hours=settings.invite_link_expire_hours)
        link_note += f"\n⏳ Срок действия ссылки: {settings.invite_link_expire_hours} ч."

    member_limit = settings.invite_link_member_limit if settings.invite_link_member_limit > 0 else None
    if member_limit == 1:
        link_note += "\n🔒 Ссылка является одноразовой (для 1 участника)."
    elif member_limit and member_limit > 1:
        link_note += f"\n👥 Лимит входов по ссылке: {member_limit}."

    # Безопасное имя ссылки (Telegram API ограничивает длину имени 32 символами)
    safe_name = f"ID {user.id}"
    clean_username = "".join(c for c in (user.username or "") if c.isalnum() or c in "_-")
    if clean_username:
        safe_name = f"{user.id} @{clean_username}"[:32]

    # Уровень 1: Полноценное создание ссылки с параметрами
    try:
        link_obj = await bot.create_chat_invite_link(
            chat_id=channel_chat_id,
            name=safe_name,
            expire_date=expire_date,
            member_limit=member_limit
        )
        if link_obj and link_obj.invite_link:
            logger.info("Успешно создана персональная ссылка для user_id=%s: %s", user.id, link_obj.invite_link)
            return link_obj.invite_link, link_note, None
    except TelegramError as te:
        last_err = str(te)
        logger.warning("Уровень 1 (create_chat_invite_link с параметрами) не удался для %s: %s", channel_chat_id, te)

    # Уровень 2: Базовый вызов create_chat_invite_link без expire_date и member_limit
    try:
        link_obj = await bot.create_chat_invite_link(chat_id=channel_chat_id)
        if link_obj and link_obj.invite_link:
            logger.info("Успешно создана базовая ссылка для user_id=%s: %s", user.id, link_obj.invite_link)
            return link_obj.invite_link, "", None
    except TelegramError as te:
        last_err = str(te)
        logger.warning("Уровень 2 (базовый create_chat_invite_link) не удался для %s: %s", channel_chat_id, te)

    # Уровень 3: Экспорт постоянной ссылки чата/канала
    try:
        exported_link = await bot.export_chat_invite_link(chat_id=channel_chat_id)
        if exported_link:
            logger.info("Успешно экспортирована постоянная ссылка чата %s: %s", channel_chat_id, exported_link)
            return exported_link, "", None
    except TelegramError as te:
        last_err = str(te)
        logger.warning("Уровень 3 (export_chat_invite_link) не удался для %s: %s", channel_chat_id, te)

    # Уровень 4: Получение существующей ссылки или публичного username через get_chat
    try:
        chat = await bot.get_chat(chat_id=channel_chat_id)
        if getattr(chat, "invite_link", None):
            return chat.invite_link, "", None
        if getattr(chat, "username", None):
            public_link = f"https://t.me/{chat.username}"
            return public_link, "", None
    except TelegramError as te:
        last_err = str(te)
        logger.warning("Уровень 4 (get_chat) не удался для %s: %s", channel_chat_id, te)

    # Уровень 5: Резервная статическая ссылка из .env
    fallback = (settings.invite_link_fallback or "").strip()
    if fallback:
        logger.info("Динамическая генерация не удалась, использована резервная ссылка INVITE_LINK_FALLBACK: %s", fallback)
        return fallback, "", None

    # Если все попытки провалились
    logger.error("Все попытки создания ссылки для канала %s провалились: %s", channel_chat_id, last_err)
    err_explanation = (
        f"\n\n⚠️ <b>Не удалось сгенерировать ссылку автоматически.</b>\n"
        f"Детали ошибки Telegram: <code>{last_err or 'Неизвестная ошибка Telegram'}</code>\n\n"
        f"🔧 <b>Как исправить:</b>\n"
        f"1. Убедитесь, что бот добавлен в целевой канал/чат <code>{channel_chat_id}</code> как <b>Администратор</b>.\n"
        f"2. Проверьте, что боту выдано право <b>«Приглашать пользователей»</b> (Invite Users via Link).\n"
        f"3. Либо укажите постоянную ссылку в .env: <code>INVITE_LINK_FALLBACK=https://t.me/+...</code>"
    )
    return None, err_explanation, last_err


async def photo_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Главный входной обработчик для входящих фото."""
    message = update.effective_message
    user = update.effective_user

    if not message or not user:
        return

    if not message.photo:
        return

    best_photo: PhotoSize = message.photo[-1]
    db: Database = context.bot_data["db"]

    # Обновляем профиль пользователя в БД
    await db.upsert_user(
        user_id=user.id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name
    )

    # Функция, вызываемая после сбора полного комплекта фото
    async def _on_complete(photos: List[PhotoSize], messages: List[Message]) -> None:
        last_msg = messages[-1] if messages else message
        await process_verification(
            user=user,
            photos=photos,
            reply_to_message=last_msg,
            context=context,
            db=db
        )

    # Функция оповещения при поштучной отправке
    async def _on_need_more(count: int, msg: Message) -> None:
        if count == 1:
            await msg.reply_text(
                "📥 <b>Первый скриншот получен!</b>\n\n"
                "Пожалуйста, отправьте <b>второй скриншот</b> в течение 45 секунд "
                "или пришлите оба скриншота одним альбомом.",
                parse_mode=ParseMode.HTML,
                reply_markup=get_admin_contact_keyboard()
            )
        else:
            await msg.reply_text(
                "⚠️ <b>Время ожидания второго скриншота истекло.</b>\n"
                "Для проверки требуется ровно 2 скриншота. Пожалуйста, отправьте оба фото заново.",
                parse_mode=ParseMode.HTML,
                reply_markup=get_admin_contact_keyboard()
            )

    # Проверяем, пришло ли фото в составе медиа-группы (альбома)
    if message.media_group_id:
        logger.info("Получено фото из медиа-группы %s от user_id=%s", message.media_group_id, user.id)
        await media_collector.add_photo_from_media_group(
            media_group_id=message.media_group_id,
            message=message,
            best_photo=best_photo,
            on_complete=_on_complete
        )
    else:
        logger.info("Получено одиночное фото от user_id=%s", user.id)
        await media_collector.add_single_photo(
            user_id=user.id,
            message=message,
            best_photo=best_photo,
            on_complete=_on_complete,
            on_need_more=_on_need_more
        )


async def process_verification(
    user,
    photos: List[PhotoSize],
    reply_to_message: Message,
    context: ContextTypes.DEFAULT_TYPE,
    db: Database
) -> None:
    """Основная логика скачивания, отправки в AI и выдачи ссылки."""
    if len(photos) < 2:
        await reply_to_message.reply_text(
            f"❌ Получено скриншотов: <b>{len(photos)}</b> из 2 необходимых.\n\n"
            "Пожалуйста, отправьте <b>ровно 2 скриншота</b> (одним альбомом или по очереди).",
            parse_mode=ParseMode.HTML,
            reply_markup=get_admin_contact_keyboard()
        )
        return

    # Берем первые 2 скриншота
    photos_to_verify = photos[:2]

    # Уведомляем пользователя о начале проверки
    if settings.ai_provider.lower() in ("grok", "xai"):
        provider_name = "xAI Grok"
    elif settings.ai_provider.lower() == "gemini":
        provider_name = "Google Gemini"
    else:
        provider_name = "GPT-4o"

    status_msg = await reply_to_message.reply_text(
        f"⏳ <b>Скриншоты получены!</b>\n"
        f"Передаю изображения в модуль <b>{provider_name}</b> для анализа... Это займет несколько секунд.",
        parse_mode=ParseMode.HTML,
        reply_markup=get_admin_contact_keyboard()
    )

    try:
        await context.bot.send_chat_action(chat_id=reply_to_message.chat_id, action=ChatAction.TYPING)

        # 1. Скачивание изображений в память с повторными попытками
        images_bytes: List[bytes] = []
        for idx, p in enumerate(photos_to_verify, start=1):
            logger.info("Скачивание фото %d/%d для user_id=%s...", idx, len(photos_to_verify), user.id)
            data = None
            last_err = None
            for attempt in range(1, 4):
                try:
                    tg_file = await context.bot.get_file(p.file_id, read_timeout=60.0, write_timeout=60.0)
                    data = await tg_file.download_as_bytearray(read_timeout=60.0, write_timeout=60.0)
                    break
                except Exception as dl_err:
                    last_err = dl_err
                    logger.warning("Попытка %d/3 скачивания фото %d не удалась: %s", attempt, idx, dl_err)
                    if attempt < 3:
                        await asyncio.sleep(1.5)

            if not data:
                raise last_err or TimeoutError("Превышено время ожидания скачивания фото от серверов Telegram.")

            images_bytes.append(bytes(data))

        # 2. AI проверка через выбранный AI-провайдер (Gemini / OpenAI)
        approved, reason = await ai_verifier.verify_screenshots(images_bytes)

        if approved:
            # 3. Генерация инвайт-ссылки в канал / чат
            invite_link, link_note, _ = await generate_channel_invite_link(
                bot=context.bot,
                channel_chat_id=settings.channel_chat_id,
                user=user
            )

            # 4. Запись успешной попытки в БД
            await db.record_attempt(
                user_id=user.id,
                approved=True,
                reason=reason,
                invite_link=invite_link
            )

            success_text = (
                "🎉 <b>Поздравляем, проверка успешно пройдена!</b>\n\n"
                f"💬 <b>Вердикт AI ({provider_name}):</b>\n<i>{reason}</i>\n\n"
            )

            if invite_link:
                success_text += (
                    f"👉 <b>Ваша ссылка для входа:</b>\n"
                    f'<a href="{invite_link}">{invite_link}</a>\n'
                    f"{link_note}\n\n"
                    "Нажмите на кнопку ниже или перейдите по ссылке, чтобы вступить! 🚀"
                )
            else:
                success_text += (
                    "✅ Условия подтверждены, обратитесь к администратору для получения ссылки."
                    f"{link_note}"
                )

            await status_msg.edit_text(
                success_text,
                parse_mode=ParseMode.HTML,
                reply_markup=get_success_keyboard(invite_link)
            )

        else:
            # AI отклонил заявку
            await db.record_attempt(
                user_id=user.id,
                approved=False,
                reason=reason
            )

            reject_text = (
                "❌ <b>К сожалению, проверка не пройдена.</b>\n\n"
                f"💬 <b>Причина отказа ({provider_name}):</b>\n{reason}\n\n"
                "🔄 <b>Что делать дальше?</b>\n"
                "1. Ознакомьтесь с замечанием AI выше.\n"
                "2. Сделайте правильные скриншоты и отправьте их заново.\n"
                "3. Если вы уверены, что выполнили все условия, или произошла ошибка — "
                "нажмите кнопку ниже для связи с администратором."
            )
            await status_msg.edit_text(
                reject_text,
                parse_mode=ParseMode.HTML,
                reply_markup=get_admin_contact_keyboard()
            )

    except Exception as e:
        logger.exception("Непредвиденная ошибка при проверке для user_id=%s: %s", user.id, e)
        await status_msg.edit_text(
            f"⚠️ <b>Произошла системная ошибка при обработке скриншотов:</b>\n<code>{str(e)}</code>\n\n"
            "Пожалуйста, попробуйте еще раз позже или свяжитесь с администратором кнопкой ниже.",
            parse_mode=ParseMode.HTML,
            reply_markup=get_admin_contact_keyboard()
        )
