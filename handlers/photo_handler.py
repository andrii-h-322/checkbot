"""
Обработчик полученных фотографий и скриншотов.
Интегрирует MediaCollector для альбомов и поштучной отправки,
AI-верификатор (Gemini / OpenAI) для проверки, Telegram API для генерации ссылок
и кнопку оперативной связи с администратором при возникновении вопросов или отказе.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional
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
    provider_name = "Google Gemini" if settings.ai_provider.lower() == "gemini" else "GPT-4o"
    status_msg = await reply_to_message.reply_text(
        f"⏳ <b>Скриншоты получены!</b>\n"
        f"Передаю изображения в модуль <b>{provider_name}</b> для анализа... Это займет несколько секунд.",
        parse_mode=ParseMode.HTML,
        reply_markup=get_admin_contact_keyboard()
    )

    try:
        await context.bot.send_chat_action(chat_id=reply_to_message.chat_id, action=ChatAction.TYPING)

        # 1. Скачивание изображений в память
        images_bytes: List[bytes] = []
        for idx, p in enumerate(photos_to_verify, start=1):
            logger.info("Скачивание фото %d/%d для user_id=%s...", idx, len(photos_to_verify), user.id)
            tg_file = await context.bot.get_file(p.file_id)
            data = await tg_file.download_as_bytearray()
            images_bytes.append(bytes(data))

        # 2. AI проверка через выбранный AI-провайдер (Gemini / OpenAI)
        approved, reason = await ai_verifier.verify_screenshots(images_bytes)

        if approved:
            # 3. Генерация инвайт-ссылки в канал
            invite_link = None
            link_note = ""

            channel_chat_id = settings.channel_chat_id
            if channel_chat_id:
                try:
                    expire_date = None
                    if settings.invite_link_expire_hours > 0:
                        expire_date = datetime.now(timezone.utc) + timedelta(hours=settings.invite_link_expire_hours)
                        link_note += f"\n⏳ Срок действия ссылки: {settings.invite_link_expire_hours} ч."

                    member_limit = settings.invite_link_member_limit if settings.invite_link_member_limit > 0 else None
                    if member_limit == 1:
                        link_note += "\n🔒 Ссылка является одноразовой (для 1 участника)."
                    elif member_limit and member_limit > 1:
                        link_note += f"\n👥 Лимит входов по ссылке: {member_limit}."

                    link_obj = await context.bot.create_chat_invite_link(
                        chat_id=channel_chat_id,
                        name=f"ID {user.id} - @{user.username or user.first_name}",
                        expire_date=expire_date,
                        member_limit=member_limit
                    )
                    invite_link = link_obj.invite_link
                    logger.info("Сгенерирована ссылка для user_id=%s: %s", user.id, invite_link)

                except TelegramError as te:
                    logger.error("Ошибка при создании invite-link в канале %s: %s", channel_chat_id, te)
                    link_note = (
                        "\n\n⚠️ <i>Не удалось сгенерировать ссылку автоматически. "
                        "Убедитесь, что бот добавлен в целевой канал как Администратор "
                        "с правом «Приглашать пользователей».</i>"
                    )
            else:
                link_note = "\n\n⚠️ <i>CHANNEL_ID не задан в конфигурации бота.</i>"

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
                    f"👉 <b>Ваша ссылка для входа в канал:</b>\n"
                    f"{invite_link}\n"
                    f"{link_note}\n\n"
                    "Добро пожаловать в наше сообщество! 🚀"
                )
            else:
                success_text += (
                    "✅ Условия подтверждены, обратитесь к администратору для получения ссылки."
                    f"{link_note}"
                )

            await status_msg.edit_text(
                success_text,
                parse_mode=ParseMode.HTML,
                reply_markup=get_admin_contact_keyboard()
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
