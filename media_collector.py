"""
Модуль буферизации и сбора медиа-групп (альбомов) и одиночных фотографий.
Решает проблему Telegram Bot API, когда фото в альбоме приходят отдельными обновлениями с media_group_id.
Также поддерживает пошаговую отправку двух фото подряд.
"""

import asyncio
import logging
import time
from typing import Callable, Dict, Any, List, Optional
from telegram import Message, PhotoSize

logger = logging.getLogger(__name__)


class MediaCollector:
    def __init__(self, album_delay: float = 1.5, single_photo_timeout: float = 45.0):
        """
        :param album_delay: Задержка в секундах для сбора всех фото из альбома (media_group_id)
        :param single_photo_timeout: Время ожидания второго фото, если отправлены поштучно
        """
        self.album_delay = album_delay
        self.single_photo_timeout = single_photo_timeout

        # media_group_id -> { "photos": [PhotoSize], "messages": [Message], "task": asyncio.Task }
        self._media_groups: Dict[str, Dict[str, Any]] = {}

        # user_id -> { "photos": [PhotoSize], "messages": [Message], "task": asyncio.Task, "updated_at": float }
        self._user_single_buffers: Dict[int, Dict[str, Any]] = {}

    async def add_photo_from_media_group(
        self,
        media_group_id: str,
        message: Message,
        best_photo: PhotoSize,
        on_complete: Callable[[List[PhotoSize], List[Message]], Any]
    ) -> None:
        """Добавить фото из альбома (media_group_id). Ждет album_delay перед вызовом on_complete."""
        if media_group_id not in self._media_groups:
            self._media_groups[media_group_id] = {
                "photos": [best_photo],
                "messages": [message],
                "task": None
            }

            async def _waiter():
                try:
                    await asyncio.sleep(self.album_delay)
                    group_data = self._media_groups.pop(media_group_id, None)
                    if group_data:
                        await on_complete(group_data["photos"], group_data["messages"])
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.exception("Ошибка в обработчике альбома media_group_id=%s: %s", media_group_id, e)

            task = asyncio.create_task(_waiter())
            self._media_groups[media_group_id]["task"] = task
        else:
            # Добавляем очередное фото альбома
            self._media_groups[media_group_id]["photos"].append(best_photo)
            self._media_groups[media_group_id]["messages"].append(message)

    async def add_single_photo(
        self,
        user_id: int,
        message: Message,
        best_photo: PhotoSize,
        on_complete: Callable[[List[PhotoSize], List[Message]], Any],
        on_need_more: Optional[Callable[[int, Message], Any]] = None
    ) -> None:
        """
        Обработка одиночного фото: для проверки требуется ровно 1 скриншот,
        поэтому сразу запускаем on_complete без ожидания и задержек.
        """
        logger.info("Получен одиночный скриншот от user_id=%s. Запуск проверки работоспособности...", user_id)
        await on_complete([best_photo], [message])


# Синглтон коллектора
media_collector = MediaCollector()
