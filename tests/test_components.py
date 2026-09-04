"""
Комплексный автоматический тест компонентов: БД, AI-парсера, медиа-буфера и веб-интерфейса.
"""

import os
import sys
import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Добавляем корневой путь в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Settings
from database import Database
from ai_verifier import AIVerifier
from media_collector import MediaCollector
from web.server import create_web_app
from fastapi.testclient import TestClient


class TestCheckBot(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Временная база данных для тестов
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        self.db = Database(self.temp_db_path)
        await self.db.init_db()

    async def asyncTearDown(self):
        os.close(self.temp_db_fd)
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except Exception:
                pass

    async def test_database_lifecycle(self):
        """Тестирование полного цикла работы с БД."""
        # 1. Upsert пользователя
        user = await self.db.upsert_user(
            user_id=123456,
            username="test_user",
            first_name="Ivan",
            last_name="Ivanov"
        )
        self.assertEqual(user["user_id"], 123456)
        self.assertEqual(user["username"], "test_user")
        self.assertEqual(user["status"], "pending")

        # 2. Неудачная попытка
        await self.db.record_attempt(
            user_id=123456,
            approved=False,
            reason="Не видна подписка"
        )
        user_after_fail = await self.db.get_user(123456)
        self.assertEqual(user_after_fail["status"], "rejected")
        self.assertEqual(user_after_fail["attempts_count"], 1)
        self.assertEqual(user_after_fail["rejection_reason"], "Не видна подписка")

        # 3. Успешная попытка
        await self.db.record_attempt(
            user_id=123456,
            approved=True,
            reason="Все условия выполнены",
            invite_link="https://t.me/+AbCdEfGh"
        )
        user_after_success = await self.db.get_user(123456)
        self.assertEqual(user_after_success["status"], "verified")
        self.assertEqual(user_after_success["attempts_count"], 2)
        self.assertEqual(user_after_success["invite_link"], "https://t.me/+AbCdEfGh")
        self.assertIsNone(user_after_success["rejection_reason"])

        # 4. Проверка логов
        logs = await self.db.get_user_logs(123456)
        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[0]["approved"], 1)  # Последний лог
        self.assertEqual(logs[1]["approved"], 0)

        # 5. Проверка статистики
        stats = await self.db.get_stats()
        self.assertEqual(stats["total_users"], 1)
        self.assertEqual(stats["verified_users"], 1)
        self.assertEqual(stats["rejected_users"], 0)
        self.assertEqual(stats["total_attempts"], 2)

    async def test_gemini_verifier_parsing(self):
        """Тестирование вызова и парсинга ответа Google Gemini API с моком."""
        verifier = AIVerifier(provider="gemini", gemini_api_key="mock_gemini_key", gemini_model="gemini-2.5-flash")

        mock_gemini_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"approved": true, "reason": "Gemini: все скриншоты в порядке."}'

        mock_gemini_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        verifier.gemini_client = mock_gemini_client

        approved, reason = await verifier.verify_screenshots(
            images_bytes=[b"fake_image_bytes_1", b"fake_image_bytes_2"],
            custom_criteria="Тестовый критерий"
        )

        self.assertTrue(approved)
        self.assertIn("Gemini: все скриншоты в порядке", reason)

    async def test_openai_verifier_parsing(self):
        """Тестирование вызова и парсинга ответа OpenAI GPT-4o с моком."""
        verifier = AIVerifier(provider="openai", openai_api_key="mock_openai_key", openai_model="gpt-4o")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"approved": true, "reason": "OpenAI: условия соблюдены."}'
                )
            )
        ]

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        verifier.openai_client = mock_openai_client

        approved, reason = await verifier.verify_screenshots(
            images_bytes=[b"fake_image_bytes_1", b"fake_image_bytes_2"],
            custom_criteria="Тестовый критерий"
        )

    async def test_grok_verifier_parsing(self):
        """Тестирование вызова и парсинга ответа xAI Grok API с моком."""
        verifier = AIVerifier(provider="grok", xai_api_key="mock_xai_key", xai_model="grok-4.20-non-reasoning")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"approved": true, "reason": "Grok: условия соблюдены."}'
                )
            )
        ]

        mock_grok_client = MagicMock()
        mock_grok_client.chat.completions.create = AsyncMock(return_value=mock_response)
        verifier.grok_client = mock_grok_client

        approved, reason = await verifier.verify_screenshots(
            images_bytes=[b"fake_image_bytes_1", b"fake_image_bytes_2"],
            custom_criteria="Тестовый критерий"
        )

        self.assertTrue(approved)
        self.assertIn("Grok: условия соблюдены", reason)

    async def test_grok_verifier_model_fallback(self):
        """Тестирование автоматического перехода на резервную модель при ошибке Model not found."""
        verifier = AIVerifier(provider="grok", xai_api_key="mock_xai_key", xai_model="deprecated-custom-model")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"approved": true, "reason": "Fallback Grok: успешно."}'
                )
            )
        ]

        mock_grok_client = MagicMock()
        # Первый вызов падает с 400 Model not found, второй (fallback) успешен
        mock_grok_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("Error code: 400 - {'code': 'invalid-argument', 'error': 'Model not found: deprecated-custom-model'}"),
                mock_response
            ]
        )
        verifier.grok_client = mock_grok_client

        approved, reason = await verifier.verify_screenshots(
            images_bytes=[b"fake_image_bytes_1", b"fake_image_bytes_2"],
            custom_criteria="Тестовый критерий"
        )

        self.assertTrue(approved)
        self.assertIn("Fallback Grok: успешно", reason)

    def test_admin_contact_button(self):
        """Тестирование генерации кнопки связи с администратором."""
        from handlers.photo_handler import get_admin_contact_keyboard
        from config import settings

        # Ссылка установлена
        settings.admin_telegram_username = "@test_admin"
        kb = get_admin_contact_keyboard()
        self.assertIsNotNone(kb)
        self.assertEqual(kb.inline_keyboard[0][0].url, "https://t.me/test_admin")

        # Ссылка не установлена
        settings.admin_telegram_username = ""
        kb_empty = get_admin_contact_keyboard()
        self.assertIsNotNone(kb_empty)
        self.assertEqual(kb_empty.inline_keyboard[0][0].callback_data, "contact_admin")

    async def test_media_collector_album(self):
        """Тестирование сборки альбома из 2 фото с небольшой задержкой."""
        collector = MediaCollector(album_delay=0.1)
        completed_results = []

        async def callback(photos, messages):
            completed_results.append((photos, messages))

        photo1 = MagicMock(file_id="photo_1")
        msg1 = MagicMock(message_id=101)

        photo2 = MagicMock(file_id="photo_2")
        msg2 = MagicMock(message_id=102)

        # Добавляем 2 фото в один media_group_id
        await collector.add_photo_from_media_group("group_abc", msg1, photo1, callback)
        await collector.add_photo_from_media_group("group_abc", msg2, photo2, callback)

        # Ждем завершения таймаута альбома
        await asyncio.sleep(0.25)

        self.assertEqual(len(completed_results), 1)
        photos, messages = completed_results[0]
        self.assertEqual(len(photos), 2)
        self.assertEqual(photos[0].file_id, "photo_1")
        self.assertEqual(photos[1].file_id, "photo_2")

    async def test_database_filtering(self):
        """Тестирование поиска и фильтрации пользователей по статусу в БД."""
        await self.db.upsert_user(user_id=1, username="alice", first_name="Alice")
        await self.db.upsert_user(user_id=2, username="bob", first_name="Bob")
        await self.db.record_attempt(user_id=1, approved=True, reason="OK", invite_link="https://t.me/+111")
        await self.db.record_attempt(user_id=2, approved=False, reason="Bad")

        # Поиск по статусу verified
        verified = await self.db.get_all_users(status="verified")
        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0]["user_id"], 1)

        # Поиск по статусу rejected
        rejected = await self.db.get_all_users(status="rejected")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["user_id"], 2)

        # Поиск по имени alice
        found_alice = await self.db.get_all_users(search="alice")
        self.assertEqual(len(found_alice), 1)
        self.assertEqual(found_alice[0]["user_id"], 1)

    def test_bot_application_creation(self):
        """Тестирование корректности создания приложения Telegram бота."""
        from bot import create_bot_application
        app = create_bot_application(self.db)
        self.assertIsNotNone(app)
        self.assertEqual(app.bot_data["db"], self.db)
        # Проверяем, что зарегистрированы обработчики
        self.assertTrue(len(app.handlers) > 0)

    def test_web_interface_auth_and_api(self):
        """Тестирование авторизации и эндпоинтов веб-панели."""
        app = create_web_app(self.db)
        client = TestClient(app)

        # Без авторизации -> 401
        res_no_auth = client.get("/")
        self.assertEqual(res_no_auth.status_code, 401)

        # С правильной авторизацией (admin:admin по умолчанию в тестах)
        auth = ("admin", "admin")
        res_home = client.get("/", auth=auth)
        self.assertEqual(res_home.status_code, 200)
        self.assertIn("CheckBot Admin", res_home.text)

        # Проверка API статистики
        res_stats = client.get("/api/stats", auth=auth)
        self.assertEqual(res_stats.status_code, 200)
        data = res_stats.json()
        self.assertIn("total_users", data)


if __name__ == "__main__":
    unittest.main()
