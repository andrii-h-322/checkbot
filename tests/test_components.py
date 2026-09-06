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

    async def test_grok_verifier_auth_error_handling(self):
        """Тестирование понятного сообщения об ошибке при неверном ключе API xAI."""
        verifier = AIVerifier(provider="grok", xai_api_key="xai-bad-key", xai_model="grok-4.20-non-reasoning")

        mock_grok_client = MagicMock()
        mock_grok_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Error code: 400 - {'code': 'invalid-argument', 'error': 'Incorrect API key provided. You can obtain an API key from https://console.x.ai.'}")
        )
        verifier.grok_client = mock_grok_client

        approved, reason = await verifier.verify_screenshots(
            images_bytes=[b"fake_image_bytes_1", b"fake_image_bytes_2"],
            custom_criteria="Тестовый критерий"
        )

        self.assertFalse(approved)
        self.assertIn("Ошибка авторизации в xAI Grok API", reason)
        self.assertIn("console.x.ai", reason)

    async def test_grok_verifier_empty_key_prevalidation(self):
        """Тестирование быстрой валидации при пустом ключе xAI."""
        verifier = AIVerifier(provider="grok", xai_api_key="", xai_model="grok-4.20-non-reasoning")

        approved, reason = await verifier.verify_screenshots(
            images_bytes=[b"fake_image_bytes_1", b"fake_image_bytes_2"],
            custom_criteria="Тестовый критерий"
        )

        self.assertFalse(approved)
        self.assertIn("Не указан API-ключ xAI Grok", reason)

    async def test_groq_cloud_decommissioned_fallback(self):
        """Тестирование автоматического перехода на резервную модель GroqCloud при ошибке model_decommissioned."""
        verifier = AIVerifier(provider="grok", xai_api_key="gsk_mock_groq_key", xai_model="llama-3.2-11b-vision-preview")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"approved": true, "reason": "GroqCloud: скриншоты проверены."}'
                )
            )
        ]

        mock_client = MagicMock()
        # Первая модель llama-3.2-11b возвращает ошибку model_decommissioned, вторая (qwen/qwen3.6-27b) успешна
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("Error code: 400 - {'error': {'message': 'The model llama-3.2-11b-vision-preview has been decommissioned and is no longer supported.', 'code': 'model_decommissioned'}}"),
                mock_response
            ]
        )
        verifier.grok_client = mock_client

        approved, reason = await verifier.verify_screenshots(
            images_bytes=[b"fake_image_bytes_1", b"fake_image_bytes_2"],
            custom_criteria="Тестовый критерий"
        )

        self.assertTrue(approved)
        self.assertIn("GroqCloud: скриншоты проверены", reason)

    async def test_groq_cloud_json_validate_failed_fallback(self):
        """Тестирование повторного вызова без response_format при ошибке json_validate_failed."""
        verifier = AIVerifier(provider="grok", xai_api_key="gsk_mock_groq_key", xai_model="qwen/qwen3.8-27b")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='<think>Размышляю...</think>```json\n{"approved": true, "reason": "Все условия соблюдены!"}\n```'
                )
            )
        ]

        mock_client = MagicMock()
        # Первый вызов падает с 400 json_validate_failed (max completion tokens reached)
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("Error code: 400 - {'error': {'message': \"Failed to generate JSON.\", 'code': 'json_validate_failed', 'failed_generation': 'max completion tokens reached before generating a valid document'}}"),
                mock_response
            ]
        )
        verifier.grok_client = mock_client

        approved, reason = await verifier.verify_screenshots(
            images_bytes=[b"fake_image_bytes_1", b"fake_image_bytes_2"],
            custom_criteria="Тестовый критерий"
        )

        self.assertTrue(approved)
        self.assertIn("Все условия соблюдены", reason)

    async def test_groq_cloud_otpm_limit_auto_reduction(self):
        """Тестирование автоматического уменьшения max_tokens до 150 при превышении OTPM лимита GroqCloud."""
        verifier = AIVerifier(provider="grok", xai_api_key="gsk_mock_groq_key", xai_model="qwen/qwen3.8-27b")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"approved": true, "reason": "OTPM auto-reduced: успешно проверено."}'
                )
            )
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                Exception("Error code: 429 - {'error': {'message': \"Request too large for model `qwen/qwen3.8-27b` on output tokens per minute (OTPM): Limit 1000, Requested 1453. The request's expected output tokens exceed the enforced limit; reduce max_tokens (or the request's expected output) and try again.\", 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"),
                mock_response
            ]
        )
        verifier.grok_client = mock_client

        approved, reason = await verifier.verify_screenshots(
            images_bytes=[b"fake_image_bytes_1", b"fake_image_bytes_2"],
            custom_criteria="Тестовый критерий"
        )

        self.assertTrue(approved)
        self.assertIn("OTPM auto-reduced: успешно проверено", reason)
        # Проверяем, что второй вызов был с max_tokens=150
        second_call_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
        self.assertEqual(second_call_kwargs.get("max_tokens"), 150)

    async def test_groq_cloud_429_all_models_rate_limit(self):
        """Тестирование понятного сообщения пользователю при исчерпании лимитов запросов 429 на всех моделях."""
        verifier = AIVerifier(provider="grok", xai_api_key="gsk_mock_groq_key", xai_model="qwen/qwen3.8-27b")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Error code: 429 - {'error': {'message': 'Rate limit exceeded: TPM limit reached.', 'code': 'rate_limit_exceeded'}}")
        )
        verifier.grok_client = mock_client

        approved, reason = await verifier.verify_screenshots(
            images_bytes=[b"fake_image_bytes_1", b"fake_image_bytes_2"],
            custom_criteria="Тестовый критерий"
        )

        self.assertFalse(approved)
        self.assertIn("Превышен лимит запросов в минуту в GroqCloud", reason)
        self.assertIn("30–60 секунд", reason)

    def test_clean_and_parse_json(self):
        """Тестирование извлечения JSON из ответов с reasoning tokens и markdown."""
        text_with_think = "<think>Длинные размышления модели...</think>{\"approved\": true, \"reason\": \"Успешно проверено\"}"
        appr, reason = AIVerifier._clean_and_parse_json(text_with_think)
        self.assertTrue(appr)
        self.assertEqual(reason, "Успешно проверено")

        text_with_markdown = "Вот ваш ответ:\n```json\n{\"approved\": false, \"reason\": \"Нет комментариев\"}\n```"
        appr, reason = AIVerifier._clean_and_parse_json(text_with_markdown)
        self.assertFalse(appr)
        self.assertEqual(reason, "Нет комментариев")

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

    def test_get_success_keyboard(self):
        """Тестирование клавиатуры с кнопкой вступления в чат/канал."""
        from handlers.photo_handler import get_success_keyboard
        from config import settings

        settings.admin_telegram_username = "@test_admin"

        # С ссылкой на канал
        kb = get_success_keyboard(invite_link="https://t.me/+test_link_123")
        self.assertEqual(len(kb.inline_keyboard), 2)
        self.assertEqual(kb.inline_keyboard[0][0].text, "🚀 Вступить в канал / чат")
        self.assertEqual(kb.inline_keyboard[0][0].url, "https://t.me/+test_link_123")
        self.assertEqual(kb.inline_keyboard[1][0].text, "💬 Связаться с администратором")

        # Без ссылки (если генерация не удалась)
        kb_no_link = get_success_keyboard(invite_link=None)
        self.assertEqual(len(kb_no_link.inline_keyboard), 1)
        self.assertEqual(kb_no_link.inline_keyboard[0][0].text, "💬 Связаться с администратором")

    async def test_generate_channel_invite_link_multilevel(self):
        """Тестирование многоуровневой генерации инвайт-ссылок."""
        from handlers.photo_handler import generate_channel_invite_link
        from telegram.error import TelegramError
        from config import settings

        mock_user = MagicMock(id=123456, username="sample_user")

        # Уровень 1: успешное создание ссылки с первого вызова
        mock_bot = MagicMock()
        mock_link = MagicMock(invite_link="https://t.me/+level1_success")
        mock_bot.create_chat_invite_link = AsyncMock(return_value=mock_link)

        link, note, err = await generate_channel_invite_link(mock_bot, -1001234567890, mock_user)
        self.assertEqual(link, "https://t.me/+level1_success")
        self.assertIsNone(err)

        # Уровень 2: Уровень 1 упал, Уровень 2 (базовый) успешен
        mock_bot_lvl2 = MagicMock()
        mock_lvl2_link = MagicMock(invite_link="https://t.me/+level2_basic")
        mock_bot_lvl2.create_chat_invite_link = AsyncMock(
            side_effect=[
                TelegramError("Bad Request: custom name not allowed"),
                mock_lvl2_link
            ]
        )
        link2, _, err2 = await generate_channel_invite_link(mock_bot_lvl2, -1001234567890, mock_user)
        self.assertEqual(link2, "https://t.me/+level2_basic")
        self.assertIsNone(err2)

        # Уровень 3: Уровни 1 и 2 упали (нет прав на создание), экспорт постоянной ссылки успешен
        mock_bot_lvl3 = MagicMock()
        mock_bot_lvl3.create_chat_invite_link = AsyncMock(side_effect=TelegramError("Not enough rights"))
        mock_bot_lvl3.export_chat_invite_link = AsyncMock(return_value="https://t.me/+level3_exported")
        link3, _, err3 = await generate_channel_invite_link(mock_bot_lvl3, -1001234567890, mock_user)
        self.assertEqual(link3, "https://t.me/+level3_exported")
        self.assertIsNone(err3)

        # Уровень 5: Резервная ссылка INVITE_LINK_FALLBACK
        mock_bot_lvl5 = MagicMock()
        mock_bot_lvl5.create_chat_invite_link = AsyncMock(side_effect=TelegramError("Chat not found"))
        mock_bot_lvl5.export_chat_invite_link = AsyncMock(side_effect=TelegramError("Chat not found"))
        mock_bot_lvl5.get_chat = AsyncMock(side_effect=TelegramError("Chat not found"))

        settings.invite_link_fallback = "https://t.me/+fallback_link"
        link5, _, err5 = await generate_channel_invite_link(mock_bot_lvl5, -1001234567890, mock_user)
        self.assertEqual(link5, "https://t.me/+fallback_link")
        self.assertIsNone(err5)
        settings.invite_link_fallback = ""

        # Уровень 6: Все упало и нет резерва -> понятная ошибка с кодом ошибки
        link6, note6, err6 = await generate_channel_invite_link(mock_bot_lvl5, -1001234567890, mock_user)
        self.assertIsNone(link6)
        self.assertIn("Chat not found", note6)
        self.assertIn("Как исправить", note6)

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

    async def test_media_collector_single_photo_immediate(self):
        """Тестирование мгновенной обработки одиночного фото для проверки 1 скриншота."""
        collector = MediaCollector()
        completed_results = []

        async def callback(photos, messages):
            completed_results.append((photos, messages))

        photo1 = MagicMock(file_id="single_photo_1")
        msg1 = MagicMock(message_id=201)

        # Отправляем одиночное фото
        await collector.add_single_photo(user_id=999, message=msg1, best_photo=photo1, on_complete=callback)

        # Обработка должна сработать сразу без задержек и таймаутов
        self.assertEqual(len(completed_results), 1)
        photos, messages = completed_results[0]
        self.assertEqual(len(photos), 1)
        self.assertEqual(photos[0].file_id, "single_photo_1")
        self.assertEqual(messages[0].message_id, 201)

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
