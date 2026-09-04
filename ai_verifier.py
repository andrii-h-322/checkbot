"""
Модуль AI-верификации скриншотов.
Поддерживает работу с Google Gemini API (google-genai) и OpenAI (GPT-4o).
Принимает список изображений в бинарном формате (байты), анализирует их на соответствие критериям
и возвращает структурированный ответ с решением (approved: bool, reason: str).
"""

import json
import base64
import logging
from typing import List, Tuple, Optional
from pydantic import BaseModel
from config import settings

logger = logging.getLogger(__name__)


class VerificationResult(BaseModel):
    """Схема структурированного ответа от AI."""
    approved: bool
    reason: str


class AIVerifier:
    def __init__(
        self,
        provider: str = "",
        gemini_api_key: str = "",
        gemini_model: str = "",
        openai_api_key: str = "",
        openai_model: str = ""
    ):
        self.provider = (provider or settings.ai_provider).lower()
        self.gemini_api_key = gemini_api_key or settings.gemini_api_key
        self.gemini_model = gemini_model or settings.gemini_model
        self.openai_api_key = openai_api_key or settings.openai_api_key
        self.openai_model = openai_model or settings.openai_model

        self._gemini_client = None
        self._openai_client = None

    @property
    def gemini_client(self):
        if self._gemini_client is None:
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY не указан! Укажите ключ в файле .env")
            from google import genai
            from google.genai import types
            self._gemini_client = genai.Client(
                api_key=self.gemini_api_key,
                http_options=types.HttpOptions(timeout=60000)
            )
        return self._gemini_client

    @gemini_client.setter
    def gemini_client(self, value) -> None:
        self._gemini_client = value

    @property
    def openai_client(self):
        if self._openai_client is None:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY не указан! Укажите ключ в файле .env")
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        return self._openai_client

    @openai_client.setter
    def openai_client(self, value) -> None:
        self._openai_client = value

    async def verify_screenshots(
        self,
        images_bytes: List[bytes],
        custom_criteria: str = ""
    ) -> Tuple[bool, str]:
        """
        Проверяет список изображений через выбранный AI-провайдер (Gemini или OpenAI).
        """
        if not images_bytes:
            return False, "Не передано ни одного изображения для проверки."

        criteria = custom_criteria or settings.verification_criteria

        if self.provider == "gemini":
            return await self._verify_with_gemini(images_bytes, criteria)
        else:
            return await self._verify_with_openai(images_bytes, criteria)

    async def _verify_with_gemini(
        self,
        images_bytes: List[bytes],
        criteria: str
    ) -> Tuple[bool, str]:
        """Проверка скриншотов через Google Gemini API (модель gemini-2.5-flash)."""
        from google.genai import types

        model_to_use = self.gemini_model
        # Автоматическая замена устаревшей модели на актуальную
        if "gemini-2.5-flash" in model_to_use:
            model_to_use = "gemini-3.6-flash"

        logger.info("Отправка %d скриншотов в Google Gemini API (%s)...", len(images_bytes), model_to_use)

        prompt = (
            "Ты профессиональный верификатор выполнения условий для предоставления доступа в закрытый Telegram-канал.\n"
            f"Тебе предоставлено {len(images_bytes)} скриншот(ов).\n"
            f"Критерии проверки:\n{criteria}\n\n"
            "Внимательно изучи каждый скриншот:\n"
            "• Если все условия выполнены — установи approved=true и подтверди выполнение.\n"
            "• Если чего-то не хватает, скриншоты обрезаны, условия не выполнены или изображения не относятся к делу — "
            "установи approved=false и вежливо, подробно объясни пользователю в поле 'reason' на русском языке, что не так и как это исправить."
        )

        contents = []
        for img in images_bytes:
            contents.append(types.Part.from_bytes(data=img, mime_type="image/jpeg"))
        contents.append(prompt)

        try:
            client = self.gemini_client
            # Вызов асинхронной модели Gemini
            response = await client.aio.models.generate_content(
                model=model_to_use,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VerificationResult,
                    temperature=0.2,
                    max_output_tokens=2048
                )
            )

            raw_text = response.text or "{}"
            logger.info("Ответ от Google Gemini API: %s", raw_text)

            parsed = json.loads(raw_text)
            approved = bool(parsed.get("approved", False))
            reason = str(parsed.get("reason", "Решение не содержит описания.")).strip()

            return approved, reason

        except Exception as e:
            logger.exception("Ошибка при обращении к Google Gemini API: %s", e)
            return False, f"Ошибка при проверке скриншотов через Google Gemini API: {str(e)}"

    async def _verify_with_openai(
        self,
        images_bytes: List[bytes],
        criteria: str
    ) -> Tuple[bool, str]:
        """Проверка скриншотов через OpenAI API (GPT-4o)."""
        logger.info("Отправка %d скриншотов в OpenAI (%s)...", len(images_bytes), self.openai_model)

        content_items = [
            {
                "type": "text",
                "text": (
                    f"Тебе предоставлено {len(images_bytes)} скриншот(ов).\n"
                    f"Критерии проверки:\n{criteria}\n\n"
                    "Проанализируй изображения. Убедись, что они соответствуют всем критериям.\n"
                    "Верни строго JSON объект следующего вида:\n"
                    "{\n"
                    '  "approved": true или false,\n'
                    '  "reason": "Четкое, вежливое объяснение на русском языке."\n'
                    "}"
                )
            }
        ]

        for img_data in images_bytes:
            b64_str = base64.b64encode(img_data).decode("utf-8")
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_str}",
                    "detail": "high"
                }
            })

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты профессиональный верификатор выполнения условий для предоставления доступа "
                    "в закрытый Telegram-канал. Всегда возвращай валидный JSON."
                )
            },
            {
                "role": "user",
                "content": content_items
            }
        ]

        try:
            client = self.openai_client
            response = await client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=600,
                temperature=0.2
            )

            raw_text = response.choices[0].message.content or "{}"
            logger.info("Ответ от OpenAI: %s", raw_text)

            parsed = json.loads(raw_text)
            approved = bool(parsed.get("approved", False))
            reason = str(parsed.get("reason", "Решение не содержит описания.")).strip()

            return approved, reason

        except Exception as e:
            logger.exception("Ошибка при обращении к OpenAI API: %s", e)
            return False, f"Произошла ошибка при анализе скриншотов через OpenAI: {str(e)}"


# Синглтон верификатора
ai_verifier = AIVerifier()
