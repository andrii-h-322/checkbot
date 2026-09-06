"""
Модуль конфигурации бота и веб-панели.
Загружает и валидирует параметры из файла .env и переменных окружения.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Telegram Bot
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    channel_id: str = Field(default="", alias="CHANNEL_ID")

    # Контакт администратора для кнопки связи
    admin_telegram_username: str = Field(default="", alias="ADMIN_TELEGRAM_USERNAME")

    # AI Provider: 'grok', 'gemini' или 'openai'
    ai_provider: str = Field(default="grok", alias="AI_PROVIDER")

    # xAI Grok API
    xai_api_key: str = Field(default="", alias="XAI_API_KEY")
    xai_model: str = Field(default="grok-4.20-non-reasoning", alias="XAI_MODEL")

    # Google Gemini API
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.6-flash", alias="GEMINI_MODEL")

    # OpenAI API
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")

    # Параметры инвайт-ссылки
    invite_link_expire_hours: int = Field(default=24, alias="INVITE_LINK_EXPIRE_HOURS")
    invite_link_member_limit: int = Field(default=0, alias="INVITE_LINK_MEMBER_LIMIT")
    invite_link_fallback: str = Field(default="", alias="INVITE_LINK_FALLBACK")

    # Критерии для проверки AI (проверка 2 скриншотов комментариев в TikTok с текстом "Мне скинул @fulyashki")
    verification_criteria: str = Field(
        default=(
            "Проверь предоставленные скриншоты (пользователь должен прислать 2 скриншота из TikTok). "
            "Критерии проверки: "
            "1) На скриншотах должен быть виден интерфейс TikTok и оставленные комментарии пользователя; "
            "2) Текст комментариев должен содержать фразу 'Мне скинул @fulyashki' или близкий по смыслу вариант (например: 'мне скинул fulyashki', 'скинул @fulyashki', 'от @fulyashki' и т.п.); "
            "3) На скриншотах должны быть разные видео/комментарии или подтверждение выполнения двух комментариев; "
            "4) Если на скриншотах четко видны комментарии в TikTok с упоминанием @fulyashki — одобри заявку (approved=true) и в поле 'reason' напиши вежливое подтверждение на русском языке; "
            "5) Если скриншоты не из TikTok, комментарии отсутствуют, не содержат нужного текста, неразборчивы или условие не выполнено — отклони (approved=false) и вежливо, понятно объясни пользователю на русском языке причину отказа и что нужно исправить."
        ),
        alias="VERIFICATION_CRITERIA"
    )

    # Web Dashboard
    web_host: str = Field(default="0.0.0.0", alias="WEB_HOST")
    web_port: int = Field(default=8080, alias="WEB_PORT")
    admin_username: str = Field(default="admin", alias="ADMIN_USERNAME")
    admin_password: str = Field(default="admin", alias="ADMIN_PASSWORD")

    # База данных
    database_path: str = Field(default="bot.db", alias="DATABASE_PATH")

    @property
    def channel_chat_id(self) -> int | str:
        """Возвращает channel_id как int, если возможно, иначе как строку (username канала)."""
        if not self.channel_id:
            return ""
        val = str(self.channel_id).strip().strip('"').strip("'")
        if not val:
            return ""
        try:
            return int(val)
        except (ValueError, TypeError):
            if not val.startswith("@") and not val.startswith("-"):
                val = f"@{val}"
            return val

    @property
    def admin_contact_url(self) -> Optional[str]:
        """Возвращает корректный URL для ссылки на диалог с администратором."""
        val = self.admin_telegram_username.strip()
        if not val:
            return None
        if val.startswith("http://") or val.startswith("https://"):
            return val
        if val.startswith("@"):
            val = val[1:]
        return f"https://t.me/{val}"


# Синглтон настроек
settings = Settings()
