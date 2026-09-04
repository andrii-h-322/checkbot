"""
Асинхронный модуль для работы с базой данных SQLite через aiosqlite.
Хранит пользователей, историю верификаций и сгенерированные ссылки доступа.
"""

import aiosqlite
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str = "bot.db"):
        self.db_path = db_path

    async def init_db(self) -> None:
        """Инициализация таблиц базы данных и включение WAL-режима."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA foreign_keys=ON;")

            # Таблица пользователей
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    status TEXT DEFAULT 'pending',
                    invite_link TEXT,
                    attempts_count INTEGER DEFAULT 0,
                    rejection_reason TEXT,
                    verified_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Таблица истории проверок скриншотов
            await db.execute("""
                CREATE TABLE IF NOT EXISTS verification_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    approved INTEGER NOT NULL,
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
            """)

            # Индексы для оптимизации выборок
            await db.execute("CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_logs_user_id ON verification_logs(user_id);")

            await db.commit()
            logger.info("База данных SQLite успешно инициализирована: %s", self.db_path)

    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Получить пользователя по Telegram ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def upsert_user(
        self,
        user_id: int,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Добавить пользователя или обновить его профиль при контакте с ботом."""
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO users (user_id, username, first_name, last_name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    username = excluded.username,
                    first_name = excluded.first_name,
                    last_name = excluded.last_name,
                    updated_at = excluded.updated_at
            """, (user_id, username, first_name, last_name, now, now))
            await db.commit()

        user = await self.get_user(user_id)
        return user or {}

    async def record_attempt(
        self,
        user_id: int,
        approved: bool,
        reason: str,
        invite_link: Optional[str] = None
    ) -> None:
        """Записать результат проверки AI и обновить статус пользователя."""
        now = datetime.now(timezone.utc).isoformat()
        status = "verified" if approved else "rejected"
        appr_int = 1 if approved else 0

        async with aiosqlite.connect(self.db_path) as db:
            # 1. Лог попытки
            await db.execute("""
                INSERT INTO verification_logs (user_id, approved, reason, created_at)
                VALUES (?, ?, ?, ?)
            """, (user_id, appr_int, reason, now))

            # 2. Обновление пользователя
            if approved:
                await db.execute("""
                    UPDATE users
                    SET status = ?,
                        invite_link = COALESCE(?, invite_link),
                        attempts_count = attempts_count + 1,
                        rejection_reason = NULL,
                        verified_at = ?,
                        updated_at = ?
                    WHERE user_id = ?
                """, (status, invite_link, now, now, user_id))
            else:
                await db.execute("""
                    UPDATE users
                    SET status = ?,
                        attempts_count = attempts_count + 1,
                        rejection_reason = ?,
                        updated_at = ?
                    WHERE user_id = ?
                """, (status, reason, now, user_id))

            await db.commit()

    async def get_all_users(
        self,
        search: str = "",
        status: str = "",
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Получение списка пользователей с фильтрацией и пагинацией для веб-интерфейса."""
        query = "SELECT * FROM users WHERE 1=1"
        params: List[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if search:
            query += " AND (CAST(user_id AS TEXT) LIKE ? OR username LIKE ? OR first_name LIKE ?)"
            s_param = f"%{search}%"
            params.extend([s_param, s_param, s_param])

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]

    async def get_total_users_count(self, search: str = "", status: str = "") -> int:
        """Подсчет общего числа пользователей с учетом фильтров."""
        query = "SELECT COUNT(*) FROM users WHERE 1=1"
        params: List[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if search:
            query += " AND (CAST(user_id AS TEXT) LIKE ? OR username LIKE ? OR first_name LIKE ?)"
            s_param = f"%{search}%"
            params.extend([s_param, s_param, s_param])

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0

    async def get_stats(self) -> Dict[str, int]:
        """Статистика для дашборда."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM users") as c:
                total = (await c.fetchone())[0]

            async with db.execute("SELECT COUNT(*) FROM users WHERE status = 'verified'") as c:
                verified = (await c.fetchone())[0]

            async with db.execute("SELECT COUNT(*) FROM users WHERE status = 'rejected'") as c:
                rejected = (await c.fetchone())[0]

            async with db.execute("SELECT COUNT(*) FROM verification_logs") as c:
                total_attempts = (await c.fetchone())[0]

            return {
                "total_users": total,
                "verified_users": verified,
                "rejected_users": rejected,
                "pending_users": total - (verified + rejected),
                "total_attempts": total_attempts
            }

    async def get_user_logs(self, user_id: int) -> List[Dict[str, Any]]:
        """Получить логи верификаций для конкретного пользователя."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM verification_logs WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]
