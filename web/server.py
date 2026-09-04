"""
Веб-сервер на FastAPI для просмотра базы данных пользователей и статистики проверок.
Защищен базовой авторизацией (HTTP Basic Auth).
"""

import os
import secrets
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, Depends, HTTPException, status, Query, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from config import settings
from database import Database

security = HTTPBasic()
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


def authenticate(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Проверка логина и пароля администратора."""
    correct_username = secrets.compare_digest(credentials.username, settings.admin_username)
    correct_password = secrets.compare_digest(credentials.password, settings.admin_password)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def create_web_app(db: Database) -> FastAPI:
    """Создает приложение FastAPI с привязкой к базе данных."""
    app = FastAPI(title="CheckBot Admin Panel", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_page(request: Request, user: str = Depends(authenticate)):
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={"admin_user": user}
        )

    @app.get("/api/stats")
    async def get_stats(user: str = Depends(authenticate)) -> Dict[str, Any]:
        return await db.get_stats()

    @app.get("/api/users")
    async def get_users(
        search: str = Query("", description="Поиск по ID или имени"),
        status: str = Query("", description="Фильтр по статусу"),
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
        user: str = Depends(authenticate)
    ) -> Dict[str, Any]:
        users = await db.get_all_users(search=search, status=status, limit=limit, offset=offset)
        total = await db.get_total_users_count(search=search, status=status)
        return {
            "users": users,
            "total": total,
            "limit": limit,
            "offset": offset
        }

    @app.get("/api/users/{user_id}/logs")
    async def get_user_logs(
        user_id: int,
        user: str = Depends(authenticate)
    ) -> List[Dict[str, Any]]:
        return await db.get_user_logs(user_id)

    return app
