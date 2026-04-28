from fastapi import FastAPI

from app.api import chat, health, ingest
from app.core.config import settings

app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(chat.router)
