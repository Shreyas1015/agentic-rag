from fastapi import FastAPI

from app.api import health
from app.core.config import settings

app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)

app.include_router(health.router)
