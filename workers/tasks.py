from celery import Celery

from backend.config import settings

celery_app = Celery(
    "pythagoras",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["workers.tasks"],
)


@celery_app.task(name="workers.tasks.healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "service": "workers"}
