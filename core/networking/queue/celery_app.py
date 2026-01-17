import os
from celery import Celery
from celery.schedules import crontab
from kombu import Queue, Exchange
from dotenv import load_dotenv
from core.utils import auto_config

# Load .env
load_dotenv(override=False)

# ================================
#       BUILD REDIS URL
# ================================
def _expanded(v: str | None) -> str | None:
    if not v:
        return v
    x = os.path.expandvars(v)
    return None if "$" in x else x

def _build_redis_url() -> str:
    raw = os.getenv("REDIS_URL")
    ex = _expanded(raw)
    if ex:
        return ex

    scheme = os.getenv("REDIS_SCHEME", "redis")
    host   = os.getenv("REDIS_HOST", "localhost")
    port   = int(os.getenv("REDIS_PORT", "6379"))
    db     = int(os.getenv("REDIS_DB", "0"))
    pwd    = os.getenv("REDIS_PASSWORD", "")

    auth   = f":{pwd}@" if pwd else ""
    return f"{scheme}://{auth}{host}:{port}/{db}"


BROKER_URL  = _build_redis_url()
BACKEND_URL = BROKER_URL

# Xóa các env override Celery
for k in (
    "CELERY_BROKER_URL",
    "CELERY_RESULT_BACKEND",
    "CELERY_BROKER_READ_URL",
    "BROKER_URL",
    "RESULT_BACKEND",
):
    if k in os.environ:
        del os.environ[k]

print(f"[celery_app] Using Redis broker: {BROKER_URL}")

# ================================
#       QUEUE DEFINITIONS (FIX)
# ================================
def _generate_task_queues() -> tuple:
    """
    TWO queues: GPU + CPU
    Each MUST have its own exchange + routing_key
    """
    queues = [
        Queue(
            name="fastvlm.gpu",
            exchange=Exchange("fastvlm", type="direct"),
            routing_key="fastvlm.gpu",
        ),
        Queue(
            name="general.cpu",
            exchange=Exchange("general", type="direct"),
            routing_key="general.cpu",
        ),
    ]

    print("[celery_app] Created queues: fastvlm.gpu, general.cpu")
    return tuple(queues)


celery_app = Celery("fastvlm")

# ================================
#       CELERY CONFIG
# ================================
celery_app.conf.update(
    broker_url=BROKER_URL,
    broker_read_url=BROKER_URL,
    result_backend=BACKEND_URL,

    include=[
        "core.networking.queue.persist_analysis_tasks",
        "core.networking.queue.analyze_image_tasks",
        "customers.classroom_vision.core.networking.queue.attendance_tasks",
    ],

    task_queues=_generate_task_queues(),

    # Default CPU queue
    task_default_queue="general.cpu",
    task_default_exchange="general",
    task_default_routing_key="general.cpu",

    # Routing rules
    task_routes={
        "core.networking.queue.analyze_image_tasks.*": {
            "queue": "fastvlm.gpu",
            "routing_key": "fastvlm.gpu",
        },
        "core.networking.queue.persist_analysis_tasks.*": {
            "queue": "general.cpu",
            "routing_key": "general.cpu",
        },
        "customers.classroom_vision.core.networking.queue.attendance_tasks.*": {
            "queue": "general.cpu",
            "routing_key": "general.cpu",
        },
    },

    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)

# Timezone
celery_app.conf.timezone = "Asia/Bangkok"
celery_app.conf.enable_utc = True
# ================================
#       BEAT SCHEDULE
# ================================
celery_app.conf.beat_schedule = {
    # chạy mỗi phút để debug
    # "plan-absence-jobs-debug": {
    #     "task": "attendance_tasks.schedule_daily_absence_checks",
    #     "schedule": crontab(minute="*/1"),
    #     "options": {"queue": "general.cpu"},
    # },
    "plan-absence-jobs-daily": {
            "task": "attendance_tasks.schedule_daily_absence_checks",
            "schedule": crontab(
                minute=5, 
                hour=0
            ),
            "options": {"queue": "general.cpu"},
        },
}

print("[celery_app] Task routing configured OK")
print("  - fastvlm.gpu queue")
print("  - general.cpu queue (default)")
