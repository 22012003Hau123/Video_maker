"""
RQ queue helpers and worker entrypoints.
"""
import asyncio
import inspect
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from redis import Redis
from rq import Queue

# RQ loads this file before app.py; load project .env so queue URL matches job_store / API.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("QUEUE_NAME", "video_jobs")
DEFAULT_JOB_TIMEOUT = int(os.getenv("QUEUE_JOB_TIMEOUT_SECONDS", "21600"))  # 6h


def get_redis_connection() -> Redis:
    return Redis.from_url(REDIS_URL)


def get_queue() -> Queue:
    return Queue(
        QUEUE_NAME,
        connection=get_redis_connection(),
        default_timeout=DEFAULT_JOB_TIMEOUT,
    )


def run_app_task(task_name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Worker entrypoint that resolves callable from app module.
    """
    import app  # Local import so worker resolves current app code.

    fn = getattr(app, task_name, None)
    if fn is None:
        raise RuntimeError(f"Task '{task_name}' not found in app module")

    if inspect.iscoroutinefunction(fn):
        return asyncio.run(fn(*args, **kwargs))

    result = fn(*args, **kwargs)
    if inspect.iscoroutine(result):
        return asyncio.run(result)
    return result


def enqueue_task(task_name: str, *args: Any, job_id: str | None = None, **kwargs: Any):
    """
    Enqueue app task for worker processing.
    """
    queue = get_queue()
    rq_job_id = f"app-job-{job_id}" if job_id else None
    return queue.enqueue_call(
        func=run_app_task,
        args=(task_name, *args),
        kwargs=kwargs,
        job_id=rq_job_id,
        timeout=DEFAULT_JOB_TIMEOUT,
        result_ttl=86400,
        failure_ttl=86400,
    )
