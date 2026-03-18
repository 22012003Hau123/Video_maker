"""
Redis-backed job storage for API/worker shared state.
"""
import json
import logging
import os
from typing import Any, Dict, Optional

from redis import Redis

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
JOB_KEY_PREFIX = os.getenv("JOB_KEY_PREFIX", "video_subtitles:job")
META_KEY_PREFIX = os.getenv("JOB_META_KEY_PREFIX", "video_subtitles:job_meta")

_redis_client: Optional[Redis] = None


def get_redis_client() -> Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _job_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}:{job_id}"


def _meta_key(job_id: str) -> str:
    return f"{META_KEY_PREFIX}:{job_id}"


def _load_json(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Failed to decode JSON payload from Redis")
        return default


def create_job(job_id: str, payload: Dict[str, Any]) -> None:
    client = get_redis_client()
    client.set(_job_key(job_id), json.dumps(payload, ensure_ascii=False))


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    client = get_redis_client()
    raw = client.get(_job_key(job_id))
    return _load_json(raw, None)


def update_job(job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    current = get_job(job_id)
    if current is None:
        return None
    current.update(updates)
    create_job(job_id, current)
    return current


def set_job_meta(job_id: str, key: str, value: Any) -> None:
    client = get_redis_client()
    meta_raw = client.get(_meta_key(job_id))
    meta = _load_json(meta_raw, {})
    meta[key] = value
    client.set(_meta_key(job_id), json.dumps(meta, ensure_ascii=False))


def get_job_meta(job_id: str, key: str, default: Any = None) -> Any:
    client = get_redis_client()
    meta_raw = client.get(_meta_key(job_id))
    meta = _load_json(meta_raw, {})
    return meta.get(key, default)
