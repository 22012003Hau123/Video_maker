#!/usr/bin/env python3
"""
Start RQ worker using REDIS_URL / QUEUE_NAME from project .env (same as uvicorn).

Usage (from repo root, venv active):
  python scripts/rq_worker.py

Avoids mismatch where the shell uses redis://localhost:6379/0 but .env points elsewhere.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    queue_name = os.environ.get("QUEUE_NAME", "video_jobs")

    print(
        f"RQ worker — queue={queue_name} REDIS_URL={redis_url}",
        file=sys.stderr,
    )
    return subprocess.call(
        ["rq", "worker", queue_name, "--url", redis_url],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    raise SystemExit(main())
