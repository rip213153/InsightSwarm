from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import orjson


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def dumps(data: Any) -> str:
    # orjson dumps to UTF-8 bytes (no ASCII escaping) with sorted keys; decode
    # to str so existing text-column callers keep working.
    return orjson.dumps(data, option=orjson.OPT_SORT_KEYS).decode("utf-8")


def loads(data: str | bytes | None, default: Any = None) -> Any:
    if data is None:
        return default
    if isinstance(data, str):
        data = data.encode("utf-8")
    return orjson.loads(data)
