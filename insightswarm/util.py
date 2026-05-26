from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True)


def loads(data: str | None, default: Any = None) -> Any:
    if data is None:
        return default
    return json.loads(data)

