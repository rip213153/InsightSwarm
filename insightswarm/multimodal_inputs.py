from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

from insightswarm.db.store import Store
from insightswarm.util import new_id


IMAGE_MIME_PREFIX = "image/"
AUDIO_MIME_PREFIX = "audio/"


def ingest_user_input_files(
    store: Store,
    run_id: str,
    *,
    file_paths: list[str] | None,
    vision_model_client: object | None = None,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for raw_path in file_paths or []:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"user input file not found: {path}")
        mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        artifact = _write_swarm_file_artifact(store, run_id, path, mime_type)
        summary = {
            "artifact_id": artifact.artifact_id,
            "type": "user_input",
            "mime_type": mime_type,
            "filename": path.name,
            "summary": f"User-provided {mime_type} input: {path.name}",
            "not_formal_evidence": True,
        }
        if mime_type.startswith(IMAGE_MIME_PREFIX):
            summary.update(_summarize_image(path, mime_type, vision_model_client))
        elif mime_type.startswith(AUDIO_MIME_PREFIX):
            summary.update(
                {
                    "modality": "audio",
                    "analysis_status": "stored_only",
                    "analysis_note": "Audio input was attached to the run, but audio transcription is not implemented in this step.",
                }
            )
        else:
            summary.update({"modality": "file", "analysis_status": "stored_only"})
        summaries.append(summary)
    if summaries:
        _write_manifest(store, run_id, summaries)
    return summaries


def _write_swarm_file_artifact(store: Store, run_id: str, path: Path, mime_type: str):
    input_dir = store.artifact_dir / run_id / "user_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    target = input_dir / f"{new_id('input')}{path.suffix}"
    target.write_bytes(path.read_bytes())
    return store.create_swarm_artifact(
        run_id,
        type="user_input",
        status="ready",
        source_task_id=None,
        payload_ref=str(target),
        summary=f"User input file: {path.name}",
    )


def _write_manifest(store: Store, run_id: str, summaries: list[dict[str, Any]]) -> None:
    manifest_dir = store.artifact_dir / run_id / "user_inputs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / "manifest.json"
    path.write_text(json.dumps({"inputs": summaries}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    store.create_swarm_artifact(
        run_id,
        type="user_input_manifest",
        status="ready",
        source_task_id=None,
        payload_ref=str(path),
        summary=f"{len(summaries)} user input attachment(s)",
    )


def _summarize_image(path: Path, mime_type: str, vision_model_client: object | None) -> dict[str, Any]:
    if vision_model_client is None or not hasattr(vision_model_client, "analyze_image"):
        return {
            "modality": "image",
            "analysis_status": "stored_only",
            "analysis_note": "Image input was attached, but no vision model client is available.",
        }
    prompt = (
        "Summarize this user-provided image for a research run. "
        "Return JSON with keys: visual_summary, visible_text, entities, relevance_to_user_question, uncertainty. "
        "This is user context only, not formal evidence."
    )
    try:
        result = vision_model_client.analyze_image(
            [{"role": "user", "content": prompt}],
            [{"path": str(path), "mime_type": mime_type}],
            response_format={"type": "json_object"},
            metadata={"role": "user_input", "tool": "summarize_user_image"},
        )
    except Exception as exc:
        return {
            "modality": "image",
            "analysis_status": "error",
            "analysis_error": f"{type(exc).__name__}: {exc}",
        }
    if getattr(result, "status", "") != "ok":
        return {
            "modality": "image",
            "analysis_status": "error",
            "analysis_error": getattr(result, "error", None) or "vision model failed",
        }
    return {
        "modality": "image",
        "analysis_status": "ok",
        "analysis_model": getattr(result, "model", None),
        "analysis": getattr(result, "json_data", None) or {"text": getattr(result, "text", "")},
    }
