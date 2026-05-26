from __future__ import annotations

from typing import Any


def validate_text_span(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    start = payload.get("start")
    end = payload.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        errors.append("text span must include integer start and end")
        return errors
    if start < 0 or end <= start:
        errors.append("text span requires 0 <= start < end")
    return errors


def validate_image_bbox_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    bbox = payload.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        errors.append("image bbox must include four normalized values")
        return errors
    if not isinstance(payload.get("original_width"), int) or payload["original_width"] <= 0:
        errors.append("image bbox original_width must be positive")
    if not isinstance(payload.get("original_height"), int) or payload["original_height"] <= 0:
        errors.append("image bbox original_height must be positive")
    try:
        values = [float(value) for value in bbox]
    except Exception:
        errors.append("image bbox values must be numeric")
        return errors
    if any(value < 0.0 or value > 1.0 for value in values):
        errors.append("image bbox values must be normalized between 0 and 1")
    if values[0] >= values[2]:
        errors.append("image bbox requires ymin < ymax")
    if values[1] >= values[3]:
        errors.append("image bbox requires xmin < xmax")
    return errors


def validate_competitor_knowledge(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not payload.get("competitor"):
        errors.append("competitor is required")
    facts = payload.get("facts")
    if not isinstance(facts, list) or not facts:
        errors.append("facts must be a non-empty list")
        return errors
    for index, fact in enumerate(facts):
        if not fact.get("field"):
            errors.append(f"facts[{index}].field is required")
        if not fact.get("value"):
            errors.append(f"facts[{index}].value is required")
        if not fact.get("quote"):
            errors.append(f"facts[{index}].quote is required")
        if not fact.get("source_url"):
            errors.append(f"facts[{index}].source_url is required")
        if fact.get("text_span"):
            errors.extend(
                f"facts[{index}].text_span.{error}"
                for error in validate_text_span(fact["text_span"])
            )
    return errors


def validate_analysis(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    inferences = payload.get("inferences")
    if not isinstance(inferences, list) or not inferences:
        errors.append("inferences must be a non-empty list")
        return errors
    for index, inference in enumerate(inferences):
        if not inference.get("claim"):
            errors.append(f"inferences[{index}].claim is required")
        evidence = inference.get("evidence_ids")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"inferences[{index}].evidence_ids must be non-empty")
        if inference.get("confidence") is not None:
            confidence = inference.get("confidence")
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                errors.append(f"inferences[{index}].confidence must be between 0 and 1")
    return errors
