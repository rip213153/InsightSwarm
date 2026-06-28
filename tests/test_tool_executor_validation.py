"""Phase 2 direct unit tests for ToolExecutor pydantic input validation.

The previous commit claimed "single source of truth for input shape validation"
but the core methods (_validate_input, _build_input_model, _json_type_to_python,
_reject_empty) had zero direct unit tests — only indirect happy-path coverage
via agent_loop tests. The failure path (failure_kind="invalid_input") was 0%
covered. These tests close that gap and fulfill the dangling docstring promise
in test_agent_loop_contract.py:25 (test_tool_executor_validates_input_shape).
"""
from __future__ import annotations

from typing import Any, get_args

import pytest
from pydantic import ValidationError

from insightswarm.agents.tool_executor import (
    ToolExecutor,
    _build_input_model,
    _json_type_to_python,
    _reject_empty,
)


def test_tool_executor_validates_input_shape():
    """Phase 2 核心论点：ToolExecutor 用 pydantic 单一来源校验 input shape。

    contract 层不再做 shape 校验（test_validate_tool_call_only_checks_structure_not_shape
    验证了 contract 放手），由 ToolExecutor._validate_input 接住。
    """
    tool_specs = [{
        "name": "search_web",
        "description": "search",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    }]
    executor = ToolExecutor(tool_specs, handlers={"search_web": lambda inp: {"ok": True}})

    # 合法输入
    result = executor.execute({"name": "search_web", "input": {"query": "hello", "limit": 5}})
    assert result.ok is True
    assert result.failure_kind is None

    # missing required
    result = executor.execute({"name": "search_web", "input": {"limit": 5}})
    assert result.ok is False
    assert result.failure_kind == "invalid_input"
    assert "query" in result.error

    # wrong type (dict for a str field — pydantic rejects in lax mode)
    result = executor.execute({"name": "search_web", "input": {"query": {"nested": "obj"}}})
    assert result.ok is False
    assert result.failure_kind == "invalid_input"

    # empty string for required str
    result = executor.execute({"name": "search_web", "input": {"query": "  "}})
    assert result.ok is False
    assert result.failure_kind == "invalid_input"


def test_json_type_to_python_mappings():
    """直接单测 _json_type_to_python：基础类型 + enum + 未知 type。"""
    # 基础类型映射
    assert _json_type_to_python({"type": "string"}) is str
    assert _json_type_to_python({"type": "integer"}) is int
    assert _json_type_to_python({"type": "number"}) is float
    assert _json_type_to_python({"type": "boolean"}) is bool
    assert _json_type_to_python({"type": "array"}) is list
    assert _json_type_to_python({"type": "object"}) is dict

    # 未知 type → Any
    assert _json_type_to_python({"type": "unknown"}) is Any
    assert _json_type_to_python({}) is Any

    # enum（全 str）→ Literal[...]
    enum_str = _json_type_to_python({"enum": ["a", "b", "c"]})
    # Literal 的 __args__ 即 enum 值
    assert get_args(enum_str) == ("a", "b", "c")

    # enum（含非 str）→ Any
    enum_mixed = _json_type_to_python({"enum": ["a", 1]})
    assert enum_mixed is Any

    # enum 优先于 type：即使带 type，有 enum 全 str 仍返回 Literal
    enum_with_type = _json_type_to_python({"type": "string", "enum": ["x", "y"]})
    assert get_args(enum_with_type) == ("x", "y")

    # 空 enum 列表 → 走 type 路径
    assert _json_type_to_python({"enum": [], "type": "string"}) is str


def test_build_input_model_required_vs_optional():
    """直接单测 _build_input_model：required/optional/empty-string/无 properties。"""
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    }
    model = _build_input_model("search_web", schema)
    assert model is not None

    # required 字段不传 → ValidationError
    with pytest.raises(ValidationError):
        model.model_validate({"limit": 5})

    # optional 字段不传 → OK（默认 None）
    instance = model.model_validate({"query": "hello"})
    assert instance.query == "hello"  # type: ignore[attr-defined]
    assert instance.limit is None  # type: ignore[attr-defined]

    # required str 空串 → ValidationError（_reject_empty）
    with pytest.raises(ValidationError):
        model.model_validate({"query": "   "})

    # required str 非空 → OK
    instance = model.model_validate({"query": "hello", "limit": 5})
    assert instance.query == "hello"  # type: ignore[attr-defined]
    assert instance.limit == 5  # type: ignore[attr-defined]

    # 无 properties 的 schema → 返回 None（走 fallback required-key 检查）
    assert _build_input_model("no_props", {"type": "object", "required": ["a"]}) is None
    assert _build_input_model("empty_props", {"type": "object", "properties": {}, "required": []}) is None
    assert _build_input_model("no_schema", {}) is None


def test_reject_empty_only_rejects_blank_strings():
    """_reject_empty：空白字符串拒绝，非空字符串和其它类型放行。"""
    assert _reject_empty("hello") == "hello"
    assert _reject_empty("  x  ") == "  x  "
    # 非字符串类型不受影响（required str 字段才用它，但函数本身对非 str 透传）
    assert _reject_empty(0) == 0
    assert _reject_empty(None) is None

    with pytest.raises(ValueError):
        _reject_empty("")
    with pytest.raises(ValueError):
        _reject_empty("   ")
    with pytest.raises(ValueError):
        _reject_empty("\t\n")


def test_validate_input_fallback_for_propertyless_schema():
    """_validate_input 在 model is None 时走 fallback required-key 检查。"""
    # 无 properties schema，只有 required 列表 → missing required 返回错误
    tool_specs_no_props = [{
        "name": "read_only",
        "description": "read",
        "input_schema": {"type": "object", "required": ["task_id"]},
    }]
    executor = ToolExecutor(tool_specs_no_props, handlers={"read_only": lambda inp: {"ok": True}})

    # missing required → 错误
    err = executor._validate_input("read_only", {})
    assert err is not None
    assert "task_id" in err
    assert "missing required input" in err

    # required 满足 → None
    assert executor._validate_input("read_only", {"task_id": "t1"}) is None

    # required 值为空串也视为 missing（fallback 行为）
    err = executor._validate_input("read_only", {"task_id": ""})
    assert err is not None
    assert "task_id" in err

    # required 值为 None 也视为 missing
    err = executor._validate_input("read_only", {"task_id": None})
    assert err is not None


def test_validate_input_error_message_includes_all_locs():
    """错误信息包含所有 error locations（避免 whack-a-mole 重试）。"""
    tool_specs = [{
        "name": "two_fields",
        "description": "test",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    }]
    executor = ToolExecutor(tool_specs, handlers={"two_fields": lambda inp: {"ok": True}})

    # 多个字段同时错误 → 错误信息包含所有 loc
    result = executor.execute({"name": "two_fields", "input": {}})
    assert result.ok is False
    assert result.failure_kind == "invalid_input"
    assert result.error is not None
    # 两个 missing 字段都应出现在错误信息中
    assert "a" in result.error
    assert "b" in result.error
    # 格式包含 "all errors" 段
    assert "all errors" in result.error

    # 单字段错误（wrong type）→ 错误信息包含该 loc
    result = executor.execute({"name": "two_fields", "input": {"a": "ok", "b": "not-a-number"}})
    assert result.ok is False
    assert result.failure_kind == "invalid_input"
    assert "b" in result.error
