"""Typed configuration for search-agent-compact training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal


@dataclass
class CompactConfig:
    enabled: bool = False
    max_context_tokens: int = 0
    reserve_tokens: int = 0
    max_compactions: int = 10


@dataclass
class SummarizerConfig:
    mode: Literal["on_policy", "off_policy"] = "on_policy"
    train_summary_session: bool = True
    off_policy_model_name: str = ""
    off_policy_base_url: str = ""
    off_policy_api_key: str = ""
    off_policy_max_tokens: int = 4096
    temperature: float = 0.1


@dataclass
class AgentControlConfig:
    max_consecutive_malformed_tool_calls: int = 8


@dataclass
class SearchAgentConfig:
    compact: CompactConfig = field(default_factory=CompactConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    agent: AgentControlConfig = field(default_factory=AgentControlConfig)


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").lower()
    if val in ("1", "true", "yes", "t"):
        return True
    if val in ("0", "false", "no", "f"):
        return False
    return default


def session_ig_enabled() -> bool:
    return _env_bool("SA_SESSION_IG_ENABLED", False)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _require_int_env(name: str) -> int:
    raw = _require_env(name)
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}: {raw!r}") from e


def _require_float_env(name: str) -> float:
    raw = _require_env(name)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}: {raw!r}") from e


def _session_ig_float_env(name: str, default: float, *, positive: bool) -> float:
    raw = os.environ.get(name)
    try:
        value = default if raw is None or raw == "" else float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}: {raw!r}") from e
    valid = value > 0.0 if positive else value >= 0.0
    if not isfinite(value) or not valid:
        requirement = "positive finite" if positive else "non-negative finite"
        raise ValueError(f"Invalid {name}: {value!r}. Must be {requirement}.")
    return value


def _data_parallel_size(args) -> int | None:
    if args is None:
        return None

    configured = getattr(args, "data_parallel_size", None)
    if configured is not None and int(configured) > 0:
        return int(configured)

    actor_nodes = getattr(args, "actor_num_nodes", None)
    actor_gpus_per_node = getattr(args, "actor_num_gpus_per_node", None)
    if actor_nodes is not None and actor_gpus_per_node is not None:
        world_size = int(actor_nodes) * int(actor_gpus_per_node)
    else:
        world_size = int(getattr(args, "world_size", 0) or 0)
    if world_size <= 0:
        return None

    model_parallel_size = (
        int(getattr(args, "tensor_model_parallel_size", 1) or 1)
        * int(getattr(args, "pipeline_model_parallel_size", 1) or 1)
        * int(getattr(args, "context_parallel_size", 1) or 1)
    )
    if world_size % model_parallel_size != 0:
        raise ValueError(
            "Cannot infer data parallel size: actor world size is not divisible by "
            "tensor, pipeline, and context parallel sizes."
        )
    return world_size // model_parallel_size


def _validate_session_ig_runtime(
    args,
    *,
    compact_enabled: bool,
    summarizer_mode: str,
    train_summary_session: bool,
) -> None:
    enabled_raw = os.environ.get("SA_SESSION_IG_ENABLED", "")
    if enabled_raw.lower() not in ("", "1", "true", "yes", "t", "0", "false", "no", "f"):
        raise ValueError(
            f"Invalid SA_SESSION_IG_ENABLED: {enabled_raw!r}. Must be a boolean string."
        )
    gating_raw = os.environ.get("SA_SESSION_IG_GATING_ENABLED", "")
    if gating_raw.lower() not in ("", "1", "true", "yes", "t", "0", "false", "no", "f"):
        raise ValueError(
            f"Invalid SA_SESSION_IG_GATING_ENABLED: {gating_raw!r}. Must be a boolean string."
        )
    if not session_ig_enabled():
        return

    _session_ig_float_env("SA_SESSION_IG_LAMBDA_NORMAL", 0.1, positive=False)
    _session_ig_float_env("SA_SESSION_IG_LAMBDA_COMPACT", 0.1, positive=False)
    _session_ig_float_env("SA_SESSION_IG_CLIP", 2.0, positive=True)
    _session_ig_float_env("SA_SESSION_IG_EPS", 1e-6, positive=True)

    if not compact_enabled:
        raise ValueError("Session-level IG requires SA_COMPACT_ENABLED=true (setting 2 semantics).")
    if summarizer_mode != "on_policy":
        raise ValueError("Session-level IG requires SA_SUMMARIZER_MODE=on_policy (setting 2 semantics).")
    if not train_summary_session:
        raise ValueError(
            "Session-level IG requires SA_SUMMARIZER_TRAIN_SUMMARY_SESSION=true "
            "(setting 2 semantics)."
        )

    dp_size = _data_parallel_size(args)
    if dp_size is not None and dp_size != 1:
        raise ValueError(
            "Session-level IG currently requires data parallel size 1 because "
            "normalization is rank-local."
        )


def build_config(args) -> SearchAgentConfig:
    """Build SearchAgentConfig from environment variables."""
    mode = _require_env("SA_SUMMARIZER_MODE")
    if mode not in ("on_policy", "off_policy"):
        raise ValueError(f"Invalid SA_SUMMARIZER_MODE: {mode!r}. Must be 'on_policy' or 'off_policy'.")

    max_context_tokens = _require_int_env("SA_COMPACT_MAX_CONTEXT_TOKENS")
    reserve_tokens = _require_int_env("SA_COMPACT_RESERVE_TOKENS")
    max_compactions = _require_int_env("SA_COMPACT_MAX_COMPACTIONS")
    off_policy_max_tokens = _require_int_env("SA_SUMMARIZER_OFF_POLICY_MAX_TOKENS")
    temperature = _require_float_env("SA_SUMMARIZER_TEMPERATURE")
    max_consecutive_malformed_tool_calls = _require_int_env(
        "SA_MAX_CONSECUTIVE_MALFORMED_TOOL_CALLS"
    )
    assert max_context_tokens > 0, "SA_COMPACT_MAX_CONTEXT_TOKENS must be positive."
    assert reserve_tokens > 0, "SA_COMPACT_RESERVE_TOKENS must be positive."
    assert (
        reserve_tokens < max_context_tokens
    ), "SA_COMPACT_RESERVE_TOKENS must be smaller than SA_COMPACT_MAX_CONTEXT_TOKENS."
    assert max_compactions >= 0, "SA_COMPACT_MAX_COMPACTIONS must be non-negative."
    assert off_policy_max_tokens > 0, "SA_SUMMARIZER_OFF_POLICY_MAX_TOKENS must be positive."
    assert temperature >= 0.0, "SA_SUMMARIZER_TEMPERATURE must be non-negative."
    assert (
        max_consecutive_malformed_tool_calls >= 1
    ), "SA_MAX_CONSECUTIVE_MALFORMED_TOOL_CALLS must be positive."

    train_summary_session_raw = _require_env("SA_SUMMARIZER_TRAIN_SUMMARY_SESSION")
    train_summary_session = _env_bool("SA_SUMMARIZER_TRAIN_SUMMARY_SESSION")
    if train_summary_session_raw.lower() not in ("1", "true", "yes", "t", "0", "false", "no", "f"):
        raise ValueError(
            "Invalid SA_SUMMARIZER_TRAIN_SUMMARY_SESSION: "
            f"{train_summary_session_raw!r}. Must be a boolean string."
        )

    compact_enabled = _env_bool("SA_COMPACT_ENABLED", False)
    _validate_session_ig_runtime(
        args,
        compact_enabled=compact_enabled,
        summarizer_mode=mode,
        train_summary_session=train_summary_session,
    )

    off_policy_model_name = os.environ.get("SA_SUMMARIZER_OFF_POLICY_MODEL_NAME", "")
    off_policy_base_url = os.environ.get("SA_SUMMARIZER_OFF_POLICY_BASE_URL", "")
    off_policy_api_key = os.environ.get("SA_SUMMARIZER_OFF_POLICY_API_KEY", "")
    if mode == "off_policy":
        _require_env("SA_SUMMARIZER_OFF_POLICY_MODEL_NAME")
        _require_env("SA_SUMMARIZER_OFF_POLICY_BASE_URL")
        _require_env("SA_SUMMARIZER_OFF_POLICY_API_KEY")

    return SearchAgentConfig(
        compact=CompactConfig(
            enabled=compact_enabled,
            max_context_tokens=max_context_tokens,
            reserve_tokens=reserve_tokens,
            max_compactions=max_compactions,
        ),
        summarizer=SummarizerConfig(
            mode=mode,
            train_summary_session=train_summary_session,
            off_policy_model_name=off_policy_model_name,
            off_policy_base_url=off_policy_base_url,
            off_policy_api_key=off_policy_api_key,
            off_policy_max_tokens=off_policy_max_tokens,
            temperature=temperature,
        ),
        agent=AgentControlConfig(
            max_consecutive_malformed_tool_calls=max_consecutive_malformed_tool_calls,
        ),
    )
