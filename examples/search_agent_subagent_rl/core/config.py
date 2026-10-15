from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import isfinite
from typing import Any


@dataclass
class ContextConfig:
    max_context_tokens: int = 0
    reserve_tokens: int = 0
    max_new_tokens: int = 0


@dataclass
class AgentControlConfig:
    max_consecutive_malformed_tool_calls: int = 8


@dataclass
class ForkConfig:
    max_subagents: int = 3
    max_fork_calls_per_rollout: int = 10


@dataclass
class SubagentSearchConfig:
    context: ContextConfig = field(default_factory=ContextConfig)
    agent: AgentControlConfig = field(default_factory=AgentControlConfig)
    fork: ForkConfig = field(default_factory=ForkConfig)


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


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "f", "off"}:
        return False
    raise ValueError(f"Invalid {name}: {raw!r}. Must be a boolean string.")


def session_ig_enabled() -> bool:
    return _bool_env("SA_SESSION_IG_ENABLED", False)


def _validate_session_ig_runtime(args: Any) -> None:
    if not session_ig_enabled():
        return
    for name, default, positive in [
        ("SA_SESSION_IG_LAMBDA_SUBAGENT", 0.1, False),
        ("SA_SESSION_IG_CLIP", 2.0, True),
        ("SA_SESSION_IG_EPS", 1e-6, True),
    ]:
        raw = os.environ.get(name)
        try:
            value = default if raw is None or raw == "" else float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid {name}: {raw!r}.") from exc
        if not isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(f"Invalid {name}: {raw!r}.")
    _bool_env("SA_SESSION_IG_GATING_ENABLED", False)

    from examples.search_agent_compact_rl.core.config import _data_parallel_size

    dp_size = _data_parallel_size(args)
    if dp_size is not None and dp_size != 1:
        raise ValueError("Subagent session IG requires data parallel size 1 because normalization is rank-local.")


def build_config(args: Any) -> SubagentSearchConfig:
    max_context_tokens = _require_int_env("SA_MAX_CONTEXT_TOKENS")
    reserve_tokens = _require_int_env("SA_RESERVE_TOKENS")
    max_new_tokens = _require_int_env("SA_MAX_NEW_TOKENS")
    max_consecutive_malformed = _require_int_env("SA_MAX_CONSECUTIVE_MALFORMED_TOOL_CALLS")
    max_subagents = _require_int_env("SA_FORK_MAX_SUBAGENTS")
    max_fork_calls = _require_int_env("SA_MAX_FORK_CALLS_PER_ROLLOUT")

    assert max_context_tokens > 0, "SA_MAX_CONTEXT_TOKENS must be positive."
    assert reserve_tokens > 0, "SA_RESERVE_TOKENS must be positive."
    assert reserve_tokens < max_context_tokens, "SA_RESERVE_TOKENS must be smaller than SA_MAX_CONTEXT_TOKENS."
    assert max_new_tokens > 0, "SA_MAX_NEW_TOKENS must be positive."
    assert max_new_tokens <= reserve_tokens, "SA_MAX_NEW_TOKENS must be no larger than SA_RESERVE_TOKENS."
    assert max_consecutive_malformed >= 1, "SA_MAX_CONSECUTIVE_MALFORMED_TOOL_CALLS must be positive."
    assert 1 <= max_subagents <= 3, "SA_FORK_MAX_SUBAGENTS must be between 1 and 3."
    assert max_fork_calls >= 0, "SA_MAX_FORK_CALLS_PER_ROLLOUT must be non-negative."
    _validate_session_ig_runtime(args)

    return SubagentSearchConfig(
        context=ContextConfig(
            max_context_tokens=max_context_tokens,
            reserve_tokens=reserve_tokens,
            max_new_tokens=max_new_tokens,
        ),
        agent=AgentControlConfig(
            max_consecutive_malformed_tool_calls=max_consecutive_malformed,
        ),
        fork=ForkConfig(
            max_subagents=max_subagents,
            max_fork_calls_per_rollout=max_fork_calls,
        ),
    )
