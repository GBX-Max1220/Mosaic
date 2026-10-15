from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from examples.search_agent_compact_rl.core.agent_loop import AgentLoop
from examples.search_agent_compact_rl.core.config import build_config
from examples.search_agent_compact_rl.eval.incremental import (
    load_incremental_eval_result,
    save_incremental_eval_result,
)

_QWEN_USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)


def _extract_original_question(prompt: str | list) -> str:
    """Return the raw user question, stripping any chat-template wrappers."""
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    if isinstance(prompt, str):
        # If the data loader already applied the chat template, extract the
        # user content so we don't double-wrap it with another <|im_start|>user
        # tag inside the actual user message.
        matches = _QWEN_USER_RE.findall(prompt)
        if matches:
            # Prefer the last user turn in case the string contains multiple.
            return matches[-1].strip()
        return prompt.strip()

    return str(prompt)


async def _run_agent(args: Any, sample: Sample) -> list[Sample]:
    config = build_config(args)
    state = GenerateState(args)

    original_question = _extract_original_question(sample.prompt)

    loop = AgentLoop(
        config=config,
        state=state,
        original_question=original_question,
        label=sample.label,
        group_index=sample.group_index,
        input_metadata=sample.metadata,
    )
    samples = await loop.run()
    # Propagate the dataset index so slime can sort samples after fan-out.
    original_index = sample.index
    for s in samples:
        s.index = original_index
    return samples


async def _generate_with_incremental_eval(args: Any, sample: Sample, *, evaluation: bool) -> list[Sample]:
    if not evaluation:
        return await _run_agent(args, sample)

    cached = await asyncio.to_thread(load_incremental_eval_result, args, sample)
    if cached is not None:
        return cached

    samples = await _run_agent(args, sample)
    await asyncio.to_thread(save_incremental_eval_result, args, sample, samples)
    return samples


async def generate_on_policy(
    args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False
) -> Sample | list[Sample]:
    """On-policy generate: summarizer uses the policy model.

    `sampling_params` is required by the slime custom generate interface but is
    ignored because sampling params come from `GenerateState`.
    """
    return await _generate_with_incremental_eval(args, sample, evaluation=evaluation)
