from __future__ import annotations

import re
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from examples.search_agent_subagent_rl.core.agent_loop import AgentLoop
from examples.search_agent_subagent_rl.core.config import build_config


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


async def generate_on_policy(
    args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False
) -> Sample | list[Sample]:
    """Run the subagent RL rollout with the policy model.

    `sampling_params` and `evaluation` are required by the slime custom generate
    interface but are currently ignored because sampling params come from
    `GenerateState`.
    """
    return await _run_agent(args, sample)


async def generate_off_policy(
    args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False
) -> Sample | list[Sample]:
    """Run the subagent RL rollout using the same agent runtime.

    `sampling_params` and `evaluation` are required by the slime custom generate
    interface but are currently ignored because sampling params come from
    `GenerateState`.
    """
    return await _run_agent(args, sample)


generate = generate_on_policy
