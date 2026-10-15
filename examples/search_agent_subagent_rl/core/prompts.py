from __future__ import annotations

from examples.search_agent_compact_rl.core.prompts import (
    FORCED_ANSWER_PROMPT,
    GENERATION_ERROR_FEEDBACK,
    MALFORMED_TOOL_FEEDBACK,
    SEARCH_TOOL_SCHEMA,
    SYSTEM_PROMPT,
    TOOL_ARGUMENTS_FEEDBACK,
    UNKNOWN_TOOL_FEEDBACK,
    VISIT_TOOL_SCHEMA,
    build_malformed_tool_feedback,
)


MAIN_SYSTEM_PROMPT = SYSTEM_PROMPT.rstrip() + """

As the main research agent, plan the investigation before executing it. For every non-trivial research request that can be meaningfully decomposed into independent or complementary lines of inquiry, you must use the fork tool at least once, preferably before performing extensive direct searches. Prefer delegating independent research to subagents instead of carrying out the entire investigation sequentially yourself.

Each fork call must create between 1 and 3 subagents. Never include more than 3 subagents in one fork call. Give each subagent a focused, self-contained task that states what to investigate and what result or evidence to return. Prefer complementary, non-overlapping assignments. If more than 3 useful subtasks exist, delegate up to 3 first, review their results, and then issue another fork call for the remaining subtasks.

You remain responsible for reviewing the returned results, resolving conflicts, filling important gaps with further research, and synthesizing the final answer. Do not treat subagent results as automatically correct.

Forking is not limited to the beginning of the investigation. After reviewing a batch of subagent results, you may fork another batch when the results reveal new independent lines of inquiry, unresolved evidence gaps, or claims that need separate verification. Use additional fork calls selectively, and do not repeat completed or substantially overlapping tasks. For simple requests that cannot be usefully decomposed, proceed directly with search and visit instead of creating unnecessary subagents.
"""


FORK_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fork",
        "description": (
            "Create 1 to 3 independent research subagents for decomposable subtasks. "
            "Each subagent receives one clear task prompt and must return its result "
            "inside <answer></answer>. The main agent will receive all subagent results "
            "inside one <subagent_results></subagent_results> tool response."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subagents": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The subtask and required result format for one subagent.",
                            },
                        },
                        "required": ["task"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["subagents"],
            "additionalProperties": False,
        },
    },
}


def build_main_tool_schemas() -> list[dict]:
    return [SEARCH_TOOL_SCHEMA, VISIT_TOOL_SCHEMA, FORK_TOOL_SCHEMA]


def build_subagent_tool_schemas() -> list[dict]:
    return [SEARCH_TOOL_SCHEMA, VISIT_TOOL_SCHEMA]


def build_subagent_user_message(task: str) -> str:
    return (
        "You are a research subagent created by a main search agent.\n"
        "Complete only the assigned task below. You may use search and visit tools. "
        "Return the final result for the main agent inside <answer></answer> tags.\n\n"
        f"Assigned task:\n{task}\n\n"
        "Final response format: <answer></answer>"
    )


def build_fork_limit_feedback() -> str:
    return (
        "The fork request failed because the maximum number of fork calls for this rollout "
        "has been reached. Continue with search, visit, or provide the final answer in "
        "<answer></answer> tags."
    )
