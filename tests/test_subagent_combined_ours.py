import importlib
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from slime.utils.types import Sample


def _ours():
    return importlib.import_module("examples.search_agent_subagent_rl.baselines.mosaic")


def _session_ig():
    return importlib.import_module(
        "examples.search_agent_subagent_rl.baselines."
        "session_level_ig_adv_dressage_broadcast"
    )


@pytest.fixture(autouse=True)
def session_ig_environment(monkeypatch):
    for name, value in {
        "SA_SESSION_IG_ENABLED": "1",
        "SA_SESSION_IG_LAMBDA_SUBAGENT": "0.1",
        "SA_SESSION_IG_CLIP": "2.0",
        "SA_SESSION_IG_EPS": "1e-6",
        "SA_SESSION_IG_GATING_ENABLED": "1",
        "SA_SESSION_AWARE_RHO": "0.2",
        "SA_SESSION_AWARE_BETA_0": "0.1",
        "SA_SESSION_AWARE_BETA_1": "0.2",
        "SA_SESSION_AWARE_SESSION_COUNT_KEY": "rollout_num_sessions",
    }.items():
        monkeypatch.setenv(name, value)


def _metadata(role, *, group=0, delta=None):
    meta = {
        "session_type": role,
        "agent_role": role,
        "is_terminal_session": role == "main",
        "group_index": group,
        "ig_before_token_ids": [10],
        "ig_after_token_ids": [10, 20],
        "ig_answer_token_ids": [30, 31, 32],
        "ig_answer_start": 1,
        "ig_answer_end": 2,
    }
    if delta is not None:
        meta["session_ig_score_results"] = {"1": {"success": True, "delta": delta}}
    return meta


def test_score_requests_exclude_main_even_when_children_come_last():
    data = {"metadata": [_metadata("main"), _metadata("subagent")]}

    requests = _session_ig().build_session_ig_score_requests(data)

    assert [(r["sample_index"], r["kind"]) for r in requests] == [(1, "before"), (1, "after")]
    assert requests[0]["tokens"] == [10, 30, 31, 32]
    assert requests[1]["tokens"] == [10, 20, 30, 31, 32]
    assert requests[0]["answer_start"] == 1
    assert requests[0]["answer_end"] == 2


def test_old_actor_scoring_attaches_child_gain_without_scoring_main():
    data = {"metadata": [_metadata("main"), _metadata("subagent")]}

    class Trainer:
        def compute_ig_request_logprobs(self, requests):
            assert len(requests) == 2
            return [-2.0, -0.5]

    _ours().compute_old_actor_session_ig_deltas(SimpleNamespace(), Trainer(), data)

    assert data["metadata"][1]["session_ig_score_results"]["1"] == {
        "before": -2.0, "after": -0.5, "delta": 1.5, "success": True,
    }
    assert not data["metadata"][0]["session_ig_score_results"]


def _rollouts():
    samples = []
    for rollout_id, (reward, count) in enumerate([(1.0, 2), (1.0, 4), (0.0, 2), (0.0, 4)]):
        for session_index in range(count):
            role = "main" if session_index == 0 else "subagent"
            meta = _metadata(role)
            meta.update(trajectory_raw_reward=reward, rollout_num_sessions=count)
            samples.append(Sample(
                index=len(samples), rollout_id=rollout_id, group_index=0,
                tokens=[1, 2, 3], response_length=2, loss_mask=[1, 0],
                reward=reward, train_metadata=meta,
                metadata={"trajectory_raw_reward": reward}, status=Sample.Status.COMPLETED,
            ))
    return samples


def test_combines_session_count_credit_with_child_ig_and_preserves_main():
    module = _ours()
    data = module.convert_samples_to_train_data(
        SimpleNamespace(advantage_estimator="grpo", grpo_std_normalization=True,
                        global_batch_size=4, reward_key=None),
        _rollouts(),
    )
    # Success/failure subgroups contain both short and long rollouts, so method
    # one produces different base advantages before method two is applied.
    expected_base = [1.2, 1.2, 0.8, 0.8, 0.8, 0.8, -1.2, -1.2, -0.8, -0.8, -0.8, -0.8]
    assert data["rewards"] == pytest.approx(expected_base, abs=1e-5)
    for idx, meta in enumerate(data["metadata"]):
        if meta["agent_role"] == "subagent":
            meta["session_ig_score_results"] = {
                str(idx): {"success": True, "delta": 1.0 if data["raw_reward"][idx] else -1.0}
            }

    module.compute_advantages_and_returns(SimpleNamespace(), data)

    expected = [1.2, 1.3, 0.8, 0.9, 0.9, 0.9, -1.2, -1.3, -0.8, -0.9, -0.9, -0.9]
    for adv, ret, value in zip(data["advantages"], data["returns"], expected, strict=True):
        torch.testing.assert_close(adv, torch.tensor([value, 0.0]), atol=1e-5, rtol=0)
        torch.testing.assert_close(ret, adv)
    assert data["session_ig_adv_metrics"]["scoring/scored_session_count"] == 8.0
    assert "session_aware_metrics" in data


def test_session_ig_alone_preserves_broadcast_base_without_session_count_credit():
    module = _session_ig()
    data = module.convert_samples_to_train_data(
        SimpleNamespace(advantage_estimator="grpo", grpo_std_normalization=True,
                        global_batch_size=4, reward_key=None),
        _rollouts(),
    )
    # Short and long trajectories have equal base credit when method one is
    # disabled; only child sessions receive method two's IG correction.
    assert data["rewards"] == pytest.approx([1.0] * 6 + [-1.0] * 6, abs=1e-5)
    assert "session_aware_metrics" not in data
    for idx, meta in enumerate(data["metadata"]):
        if meta["agent_role"] == "subagent":
            meta["session_ig_score_results"] = {
                str(idx): {"success": True, "delta": 1.0 if data["raw_reward"][idx] else -1.0}
            }

    module.compute_advantages_and_returns(SimpleNamespace(), data)

    expected = [1.0, 1.1, 1.0, 1.1, 1.1, 1.1, -1.0, -1.1, -1.0, -1.1, -1.1, -1.1]
    for adv, ret, value in zip(data["advantages"], data["returns"], expected, strict=True):
        torch.testing.assert_close(adv, torch.tensor([value, 0.0]), atol=1e-5, rtol=0)
        torch.testing.assert_close(ret, adv)
    assert data["session_ig_adv_metrics"]["scoring/scored_session_count"] == 8.0


@pytest.mark.parametrize("base,raw,deltas,expected", [
    ([0.0, 0.0], [0.0, 0.0], [-1.0, 1.0], [-0.1, 0.0]),
    ([0.05, 0.05], [1.0, 1.0], [-1.0, 1.0], [0.05, 0.15]),
])
def test_child_ig_gating_prevents_failed_group_credit_and_sign_reversal(base, raw, deltas, expected):
    metadata = [_metadata("subagent"), _metadata("subagent")]
    for idx, delta in enumerate(deltas):
        metadata[idx]["session_ig_score_results"] = {str(idx): {"success": True, "delta": delta}}
    data = {"rewards": base, "raw_reward": raw, "response_lengths": [1, 1],
            "loss_masks": [[1], [1]], "metadata": metadata}

    _ours().compute_advantages_and_returns(SimpleNamespace(), data)

    assert [a.item() for a in data["advantages"]] == pytest.approx(expected, abs=1e-5)
    assert data["session_ig_adv_metrics"]["gating/total/count"] == 1.0


def test_missing_child_score_falls_back_to_trajectory_advantage():
    data = {"rewards": [0.4, -0.4], "raw_reward": [1.0, 0.0],
            "response_lengths": [1, 1], "loss_masks": [[1], [1]],
            "metadata": [_metadata("main"), _metadata("subagent")]}

    _ours().compute_advantages_and_returns(SimpleNamespace(), data)

    assert [a.item() for a in data["advantages"]] == pytest.approx([0.4, -0.4])
    assert data["session_ig_adv_metrics"]["scoring/skipped_session_count"] == 1.0


def test_child_handoff_preserves_sampled_qwen_history():
    from transformers import AutoTokenizer

    from examples.search_agent_subagent_rl.core.agent_loop import AgentLoop, PendingFork
    from examples.search_agent_subagent_rl.core.config import SubagentSearchConfig
    from examples.search_agent_subagent_rl.core.session import SessionSegment, SubagentResult

    model_path = Path("/mnt/models/Qwen/Qwen3.5-4B")
    if not model_path.exists():
        pytest.skip("Local Qwen3.5 tokenizer unavailable")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    state = SimpleNamespace(tokenizer=tokenizer, sampling_params={}, args=SimpleNamespace())
    loop = AgentLoop(SubagentSearchConfig(), state, "Original question", "truth", 0)

    def append_sampled_turn(text):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        loop._append_turn(text, {"tokens": tokens, "log_probs": [-0.1] * len(tokens)})

    append_sampled_turn('earlier reasoning\n</think>\n<tool_call><function=search><parameter=query>earlier</parameter></function></tool_call>')
    loop._append_user_message("<tool_response>EARLIER_OBSERVATION</tool_response>")
    before = loop._current_generation_prompt_ids()
    append_sampled_turn('fork reasoning\n</think>\n<tool_call><function=fork><parameter=subagents>[{"task":"child"}]</parameter></function></tool_call>')
    exact_prefix = list(loop.current_session.prompt_token_ids + loop.current_session.response_token_ids)
    child = SessionSegment(prompt_text="p", prompt_token_ids=[1], agent_role="subagent")
    result = SubagentResult(agent_id="child", subagent_index=0, task="child", success=True,
                            answer="CHILD_OBSERVATION", failure=None, session=child)

    loop._capture_fork_session_ig(PendingFork("", [result], is_successful=True), before)

    assert child.ig_before_token_ids == before
    assert child.ig_after_token_ids[:len(exact_prefix)] == exact_prefix
    assert "EARLIER_OBSERVATION" in tokenizer.decode(child.ig_after_token_ids)
    assert "CHILD_OBSERVATION" in tokenizer.decode(child.ig_after_token_ids)


@pytest.mark.parametrize("value,expected", [("y", True), ("t", None), ("off", False)])
def test_gating_boolean_validation_agrees_with_shared_math(monkeypatch, value, expected):
    from examples.search_agent_subagent_rl.core.config import _bool_env
    from examples.search_agent_compact_rl.baselines.session_level_ig_adv_dressage_broadcast import _bool_env as math_bool

    monkeypatch.setenv("SA_SESSION_IG_GATING_ENABLED", value)
    if expected is None:
        with pytest.raises(ValueError):
            _bool_env("SA_SESSION_IG_GATING_ENABLED", False)
    else:
        assert _bool_env("SA_SESSION_IG_GATING_ENABLED", False) is expected
        assert math_bool("SA_SESSION_IG_GATING_ENABLED") is expected


class TextTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


@pytest.mark.parametrize("enabled", [True, False])
def test_accepted_fork_exports_each_child_handoff_without_sibling_or_final_answer(monkeypatch, enabled):
    from examples.search_agent_subagent_rl.core.agent_loop import AgentLoop, PendingFork, format_subagent_results
    from examples.search_agent_subagent_rl.core.config import SubagentSearchConfig
    from examples.search_agent_subagent_rl.core.session import SessionSegment, SubagentResult

    monkeypatch.setenv("SA_SESSION_IG_ENABLED", "1" if enabled else "0")
    config = SubagentSearchConfig()
    config.context.max_context_tokens = 30000
    config.context.reserve_tokens = 128
    config.context.max_new_tokens = 16
    state = SimpleNamespace(tokenizer=TextTokenizer(), sampling_params={"max_new_tokens": 16},
                            args=SimpleNamespace(rollout_max_context_len=30000))
    loop = AgentLoop(config, state, "Original question", "GLOBAL_GROUND_TRUTH", 7)
    initial_prefix = loop._current_generation_prompt_ids()
    results = [
        SubagentResult(agent_id=f"child-{i}", subagent_index=i, task=f"task-{i}",
                       success=True, answer=f"CHILD_RESULT_{i}", failure=None,
                       session=SessionSegment(prompt_text="p", prompt_token_ids=[1], response_text="r",
                                              response_token_ids=[2], loss_mask=[1], rollout_log_probs=[-0.1],
                                              agent_role="subagent", agent_id=f"child-{i}"))
        for i in range(2)
    ]
    responses = iter([
        '<think>Delegate</think><tool_call><function=fork><parameter=subagents>'
        '[{"task":"task-0"},{"task":"task-1"}]</parameter></function></tool_call>',
        '<think>Finish</think><answer>MAIN_FINAL_ANSWER</answer>',
    ])

    async def generate(*args, **kwargs):
        text = next(responses)
        return {"text": text, "tokens": state.tokenizer.encode(text),
                "log_probs": [-0.1] * len(text), "error": None}

    async def fork(*args, **kwargs):
        return PendingFork(format_subagent_results(results), results, is_successful=True)

    async def reward():
        loop.total_reward = 1.0
        for session in [loop.current_session] + loop.subagent_sessions:
            session.reward = 1.0

    monkeypatch.setattr(loop, "_generate_once", generate)
    monkeypatch.setattr(loop, "_stage_fork", fork)
    monkeypatch.setattr(loop, "_compute_and_broadcast_reward", reward)
    samples = asyncio.run(loop.run())

    assert len(samples) == 3
    if not enabled:
        assert all("ig_before_token_ids" not in sample.train_metadata for sample in samples)
        return
    assert samples[0].train_metadata["is_terminal_session"] is True
    assert samples[0].train_metadata["session_type"] == "main"
    for i, sample in enumerate(samples[1:]):
        meta = sample.train_metadata
        assert meta["group_index"] == 7
        assert meta["session_type"] == "subagent"
        assert meta["is_terminal_session"] is False
        assert meta["ig_before_token_ids"] == initial_prefix
        after = "".join(chr(t) for t in meta["ig_after_token_ids"])
        assert f"CHILD_RESULT_{i}" in after
        assert f"CHILD_RESULT_{1-i}" not in after
        assert "MAIN_FINAL_ANSWER" not in after
        answer = meta["ig_answer_token_ids"][meta["ig_answer_start"]:meta["ig_answer_end"]]
        assert "".join(chr(t) for t in answer) == "GLOBAL_GROUND_TRUTH"


@pytest.mark.parametrize("field,value", [
    ("SA_SESSION_IG_LAMBDA_SUBAGENT", "-1"),
    ("SA_SESSION_IG_LAMBDA_SUBAGENT", "nan"),
    ("SA_SESSION_IG_EPS", "0"),
    ("SA_SESSION_IG_CLIP", "inf"),
    ("SA_SESSION_IG_GATING_ENABLED", "invalid"),
])
def test_subagent_config_rejects_invalid_session_ig_options(monkeypatch, field, value):
    from examples.search_agent_subagent_rl.core.config import build_config

    for key, setting in {"SA_MAX_CONTEXT_TOKENS": "131072", "SA_RESERVE_TOKENS": "16384",
                         "SA_MAX_NEW_TOKENS": "16384", "SA_MAX_CONSECUTIVE_MALFORMED_TOOL_CALLS": "8",
                         "SA_FORK_MAX_SUBAGENTS": "3", "SA_MAX_FORK_CALLS_PER_ROLLOUT": "10"}.items():
        monkeypatch.setenv(key, setting)
    monkeypatch.setenv(field, value)
    with pytest.raises(ValueError, match=field):
        build_config(SimpleNamespace(data_parallel_size=1))


def test_mosaic_script_submits_both_method_hooks_and_session_ig_environment(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for command in ("pkill", "sleep", "mkdir"):
        stub = fake_bin / command
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    ray = fake_bin / "ray"
    ray.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
if sys.argv[1:3] == ["job", "submit"]:
    args = sys.argv[3:]
    runtime = json.loads(next(a.split("=", 1)[1] for a in args if a.startswith("--runtime-env-json=")))
    selected = {k: v for k, v in runtime["env_vars"].items() if k.startswith("SA_SESSION_")}
    clean_args = [a for a in args if not a.startswith("--runtime-env-json=")]
    with open(os.environ["MSRL_TEST_CAPTURE"], "w") as handle:
        json.dump({"args": clean_args, "env": selected}, handle)
''')
    ray.chmod(0o755)
    capture = tmp_path / "submission.json"
    script = Path(__file__).resolve().parents[1] / "examples/search_agent_subagent_rl/scripts/train/run_qwen3.5_4B_mosaic.sh"
    env = {k: v for k, v in os.environ.items() if not k.startswith("SA_SESSION_")}
    env.update(PATH=f"{fake_bin}:{env['PATH']}", MSRL_TEST_CAPTURE=str(capture))

    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True)

    assert result.returncode == 0
    submission = json.loads(capture.read_text())
    args = submission["args"]
    module = "examples.search_agent_subagent_rl.baselines.mosaic"
    assert args[args.index("--custom-convert-samples-to-train-data-path") + 1] == module + ".convert_samples_to_train_data"
    assert args[args.index("--custom-pre-advantage-function-path") + 1] == module + ".compute_old_actor_session_ig_deltas"
    assert args[args.index("--custom-advantage-function-path") + 1] == module + ".compute_advantages_and_returns"
    assert submission["env"]["SA_SESSION_IG_ENABLED"] == "1"
    assert submission["env"]["SA_SESSION_IG_LAMBDA_SUBAGENT"] == "0.1"
    assert submission["env"]["SA_SESSION_IG_GATING_ENABLED"] == "1"
    assert args[args.index("--save") + 1].endswith("/setting-subagent-qwen3.5-4B-mosaic")
