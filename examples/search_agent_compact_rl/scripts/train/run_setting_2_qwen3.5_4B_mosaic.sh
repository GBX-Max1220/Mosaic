#!/bin/bash

# Setting 2 Qwen3.5-4B with session-aware trajectory advantage and gated
# session-level IG correction, using Dressage prompt-equal broadcast.

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
sleep 3
pkill -9 ray

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="${SCRIPT_DIR}/../../../.."

cd "$REPO_DIR"

if [ -f "${REPO_DIR}/.env" ]; then
    source "${REPO_DIR}/.env"
fi

source "${REPO_DIR}/scripts/models/qwen3.5-4B.sh"

# Replace these placeholder paths before running training.
MODEL="path/to/Qwen3.5-4B"
TORCH_DIST="path/to/Qwen3.5-4B_torch_dist"
DATA="path/to/train.parquet"
SAVE="path/to/checkpoints/setting-2-qwen3.5-4B-mosaic"
MEGATRON_LM_PATH="path/to/Megatron-LM"
mkdir -p "$SAVE"

MAX_CONTEXT_LEN=131072
SGLANG_CONTEXT_LEN=${SGLANG_CONTEXT_LEN:-262144}
RESERVE_TOKENS=16384
MAX_COMPACTIONS=10
MAX_CONSECUTIVE_MALFORMED_TOOL_CALLS=8
ROLLOUT_MAX_RESPONSE_LEN=16384
SUMMARY_MAX_TOKENS=16384
SA_SUMMARIZER_TEMPERATURE=0.1
JUDGE_MODEL="deepseek-v4-flash"
JUDGE_MAX_CONCURRENCY=8
JUDGE_TIMEOUT_SECONDS=120
JUDGE_MAX_RETRIES=1
SA_SESSION_AWARE_RHO=${SA_SESSION_AWARE_RHO:-0.2}
SA_SESSION_AWARE_BETA_0=${SA_SESSION_AWARE_BETA_0:-0.1}
SA_SESSION_AWARE_BETA_1=${SA_SESSION_AWARE_BETA_1:-0.2}
SA_SESSION_AWARE_SESSION_COUNT_KEY=${SA_SESSION_AWARE_SESSION_COUNT_KEY:-rollout_num_sessions}
SA_SESSION_IG_LAMBDA_NORMAL=${SA_SESSION_IG_LAMBDA_NORMAL:-0.1}
SA_SESSION_IG_LAMBDA_COMPACT=${SA_SESSION_IG_LAMBDA_COMPACT:-0.1}
SA_SESSION_IG_CLIP=${SA_SESSION_IG_CLIP:-2.0}
SA_SESSION_IG_EPS=${SA_SESSION_IG_EPS:-1e-6}
SA_SESSION_IG_GATING_ENABLED=${SA_SESSION_IG_GATING_ENABLED:-1}

NOW=$(date "+%Y%m%d-%H%M%S")
EXP_NAME="search-agent-setting-2-qwen3.5-4B-mosaic-ctx${MAX_CONTEXT_LEN}-${NOW}"

CKPT_ARGS=(
   --hf-checkpoint "$MODEL"
   --ref-load "$TORCH_DIST"
   --load "$TORCH_DIST"
   --save "$SAVE"
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data "$DATA"
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 60
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len $ROLLOUT_MAX_RESPONSE_LEN
   --rollout-max-context-len $MAX_CONTEXT_LEN
   --rollout-temperature 1.0

   --global-batch-size 128
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 4
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 6144
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-tensorboard
   --tb-project-name ./tensorboard_log
   --tb-experiment-name $EXP_NAME
   --log-multi-turn
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.65
   --sglang-context-length $SGLANG_CONTEXT_LEN
   --sglang-disable-custom-all-reduce
   --sglang-disable-overlap-schedule
   --sglang-disable-cuda-graph
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.search_agent_compact_rl.runtime.generate.generate_on_policy
   --custom-rm-path examples.search_agent_compact_rl.runtime.reward.browsecomp_judge
   --custom-reward-post-process-path examples.search_agent_compact_rl.runtime.grpo_hook.grpo_normalize_by_group_index
   --custom-convert-samples-to-train-data-path examples.search_agent_compact_rl.baselines.mosaic.convert_samples_to_train_data
   --custom-pre-advantage-function-path examples.search_agent_compact_rl.baselines.mosaic.compute_old_actor_session_ig_deltas
   --custom-advantage-function-path examples.search_agent_compact_rl.baselines.mosaic.compute_advantages_and_returns
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${MEGATRON_LM_PATH}:${REPO_DIR}:${SCRIPT_DIR}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "SA_COMPACT_ENABLED": "true",
    "SA_COMPACT_MAX_CONTEXT_TOKENS": "${MAX_CONTEXT_LEN}",
    "SA_COMPACT_RESERVE_TOKENS": "${RESERVE_TOKENS}",
    "SA_COMPACT_MAX_COMPACTIONS": "${MAX_COMPACTIONS}",
    "SA_MAX_CONSECUTIVE_MALFORMED_TOOL_CALLS": "${MAX_CONSECUTIVE_MALFORMED_TOOL_CALLS}",
    "SA_SUMMARIZER_MODE": "on_policy",
    "SA_SUMMARIZER_TRAIN_SUMMARY_SESSION": "true",
    "SA_SUMMARIZER_OFF_POLICY_MAX_TOKENS": "${SUMMARY_MAX_TOKENS}",
    "SA_SUMMARIZER_TEMPERATURE": "${SA_SUMMARIZER_TEMPERATURE}",
    "SA_SESSION_AWARE_RHO": "${SA_SESSION_AWARE_RHO}",
    "SA_SESSION_AWARE_BETA_0": "${SA_SESSION_AWARE_BETA_0}",
    "SA_SESSION_AWARE_BETA_1": "${SA_SESSION_AWARE_BETA_1}",
    "SA_SESSION_AWARE_SESSION_COUNT_KEY": "${SA_SESSION_AWARE_SESSION_COUNT_KEY}",
    "SA_SESSION_IG_ENABLED": "1",
    "SA_SESSION_IG_LAMBDA_NORMAL": "${SA_SESSION_IG_LAMBDA_NORMAL}",
    "SA_SESSION_IG_LAMBDA_COMPACT": "${SA_SESSION_IG_LAMBDA_COMPACT}",
    "SA_SESSION_IG_CLIP": "${SA_SESSION_IG_CLIP}",
    "SA_SESSION_IG_EPS": "${SA_SESSION_IG_EPS}",
    "SA_SESSION_IG_GATING_ENABLED": "${SA_SESSION_IG_GATING_ENABLED}",
    "JUDGE_API_KEY": "${JUDGE_API_KEY:-}",
    "JUDGE_BASE_URL": "${JUDGE_BASE_URL:-https://api.openai.com/v1}",
    "JUDGE_MODEL": "${JUDGE_MODEL}",
    "JUDGE_MAX_CONCURRENCY": "${JUDGE_MAX_CONCURRENCY}",
    "JUDGE_TIMEOUT_SECONDS": "${JUDGE_TIMEOUT_SECONDS}",
    "JUDGE_MAX_RETRIES": "${JUDGE_MAX_RETRIES}",
    "SERPER_API_KEY": "${SERPER_API_KEY:-}",
    "SERPER_BASE_URL": "${SERPER_BASE_URL:-}",
    "JINA_API_KEY": "${JINA_API_KEY:-}",
    "JINA_BASE_URL": "${JINA_BASE_URL:-}",
    "SA_SEARCH_TRACKING_FILE": "${SA_SEARCH_TRACKING_FILE:-}",
    "SA_VISIT_TRACKING_FILE": "${SA_VISIT_TRACKING_FILE:-}",
    "SA_JUDGE_TRACKING_FILE": "${SA_JUDGE_TRACKING_FILE:-}"
  }
}
EOF
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
