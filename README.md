<div align="center">

# Mosaic

**Mosaic: Multi-Session Agentic Reinforcement Learning**

<!-- Replace the paper URL placeholder with the release link. -->
<a href="PAPER_URL"><img src="https://img.shields.io/badge/Paper-PDF-yellow?style=for-the-badge&logo=readthedocs" alt="Paper"></a>
<a href="https://arxiv.org/abs/2610.22174"><img src="https://img.shields.io/badge/ArXiv-Paper-red?style=for-the-badge&logo=arxiv" alt="arXiv"></a>
<a href="https://lark-ai-lab.github.io/Mosaic/"><img src="https://img.shields.io/badge/Homepage-Website-blue?style=for-the-badge&logo=googlechrome" alt="Homepage"></a>

*Before reading our paper, we highly recommend visiting our [Blog](https://lark-ai-lab.github.io/Mosaic/) for a visual introduction to Multi-Session Agentic RL—explore the topic through illustrations rather than dry prose.*

---

</div>

## What's Multi-Session Agentic RL?

**Multi-Session Agentic Reinforcement Learning** is a setting where a single rollout can span multiple sessions. Each session consists of agent–environment interactions within one continuously accumulated context and becomes a separate training segment with its own assigned advantage. Context compaction and subagent forking are two ways to create multiple sessions within one rollout, as they both can introduce discontinuities in the context prefix and create session boundaries.

## Why Mosaic?

**Broadcasting** is a common approach to Multi-Session Agentic RL: it computes a trajectory-level advantage from the overall task outcome, then applies that same advantage to every session and every trainable token within it. However, this gives useful, redundant, and unhelpful sessions the same credit, and does not distinguish trajectories that reach the same outcome using different numbers of sessions.

We propose Mosaic, a general method for multi-session agentic RL, which combines two levels of advantage shaping:

- **Session-aware trajectory advantage:** adjusts outcome-based advantages using relative session counts within success/failure subgroups, balancing efficiency and exploration.
- **Session-level marginal gain:** measures each session's marginal information gain through changes in the ground-truth answer's log-likelihood, then adds a type-conditioned normalization for different sessions within each prompt group.

Our implementation supports both context compaction and subagent forking, two techniques widely used in agent harnesses such as Claude Code.

## Installation

Follow the [slime environment setup](https://thudm.github.io/slime/get_started/quick_start.html) to prepare a compatible CUDA, PyTorch, SGLang, and Megatron-LM environment. Then install this checkout in that environment:

```bash
git clone https://github.com/LARK-AI-Lab/Mosaic.git
cd Mosaic
pip install -e . --no-deps
cp .env.example .env
```

Fill in `.env` with your judge API credentials and endpoint, Serper search credentials, and Jina Reader credentials. Set `JUDGE_MODEL` in the selected training script to a model available at your judge endpoint.


## Model and Data Preparation

The supplied scripts use [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). Download its Hugging Face checkpoint and convert it to Megatron `torch_dist` format:

```bash
hf download Qwen/Qwen3.5-4B --local-dir path/to/Qwen3.5-4B
source scripts/models/qwen3.5-4B.sh
PYTHONPATH="path/to/Megatron-LM:$PWD" \
  torchrun --nproc_per_node=8 tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint path/to/Qwen3.5-4B \
  --save path/to/Qwen3.5-4B_torch_dist
```

Prepare a Parquet training dataset with `problem` and `answer` columns, such as [REDSearcher_RL_1K](https://huggingface.co/datasets/Zchu/REDSearcher_RL_1K):

```bash
hf download --repo-type dataset Zchu/REDSearcher_RL_1K \
  --local-dir path/to/REDSearcher_RL_1K
```

Before training, replace `MODEL`, `TORCH_DIST`, `DATA`, `SAVE`, and `MEGATRON_LM_PATH` in the selected script with actual paths. `DATA` must point to the training Parquet file. The supplied launchers target one node with eight GPUs.


## How to run

Run from the repository root:

```bash
# For context compaction
bash examples/search_agent_compact_rl/scripts/train/run_setting_2_qwen3.5_4B_mosaic.sh

# For forked subagents
bash examples/search_agent_subagent_rl/scripts/train/run_qwen3.5_4B_mosaic.sh
```


## Acknowledgements

We thank [slime](https://github.com/THUDM/slime), [DR-Venus](https://github.com/inclusionAI/DR-Venus), [Dressage](https://github.com/Accio-Lab/Dressage), and [REDSearcher](https://huggingface.co/datasets/Zchu/REDSearcher_RL_1K) for their inspiration and support.


## Citation

```bibtex
@article{xu2026mosaic,
  title={Mosaic: Multi-Session Agentic Reinforcement Learning},
  author={Xu, Yilong and Xu, Minrui and Zhou, Xinyu and Long, Xiang and Tan, Haochen and Guo Zhijiang},
  journal={arXiv preprint},
  year={2026}
}
```
