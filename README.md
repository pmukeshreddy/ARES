# RLCR v2: Reinforcement Learning for Code Review

A reward model that learns `f(diff, comment) → quality_score` to predict whether a developer will act on a code review comment. This serves as the foundation for per-team DAPO fine-tuning.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download & preprocess data
python scripts/00_download_data.py

# 3. Train reward model
python scripts/01_train_reward.py

# 4. Evaluate
python scripts/02_evaluate_reward.py
```

## Smoke Test (CPU, no GPU needed)

```bash
# Download a small subset
python scripts/00_download_data.py --max-samples 500 --skip-zenodo --skip-embeddings

# Quick training run
python scripts/01_train_reward.py --max-steps 10 --batch-size 4 --eval-steps 5

# Evaluate
python scripts/02_evaluate_reward.py --batch-size 4
```

## Project Structure

```
configs/default.yaml          # All hyperparameters
src/data/
  download.py                 # Download CodeReviewer + HF datasets
  preprocessing.py            # Filter, clean, split (80/10/10)
  dataset.py                  # PyTorch Dataset: [diff] [SEP] [comment]
  embeddings.py               # MiniLM embedding precomputation
src/training/
  reward_model.py             # Qwen2.5-Coder + LoRA + classification head
  train_reward.py             # Training loop: BCE, AdamW, AUROC validation
scripts/
  00_download_data.py         # CLI: data pipeline
  01_train_reward.py          # CLI: train reward model
  02_evaluate_reward.py       # CLI: evaluate on test set
```

## Pipeline Overview

| Phase | What | Runtime |
|-------|------|---------|
| **Subpart 0** | Data: download, filter, split, embed | ~15 min |
| **Subpart 1** | Reward model: binary classifier on 100K+ samples | ~15-20 min (H100) |
| **Subpart 2** | DAPO: per-team RL with 20-50 samples (future) | ~5 min |
| **Subpart 3** | KD: compress to MiniLM for production (future) | ~15-20 min |
| **Subpart 4** | Evaluation: vs cosine similarity baseline (future) | ~5 min |

## Configuration

All settings are in `configs/default.yaml`. Key options:

- `reward_model.model_name`: Default `Qwen/Qwen2.5-Coder-1.5B`, use `7B` for better quality
- `reward_model.bf16`: Set `false` if no bf16 support
- `reward_model.batch_size`: Reduce if OOM
- `data.max_diff_tokens`: Max diff length in tokens

## Requirements

- Python 3.10+
- PyTorch 2.1+
- GPU with ≥16GB VRAM (for training)
- ~5GB disk space for datasets
