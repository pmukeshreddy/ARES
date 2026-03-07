#!/usr/bin/env python3
"""
Train the reward model (Phase 1) for RLCR v2.

Usage:
    python scripts/01_train_reward.py
    python scripts/01_train_reward.py --max-steps 10 --batch-size 4   # Smoke test
    python scripts/01_train_reward.py --config configs/custom.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import yaml
import torch
from src.training.reward_model import RewardModel
from src.training.train_reward import train_reward_model
from src.data.dataset import create_dataloaders


def main():
    parser = argparse.ArgumentParser(description="RLCR v2: Train Reward Model")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps (debug)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max data samples (debug)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Override num epochs")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override eval frequency")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("train_reward")
    
    # Load config
    config_path = Path(PROJECT_ROOT) / args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Apply CLI overrides
    if args.batch_size:
        config["reward_model"]["batch_size"] = args.batch_size
    if args.lr:
        config["reward_model"]["learning_rate"] = args.lr
    if args.epochs:
        config["reward_model"]["num_epochs"] = args.epochs
    if args.eval_steps:
        config["reward_model"]["eval_steps"] = args.eval_steps
    
    # Check data exists
    processed_dir = Path(PROJECT_ROOT) / config["data"]["processed_dir"]
    train_path = processed_dir / "train_small.jsonl"
    val_path = processed_dir / "val_small.jsonl"
    
    if not train_path.exists() or not val_path.exists():
        logger.error(
            f"Processed data not found in {processed_dir}.\n"
            f"Run 'python scripts/00_download_data.py' first."
        )
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("RLCR v2: REWARD MODEL TRAINING (Phase 1)")
    logger.info("=" * 60)
    
    # Build model
    logger.info("\n--- Building Reward Model ---")
    model = RewardModel.from_config(config)
    
    # Create data loaders
    logger.info("\n--- Loading Data ---")
    train_loader, val_loader = create_dataloaders(
        str(train_path),
        str(val_path),
        model.tokenizer,
        config,
        max_samples=args.max_samples,
    )
    
    # Train
    logger.info("\n--- Starting Training ---")
    final_metrics = train_reward_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        project_root=PROJECT_ROOT,
        max_steps=args.max_steps,
    )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL METRICS")
    for k, v in final_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 60)
    
    checkpoint_dir = Path(PROJECT_ROOT) / config["reward_model"]["output_dir"] / "best"
    logger.info(f"\nBest checkpoint: {checkpoint_dir}")
    logger.info("Next: python scripts/02_evaluate_reward.py")


if __name__ == "__main__":
    main()
