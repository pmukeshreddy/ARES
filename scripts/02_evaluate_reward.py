#!/usr/bin/env python3
"""
Evaluate trained reward model on test set.

Usage:
    python scripts/02_evaluate_reward.py
    python scripts/02_evaluate_reward.py --checkpoint checkpoints/reward_model/best
    python scripts/02_evaluate_reward.py --split test
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import yaml
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.training.reward_model import RewardModel
from src.data.dataset import RewardModelDataset


def main():
    parser = argparse.ArgumentParser(description="RLCR v2: Evaluate Reward Model")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Eval split")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--output", default=None, help="Save predictions to JSON")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("evaluate_reward")
    
    # Load config
    config_path = Path(PROJECT_ROOT) / args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Find checkpoint
    checkpoint_dir = args.checkpoint
    if checkpoint_dir is None:
        checkpoint_dir = str(
            Path(PROJECT_ROOT) / config["reward_model"]["output_dir"] / "best"
        )
    
    if not Path(checkpoint_dir).exists():
        logger.error(f"Checkpoint not found: {checkpoint_dir}")
        logger.error("Run 'python scripts/01_train_reward.py' first.")
        sys.exit(1)
    
    # Find data
    data_path = (
        Path(PROJECT_ROOT) / config["data"]["processed_dir"] / f"{args.split}.jsonl"
    )
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("RLCR v2: REWARD MODEL EVALUATION")
    logger.info(f"  Checkpoint: {checkpoint_dir}")
    logger.info(f"  Split: {args.split}")
    logger.info(f"  Data: {data_path}")
    logger.info("=" * 60)
    
    # Load model
    logger.info("\nLoading model...")
    model = RewardModel.load_checkpoint(checkpoint_dir, config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load data
    dataset = RewardModelDataset(
        str(data_path),
        model.tokenizer,
        max_length=config["data"]["max_input_tokens"],
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Run inference
    all_labels = []
    all_probs = []
    
    logger.info("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            
            logits = model(input_ids, attention_mask).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    
    # Compute metrics
    auroc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"  AUROC:     {auroc:.4f}")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1:        {f1:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    logger.info(f"  FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
    
    logger.info(f"\nClassification Report:")
    logger.info("\n" + classification_report(all_labels, all_preds, 
                                              target_names=["ignored", "acted_on"]))
    
    # AUROC verdict
    target = config["reward_model"]["auroc_target"]
    if auroc >= target:
        logger.info(f"✅ AUROC {auroc:.4f} ≥ {target} — READY FOR PHASE 2")
    else:
        logger.warning(f"⚠️  AUROC {auroc:.4f} < {target} — needs improvement")
    
    # Score distribution analysis
    pos_scores = all_probs[all_labels == 1]
    neg_scores = all_probs[all_labels == 0]
    logger.info(f"\nScore Distribution:")
    logger.info(f"  Positive (acted on): mean={pos_scores.mean():.3f}, "
                f"std={pos_scores.std():.3f}, median={np.median(pos_scores):.3f}")
    logger.info(f"  Negative (ignored):  mean={neg_scores.mean():.3f}, "
                f"std={neg_scores.std():.3f}, median={np.median(neg_scores):.3f}")
    
    # Save predictions if requested
    if args.output:
        output_path = Path(args.output)
        results = {
            "metrics": {
                "auroc": float(auroc),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            },
            "predictions": [
                {"label": int(l), "prob": float(p), "pred": int(d)}
                for l, p, d in zip(all_labels, all_probs, all_preds)
            ],
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
