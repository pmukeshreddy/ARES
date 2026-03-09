#!/usr/bin/env python3
"""
Prepare and subsample Phase 1 Reward Model training data efficiently without OOMing.

Usage:
    python scripts/00a_prepare_rm_data.py
    python scripts/00a_prepare_rm_data.py --train-samples 38000 --val-samples 5000
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import yaml
from tqdm import tqdm


def count_lines(filepath):
    """Counts lines in a large file efficiently without loading it fully into RAM."""
    def _make_gen(reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024 * 1024)
            
    with open(filepath, 'rb') as f:
        count = sum(buf.count(b'\n') for buf in _make_gen(f.raw.read))
    return count


def reservoir_sample(filepath, k, total_lines=None):
    """Uses Reservoir Sampling to sample exactly k random lines in O(N) time and O(k) memory!"""
    sample = []
    
    # Fast path if file is shorter than k
    if total_lines is not None and total_lines <= k:
        with open(filepath, "r") as f:
            return f.readlines()
            
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f), total=total_lines, desc="Sampling..."):
            if i < k:
                sample.append(line)
            else:
                j = random.randint(0, i)
                if j < k:
                    sample[j] = line
                    
    return sample


def main():
    parser = argparse.ArgumentParser(description="RLCR v2: Efficiently Prepare Phase 1 Data")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--train-samples", type=int, default=38000, help="Number of train samples")
    parser.add_argument("--val-samples", type=int, default=5000, help="Number of val samples")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("prepare_rm_data")
    
    config_path = Path(PROJECT_ROOT) / args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    processed_dir = Path(PROJECT_ROOT) / config.get("data", {}).get("processed_dir", "data/processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    raw_train = processed_dir / "train.jsonl"
    raw_val = processed_dir / "val.jsonl"
    
    out_train = processed_dir / "train_small.jsonl"
    out_val = processed_dir / "val_small.jsonl"
    
    if not raw_train.exists() or not raw_val.exists():
        logger.error(f"Cannot find raw datasets in {processed_dir}. Run 'python scripts/00_download_data.py' first!")
        sys.exit(1)
        
    logger.info("=" * 60)
    logger.info("PHASE 1: MEMORY-EFFICIENT DATA SUBSAMPLING")
    logger.info("=" * 60)
    
    # 1. Process Validation Set Focus
    logger.info(f"Targeting {args.val_samples} validation samples...")
    val_lines = count_lines(raw_val)
    val_sample = reservoir_sample(raw_val, args.val_samples, total_lines=val_lines)
    with open(out_val, "w", encoding="utf-8") as f:
        f.writelines(val_sample)
    logger.info(f"Saved {len(val_sample)} samples to {out_val}")
    
    # 2. Process Training Set Focus
    logger.info(f"Targeting {args.train_samples} training samples...")
    train_lines = count_lines(raw_train)
    train_sample = reservoir_sample(raw_train, args.train_samples, total_lines=train_lines)
    with open(out_train, "w", encoding="utf-8") as f:
        f.writelines(train_sample)
    logger.info(f"Saved {len(train_sample)} samples to {out_train}")
    
    logger.info("\nData preparation complete! Memory usage was capped successfully.")
    logger.info(f"Next step: python scripts/01_train_reward.py")

if __name__ == "__main__":
    main()
