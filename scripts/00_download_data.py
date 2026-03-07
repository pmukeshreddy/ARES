#!/usr/bin/env python3
"""
Download and preprocess datasets for RLCR v2.

Usage:
    python scripts/00_download_data.py
    python scripts/00_download_data.py --max-samples 1000   # Debug mode
    python scripts/00_download_data.py --skip-zenodo         # HF only
    python scripts/00_download_data.py --skip-embeddings     # No MiniLM
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import yaml
from src.data.download import download_codereviewer, download_hf_dataset
from src.data.preprocessing import preprocess_and_split
from src.data.embeddings import precompute_embeddings


def main():
    parser = argparse.ArgumentParser(description="RLCR v2: Download & Preprocess Data")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (debug)")
    parser.add_argument("--skip-zenodo", action="store_true", help="Skip CodeReviewer download")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HF dataset download")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip MiniLM embeddings")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("download_data")
    
    # Load config
    config_path = Path(PROJECT_ROOT) / args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 60)
    logger.info("RLCR v2: DATA PIPELINE")
    logger.info("=" * 60)
    
    input_paths = []
    
    # Step 1: Download CodeReviewer from Zenodo
    if not args.skip_zenodo:
        logger.info("\n--- Step 1: Download CodeReviewer (Zenodo) ---")
        try:
            cr_path = download_codereviewer(config, PROJECT_ROOT)
            input_paths.append(cr_path)
        except Exception as e:
            logger.error(f"Failed to download CodeReviewer: {e}")
            logger.info("Continuing with HF dataset only...")
    else:
        logger.info("Skipping CodeReviewer download (--skip-zenodo)")
    
    # Step 2: Download HF dataset
    if not args.skip_hf:
        logger.info("\n--- Step 2: Download ronantakizawa/github-codereview (HF) ---")
        try:
            hf_path = download_hf_dataset(config, PROJECT_ROOT)
            input_paths.append(hf_path)
        except Exception as e:
            logger.error(f"Failed to download HF dataset: {e}")
    else:
        logger.info("Skipping HF dataset download (--skip-hf)")
    
    if not input_paths:
        logger.error("No data downloaded! Exiting.")
        sys.exit(1)
    
    # Step 3: Filter & Split
    logger.info("\n--- Step 3: Filter & Split ---")
    split_paths = preprocess_and_split(
        input_paths, config, PROJECT_ROOT, max_samples=args.max_samples
    )
    
    logger.info("\nSplit files:")
    for split_name, path in split_paths.items():
        count = sum(1 for _ in open(path))
        logger.info(f"  {split_name}: {path} ({count} samples)")
    
    # Step 4: Precompute embeddings
    if not args.skip_embeddings:
        logger.info("\n--- Step 4: Precompute MiniLM Embeddings ---")
        emb_path, ids_path = precompute_embeddings(
            list(split_paths.values()), config, PROJECT_ROOT
        )
        logger.info(f"Embeddings: {emb_path}")
        logger.info(f"IDs: {ids_path}")
    else:
        logger.info("Skipping embeddings (--skip-embeddings)")
    
    logger.info("\n" + "=" * 60)
    logger.info("DATA PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
