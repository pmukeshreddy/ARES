#!/usr/bin/env python3
"""
GRPO DAPO Per-Team Training (Phase 2) for RLCR v2.

Usage:
    python scripts/03_train_dapo.py
"""

import argparse
import logging
import os
import subprocess
import sys
import time
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import yaml
import torch
from src.training.reward_model import RewardModel
from src.data.team_dataset import simulate_team_datasets
from src.training.dapo_trainer import DAPOTrainer

def start_sglang_server(model_name: str, port: int) -> subprocess.Popen:
    """Starts the SGLang generation engine as a background process."""
    logger = logging.getLogger("dapo_main")
    logger.info(f"Starting SGLang server for {model_name} on port {port}...")
    
    # We use subprocess to isolate SGLang memory from PyTorch memory safely.
    # --disable-cuda-graph is often needed for dynamic LoRAs if memory is extremely tight
    # --lora-paths allows dynamic swapping later via the API.
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", model_name,
        "--port", str(port),
        "--trust-remote-code",
        # For H100
        "--dtype", "bfloat16"
    ]
    
    # Suppress STDOUT to not spam the training logs, but keep STDERR for crashes.
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    
    # Wait for boot
    import requests
    max_retries = 60
    for i in range(max_retries):
        try:
            resp = requests.get(f"http://localhost:{port}/health")
            if resp.status_code == 200:
                logger.info("SGLang server is fully booted and ready!")
                return process
        except Exception:
            pass
        time.sleep(2)
        
    # If we get here, it probably crashed. Let's dump stderr
    logger.error("SGLang server failed to start within 120 seconds.")
    _, stderr = process.communicate()
    logger.error(f"SGLang STDERR: {stderr}")
    process.terminate()
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="RLCR v2: Phase 2 DAPO Per-Team Training")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--sglang-port", type=int, default=30000, help="Local port for SGLang engine")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("dapo_main")
    
    # Load config
    config_path = Path(PROJECT_ROOT) / args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    logger.info("=" * 60)
    logger.info("RLCR v2: PHASE 2 DAPO PER-TEAM TRAINING")
    logger.info("=" * 60)
    
    # 1. Ensure Data Exists
    teams_dir = Path(PROJECT_ROOT) / "data" / "teams"
    if not teams_dir.exists() or len(list(teams_dir.glob("*"))) == 0:
        logger.info("Simulated team data not found, generating now...")
        processed_dir = Path(PROJECT_ROOT) / config["data"]["processed_dir"]
        base_data = processed_dir / "train.jsonl"
        # Use small if available, else primary
        if (processed_dir / "train_small.jsonl").exists():
            base_data = processed_dir / "train_small.jsonl"
            
        simulate_team_datasets(str(base_data), str(teams_dir))
        
    teams = [d.name for d in teams_dir.iterdir() if d.is_dir()]
    logger.info(f"Discovered {len(teams)} teams: {', '.join(teams)}")
    
    # 2. Load Phase 1 Reward Model (Frozen)
    rm_path = Path(PROJECT_ROOT) / config["reward_model"]["output_dir"] / "best"
    if not rm_path.exists():
        logger.error(f"Phase 1 Reward Model checkpoint not found at {rm_path}")
        logger.error("Run 'python scripts/01_train_reward.py' first!")
        sys.exit(1)
        
    logger.info("Loading Phase 1 Reward Model for R1 / R3 computation...")
    # Load base model from HF, adapters from local
    rm_model = RewardModel.from_pretrained(str(rm_path))
    rm_model = rm_model.to("cuda")
    rm_model.eval()
    
    # 3. Start SGLang Subprocess
    dapo_config = config.get("dapo", {})
    # Inject port
    config.setdefault("dapo", {})["sglang_port"] = args.sglang_port
    
    # 3B coder instruct base
    base_model_name = dapo_config.get("model_name", "Qwen/Qwen2.5-Coder-3B-Instruct")
    
    sglang_process = start_sglang_server(base_model_name, args.sglang_port)
    
    try:
        # 4. Initialize Trainer
        trainer = DAPOTrainer(config, rm_model, rm_model.tokenizer)
        
        # 5. Train Per Team
        for team in teams:
            team_train_path = teams_dir / team / "train.jsonl"
            if not team_train_path.exists():
                logger.warning(f"No train data for team {team}, skipping.")
                continue
                
            # Load dataset
            with open(team_train_path, "r") as f:
                train_data = [json.loads(line) for line in f]
                
            logger.info(f"Team {team} dataset loaded: {len(train_data)} samples.")
            trainer.train_team(team, train_data)
            
        logger.info("\n" + "=" * 60)
        logger.info("ALL TEAMS FINISHED DAPO TRAINING SUCCESSFULLY!")
        logger.info("Models saved to disk.")
        logger.info("=" * 60)
            
    except Exception as e:
        logger.error(f"Training crashed: {e}", exc_info=True)
    finally:
        # 6. Shutdown SGLang gracefully
        logger.info("Shutting down SGLang engine...")
        sglang_process.terminate()
        sglang_process.wait()

if __name__ == "__main__":
    main()
