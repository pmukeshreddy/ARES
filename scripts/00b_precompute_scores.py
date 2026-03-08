import json
import logging
import argparse
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.reward_model import RewardModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description="Precompute Phase 1 RM scores for unlabeled data")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--max-samples", type=int, default=100000, help="Max samples to process")
    args = parser.parse_args()
    
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    rm_path = PROJECT_ROOT / config["reward_model"]["output_dir"] / "best"
    if not rm_path.exists():
        logger.error(f"Phase 1 RM checkpoint not found at {rm_path}")
        sys.exit(1)
        
    logger.info("Loading Phase 1 Reward Model...")
    rm_model = RewardModel.load_checkpoint(str(rm_path), config).to("cuda")
    rm_model.eval()
    tokenizer = rm_model.tokenizer
    
    # Load unlabeled data - fallback to train.jsonl
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    base_data = processed_dir / "train.jsonl"
    
    if not base_data.exists():
        logger.error(f"Base data {base_data} not found.")
        sys.exit(1)
        
    logger.info(f"Loading data from {base_data}")
    unlabeled_examples = []
    
    import hashlib
    with open(base_data, "r") as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            data = json.loads(line)
            # Create a deterministic content hash as example_id
            diff = data.get("diff_hunk", "")
            comment = data.get("comment", "")
            example_id = data.get("example_id", hashlib.md5(f"{diff}_{comment}".encode('utf-8')).hexdigest())
            unlabeled_examples.append({
                "example_id": example_id,
                "diff": diff,
                "comment": comment
            })

    output_scores_path = PROJECT_ROOT / config["dapo"]["precomputed_scores_path"]
    output_unlabeled_path = PROJECT_ROOT / config["dapo"]["unlabeled_data_path"]
    
    output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    
    precomputed_scores = {}
    
    logger.info(f"Scoring {len(unlabeled_examples)} examples with Reward Model...")
    
    batch_size = 32
    for i in tqdm(range(0, len(unlabeled_examples), batch_size)):
        batch = unlabeled_examples[i:i+batch_size]
        prompts = [f"{ex['diff'][:2048]} [SEP] {ex['comment'][:512]}" for ex in batch]
        
        inputs = tokenizer(prompts, truncation=True, max_length=1024, padding=True, return_tensors="pt").to(rm_model.backbone.device)
        
        with torch.no_grad():
            outputs = rm_model(inputs["input_ids"], inputs["attention_mask"])
            scores = torch.sigmoid(outputs).squeeze(-1).tolist()
            
        for ex, score in zip(batch, scores):
            precomputed_scores[ex["example_id"]] = round(score, 4)
            
    logger.info(f"Writing {len(precomputed_scores)} scores to {output_scores_path}")
    with open(output_scores_path, "w") as f:
        json.dump(precomputed_scores, f)
        
    logger.info(f"Writing {len(unlabeled_examples)} unlabeled pairs to {output_unlabeled_path}")
    with open(output_unlabeled_path, "w") as f:
        for ex in unlabeled_examples:
            f.write(json.dumps(ex) + "\n")
            
    logger.info("Precomputation complete!")

if __name__ == "__main__":
    main()
