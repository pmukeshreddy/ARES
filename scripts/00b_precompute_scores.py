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
    output_scores_path = PROJECT_ROOT / config["dapo"]["precomputed_scores_path"]
    output_unlabeled_path = PROJECT_ROOT / config["dapo"]["unlabeled_data_path"]
    
    output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    
    precomputed_scores = {}
    
    logger.info(f"Targeting up to {args.max_samples} examples. Processing in streaming batches...")
    
    batch_size = 128
    current_batch = []
    processed_count = 0
    
    # Open both files: input for reading, output for appending JSONL
    # Open in write mode first to clear it
    with open(output_unlabeled_path, "w") as f:
        pass
        
    with open(base_data, "r") as infile, open(output_unlabeled_path, "a") as outfile:
        # Create a progress bar using the arg size or file size
        pbar = tqdm(total=args.max_samples, desc="Precomputing Scores")
        
        for i, line in enumerate(infile):
            if i >= args.max_samples:
                break
                
            data = json.loads(line)
            diff = data.get("diff_hunk", "")
            comment = data.get("comment", "")
            import hashlib
            example_id = data.get("example_id", hashlib.md5(f"{diff}_{comment}".encode('utf-8')).hexdigest())
            
            ex = {
                "example_id": example_id,
                "diff": diff,
                "comment": comment
            }
            current_batch.append(ex)
            
            # When batch is full, process it
            if len(current_batch) >= batch_size:
                prompts = [f"{ex['diff'][:2048]} [SEP] {ex['comment'][:512]}" for ex in current_batch]
                inputs = tokenizer(prompts, truncation=True, max_length=1024, padding=True, return_tensors="pt").to(rm_model.backbone.device)
                
                with torch.no_grad():
                    outputs = rm_model(inputs["input_ids"], inputs["attention_mask"])
                    scores = torch.sigmoid(outputs).squeeze(-1).tolist()
                    
                    # If batch is 1, it might not return a list
                    if not isinstance(scores, list):
                        scores = [scores]
                
                for ex_obj, score in zip(current_batch, scores):
                    precomputed_scores[ex_obj["example_id"]] = round(score, 4)
                    outfile.write(json.dumps(ex_obj) + "\n")
                
                processed_count += len(current_batch)
                pbar.update(len(current_batch))
                current_batch = []
                
        # Process any remaining items in the final partial batch
        if current_batch:
            prompts = [f"{ex['diff'][:2048]} [SEP] {ex['comment'][:512]}" for ex in current_batch]
            inputs = tokenizer(prompts, truncation=True, max_length=1024, padding=True, return_tensors="pt").to(rm_model.backbone.device)
            
            with torch.no_grad():
                outputs = rm_model(inputs["input_ids"], inputs["attention_mask"])
                scores = torch.sigmoid(outputs).squeeze(-1).tolist()
                
                if not isinstance(scores, list):
                    scores = [scores]
                    
            for ex_obj, score in zip(current_batch, scores):
                precomputed_scores[ex_obj["example_id"]] = round(score, 4)
                outfile.write(json.dumps(ex_obj) + "\n")
                
            processed_count += len(current_batch)
            pbar.update(len(current_batch))
            
        pbar.close()
            
    logger.info(f"Writing {len(precomputed_scores)} final dictionary scores to {output_scores_path}")
    with open(output_scores_path, "w") as f:
        json.dump(precomputed_scores, f)
            
    logger.info(f"Precomputation complete! Processed {processed_count} lines directly via streaming.")

if __name__ == "__main__":
    main()
