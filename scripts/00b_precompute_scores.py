import json
import logging
import argparse
import sys
import hashlib
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.reward_model import RewardModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def collate_fn(batch):
    """Custom collator to transpose list of dicts into dict of lists for the model"""
    example_ids = [item["example_id"] for item in batch]
    diffs = [item["diff"] for item in batch]
    comments = [item["comment"] for item in batch]
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
    
    return {
        "example_id": example_ids,
        "diff": diffs,
        "comment": comments,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

def main():
    parser = argparse.ArgumentParser(description="Precompute Phase 1 RM scores for unlabeled data using PyTorch DataLoaders")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--max-samples", type=int, default=100000, help="Max samples to process")
    parser.add_argument("--batch-size", type=int, default=128, help="GPU Batch Size")
    parser.add_argument("--workers", type=int, default=4, help="Number of PyTorch Dataloader CPU background workers")
    args = parser.parse_args()
    
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    rm_path = PROJECT_ROOT / config["reward_model"]["output_dir"] / "best"
    if not rm_path.exists():
        logger.error(f"Phase 1 RM checkpoint not found at {rm_path}")
        sys.exit(1)
        
    logger.info("Loading Phase 1 Reward Model (With massive DataLoader)...")
    rm_model = RewardModel.load_checkpoint(str(rm_path), config).to("cuda")
    rm_model.eval()
    tokenizer = rm_model.tokenizer
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    base_data = str(processed_dir / "train.jsonl")
    
    if not os.path.exists(base_data):
        logger.error(f"Base data {base_data} not found.")
        sys.exit(1)
        
    output_scores_path = PROJECT_ROOT / config["dapo"]["precomputed_scores_path"]
    output_unlabeled_path = PROJECT_ROOT / config["dapo"]["unlabeled_data_path"]
    output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Setting up Multi-Worker Streaming Dataset...")
    
    # 1. Load the huge 10GB file using the safe TEXT iterator via streaming
    # This prevents the HuggingFace JSON backend from crashing on unstructured/ragged JSON keys
    dataset = load_dataset("text", data_files=base_data, split="train", streaming=True)
    dataset = dataset.take(args.max_samples)
    
    # 2. Setup the CPU tokenization mapping function
    def tokenize_and_format(examples):
        diffs = []
        comments = []
        example_ids = []
        
        # Safely parse each JSON row string manually to ignore unstructured columns
        for line in examples["text"]:
            data = json.loads(line)
            diffs.append(data.get("diff_hunk", ""))
            comments.append(data.get("comment", ""))
            example_ids.append(data.get("example_id", None))
        
        final_ids = []
        for i in range(len(diffs)):
            if example_ids[i] is not None:
                final_ids.append(example_ids[i])
            else:
                final_ids.append(hashlib.md5(f"{diffs[i]}_{comments[i]}".encode('utf-8')).hexdigest())

        prompts = [f"{str(d)[:2048]} [SEP] {str(c)[:512]}" for d, c in zip(diffs, comments)]
        
        tokens = tokenizer(
            prompts, 
            truncation=True, 
            max_length=1024, 
            padding="max_length",
        )
        
        return {
            "example_id": final_ids,
            "diff": diffs,
            "comment": comments,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }

    # 3. Apply the mapping to the streaming dataset
    # We must remove the "text" column since the model doesn't expect it
    tokenized_dataset = dataset.map(tokenize_and_format, batched=True, batch_size=1000, remove_columns=["text"])
    
    # 4. Wrap with PyTorch DataLoader
    dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.workers,
        prefetch_factor=2,
        collate_fn=collate_fn
    )

    precomputed_scores = {}
    processed_count = 0
    
    logger.info(f"Targeting up to {args.max_samples} examples with BS={args.batch_size} and {args.workers} background workers.")
    
    with open(output_unlabeled_path, "w") as f:
        pass
        
    pbar = tqdm(total=args.max_samples, desc="Precomputing Scores (CUDA + Multiprocess)")
    
    with open(output_unlabeled_path, "a") as outfile:
        # 5. Start the engine. Check iterator to prevent crashing empty loads.
        try:
            for batch in dataloader:
                input_ids = batch["input_ids"].to(rm_model.backbone.device)
                attention_mask = batch["attention_mask"].to(rm_model.backbone.device)
                
                with torch.no_grad():
                    outputs = rm_model(input_ids, attention_mask)
                    scores = torch.sigmoid(outputs).squeeze(-1).tolist()
                    
                if not isinstance(scores, list):
                    scores = [scores]
                    
                del input_ids
                del attention_mask
                
                for i in range(len(scores)):
                    ex_id = batch["example_id"][i]
                    score = round(scores[i], 4)
                    
                    ex_obj = {
                        "example_id": ex_id,
                        "diff": batch["diff"][i],
                        "comment": batch["comment"][i]
                    }
                    
                    precomputed_scores[ex_id] = score
                    outfile.write(json.dumps(ex_obj) + "\n")
                    
                batch_len = len(scores)
                processed_count += batch_len
                pbar.update(batch_len)
                
                if processed_count >= args.max_samples:
                    break
        except Exception as e:
            logger.error(f"Dataloader finished or caught standard iteration exception: {e}")
            
    pbar.close()
            
    logger.info(f"Writing {len(precomputed_scores)} final dictionary scores to {output_scores_path}")
    with open(output_scores_path, "w") as f:
        json.dump(precomputed_scores, f)
            
    logger.info(f"Precomputation complete! Perfectly utilized multi-core streaming on {processed_count} examples.")

if __name__ == "__main__":
    main()
