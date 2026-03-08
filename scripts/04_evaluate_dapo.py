"""
Evaluate Phase 2 DAPO Models.

This script runs the trained Qwen2.5-Coder-3B-Instruct model with its
team-specific LoRA adapters on the corresponding test sets.
It records the model's <decision> (SURFACE/FILTER) and <score> outputs,
and calculates metrics like Accuracy, F1, and Precision/Recall.
"""

import os
import sys
import json
import logging
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.rewards import parse_completion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def evaluate_team_lora(model_name: str, team_name: str, test_file: str, lora_path: str, max_samples: int = 100):
    """Evaluates a single team's LoRA adapter on its test dataset."""
    logger.info(f"\n{'='*50}\nEvaluating team: {team_name}\n{'='*50}")
    
    # 1. Load Data
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return None
        
    dataset = []
    with open(test_file, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
            
    # Limit samples for quicker evaluation
    dataset = dataset[:max_samples]
    if len(dataset) == 0:
        return None
        
    logger.info(f"Loaded {len(dataset)} test samples for {team_name}")
    
    # 2. Load Model & Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Loading Base Model: {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None
    )
    
    if os.path.exists(lora_path):
        logger.info(f"Loading LoRA weights from {lora_path}...")
        model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        logger.warning(f"LoRA weights not found at {lora_path}. Using base model.")
        model = base_model
        
    model.eval()
    
    # 3. Generate Predictions
    results = []
    batch_size = 4
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {team_name}"):
            batch = dataset[i:i+batch_size]
            prompts = [item["prompt"] for item in batch]
            
            inputs = tokenizer(prompts, padding=True, truncation=True, max_length=1536, return_tensors="pt").to(model.device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.1,  # Low temp for deterministic evaluation
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Extract only the generated text (ignore prompt)
            generated_ids = [output[len(input_id):] for input_id, output in zip(inputs.input_ids, outputs)]
            completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Parse responses
            for idx, completion in enumerate(completions):
                item = batch[idx]
                parsed = parse_completion(completion)
                
                # Convert string decision to binary prediction (1=SURFACE, 0=FILTER)
                prediction = 1 if parsed["decision"] == "SURFACE" else 0
                
                results.append({
                    "team": team_name,
                    "prompt": item["prompt"],
                    "diff": item["diff"],
                    "comment": item["comment"],
                    "ground_truth_label": item["label"],
                    "predicted_label": prediction,
                    "predicted_score": parsed["score"],
                    "raw_completion": completion
                })
                
    # 4. Calculate Metrics
    correct = sum(1 for r in results if r["ground_truth_label"] == r["predicted_label"])
    accuracy = correct / len(results)
    
    true_positives = sum(1 for r in results if r["ground_truth_label"] == 1 and r["predicted_label"] == 1)
    false_positives = sum(1 for r in results if r["ground_truth_label"] == 0 and r["predicted_label"] == 1)
    false_negatives = sum(1 for r in results if r["ground_truth_label"] == 1 and r["predicted_label"] == 0)
    
    precision = true_positives / max(1, (true_positives + false_positives))
    recall = true_positives / max(1, (true_positives + false_negatives))
    f1 = 2 * (precision * recall) / max(1e-6, (precision + recall))
    
    surface_ratio = sum(1 for r in results if r["predicted_label"] == 1) / len(results)
    
    metrics = {
        "team": team_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "surface_ratio": surface_ratio,
        "total_samples": len(results)
    }
    
    logger.info(f"Team {team_name} Metrics: Acc: {accuracy:.2f}, F1: {f1:.2f}, Surface Ratio: {surface_ratio:.2f}")
    return metrics, results

def main():
    import yaml
    with open("configs/default.yaml", "r") as f:
        config_data = yaml.safe_load(f)
    
    model_name = config_data["dapo"]["model_name"]
    teams_dir = Path("data/teams")
    checkpoints_dir = Path("checkpoints/dapo")
    
    all_metrics = []
    
    # Process each team
    for team_dir in teams_dir.iterdir():
        if not team_dir.is_dir():
            continue
            
        team_name = team_dir.name
        test_file = str(team_dir / "test.jsonl")
        lora_path = str(checkpoints_dir / f"dapo_lora_{team_name}")
        
        metrics, results = evaluate_team_lora(model_name, team_name, test_file, lora_path, max_samples=50)
        
        if metrics:
            all_metrics.append(metrics)
            
            # Save raw predictions for analysis
            out_file = f"data/teams/{team_name}/predictions.jsonl"
            with open(out_file, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
                    
    # Print summary
    print("\n" + "="*50)
    print("DAPO PHASE 2 GENERALIZATION EVALUATION SUMMARY")
    print("="*50)
    for m in all_metrics:
        print(f"Team: {m['team']:<20} | Acc: {m['accuracy']:.2f} | P: {m['precision']:.2f} | R: {m['recall']:.2f} | F1: {m['f1']:.2f} | Surface%: {m['surface_ratio']:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()
