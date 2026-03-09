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

def evaluate_team_lora(model, tokenizer, team_name: str, test_file: str, max_samples: int = 100):
    """Evaluates a single team's LoRA adapter on its test dataset."""
    logger.info(f"\n{'='*50}\nEvaluating team: {team_name}\n{'='*50}")
    
    # 1. Load Data
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return None, None
        
    dataset = []
    with open(test_file, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
            
    # Limit samples for quicker evaluation
    dataset = dataset[:max_samples]
    if len(dataset) == 0:
        return None, None
        
    logger.info(f"Loaded {len(dataset)} test samples for {team_name}")
    
    model.eval()
    
    # 3. Generate Predictions with majority voting
    # Training generates N=16 completions per prompt — evaluating with just 1
    # misses the diversity. Generate N completions and take majority vote.
    num_votes = 8  # Number of completions per prompt for majority voting
    results = []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Evaluating {team_name}"):
            item = dataset[i]
            messages = [
                {"role": "system", "content": "You are a helpful AI code reviewer."},
                {"role": "user", "content": item["prompt"]}
            ]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer([formatted], padding=True, truncation=True, max_length=1536, return_tensors="pt").to(model.device)
            
            # Generate all N completions in ONE forward pass
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=1.0,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=num_votes,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Parse all completions
            votes_surface = 0
            votes_filter = 0
            all_completions = []
            prompt_len = len(inputs.input_ids[0])
            
            for v in range(num_votes):
                generated_ids = outputs[v][prompt_len:]
                completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
                parsed = parse_completion(completion)
                all_completions.append(completion)
                
                if parsed["decision"] == "SURFACE":
                    votes_surface += 1
                elif parsed["decision"] == "FILTER":
                    votes_filter += 1
            
            # Majority vote
            prediction = 1 if votes_surface > votes_filter else 0
            
            results.append({
                "team": team_name,
                "prompt": item["prompt"],
                "diff": item["diff"],
                "comment": item["comment"],
                "ground_truth_label": item["label"],
                "predicted_label": prediction,
                "predicted_score": None,
                "raw_completion": all_completions[0],
                "votes_surface": votes_surface,
                "votes_filter": votes_filter
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
    
    # Per-sample diagnostics
    print(f"\n  Per-sample results ({team_name}, {num_votes}-vote majority):")
    for i, r in enumerate(results[:20]):
        gt = "SURFACE" if r["ground_truth_label"] == 1 else "FILTER"
        pred = "SURFACE" if r["predicted_label"] == 1 else "FILTER"
        match = "✓" if gt == pred else "✗"
        vs = r["votes_surface"]
        vf = r["votes_filter"]
        snippet = r["raw_completion"][:100].replace('\n', ' ')
        print(f"    [{match}] GT={gt:7s} Pred={pred:7s} ({vs}S/{vf}F) | {snippet}...")
    
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
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="RLCR v2: Evaluate DAPO/KD Models")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--teams", nargs="*", default=None, help="Teams to evaluate (default: all). E.g. --teams pragmatic_shippers")
    parser.add_argument("--checkpoint-type", choices=["dapo", "kd"], default="dapo",
                        help="Which checkpoint to evaluate: 'dapo' (per-team LoRA) or 'kd' (unified distilled LoRA)")
    parser.add_argument("--max-samples", type=int, default=50, help="Max test samples per team")
    parser.add_argument("--sft", action="store_true", help="Evaluate SFT checkpoint instead of DAPO")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)
    
    model_name = config_data["dapo"]["model_name"]
    teams_dir = Path("data/teams")
    
    # Discover teams
    all_teams = sorted([d.name for d in teams_dir.iterdir() if d.is_dir()])
    if args.teams:
        teams = [t for t in args.teams if t in all_teams]
        missing = [t for t in args.teams if t not in all_teams]
        if missing:
            logger.warning(f"Teams not found: {missing}. Available: {all_teams}")
    else:
        teams = all_teams
    
    logger.info(f"Evaluating {len(teams)} teams with checkpoint type: {args.checkpoint_type}")
    
    all_metrics = []
    
    # Load Base Model & Tokenizer ONCE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
        
    logger.info(f"Loading Base Model: {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None
    )
    
    model = None
    
    if args.checkpoint_type == "kd":
        # Load unified KD LoRA once for all teams
        kd_path = Path("checkpoints/kd_unified")
        if kd_path.exists():
            logger.info(f"Loading unified KD LoRA from {kd_path}...")
            model = PeftModel.from_pretrained(base_model, str(kd_path))
        else:
            logger.error(f"KD checkpoint not found at {kd_path}. Run 03b_knowledge_distill.py first.")
            sys.exit(1)
    
    # Process each team
    for team_name in teams:
        team_dir = teams_dir / team_name
        test_file = str(team_dir / "test.jsonl")
        
        if args.checkpoint_type == "dapo":
            if args.sft:
                lora_path = str(Path("checkpoints/sft_warmup") / f"sft_warmup_{team_name}")
                label = "SFT"
            else:
                lora_path = str(Path("checkpoints/dapo") / f"dapo_lora_{team_name}")
                label = "DAPO"
            
            if os.path.exists(lora_path):
                logger.info(f"Loading {label} LoRA weights from {lora_path} for {team_name}...")
                if model is None:
                    model = PeftModel.from_pretrained(base_model, lora_path, adapter_name=team_name)
                else:
                    model.load_adapter(lora_path, adapter_name=team_name)
                model.set_adapter(team_name)
            else:
                logger.warning(f"{label} weights not found at {lora_path}. Using base model.")
                model = base_model
        
        # For KD, model is already loaded — same LoRA for all teams
            
        metrics, results = evaluate_team_lora(model, tokenizer, team_name, test_file, max_samples=args.max_samples)
        
        if metrics:
            metrics["checkpoint_type"] = args.checkpoint_type
            all_metrics.append(metrics)
            
            # Save raw predictions for analysis
            out_file = f"data/teams/{team_name}/predictions_{args.checkpoint_type}.jsonl"
            with open(out_file, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
                    
    # Print summary
    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY ({args.checkpoint_type.upper()} checkpoints)")
    print("="*60)
    for m in all_metrics:
        print(f"Team: {m['team']:<20} | Acc: {m['accuracy']:.2f} | P: {m['precision']:.2f} | R: {m['recall']:.2f} | F1: {m['f1']:.2f} | Surface%: {m['surface_ratio']:.2f}")
    print("="*60)

if __name__ == "__main__":
    main()

