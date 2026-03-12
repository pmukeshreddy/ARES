"""
Evaluate Phase 2 DAPO Models using SGLang (same engine as training).

Uses the same SGLang server + LoRA loading as DAPO training to ensure
generation behavior matches exactly.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import time
import signal
from pathlib import Path
from tqdm import tqdm

import yaml
import requests

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.rewards import parse_completion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SGLANG_PORT = 30000
SGLANG_URL = f"http://localhost:{SGLANG_PORT}"

# Load tokenizer once at module level
_tokenizer = None
def get_tokenizer(model_name):
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return _tokenizer


def kill_existing_sglang():
    """Kill any existing SGLang server on our port — forcefully."""
    logger.info("Killing any existing SGLang processes...")
    # Kill by process name
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
    # Kill by port
    subprocess.run(f"fuser -k {SGLANG_PORT}/tcp", shell=True, capture_output=True)
    time.sleep(5)
    
    # Verify port is free
    try:
        resp = requests.get(f"{SGLANG_URL}/health", timeout=2)
        if resp.status_code == 200:
            logger.warning("Server STILL running after kill! Trying again...")
            subprocess.run(["pkill", "-9", "-f", "sglang"], capture_output=True)
            time.sleep(5)
    except Exception:
        logger.info("Port is free.")


def start_sglang_server(model_name: str, dapo_config: dict):
    """Starts SGLang server identical to training."""
    kill_existing_sglang()
    
    logger.info(f"Starting SGLang server for {model_name} on port {SGLANG_PORT}...")
    
    lora_rank = str(dapo_config.get("lora_r", 16))
    lora_targets = dapo_config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    sglang_python = sys.executable
    
    cmd = [
        sglang_python, "-m", "sglang.launch_server",
        "--model-path", model_name,
        "--port", str(SGLANG_PORT),
        "--trust-remote-code",
        "--enable-lora",
        "--max-lora-rank", lora_rank,
        "--max-loras-per-batch", "2",
        "--lora-target-modules"
    ] + lora_targets + [
        "--mem-fraction-static", "0.4",
        "--dtype", "bfloat16"
    ]
    
    log_file = open("sglang_eval.log", "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, text=True)
    
    logger.info("Waiting for SGLang to boot (up to 120s)...")
    for i in range(60):
        try:
            resp = requests.get(f"{SGLANG_URL}/health", timeout=2)
            if resp.status_code == 200:
                logger.info(f"SGLang server is ready! (took {(i+1)*2}s)")
                return process
        except Exception:
            pass
        time.sleep(2)
    
    logger.error("SGLang failed to start within 120s.")
    try:
        with open("sglang_eval.log") as f:
            logger.error(f"Log tail:\n{f.read()[-2000:]}")
    except Exception:
        pass
    process.terminate()
    sys.exit(1)


def load_lora(adapter_name: str, lora_path: str):
    """Load a LoRA adapter into SGLang — same API as training."""
    url = f"{SGLANG_URL}/load_lora_adapter"
    payload = {"lora_name": adapter_name, "lora_path": lora_path}
    logger.info(f"Loading LoRA '{adapter_name}' from {lora_path}")
    resp = requests.post(url, json=payload)
    logger.info(f"LoRA response [{resp.status_code}]: {resp.text[:500]}")
    if resp.status_code != 200:
        logger.error(f"LoRA load FAILED!")
        return False
    return True


def generate_completions(prompts: list, lora_name: str, tokenizer, n: int = 8, max_tokens: int = 256):
    """Generate N completions per prompt using SGLang — identical to training."""
    url = f"{SGLANG_URL}/generate"
    all_results = []
    
    for p_idx, p in enumerate(prompts):
        messages = [
            {"role": "system", "content": "You are a helpful AI code reviewer."},
            {"role": "user", "content": p}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        payload = {
            "text": formatted,
            "sampling_params": {
                "temperature": 0.8,
                "top_p": 0.95,
                "n": n,
                "max_new_tokens": max_tokens
            },
            "lora_path": lora_name
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp_json = resp.json()
            
            if isinstance(resp_json, list):
                completions = [item.get("text", "") if isinstance(item, dict) else str(item) for item in resp_json]
            elif isinstance(resp_json, dict):
                # Check for error
                if "error" in resp_json or resp_json.get("error_message"):
                    err = resp_json.get("error", resp_json.get("error_message", "unknown"))
                    logger.error(f"SGLang error for prompt {p_idx}: {err}")
                    all_results.append([""] * n)
                    continue
                completions = resp_json.get("text", [])
                if not isinstance(completions, list):
                    completions = [completions]
            else:
                logger.error(f"Unexpected response type: {type(resp_json)}")
                completions = []
            
            # Log first result for debugging
            if p_idx == 0 and completions:
                logger.info(f"Sample completion ({len(completions)} total): {completions[0][:150]}...")
            
            while len(completions) < n:
                completions.append("")
            all_results.append(completions[:n])
            
        except Exception as e:
            logger.error(f"SGLang generation failed for prompt {p_idx}: {e}")
            all_results.append([""] * n)
    
    return all_results


def evaluate_team(team_name: str, test_file: str, lora_path: str,
                  tokenizer, num_votes: int = 8, max_samples: int = 50, max_tokens: int = 256):
    """Evaluate a team using SGLang with proper majority voting per prompt.
    
    Invalid completions (failed to parse) are excluded from the vote count
    instead of being silently counted as FILTER.
    """
    logger.info(f"\n{'='*50}\nEvaluating team: {team_name}\n{'='*50}")
    
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return None, None
    
    import random
    dataset = []
    with open(test_file, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
            
    if not dataset:
        return None, None
        
    eval_rng = random.Random(42)
    dataset = eval_rng.sample(dataset, min(max_samples, len(dataset)))
    
    # Regenerate prompts from current template (same as 03_train_dapo.py)
    from src.data.team_dataset import generate_prompt
    for item in dataset:
        item["prompt"] = generate_prompt(item["diff"], item["comment"], team_name)
    
    # Count ground truth
    n_gt_surface = sum(1 for d in dataset if d["label"] == 1)
    n_gt_filter = sum(1 for d in dataset if d["label"] == 0)
    logger.info(f"Loaded {len(dataset)} test samples (GT: {n_gt_surface} SURFACE / {n_gt_filter} FILTER)")
    
    # Load LoRA into SGLang
    adapter_name = f"eval_{team_name}"
    if not load_lora(adapter_name, lora_path):
        logger.error("LoRA failed to load, aborting evaluation")
        return None, None
    
    # Generate completions in batches
    batch_size = 8
    # Store per-prompt completion groups for majority voting
    prompt_results = []  # list of {gt_label, completions: [{decision, raw}]}
    all_completions = []  # flat list for individual stats
    
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {team_name}"):
        batch = dataset[i:i+batch_size]
        prompts = [item["prompt"] for item in batch]
        
        completions_grouped = generate_completions(
            prompts, lora_name=adapter_name, tokenizer=tokenizer,
            n=num_votes, max_tokens=max_tokens
        )
        
        for b_idx, item in enumerate(batch):
            prompt_comps = []
            for comp in completions_grouped[b_idx]:
                parsed = parse_completion(comp)
                decision = parsed["decision"]  # "SURFACE", "FILTER", or None
                prompt_comps.append({"decision": decision, "raw": comp})
                all_completions.append({
                    "gt": item["label"],
                    "decision": decision,
                    "raw": comp
                })
            prompt_results.append({
                "gt_label": item["label"],
                "completions": prompt_comps
            })
    
    # === Individual completion stats (excludes invalid) ===
    valid_comps = [c for c in all_completions if c["decision"] is not None]
    invalid_comps = [c for c in all_completions if c["decision"] is None]
    n_surface_votes = sum(1 for c in valid_comps if c["decision"] == "SURFACE")
    n_filter_votes = sum(1 for c in valid_comps if c["decision"] == "FILTER")
    n_invalid = len(invalid_comps)
    total_comps = len(all_completions)
    
    print(f"\n  Individual completion stats: {n_surface_votes}S/{n_filter_votes}F/{n_invalid}inv "
          f"({n_surface_votes / max(1, n_surface_votes + n_filter_votes) * 100:.0f}% SURFACE "
          f"across {n_surface_votes + n_filter_votes} valid of {total_comps} total)")
    
    if n_surface_votes + n_filter_votes == 0:
        logger.error("ALL completions were invalid! SGLang generation is broken.")
        return None, None
    
    # === Majority Voting per prompt ===
    correct = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    n_tie = 0
    
    print(f"\n  Per-prompt majority vote ({team_name}, {num_votes} completions per prompt):")
    for pr in prompt_results:
        gt = pr["gt_label"]
        votes = [c["decision"] for c in pr["completions"] if c["decision"] is not None]
        n_s = sum(1 for v in votes if v == "SURFACE")
        n_f = sum(1 for v in votes if v == "FILTER")
        n_valid = n_s + n_f
        
        if n_valid == 0:
            # All completions invalid — skip this prompt
            n_tie += 1
            continue
        
        # Majority vote (ties go to FILTER since it's the conservative choice)
        majority = 1 if n_s > (n_valid / 2) else 0
        
        gt_str = "SURFACE" if gt == 1 else "FILTER"
        pred_str = "SURFACE" if majority == 1 else "FILTER"
        match = "✓" if gt == majority else "✗"
        
        if gt == majority:
            correct += 1
        
        if gt == 1 and majority == 1:
            true_positives += 1
        elif gt == 0 and majority == 1:
            false_positives += 1
        elif gt == 1 and majority == 0:
            false_negatives += 1
        else:
            true_negatives += 1
        
        print(f"    [{match}] GT={gt_str:7s} Majority={pred_str:7s} ({n_s}/{n_valid}S)")
    
    n_voted = correct + (true_positives + false_positives + false_negatives + true_negatives - correct)
    n_voted = len([pr for pr in prompt_results 
                   if any(c["decision"] is not None for c in pr["completions"])])
    
    accuracy = correct / max(1, n_voted)
    
    # Address Rate = TP / (TP + FP) — only among prompts the model chose to SURFACE
    address_rate = (true_positives / max(1, (true_positives + false_positives))) * 100
    
    print(f"\n  Majority Vote Accuracy: {correct}/{n_voted} ({accuracy*100:.0f}%)")
    print(f"  Address Rate: {address_rate:.1f}% (TP={true_positives}, FP={false_positives})")
    if n_tie > 0:
        print(f"  Skipped prompts (all invalid): {n_tie}")
    
    metrics = {
        "team": team_name,
        "accuracy": accuracy,
        "address_rate": address_rate,
        "total_samples": len(dataset),
        "total_completions": total_comps,
        "valid_completions": n_surface_votes + n_filter_votes,
        "invalid_completions": n_invalid,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }
    
    return metrics, prompt_results


def main():
    parser = argparse.ArgumentParser(description="RLCR v2: Evaluate DAPO Models (SGLang)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--teams", nargs="*", default=None)
    parser.add_argument("--sft", action="store_true", help="Evaluate SFT checkpoint instead of DAPO")
    parser.add_argument("--train-data", action="store_true", help="Evaluate on train.jsonl (overfitting check)")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--num-votes", type=int, default=8)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)
    
    dapo_config = config_data["dapo"]
    model_name = dapo_config["model_name"]
    teams_dir = Path("data/teams")
    
    # Discover teams
    all_teams = sorted([d.name for d in teams_dir.iterdir() if d.is_dir()])
    if args.teams:
        teams = [t for t in args.teams if t in all_teams]
    else:
        teams = all_teams
    
    # Load tokenizer
    tokenizer = get_tokenizer(model_name)
    
    # Start SGLang server (kills any stale ones first)
    sglang_process = start_sglang_server(model_name, dapo_config)
    
    try:
        all_metrics = []
        
        for team_name in teams:
            test_file = str(teams_dir / team_name / ("train.jsonl" if args.train_data else "test.jsonl"))
            data_label = "TRAIN" if args.train_data else "TEST"
            
            if args.sft:
                lora_path = str(Path("checkpoints/sft_warmup") / f"sft_warmup_{team_name}")
                label = "SFT"
            else:
                lora_path = str(Path("checkpoints/dapo") / f"dapo_lora_{team_name}")
                label = "DAPO"
            
            if not os.path.exists(lora_path):
                logger.warning(f"{label} checkpoint not found: {lora_path}")
                continue
            
            # Convert to absolute path for SGLang
            lora_path = str(Path(lora_path).resolve())
            logger.info(f"Using {label} LoRA: {lora_path} on {data_label} data")
            
            print(f"\n{'='*70}")
            print(f"🚀 EVALUATING WITH CHECKPOINT:")
            print(f"👉 {lora_path}")
            print(f"{'='*70}\n")
            
            max_tokens = dapo_config.get("max_new_tokens", 256)
            metrics, results = evaluate_team(
                team_name, test_file, lora_path, tokenizer,
                num_votes=args.num_votes, max_samples=args.max_samples,
                max_tokens=max_tokens
            )
            
            if metrics:
                metrics["checkpoint_type"] = label
                all_metrics.append(metrics)
        
        # Summary
        print("\n" + "=" * 60)
        checkpoint_label = "SFT" if args.sft else "DAPO"
        print(f"EVALUATION SUMMARY ({checkpoint_label}, SGLang, {args.num_votes}-vote)")
        print("=" * 70)
        for m in all_metrics:
            print(f"Team: {m['team']:<20} | Acc: {m['accuracy']:.2f} | Address Rate: {m['address_rate']:.2f}")
        print("=" * 70)
    
    finally:
        logger.info("Shutting down SGLang server...")
        sglang_process.terminate()
        sglang_process.wait(timeout=10)


if __name__ == "__main__":
    main()
