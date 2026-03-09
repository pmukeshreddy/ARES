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
    
    sglang_python = "/opt/sglang_venv/bin/python3"
    
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
        "--mem-fraction-static", "0.7",
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


def generate_completions(prompts: list, lora_name: str, tokenizer, n: int = 8, max_tokens: int = 512):
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
                "temperature": 1.0,
                "top_p": 0.95,
                "n": n,
                "max_new_tokens": max_tokens
            },
            "lora_path": lora_name
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=60)
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
                  tokenizer, num_votes: int = 8, max_samples: int = 50):
    """Evaluate a team using SGLang with majority voting."""
    logger.info(f"\n{'='*50}\nEvaluating team: {team_name}\n{'='*50}")
    
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return None, None
    
    dataset = []
    with open(test_file, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    dataset = dataset[:max_samples]
    
    if not dataset:
        return None, None
    
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
    results = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {team_name}"):
        batch = dataset[i:i+batch_size]
        prompts = [item["prompt"] for item in batch]
        
        completions_grouped = generate_completions(
            prompts, lora_name=adapter_name, tokenizer=tokenizer,
            n=num_votes
        )
        
        for b_idx, item in enumerate(batch):
            for comp in completions_grouped[b_idx]:
                parsed = parse_completion(comp)
                pred = 1 if parsed["decision"] == "SURFACE" else 0
                
                results.append({
                    "team": team_name,
                    "ground_truth_label": item["label"],
                    "predicted_label": pred,
                    "raw_completion": comp,
                })
    
    # Individual completion stats (matches training S:F reporting)
    total_surface_votes = sum(1 for r in results if r["predicted_label"] == 1)
    total_filter_votes = sum(1 for r in results if r["predicted_label"] == 0)
    total_votes = total_surface_votes + total_filter_votes
    completion_surface_pct = total_surface_votes / max(1, total_votes) * 100
    print(f"\n  Individual completion stats: {total_surface_votes}S/{total_filter_votes}F ({completion_surface_pct:.0f}% SURFACE across all {total_votes} completions)")
    
    # Check if generation actually worked
    if total_votes == 0:
        logger.error("ALL completions were empty! SGLang generation is broken.")
        logger.error("Check sglang_eval.log for errors.")
        return None, None
    
    # Metrics
    correct = sum(1 for r in results if r["ground_truth_label"] == r["predicted_label"])
    accuracy = correct / len(results)
    
    true_positives = sum(1 for r in results if r["ground_truth_label"] == 1 and r["predicted_label"] == 1)
    false_positives = sum(1 for r in results if r["ground_truth_label"] == 0 and r["predicted_label"] == 1)
    false_negatives = sum(1 for r in results if r["ground_truth_label"] == 1 and r["predicted_label"] == 0)
    
    # Address Rate = (Comments devs acted on) / (Total comments Greptile posted) × 100
    # Comments devs acted on = true_positives (GT=1, Pred=1)
    # Total comments Greptile posted = true_positives + false_positives (Predictions=1)
    address_rate = (true_positives / max(1, (true_positives + false_positives))) * 100
    
    # Per-sample diagnostics (showing individual completions now)
    print(f"\n  Per-completion results ({team_name}, {num_votes} completions per prompt):")
    for r in results[:20]:
        gt = "SURFACE" if r["ground_truth_label"] == 1 else "FILTER"
        pred = "SURFACE" if r["predicted_label"] == 1 else "FILTER"
        match = "✓" if gt == pred else "✗"
        snippet = r["raw_completion"][:100].replace('\n', ' ')
        print(f"    [{match}] GT={gt:7s} Pred={pred:7s} | {snippet}...")
    
    metrics = {
        "team": team_name,
        "accuracy": accuracy,
        "address_rate": address_rate,
        "total_samples": len(dataset),
        "total_completions": len(results)
    }
    
    return metrics, results


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
            
            metrics, results = evaluate_team(
                team_name, test_file, lora_path, tokenizer,
                num_votes=args.num_votes, max_samples=args.max_samples
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
    print(f"TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")
