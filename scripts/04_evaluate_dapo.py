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
    """Kill any existing SGLang server on our port."""
    try:
        # Check if anything is running on the port
        resp = requests.get(f"{SGLANG_URL}/health", timeout=2)
        if resp.status_code == 200:
            logger.info("Found existing SGLang server, killing it...")
            subprocess.run(["pkill", "-f", "sglang.launch_server"], capture_output=True)
            time.sleep(3)
    except Exception:
        pass  # No server running, that's fine


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
    """Load a LoRA adapter into SGLang and verify it loaded."""
    url = f"{SGLANG_URL}/load_lora"
    payload = {"lora_name": adapter_name, "lora_path": lora_path}
    resp = requests.post(url, json=payload)
    resp_data = resp.json()
    success = resp_data.get("success", False)
    logger.info(f"LoRA load '{adapter_name}' -> success={success}, loaded={list(resp_data.get('loaded_adapters', {}).keys())}")
    if not success:
        logger.error(f"LoRA load FAILED: {resp_data.get('error_message', 'unknown')}")
    return success


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
            votes_surface = 0
            votes_filter = 0
            first_completion = None
            
            for comp in completions_grouped[b_idx]:
                parsed = parse_completion(comp)
                if first_completion is None:
                    first_completion = comp
                if parsed["decision"] == "SURFACE":
                    votes_surface += 1
                elif parsed["decision"] == "FILTER":
                    votes_filter += 1
            
            prediction = 1 if votes_surface > votes_filter else 0
            
            results.append({
                "team": team_name,
                "ground_truth_label": item["label"],
                "predicted_label": prediction,
                "raw_completion": first_completion or "",
                "votes_surface": votes_surface,
                "votes_filter": votes_filter
            })
    
    # Check if generation actually worked
    total_votes = sum(r["votes_surface"] + r["votes_filter"] for r in results)
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
    
    precision = true_positives / max(1, (true_positives + false_positives))
    recall = true_positives / max(1, (true_positives + false_negatives))
    f1 = 2 * (precision * recall) / max(1e-6, (precision + recall))
    surface_ratio = sum(1 for r in results if r["predicted_label"] == 1) / len(results)
    
    # Per-sample diagnostics
    print(f"\n  Per-sample results ({team_name}, {num_votes}-vote majority):")
    for r in results[:20]:
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
    
    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="RLCR v2: Evaluate DAPO Models (SGLang)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--teams", nargs="*", default=None)
    parser.add_argument("--sft", action="store_true", help="Evaluate SFT checkpoint instead of DAPO")
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
            test_file = str(teams_dir / team_name / "test.jsonl")
            
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
            logger.info(f"Using {label} LoRA: {lora_path}")
            
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
        print("=" * 60)
        for m in all_metrics:
            print(f"Team: {m['team']:<20} | Acc: {m['accuracy']:.2f} | P: {m['precision']:.2f} | R: {m['recall']:.2f} | F1: {m['f1']:.2f} | Surface%: {m['surface_ratio']:.2f}")
        print("=" * 60)
    
    finally:
        logger.info("Shutting down SGLang server...")
        sglang_process.terminate()
        sglang_process.wait(timeout=10)


if __name__ == "__main__":
    main()
