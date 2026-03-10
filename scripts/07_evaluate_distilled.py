"""
Evaluate the Distilled 3B Student Model.

Boots SGLang with the 3B base model, loads the distilled LoRA, and runs
the same accuracy + address rate metrics as 04_evaluate_dapo.py.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import time
from pathlib import Path
from tqdm import tqdm

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.training.rewards import parse_completion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SGLANG_PORT = 30000
SGLANG_URL = f"http://localhost:{SGLANG_PORT}"


def kill_existing_sglang():
    logger.info("Killing any existing SGLang processes...")
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
    subprocess.run(f"fuser -k {SGLANG_PORT}/tcp", shell=True, capture_output=True)
    time.sleep(5)

    try:
        resp = requests.get(f"{SGLANG_URL}/health", timeout=2)
        if resp.status_code == 200:
            logger.warning("Server STILL running after kill! Trying again...")
            subprocess.run(["pkill", "-9", "-f", "sglang"], capture_output=True)
            time.sleep(5)
    except Exception:
        logger.info("Port is free.")


def start_sglang_server(model_name: str, lora_path: str):
    """Boot the 3B student model in SGLang with LoRA support."""
    kill_existing_sglang()

    logger.info(f"Starting SGLang server for {model_name} on port {SGLANG_PORT}...")

    sglang_python = "/opt/sglang_venv/bin/python3"

    cmd = [
        sglang_python, "-m", "sglang.launch_server",
        "--model-path", model_name,
        "--port", str(SGLANG_PORT),
        "--trust-remote-code",
        "--enable-lora",
        "--max-lora-rank", "16",
        "--max-loras-per-batch", "1",
        "--lora-target-modules", "q_proj", "k_proj", "v_proj", "o_proj",
        "--mem-fraction-static", "0.5",
        "--dtype", "bfloat16"
    ]

    log_file = open("sglang_distilled_eval.log", "w")
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
        with open("sglang_distilled_eval.log") as f:
            logger.error(f"Log tail:\n{f.read()[-2000:]}")
    except Exception:
        pass
    process.terminate()
    sys.exit(1)


def load_lora(adapter_name: str, lora_path: str):
    url = f"{SGLANG_URL}/load_lora_adapter"
    payload = {"lora_name": adapter_name, "lora_path": lora_path}
    logger.info(f"Loading LoRA '{adapter_name}' from {lora_path}")
    resp = requests.post(url, json=payload)
    logger.info(f"LoRA response [{resp.status_code}]: {resp.text[:500]}")
    if resp.status_code != 200:
        logger.error("LoRA load FAILED!")
        return False
    return True


def generate_completions(prompts: list, lora_name: str, tokenizer, n: int = 8, max_tokens: int = 512):
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
            resp = requests.post(url, json=payload, timeout=300)
            resp_json = resp.json()

            if isinstance(resp_json, list):
                completions = [item.get("text", "") if isinstance(item, dict) else str(item) for item in resp_json]
            elif isinstance(resp_json, dict):
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

            if p_idx == 0 and completions:
                logger.info(f"Sample completion ({len(completions)} total): {completions[0][:150]}...")

            while len(completions) < n:
                completions.append("")
            all_results.append(completions[:n])

        except Exception as e:
            logger.error(f"SGLang generation failed for prompt {p_idx}: {e}")
            all_results.append([""] * n)

    return all_results


def evaluate(test_file: str, lora_path: str, tokenizer, num_votes: int = 8, max_samples: int = 50):
    """Evaluate the distilled model. Same metrics as 04_evaluate_dapo.py."""
    logger.info(f"\n{'='*50}\nEvaluating Distilled 3B Student\n{'='*50}")

    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return None

    dataset = []
    with open(test_file, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    dataset = dataset[:max_samples]

    if not dataset:
        return None

    n_gt_surface = sum(1 for d in dataset if d["label"] == 1)
    n_gt_filter = sum(1 for d in dataset if d["label"] == 0)
    logger.info(f"Loaded {len(dataset)} test samples (GT: {n_gt_surface} SURFACE / {n_gt_filter} FILTER)")

    adapter_name = "distilled_student"
    if not load_lora(adapter_name, lora_path):
        logger.error("LoRA failed to load, aborting evaluation")
        return None

    batch_size = 8
    results = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating distilled model"):
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
                    "ground_truth_label": item["label"],
                    "predicted_label": pred,
                    "raw_completion": comp,
                })

    # Individual completion stats
    total_surface_votes = sum(1 for r in results if r["predicted_label"] == 1)
    total_filter_votes = sum(1 for r in results if r["predicted_label"] == 0)
    total_votes = total_surface_votes + total_filter_votes
    completion_surface_pct = total_surface_votes / max(1, total_votes) * 100
    print(f"\n  Individual completion stats: {total_surface_votes}S/{total_filter_votes}F ({completion_surface_pct:.0f}% SURFACE across all {total_votes} completions)")

    if total_votes == 0:
        logger.error("ALL completions were empty! SGLang generation is broken.")
        logger.error("Check sglang_distilled_eval.log for errors.")
        return None

    # Metrics — identical to 04_evaluate_dapo.py
    correct = sum(1 for r in results if r["ground_truth_label"] == r["predicted_label"])
    accuracy = correct / len(results)

    true_positives = sum(1 for r in results if r["ground_truth_label"] == 1 and r["predicted_label"] == 1)
    false_positives = sum(1 for r in results if r["ground_truth_label"] == 0 and r["predicted_label"] == 1)
    false_negatives = sum(1 for r in results if r["ground_truth_label"] == 1 and r["predicted_label"] == 0)

    # Address Rate = TP / (TP + FP) × 100
    address_rate = (true_positives / max(1, (true_positives + false_positives))) * 100

    # Per-completion diagnostics
    print(f"\n  Per-completion results ({num_votes} completions per prompt):")
    for r in results[:20]:
        gt = "SURFACE" if r["ground_truth_label"] == 1 else "FILTER"
        pred = "SURFACE" if r["predicted_label"] == 1 else "FILTER"
        match = "✓" if gt == pred else "✗"
        snippet = r["raw_completion"][:100].replace('\n', ' ')
        print(f"    [{match}] GT={gt:7s} Pred={pred:7s} | {snippet}...")

    metrics = {
        "accuracy": accuracy,
        "address_rate": address_rate,
        "total_samples": len(dataset),
        "total_completions": len(results),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate distilled 3B student model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct", help="Student base model")
    parser.add_argument("--lora-path", default="checkpoints/distilled_student_3B", help="Path to distilled LoRA")
    parser.add_argument("--test-data", default="data/teams/pragmatic_shippers/test.jsonl", help="Test data file")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--num-votes", type=int, default=8)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Resolve LoRA path
    lora_path = str(Path(args.lora_path).resolve())
    if not os.path.exists(lora_path):
        logger.error(f"Distilled LoRA not found at {lora_path}")
        sys.exit(1)

    sglang_process = start_sglang_server(args.model, lora_path)

    try:
        test_file = str(PROJECT_ROOT / args.test_data)
        metrics = evaluate(test_file, lora_path, tokenizer,
                           num_votes=args.num_votes, max_samples=args.max_samples)

        if metrics:
            print("\n" + "=" * 60)
            print("DISTILLED 3B STUDENT EVALUATION RESULTS")
            print("=" * 60)
            print(f"  Accuracy:     {metrics['accuracy']:.4f}")
            print(f"  Address Rate: {metrics['address_rate']:.2f}%")
            print(f"  Samples:      {metrics['total_samples']}")
            print(f"  Completions:  {metrics['total_completions']}")
            print(f"  TP: {metrics['true_positives']}  FP: {metrics['false_positives']}  FN: {metrics['false_negatives']}")
            print("=" * 60)
        else:
            logger.error("Evaluation failed.")

    finally:
        logger.info("Shutting down SGLang server...")
        sglang_process.terminate()
        sglang_process.wait(timeout=10)


if __name__ == "__main__":
    main()
