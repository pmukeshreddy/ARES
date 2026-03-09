"""
Greptile-Style Baseline Evaluation (RAG / Vector Database approach).

This establishes the industry-standard baseline:
1. Generate a raw code review comment using a general instruction-tuned model (Qwen 14B).
2. Embed the generated comment using `sentence-transformers`.
3. Compare the embedding's cosine similarity against the Team's historical accepted (SURFACE) and ignored (FILTER) comments from their specific `train.jsonl` vector database.
4. If the comment is mathematically closer to the 'FILTER' cluster, drop it silently. Otherwise, SURFACE it.
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

import yaml
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.rewards import parse_completion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SGLANG_PORT = 30005  # Different port to avoid conflicts
SGLANG_URL = f"http://localhost:{SGLANG_PORT}"

def kill_existing_sglang():
    """Kill any existing SGLang server on our port."""
    logger.info("Killing any existing SGLang processes on our port...")
    subprocess.run(f"fuser -k {SGLANG_PORT}/tcp", shell=True, capture_output=True)
    time.sleep(3)

def start_sglang_server(model_name: str):
    """Starts SGLang server for the raw baseline model (no LoRA)."""
    kill_existing_sglang()
    
    logger.info(f"Starting Base Model SGLang server for {model_name} on port {SGLANG_PORT}...")
    sglang_python = "/opt/sglang_venv/bin/python3"
    
    cmd = [
        sglang_python, "-m", "sglang.launch_server",
        "--model-path", model_name,
        "--port", str(SGLANG_PORT),
        "--trust-remote-code",
        # 40% memory so we have room for the SentenceTransformer encoding
        "--mem-fraction-static", "0.4",
        "--dtype", "bfloat16"
    ]
    
    log_file = open("sglang_greptile.log", "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, text=True)
    
    logger.info("Waiting for SGLang to boot (up to 120s)...")
    for _ in range(60):
        try:
            resp = requests.get(f"{SGLANG_URL}/health", timeout=2)
            if resp.status_code == 200:
                logger.info("SGLang server is fully booted and ready!")
                return process
        except Exception:
            pass
        time.sleep(2)
    
    logger.error("SGLang failed to start within 120s.")
    process.terminate()
    sys.exit(1)

def generate_base_completion(prompts: list, tokenizer, max_tokens: int = 512):
    """Generate 1 standard completion per prompt using SGLang (greedy decoding)."""
    url = f"{SGLANG_URL}/generate"
    completions = []
    
    for p_idx, p in enumerate(prompts):
        messages = [
            {"role": "system", "content": "You are a helpful AI code reviewer. Please review the following code diff and provide a constructive comment."},
            {"role": "user", "content": p}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        payload = {
            "text": formatted,
            "sampling_params": {
                "temperature": 0.0, # Greedy generation for the standard "Greptile" approach
                "max_new_tokens": max_tokens
            }
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp_json = resp.json()
            
            text = resp_json.get("text", "") if isinstance(resp_json, dict) else ""
            completions.append(text)
            
        except Exception as e:
            logger.error(f"SGLang generation failed for prompt {p_idx}: {e}")
            completions.append("")
            
    return completions

def compute_team_embeddings(team_name: str, embedder, config_data):
    """Builds the Vector Database (average centroids) for the team's historical SURFACE and FILTER comments."""
    train_file = PROJECT_ROOT / "data" / "teams" / team_name / "train.jsonl"
    if not train_file.exists():
        logger.error(f"Training data (Vector DB) not found for team {team_name}")
        return None, None
        
    surface_comments = []
    filter_comments = []
    
    with open(train_file, "r") as f:
        for line in f:
            item = json.loads(line)
            if item["label"] == 1:
                surface_comments.append(item["comment"])
            else:
                filter_comments.append(item["comment"])
                
    logger.info(f"Team '{team_name}' Vector Database: {len(surface_comments)} SURFACE comments, {len(filter_comments)} FILTER comments.")
    
    if not surface_comments or not filter_comments:
        return None, None
        
    # Create the Vector Database
    logger.info("Computing embeddings for historical Vector DB...")
    with torch.no_grad():
        surface_embeddings = embedder.encode(surface_comments, convert_to_tensor=True)
        filter_embeddings = embedder.encode(filter_comments, convert_to_tensor=True)
        
        # We compute the 'Centroid' (average vector location) of the two clusters
        # A real vector DB does KNN on individual points, but cosine to centroid is mathematically identical for scaled categorization
        surface_centroid = surface_embeddings.mean(dim=0)
        filter_centroid = filter_embeddings.mean(dim=0)
        
        # Normalize centroids for cosine similarity
        surface_centroid = F.normalize(surface_centroid, p=2, dim=0)
        filter_centroid = F.normalize(filter_centroid, p=2, dim=0)
        
    return surface_centroid, filter_centroid

def evaluate_greptile_baseline(team_name: str, test_file: str, tokenizer, embedder, surface_centroid, filter_centroid, max_samples: int = 50):
    """Runs the RAG similarity search evaluation."""
    logger.info(f"\n{'='*50}\nEvaluating Greptile Architecture for team: {team_name}\n{'='*50}")
    
    dataset = []
    with open(test_file, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    dataset = dataset[:max_samples]
    
    n_gt_surface = sum(1 for d in dataset if d["label"] == 1)
    n_gt_filter = sum(1 for d in dataset if d["label"] == 0)
    logger.info(f"Loaded {len(dataset)} test samples (GT: {n_gt_surface} SURFACE / {n_gt_filter} FILTER)")
    
    results = []
    batch_size = 4
    
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating Greptile for {team_name}"):
        batch = dataset[i:i+batch_size]
        prompts = [item["prompt"] for item in batch]
        
        # 1. Base Model Generation (No LoRA knowledge of the team)
        generated_comments = generate_base_completion(prompts, tokenizer)
        
        # 2. Embedding similarity matching
        with torch.no_grad():
            gen_embeddings = embedder.encode(generated_comments, convert_to_tensor=True)
            gen_embeddings = F.normalize(gen_embeddings, p=2, dim=1)
            
            # 3. Compute Cosine Similarity to both clusters
            sim_to_surface = torch.matmul(gen_embeddings, surface_centroid)
            sim_to_filter = torch.matmul(gen_embeddings, filter_centroid)
            
        for b_idx, item in enumerate(batch):
            sim_S = sim_to_surface[b_idx].item()
            sim_F = sim_to_filter[b_idx].item()
            
            # The architectural threshold logic:
            # If the generated comment is mathematically closer in semantic 
            # space to historically ignored comments, silently filter it.
            if sim_S > sim_F:
                pred = 1
                decision = "SURFACE"
            else:
                pred = 0
                decision = "FILTER"
                
            results.append({
                "team": team_name,
                "ground_truth_label": item["label"],
                "predicted_label": pred,
                "sim_S": sim_S,
                "sim_F": sim_F,
                "raw_completion": generated_comments[b_idx],
            })
            
    # Metrics Calculation
    correct = sum(1 for r in results if r["ground_truth_label"] == r["predicted_label"])
    accuracy = correct / len(results)
    
    true_positives = sum(1 for r in results if r["ground_truth_label"] == 1 and r["predicted_label"] == 1)
    false_positives = sum(1 for r in results if r["ground_truth_label"] == 0 and r["predicted_label"] == 1)
    
    address_rate = (true_positives / max(1, (true_positives + false_positives))) * 100
    
    # Raw stats
    total_surface_votes = sum(1 for r in results if r["predicted_label"] == 1)
    total_filter_votes = sum(1 for r in results if r["predicted_label"] == 0)
    
    print(f"\n  Individual completion stats: {total_surface_votes}S/{total_filter_votes}F")
    print(f"\n  Greptile RAG Vector Search Trace for {team_name}:")
    
    for r in results[:20]:
        gt = "SURFACE" if r["ground_truth_label"] == 1 else "FILTER"
        pred = "SURFACE" if r["predicted_label"] == 1 else "FILTER"
        match = "✓" if gt == pred else "✗"
        # Extract just the first line or meaningful snippet of the raw generation
        snippet = r["raw_completion"].split('\n')[0][:80]
        
        # Trace the mathematical vector decision
        vector_logic = f"[S:{r['sim_S']:.2f} | F:{r['sim_F']:.2f}]"
        
        print(f"    [{match}] GT={gt:7s} | Vector Math={vector_logic} -> Decision: {pred:7s} | {snippet}...")
        
    metrics = {
        "team": team_name,
        "accuracy": accuracy,
        "address_rate": address_rate,
        "total_samples": len(dataset)
    }
    
    return metrics, results

def main():
    parser = argparse.ArgumentParser(description="RLCR v2: Greptile RAG/Vector Embedding Baseline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--teams", nargs="*", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)
        
    model_name = config_data["dapo"]["model_name"]
    embedder_name = config_data["embeddings"]["model_name"]
    
    teams_dir = PROJECT_ROOT / "data" / "teams"
    all_teams = sorted([d.name for d in teams_dir.iterdir() if d.is_dir()])
    teams = [t for t in args.teams if t in all_teams] if args.teams else all_teams
    
    # Initialize Embedder (the "Greptile Vector Database" engine)
    logger.info(f"Loading SentenceTransformer for Vector Search: {embedder_name}")
    embedder = SentenceTransformer(embedder_name, device=args.device)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    sglang_process = start_sglang_server(model_name)
    
    try:
        all_metrics = []
        for team_name in teams:
            test_file = str(teams_dir / team_name / "test.jsonl")
            
            # Pre-compute internal clustering centroids acting as the Vector DB historical trace
            surface_c, filter_c = compute_team_embeddings(team_name, embedder, config_data)
            
            if surface_c is None:
                continue
                
            metrics, _ = evaluate_greptile_baseline(
                team_name, test_file, tokenizer, embedder, 
                surface_c, filter_c
            )
            
            if metrics:
                all_metrics.append(metrics)
                
        # Print Summary Panel identically to DAPO eval
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY (Greptile RAG Baseline)")
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
