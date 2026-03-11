"""
Greptile-Style Baseline Evaluation (RAG / Vector Database approach).

This establishes the industry-standard baseline:
1. Generate a raw code review comment using a general instruction-tuned model.
2. Embed the generated comment using `sentence-transformers`.
3. Compare the embedding's cosine similarity against the Team's historical accepted (SURFACE) and ignored (FILTER) comments from the Vector DB.
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
from transformers import AutoTokenizer

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

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
    sglang_python = sys.executable
    
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

def load_team_data(file_path: Path):
    """Loads datapoints from a team-specific JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

def compute_team_embeddings(team_name: str, train_file: Path, embedder):
    """Builds the KNN Vector Database from team's historical comments."""
    if not train_file.exists():
        logger.error(f"Training data not found at {train_file}")
        return None, None
        
    team_data = load_team_data(train_file)
    
    comments = [item["comment"] for item in team_data]
    labels = [item["label"] for item in team_data]
                
    n_surface = sum(1 for l in labels if l == 1)
    n_filter = sum(1 for l in labels if l == 0)
    logger.info(f"Team '{team_name}' Vector Database: {n_surface} SURFACE, {n_filter} FILTER individual comment embeddings.")
    
    if not comments:
        logger.warning(f"No training data found for team '{team_name}'. Skipping.")
        return None, None
        
    logger.info("Computing individual embeddings for KNN Vector DB...")
    with torch.no_grad():
        embeddings = embedder.encode(comments, convert_to_tensor=True)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return embeddings, labels_tensor

def evaluate_greptile_baseline(team_name: str, test_file: Path, embedder, db_embeddings, db_labels, max_samples: int = 250, k: int = 3):
    """Runs KNN k=3 evaluation using existing comments from the test set (same data as DAPO eval)."""
    logger.info(f"\n{'='*50}\nEvaluating Greptile Architecture for team: {team_name}\n{'='*50}")
    
    dataset = load_team_data(test_file)
    dataset = dataset[:max_samples]
    
    if not dataset:
         logger.warning(f"No testing data found for team '{team_name}'. Skipping.")
         return None, None
    
    n_gt_surface = sum(1 for d in dataset if d["label"] == 1)
    n_gt_filter = sum(1 for d in dataset if d["label"] == 0)
    logger.info(f"Loaded {len(dataset)} test samples (GT: {n_gt_surface} SURFACE / {n_gt_filter} FILTER)")
    
    results = []
    
    # Use existing comments from the dataset (same comments DAPO evaluates)
    comments = [item["comment"] for item in dataset]
    
    # Embed the test comments for querying the Vector DB
    logger.info("Embedding test comments for KNN lookup...")
    with torch.no_grad():
        test_embeddings = embedder.encode(comments, convert_to_tensor=True)
        test_embeddings = F.normalize(test_embeddings, p=2, dim=1)
        
        # Move db tensors to same device as test embeddings
        db_embeddings = db_embeddings.to(test_embeddings.device)
        db_labels = db_labels.to(test_embeddings.device)
        
        # Compute cosine similarity matrix: (n_test, n_db)
        sim_matrix = torch.matmul(test_embeddings, db_embeddings.T)
    
    for t_idx, item in enumerate(tqdm(dataset, desc=f"KNN lookup for {team_name}")):
        # Get top-k nearest neighbors from the historical DB
        sims = sim_matrix[t_idx]  # shape: (n_db,)
        topk_indices = torch.topk(sims, k=min(k, len(db_labels))).indices
        topk_labels = db_labels[topk_indices]
        
        n_filter_neighbors = (topk_labels == 0).sum().item()
        n_surface_neighbors = (topk_labels == 1).sum().item()
        
        # Only suppress if majority of k neighbors are FILTER (Greptile pass-through default = SURFACE)
        if n_filter_neighbors >= n_surface_neighbors:
            pred = 0
            decision = "FILTER"
        else:
            pred = 1
            decision = "SURFACE"  # Default: pass through
        
        top_sim = sims[topk_indices[0]].item()
        
        snippet = comments[t_idx].split('\n')[0][:80] if comments[t_idx] else "<EMPTY>"
        
        results.append({
            "team": team_name,
            "ground_truth_label": item["label"],
            "predicted_label": pred,
            "top_sim": top_sim,
            "knn_votes": f"{n_surface_neighbors}S/{n_filter_neighbors}F",
            "comment_snippet": snippet,
        })
            
    # Metrics Calculation
    correct = sum(1 for r in results if r["ground_truth_label"] == r["predicted_label"])
    accuracy = correct / max(1, len(results))
    
    true_positives = sum(1 for r in results if r["ground_truth_label"] == 1 and r["predicted_label"] == 1)
    false_positives = sum(1 for r in results if r["ground_truth_label"] == 0 and r["predicted_label"] == 1)
    
    address_rate = (true_positives / max(1, (true_positives + false_positives))) * 100
    
    # Raw stats
    total_surface_votes = sum(1 for r in results if r["predicted_label"] == 1)
    total_filter_votes = sum(1 for r in results if r["predicted_label"] == 0)
    
    print(f"\n  Individual completion stats: {total_surface_votes}S/{total_filter_votes}F")
    print(f"  Address Rate: {address_rate:.1f}% (TP={true_positives}, FP={false_positives})")
    print(f"  Accuracy: {correct}/{len(results)} ({accuracy*100:.0f}%)")
    print(f"\n  Greptile RAG Vector Search Trace for {team_name}:")
    
    for r in results[:30]:
        gt = "SURFACE" if r["ground_truth_label"] == 1 else "FILTER"
        pred = "SURFACE" if r["predicted_label"] == 1 else "FILTER"
        match = "✓" if gt == pred else "✗"
        print(f"    [{match}] GT={gt:7s} | KNN={r['knn_votes']:7s} | TopSim={r['top_sim']:.2f} -> Decision: {pred:7s} | {r['comment_snippet']}...")
        
    metrics = {
        "team": team_name,
        "accuracy": accuracy,
        "address_rate": address_rate,
        "total_samples": len(dataset)
    }
    
    return metrics, results

def main():
    parser = argparse.ArgumentParser(description="RLCR v2: Greptile KNN Baseline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--teams", nargs="*", default=None, help="Specific teams to evaluate (e.g., pragmatic_shippers)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=int, default=3, help="KNN k neighbors")
    parser.add_argument("--max-samples", type=int, default=250, help="Max test samples")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)
        
    embedder_name = config_data["embeddings"]["model_name"]
    
    # Use team-specific data directories (same as DAPO eval)
    teams_dir = Path("data/teams")
    
    # Discover teams
    all_teams = sorted([d.name for d in teams_dir.iterdir() if d.is_dir()])
    if args.teams:
        teams = [t for t in args.teams if t in all_teams]
    else:
        teams = all_teams
    
    if not teams:
        logger.error(f"No matching teams found. Available teams: {all_teams}")
        sys.exit(1)
        
    logger.info(f"Will evaluate teams: {teams}")
    
    # Initialize Embedder (the "Greptile Vector Database" engine)
    logger.info(f"Loading SentenceTransformer for KNN Vector Search: {embedder_name}")
    embedder = SentenceTransformer(embedder_name, device=args.device)
    
    all_metrics = []
    
    for team_name in teams:
        train_file = teams_dir / team_name / "train.jsonl"
        test_file = teams_dir / team_name / "test.jsonl"
        
        if not test_file.exists():
            logger.warning(f"Test file not found: {test_file}")
            continue
        
        # Pre-compute individual embeddings for KNN Vector DB from team train data
        db_embeddings, db_labels = compute_team_embeddings(team_name, train_file, embedder)
        
        if db_embeddings is None:
            continue
            
        metrics, _ = evaluate_greptile_baseline(
            team_name, test_file, embedder,
            db_embeddings=db_embeddings, db_labels=db_labels, k=args.k,
            max_samples=args.max_samples
        )
        
        if metrics:
            all_metrics.append(metrics)
            
    # Print Summary Panel identically to DAPO eval
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (Greptile KNN k=3 Baseline)")
    print("=" * 70)
    for m in all_metrics:
        print(f"Team: {m['team']:<20} | Acc: {m['accuracy']:.4f} | Address Rate: {m['address_rate']:.2f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
