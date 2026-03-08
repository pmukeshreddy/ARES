import json
import random
from pathlib import Path
from typing import Dict, List
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 5 Simulated Teams
TEAM_PROFILES = {
    "Pragmatic-Shippers": {
        "context": "We are an early-stage startup team. We only care about catastrophic bugs, logic errors that break the app, or complete architectural failures. We ignore everything else to ship faster.",
        "rm_threshold": 0.85  # Only surface extremely actionable/critical items
    },
    "Thorough-Mentors": {
        "context": "We are an open-source maintainer team. We care about teaching. We surface all suggestions, alternative approaches, design patterns, and edge cases.",
        "rm_threshold": 0.30  # Surface almost everything that has any utility
    },
    "Security-First": {
        "context": "We are a high-security backend team. We care deeply about vulnerabilities, SQL injection, buffer overflows, and input validation. We ignore style nits.",
        "rm_threshold": 0.70  # High bar, but not as extreme as pragmatic
    },
    "Performance-Obsessed": {
        "context": "We are a low-latency trading systems team. We care about O(N) complexity, memory allocations, caching, and CPU cycles. We ignore minor refactoring unless it speeds up code.",
        "rm_threshold": 0.65
    },
    "Style-Sticklers": {
        "context": "We are a frontend foundational UI team. Consistency is god. We care about naming conventions, strict typing, linting rules, formatting, and documentation.",
        "rm_threshold": 0.50  # Average bar
    }
}

# Reverse map: snake_case folder name -> original TEAM_PROFILES key
_TEAM_NAME_LOOKUP = {k.lower().replace("-", "_"): k for k in TEAM_PROFILES}

def generate_prompt(diff: str, comment: str, team_name: str) -> str:
    """Formats the input prompt for the DAPO Qwen2.5-Coder-3B-Instruct model."""
    # Resolve snake_case folder names (e.g. "pragmatic_shippers") to profile keys ("Pragmatic-Shippers")
    canonical = _TEAM_NAME_LOOKUP.get(team_name, team_name)
    context = TEAM_PROFILES[canonical]["context"]
    
    prompt = (
        f"You are an AI code reviewer acting as a filter for a specific engineering team.\n\n"
        f"<team_context>\n{context}\n</team_context>\n\n"
        f"Below is a code diff and a proposed review comment.\n"
        f"<diff>\n{diff[:1500]}\n</diff>\n\n"
        f"<comment>\n{comment}\n</comment>\n\n"
        f"Analyze if this team would care about this comment. "
        f"Output your reasoning inside <think> tags, followed by a probability score between 0.0 and 1.0 in <score> tags, "
        f"and finally your decision (<decision>SURFACE</decision> or <decision>FILTER</decision>).\n"
    )
    return prompt


def simulate_team_datasets(hf_dataset_path: str, output_dir: str, rm_model=None, rm_tokenizer=None):
    """
    Reads the processed HF dataset and distributes samples into 5 team buckets.
    Uses the Phase 1 Reward Model (F1=0.99) to score each comment's actionability,
    then applies team-specific thresholds to generate high-quality ground truth labels.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize buckets for balancing
    team_data_surface = {team: [] for team in TEAM_PROFILES.keys()}
    team_data_filter = {team: [] for team in TEAM_PROFILES.keys()}
    
    logger.info(f"Loading base dataset from {hf_dataset_path}")
    # Load dataset
    with open(hf_dataset_path, "r") as f:
        lines = f.readlines()
        random.shuffle(lines)
        
    logger.info(f"Loaded {len(lines)} total items. Beginning RM scoring and dynamic distribution...")
    
    # We need 125 SURFACE and 125 FILTER samples per team (250 total = 50 train / 200 test)
    TARGET_PER_CLASS = 125
    
    for line in tqdm(lines, desc="Generating Team Labels"):
        # Check if all teams are fully populated
        all_full = True
        for t in TEAM_PROFILES.keys():
            if len(team_data_surface[t]) < TARGET_PER_CLASS or len(team_data_filter[t]) < TARGET_PER_CLASS:
                all_full = False
                break
                
        if all_full:
            logger.info("All team buckets perfectly filled (125 SURFACE / 125 FILTER)! Stopping early.")
            break
            
        data = json.loads(line)
        comment = data.get("comment", "").lower()
        diff = data.get("diff_hunk", "")
        # Real world label (did it get acted on?)
        original_label = data.get("label", 0)
        
        # 5. Generate Phase 1 RM Score
        rm_score = 0.0
        if rm_model is not None and rm_tokenizer is not None:
            import torch
            diff_text = diff[:2048]
            comment_text = comment[:512]
            prompt_text = f"{diff_text} [SEP] {comment_text}"
            inputs = rm_tokenizer(prompt_text, truncation=True, max_length=1024, return_tensors="pt").to(rm_model.backbone.device)
            with torch.no_grad():
                output = rm_model(inputs["input_ids"], inputs["attention_mask"])
                rm_score = torch.sigmoid(output).item()
        else:
            # Fallback to original label if RM not provided
            rm_score = float(original_label)
            
        # 6. Apply Team Thresholds (one RM score powers all teams!)
        for team_name, profile in TEAM_PROFILES.items():
            threshold = profile["rm_threshold"]
            team_label = 1 if rm_score >= threshold else 0
            
            # Only generate and store if the bucket still needs it
            if team_label == 1 and len(team_data_surface[team_name]) < TARGET_PER_CLASS:
                sample = {
                    "prompt": generate_prompt(diff, comment, team_name),
                    "diff": diff,
                    "comment": comment,
                    "label": team_label,
                    "rm_score": round(rm_score, 4),
                    "team": team_name
                }
                team_data_surface[team_name].append(sample)
                
            elif team_label == 0 and len(team_data_filter[team_name]) < TARGET_PER_CLASS:
                sample = {
                    "prompt": generate_prompt(diff, comment, team_name),
                    "diff": diff,
                    "comment": comment,
                    "label": team_label,
                    "rm_score": round(rm_score, 4),
                    "team": team_name
                }
                team_data_filter[team_name].append(sample)
    
    # Save datasets
    # The requirement is 20-50 train samples, 200+ test samples per team
    for team_name in TEAM_PROFILES.keys():
        surfaces = team_data_surface[team_name]
        filters = team_data_filter[team_name]
        
        # Balance 50/50
        min_len = min(len(surfaces), len(filters))
        if min_len < 125: # Need 250 total (125 each) for 50 train / 200 test
            logger.warning(f"Team {team_name} only got {min_len*2} balanced samples. Simulation might be weak.")
            
        random.shuffle(surfaces)
        random.shuffle(filters)
        
        # Take equal amounts
        samples = surfaces[:min_len] + filters[:min_len]
        
        # Shuffle matched dataset
        random.shuffle(samples)
        
        # Take 50 for train, 200 for test (or whatever is available)
        train_samples = samples[:50]
        test_samples = samples[50:250] if len(samples) >= 250 else samples[50:]
        
        team_dir = out_dir / team_name.lower().replace("-", "_")
        team_dir.mkdir(exist_ok=True)
        
        with open(team_dir / "train.jsonl", "w") as f:
            for s in train_samples:
                f.write(json.dumps(s) + "\n")
                
        with open(team_dir / "test.jsonl", "w") as f:
            for s in test_samples:
                f.write(json.dumps(s) + "\n")
                
        logger.info(f"Saved {team_name}: {len(train_samples)} Train, {len(test_samples)} Test")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    simulate_team_datasets("data/processed/train_small.jsonl", "data/teams/")
