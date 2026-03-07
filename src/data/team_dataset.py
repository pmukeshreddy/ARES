import json
import random
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# 5 Simulated Teams
TEAM_PROFILES = {
    "Security-First": {
        "context": "We are a high-security backend team. We care deeply about vulnerabilities, SQL injection, buffer overflows, and input validation. We ignore style nits.",
        "keywords": ["security", "vulnerability", "sql", "inject", "auth", "validation", "buffer", "overflow", "leak", "secret", "password", "token", "crypto", "safe", "exploit"]
    },
    "Performance-Obsessed": {
        "context": "We are a low-latency trading systems team. We care about O(N) complexity, memory allocations, caching, and CPU cycles. We ignore minor refactoring unless it speeds up code.",
        "keywords": ["performance", "speed", "latency", "memory", "allocation", "cache", "o(n)", "complexity", "slow", "fast", "optimize", "loop", "bottleneck", "thread"]
    },
    "Style-Sticklers": {
        "context": "We are a frontend foundational UI team. Consistency is god. We care about naming conventions, strict typing, linting rules, formatting, and documentation.",
        "keywords": ["style", "format", "name", "camelcase", "lint", "type", "docstring", "comment", "document", "readability", "pep8", "consistent", "convention", "clear"]
    },
    "Pragmatic-Shippers": {
        "context": "We are an early-stage startup team. We only care about catastrophic bugs, logic errors that break the app, or complete architectural failures. We ignore everything else to ship faster.",
        "keywords": ["bug", "error", "crash", "break", "fail", "null", "exception", "undefined", "logic", "wrong", "fix", "issue"]
    },
    "Thorough-Mentors": {
        "context": "We are an open-source maintainer team. We care about teaching. We surface surface all suggestions, alternative approaches, design patterns, and edge cases.",
        "keywords": ["suggest", "alternative", "pattern", "design", "edge case", "corner case", "better", "refactor", "abstract", "extract", "cleaner", "solid", "dry"]
    }
}

def generate_prompt(diff: str, comment: str, team_name: str) -> str:
    """Formats the input prompt for the DAPO Qwen2.5-Coder-3B-Instruct model."""
    context = TEAM_PROFILES[team_name]["context"]
    
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


def simulate_team_datasets(hf_dataset_path: str, output_dir: str):
    """
    Reads the processed HF dataset and distributes samples into 5 team buckets
    based on keyword matching to simulate different team preferences.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize buckets
    team_data = {team: [] for team in TEAM_PROFILES.keys()}
    
    logger.info(f"Loading base dataset from {hf_dataset_path}")
    # Load dataset
    with open(hf_dataset_path, "r") as f:
        # We only need string manipulation, load up to 50k to be fast
        lines = f.readlines()
        random.shuffle(lines)
    
    logger.info("Distributing samples to simulated teams based on keywords...")
    missing_count = 0
    
    for line in lines:
        data = json.loads(line)
        comment = data.get("comment", "").lower()
        diff = data.get("diff_hunk", "")
        # Real world label (did it get acted on?)
        original_label = data.get("label", 0)
        
        assigned = False
        # Very naive assignment based on keyword hit for simulation
        for team_name, profile in TEAM_PROFILES.items():
            if any(kw in comment for kw in profile["keywords"]):
                
                # If the team cares about this topic AND the original author acted on it, it's a SURFACE
                # If they care about it, but original didn't act, maybe it's still a SURFACE for this team
                # For simplicity, we just inherit the ground truth label.
                
                prompt = generate_prompt(diff, comment, team_name)
                
                team_data[team_name].append({
                    "prompt": prompt,
                    "diff": diff,
                    "comment": comment,
                    "label": original_label,
                    "team": team_name
                })
                assigned = True
                break
                
        if not assigned:
            missing_count += 1
            
    logger.info(f"Ignored {missing_count} samples that didn't match any team keywords.")
    
    # Save datasets
    # The requirement is 20-50 train samples, 200+ test samples per team
    for team_name, samples in team_data.items():
        if len(samples) < 250:
            logger.warning(f"Team {team_name} only got {len(samples)} samples. Simulation might be weak.")
        
        # Shuffle
        random.shuffle(samples)
        
        # Take 50 for train, 200 for test
        train_samples = samples[:50]
        test_samples = samples[50:250]
        
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
