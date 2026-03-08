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
    
    # Initialize buckets for balancing
    team_data_surface = {team: [] for team in TEAM_PROFILES.keys()}
    team_data_filter = {team: [] for team in TEAM_PROFILES.keys()}
    
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
        
        # 5. Generate team-specific labels
        # Assign to a random team to balance dataset sizes
        team_name = random.choice(list(TEAM_PROFILES.keys()))
        profile = TEAM_PROFILES[team_name]
        
        cares = any(kw in comment for kw in profile["keywords"])
        # If the team cares about this topic, the label inherits the original author's action (1 or 0)
        # If the team does NOT care about this topic, it is always a FILTER (0)
        team_label = original_label if cares else 0
        
        prompt = generate_prompt(diff, comment, team_name)
        
        sample = {
            "prompt": prompt,
            "diff": diff,
            "comment": comment,
            "label": team_label,
            "team": team_name
        }
        
        if team_label == 1:
            team_data_surface[team_name].append(sample)
        else:
            team_data_filter[team_name].append(sample)
            
    logger.info(f"Ignored {missing_count} samples that didn't match any team keywords.")
    
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
