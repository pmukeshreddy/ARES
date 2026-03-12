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
        "context": (
            "Our priorities (SURFACE any comment matching these):\n"
            "- Bug reports: logic errors, off-by-one, null handling, race conditions\n"
            "- Correctness: wrong behavior, missing validation, broken edge cases\n"
            "- Architecture: poor abstractions, coupling issues, scaling concerns\n"
            "- Data safety: potential data loss, corruption, or security issues\n"
            "- Subtle issues: comments that look minor but point to real problems\n\n"
            "Lower priority (FILTER these):\n"
            "- Pure style/formatting with no functional impact\n"
            "- Subjective preferences with no clear improvement"
        ),
        "rm_threshold": 0.85
    },
    "Thorough-Mentors": {
        "context": (
            "Our priorities (SURFACE any comment matching these):\n"
            "- Learning opportunities: alternative approaches, design patterns\n"
            "- Best practices: suggestions that help contributors grow\n"
            "- Edge cases: any scenario the author may not have considered\n"
            "- Code quality: readability, maintainability, documentation improvements\n"
            "- Community standards: anything that helps align with project norms\n\n"
            "Lower priority (FILTER these):\n"
            "- Obviously trivial or auto-fixable issues (trailing whitespace, etc.)"
        ),
        "rm_threshold": 0.30
    },
    "Security-First": {
        "context": (
            "Our priorities (SURFACE any comment matching these):\n"
            "- Vulnerabilities: SQL injection, XSS, buffer overflows, SSRF\n"
            "- Auth issues: broken access control, missing validation, token handling\n"
            "- Input validation: unsanitized user input, missing bounds checks\n"
            "- Data exposure: leaking secrets, PII, or internal details\n"
            "- Unsafe defaults: permissive configs, disabled security features\n\n"
            "Lower priority (FILTER these):\n"
            "- Pure style comments with no security implications"
        ),
        "rm_threshold": 0.70
    },
    "Performance-Obsessed": {
        "context": (
            "Our priorities (SURFACE any comment matching these):\n"
            "- Complexity: O(N²) loops, unnecessary iterations, algorithmic issues\n"
            "- Memory: unnecessary allocations, copies, or leaks\n"
            "- Concurrency: lock contention, thread safety, deadlock risks\n"
            "- Caching: missed caching opportunities, cache invalidation bugs\n"
            "- GC pressure: object churn, large temporary allocations\n\n"
            "Lower priority (FILTER these):\n"
            "- Minor refactoring that doesn't impact performance"
        ),
        "rm_threshold": 0.65
    },
    "Style-Sticklers": {
        "context": (
            "Our priorities (SURFACE any comment matching these):\n"
            "- Naming: inconsistent or unclear variable/function names\n"
            "- Types: missing or incorrect type annotations\n"
            "- Formatting: violations of team lint rules or style guide\n"
            "- Documentation: missing/outdated docstrings, unclear comments\n"
            "- Consistency: patterns that diverge from codebase conventions\n\n"
            "Lower priority (FILTER these):\n"
            "- Comments that are purely about logic/architecture with no style aspect"
        ),
        "rm_threshold": 0.50
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
        f"You are an AI code review routing assistant. You classify review comments as "
        f"SURFACE (the team should see this) or FILTER (the team can skip this).\n\n"
        f"<team_priorities>\n{context}\n</team_priorities>\n\n"
        f"<diff>\n{diff[:1500]}\n</diff>\n\n"
        f"<comment>\n{comment}\n</comment>\n\n"
        f"Evaluate the comment against the team's priorities. In your <think> block:\n"
        f"1. What specific issue does the comment raise? (factual summary)\n"
        f"2. Does it match any of the team's stated priorities? (check each)\n"
        f"3. Is the issue actionable — does it point to a concrete problem or improvement?\n\n"
        f"Based on your evaluation, output a confidence score in <score> tags and your "
        f"decision in <decision> tags (SURFACE or FILTER).\n\n"
        f"Note: Approximately half of comments are worth surfacing. When in doubt and "
        f"the comment raises a plausible technical concern, lean toward SURFACE.\n"
    )
    return prompt


def simulate_team_datasets(hf_dataset_path: str, output_dir: str, rm_model=None, rm_tokenizer=None, precomputed_scores_path: str = None):
    """
    Reads the processed HF dataset and distributes samples into 5 team buckets.
    Uses the Phase 1 Reward Model (F1=0.99) to score each comment's actionability,
    then applies team-specific thresholds to generate high-quality ground truth labels.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load precomputed RM scores if available (for continuous thresholds)
    precomputed_scores = {}
    if precomputed_scores_path and Path(precomputed_scores_path).exists():
        import hashlib
        with open(precomputed_scores_path, "r") as f:
            precomputed_scores = json.load(f)
        logger.info(f"Loaded {len(precomputed_scores)} precomputed RM scores for continuous thresholds")
    elif rm_model is None:
        logger.warning("No RM model or precomputed scores — team thresholds will be ineffective (binary labels)")
    
    # Initialize buckets for balancing
    team_data_surface = {team: [] for team in TEAM_PROFILES.keys()}
    team_data_filter = {team: [] for team in TEAM_PROFILES.keys()}
    
    # Track seen comments for deduplication
    seen_comments = set()
    skipped_dupes = 0
    
    logger.info(f"Loading base dataset from {hf_dataset_path}")
    # Load dataset
    with open(hf_dataset_path, "r") as f:
        lines = f.readlines()
        random.shuffle(lines)
        
    logger.info(f"Loaded {len(lines)} total items. Beginning RM scoring and dynamic distribution...")
    
    # We need 175 SURFACE and 175 FILTER samples per team (350 total = 100 train / 250 test)
    TARGET_PER_CLASS = 175
    
    for line in tqdm(lines, desc="Generating Team Labels"):
        # Check if all teams are fully populated
        all_full = True
        for t in TEAM_PROFILES.keys():
            if len(team_data_surface[t]) < TARGET_PER_CLASS or len(team_data_filter[t]) < TARGET_PER_CLASS:
                all_full = False
                break
                
        if all_full:
            logger.info("All team buckets perfectly filled (175 SURFACE / 175 FILTER)! Stopping early.")
            break
            
        data = json.loads(line)
        comment = data.get("comment", "")
        diff = data.get("diff_hunk", "")
        # Real world label (did it get acted on?)
        original_label = data.get("label", 0)
        
        # Deduplicate by comment text (prevents "no issues found" from flooding FILTER)
        comment_key = comment.strip().lower()
        if comment_key in seen_comments:
            skipped_dupes += 1
            continue
        seen_comments.add(comment_key)
        
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
        elif precomputed_scores:
            # Use precomputed continuous RM score for meaningful team thresholds
            import hashlib
            example_id = data.get("example_id", hashlib.md5(f"{diff}_{comment}".encode('utf-8')).hexdigest())
            
            # CRITICAL FIX: Skip examples that haven't been scored by 00b_precompute_scores.py
            if example_id not in precomputed_scores:
                continue
                
            rm_score = precomputed_scores[example_id]
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
    
    logger.info(f"Skipped {skipped_dupes} duplicate comments during data generation")
    
    # Save datasets
    # The requirement is 100 train samples, 200+ test samples per team
    for team_name in TEAM_PROFILES.keys():
        surfaces = team_data_surface[team_name]
        filters = team_data_filter[team_name]
        
        # Balance 50/50
        min_len = min(len(surfaces), len(filters))
        if min_len < 175: # Need 350 total (175 each) for 100 train / 250 test
            logger.warning(f"Team {team_name} only got {min_len*2} balanced samples. Simulation might be weak.")
            
        random.shuffle(surfaces)
        random.shuffle(filters)
        
        # Take equal amounts
        samples = surfaces[:min_len] + filters[:min_len]
        
        # Shuffle matched dataset
        random.shuffle(samples)
        
        # Take 100 for train, 250 for test (or whatever is available)
        train_samples = samples[:100]
        test_samples = samples[100:350] if len(samples) >= 350 else samples[100:]
        
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
    simulate_team_datasets(
        "data/processed/train.jsonl", "data/teams/",
        precomputed_scores_path="data/precomputed_rm_scores.json"
    )
