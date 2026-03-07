"""
Data preprocessing module for RLCR v2.

Handles:
- Filtering: bots, short comments, long diffs, non-English
- Label validation
- Train/val/test splitting (stratified)
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Common bot patterns in code review
BOT_PATTERNS = [
    r"^(dependabot|renovate|codecov|sonarcloud|github-actions)",
    r"\bbot\b",
    r"^\[automated\]",
    r"^(lgtm|nit:?\s*$)",
]

# Compiled for efficiency
BOT_RE = [re.compile(p, re.IGNORECASE) for p in BOT_PATTERNS]


def _is_bot_comment(comment: str) -> bool:
    """Check if comment appears to be from a bot."""
    comment_lower = comment.strip().lower()
    for pattern in BOT_RE:
        if pattern.search(comment_lower):
            return True
    return False


def _is_english(text: str) -> bool:
    """Quick heuristic English check (fast, no langdetect dependency required)."""
    try:
        from langdetect import detect
        if len(text.strip()) < 20:
            # Too short for reliable detection, assume English
            return True
        lang = detect(text)
        return lang == "en"
    except Exception:
        # If langdetect fails or isn't installed, use heuristic
        # Check if text is mostly ASCII (crude but fast)
        ascii_count = sum(1 for c in text if ord(c) < 128)
        return (ascii_count / max(len(text), 1)) > 0.8


def _count_tokens_approx(text: str) -> int:
    """Approximate token count (whitespace + punctuation splitting)."""
    # Rough approximation: 1 token ≈ 4 chars for code
    return len(text) // 4


def filter_sample(
    sample: dict,
    min_comment_words: int = 3,
    max_diff_tokens: int = 2048,
    remove_bots: bool = True,
    remove_non_english: bool = True,
) -> tuple[bool, str]:
    """
    Check if a sample passes all filters.
    
    Returns:
        (keep, reason) - True if sample should be kept, else reason for removal
    """
    comment = sample.get("comment", "").strip()
    diff = sample.get("diff_hunk", "").strip()
    
    # Empty fields
    if not comment:
        return False, "empty_comment"
    if not diff:
        return False, "empty_diff"
    
    # Minimum comment length
    word_count = len(comment.split())
    if word_count < min_comment_words:
        return False, "short_comment"
    
    # Bot detection
    if remove_bots and _is_bot_comment(comment):
        return False, "bot_comment"
    
    # Diff token limit
    diff_tokens = _count_tokens_approx(diff)
    if diff_tokens > max_diff_tokens:
        return False, "long_diff"
    
    # English check (only for longer comments, short ones are unreliable)
    if remove_non_english and len(comment) > 50 and not _is_english(comment):
        return False, "non_english"
    
    return True, "ok"


def preprocess_and_split(
    input_paths: list[str],
    config: dict,
    project_root: str,
    max_samples: Optional[int] = None,
) -> dict[str, str]:
    """
    Load raw JSONL files, filter, and split into train/val/test.
    
    Args:
        input_paths: List of paths to raw JSONL files
        config: Full config dict
        project_root: Project root directory
        max_samples: Optional limit for debugging
    
    Returns:
        Dict mapping split name → JSONL file path
    """
    data_config = config["data"]
    processed_dir = Path(project_root) / data_config["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    output_paths = {
        "train": processed_dir / "train.jsonl",
        "val": processed_dir / "val.jsonl",
        "test": processed_dir / "test.jsonl",
    }
    
    if all(p.exists() for p in output_paths.values()):
        logger.info("Processed data already exists, loading stats...")
        for split_name, path in output_paths.items():
            count = sum(1 for _ in open(path))
            logger.info(f"  {split_name}: {count} samples")
        return {k: str(v) for k, v in output_paths.items()}
    
    # Load all samples
    all_samples = []
    for input_path in input_paths:
        if not Path(input_path).exists():
            logger.warning(f"Input file not found: {input_path}")
            continue
        
        logger.info(f"Loading {input_path}...")
        with open(input_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {Path(input_path).name}"):
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        all_samples.append(sample)
                    except json.JSONDecodeError:
                        continue
    
    logger.info(f"Loaded {len(all_samples)} total samples")
    
    if max_samples:
        all_samples = all_samples[:max_samples]
        logger.info(f"Limited to {max_samples} samples for debugging")
    
    # Filter
    filter_stats = {}
    filtered_samples = []
    
    for sample in tqdm(all_samples, desc="Filtering"):
        keep, reason = filter_sample(
            sample,
            min_comment_words=data_config["min_comment_words"],
            max_diff_tokens=data_config["max_diff_tokens"],
            remove_bots=data_config["remove_bots"],
            remove_non_english=data_config["remove_non_english"],
        )
        filter_stats[reason] = filter_stats.get(reason, 0) + 1
        if keep:
            filtered_samples.append(sample)
    
    logger.info(f"Filtering: {len(all_samples)} → {len(filtered_samples)} samples")
    logger.info(f"Filter stats: {filter_stats}")
    
    # Label distribution
    n_pos = sum(1 for s in filtered_samples if s["label"] == 1)
    n_neg = len(filtered_samples) - n_pos
    logger.info(f"Label distribution: pos={n_pos} ({n_pos/len(filtered_samples)*100:.1f}%), "
                f"neg={n_neg} ({n_neg/len(filtered_samples)*100:.1f}%)")
    
    # Stratified split
    seed = data_config["seed"]
    rng = np.random.RandomState(seed)
    
    # Separate by label for stratified splitting
    pos_indices = [i for i, s in enumerate(filtered_samples) if s["label"] == 1]
    neg_indices = [i for i, s in enumerate(filtered_samples) if s["label"] == 0]
    
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)
    
    train_ratio = data_config["train_ratio"]
    val_ratio = data_config["val_ratio"]
    
    def split_indices(indices):
        n = len(indices)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }
    
    pos_splits = split_indices(pos_indices)
    neg_splits = split_indices(neg_indices)
    
    # Combine and shuffle within each split
    split_data = {}
    for split_name in ["train", "val", "test"]:
        indices = pos_splits[split_name] + neg_splits[split_name]
        rng.shuffle(indices)
        split_data[split_name] = [filtered_samples[i] for i in indices]
    
    # Save
    for split_name, samples in split_data.items():
        output_path = output_paths[split_name]
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        n_pos = sum(1 for s in samples if s["label"] == 1)
        logger.info(f"  {split_name}: {len(samples)} samples "
                    f"(pos={n_pos}, neg={len(samples)-n_pos})")
    
    # Save filtering metadata
    meta = {
        "total_raw": len(all_samples),
        "total_filtered": len(filtered_samples),
        "filter_stats": filter_stats,
        "splits": {k: len(v) for k, v in split_data.items()},
        "seed": seed,
    }
    with open(processed_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    return {k: str(v) for k, v in output_paths.items()}
