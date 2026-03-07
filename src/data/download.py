"""
Data download module for RLCR v2.

Downloads:
1. CodeReviewer dataset from Zenodo (Code_Refinement + Comment_Generation)
2. ronantakizawa/github-codereview from HuggingFace
"""

import os
import json
import zipfile
import logging
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_file(url: str, dest_path: str, chunk_size: int = 8192) -> str:
    """Download a file with progress bar."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        logger.info(f"File already exists: {dest_path}")
        return str(dest_path)
    
    logger.info(f"Downloading {url} → {dest_path}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    
    total = int(resp.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest_path.name
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))
    
    return str(dest_path)


def extract_zip(zip_path: str, extract_dir: str) -> str:
    """Extract a zip file."""
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting {zip_path} → {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    
    return str(extract_dir)


def _parse_codereviewer_files(data_dir: str, split: str) -> list[dict]:
    """
    Parse CodeReviewer's line-aligned text files into structured records.
    
    CodeReviewer stores data in separate files:
    - {split}.diff_hunk.src    → diff hunks (one per line)
    - {split}.msg.src          → review comments (one per line)  
    - {split}.code.src         → old code (one per line)
    - {split}.code.tgt         → refined code / target (one per line)
    
    Lines are aligned: line N in each file corresponds to the same sample.
    """
    data_dir = Path(data_dir)
    records = []
    
    # Try multiple possible directory structures
    possible_dirs = [
        data_dir,
        data_dir / "Code_Refinement",
        data_dir / "code_refinement",
        data_dir / "Refinement",
    ]
    
    src_dir = None
    for d in possible_dirs:
        # Check for the text files in this directory or its subdirectories
        diff_file = None
        for pattern in [
            f"{split}.diff_hunk.src",
            f"{split}.diff.src",
            f"ref-{split}.diff_hunk.src",
        ]:
            candidate = d / pattern
            if candidate.exists():
                diff_file = candidate
                break
            # Also check subdirectories
            for sub in d.iterdir() if d.exists() else []:
                if sub.is_dir():
                    candidate = sub / pattern
                    if candidate.exists():
                        diff_file = candidate
                        break
        
        if diff_file is not None:
            src_dir = diff_file.parent
            break
    
    if src_dir is None:
        logger.warning(
            f"Could not find CodeReviewer {split} files in {data_dir}. "
            f"Searched: {[str(d) for d in possible_dirs]}"
        )
        # List what's actually in the directory for debugging
        if data_dir.exists():
            logger.info(f"Contents of {data_dir}: {list(data_dir.rglob('*'))[:20]}")
        return []
    
    logger.info(f"Found CodeReviewer data in: {src_dir}")
    
    # Read all aligned files
    def read_lines(filename_patterns: list[str]) -> list[str]:
        for pattern in filename_patterns:
            path = src_dir / pattern
            if path.exists():
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    return [line.strip() for line in f]
        return []
    
    diffs = read_lines([
        f"{split}.diff_hunk.src", f"{split}.diff.src",
        f"ref-{split}.diff_hunk.src"
    ])
    comments = read_lines([
        f"{split}.msg.src", f"{split}.comment.src",
        f"ref-{split}.msg.src"
    ])
    old_codes = read_lines([
        f"{split}.code.src", f"ref-{split}.code.src"
    ])
    targets = read_lines([
        f"{split}.code.tgt", f"ref-{split}.code.tgt"
    ])
    
    if not diffs:
        logger.warning(f"No diff data found for split '{split}' in {src_dir}")
        return []
    
    n = len(diffs)
    comments = comments if len(comments) == n else [""] * n
    old_codes = old_codes if len(old_codes) == n else [""] * n
    targets = targets if len(targets) == n else [""] * n
    
    for i in range(n):
        target = targets[i]
        # Label: 1 if developer acted on comment (non-empty target different from source)
        has_target = bool(target and target.strip() and target.strip() != old_codes[i].strip())
        
        records.append({
            "diff_hunk": diffs[i],
            "comment": comments[i],
            "old_code": old_codes[i],
            "target": target,
            "label": 1 if has_target else 0,
            "source": "codereviewer",
        })
    
    logger.info(f"Parsed {len(records)} CodeReviewer {split} samples "
                f"(label=1: {sum(r['label'] for r in records)}, "
                f"label=0: {sum(1 - r['label'] for r in records)})")
    
    return records


def download_codereviewer(config: dict, project_root: str) -> str:
    """
    Download and parse CodeReviewer dataset from Zenodo.
    
    Returns path to the combined JSONL file.
    """
    raw_dir = Path(project_root) / config["data"]["raw_dir"] / "codereviewer"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = raw_dir / "codereviewer_parsed.jsonl"
    if output_path.exists():
        logger.info(f"CodeReviewer already parsed: {output_path}")
        return str(output_path)
    
    # Download Code_Refinement.zip (has diff, comment, old_code, target)
    refinement_url = config["data"]["codereviewer_url"]
    zip_path = raw_dir / "Code_Refinement.zip"
    download_file(refinement_url, str(zip_path))
    
    # Extract
    extract_dir = raw_dir / "extracted"
    extract_zip(str(zip_path), str(extract_dir))
    
    # Also download Comment_Generation.zip for additional (diff, comment) pairs
    comment_url = config["data"]["codereviewer_comment_url"]
    comment_zip_path = raw_dir / "Comment_Generation.zip"
    download_file(comment_url, str(comment_zip_path))
    extract_zip(str(comment_zip_path), str(extract_dir))
    
    # Parse all splits
    all_records = []
    for split in ["train", "valid", "test"]:
        # Check in refinement data
        for subdir in ["Code_Refinement", "Comment_Generation"]:
            records = _parse_codereviewer_files(str(extract_dir / subdir), split)
            if records:
                all_records.extend(records)
        
        # Also try root extract dir
        if not any(r for r in all_records if True):  # only if nothing found
            records = _parse_codereviewer_files(str(extract_dir), split)
            if records:
                all_records.extend(records)
    
    # If we still have nothing, try to find files recursively
    if not all_records:
        logger.warning("Standard paths didn't work, searching recursively...")
        for txt_file in extract_dir.rglob("*.src"):
            logger.info(f"Found: {txt_file}")
        for txt_file in extract_dir.rglob("*.tgt"):
            logger.info(f"Found: {txt_file}")
    
    # Save as JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(all_records)} CodeReviewer records to {output_path}")
    return str(output_path)


def download_hf_dataset(config: dict, project_root: str) -> str:
    """
    Download ronantakizawa/github-codereview from HuggingFace.
    
    Returns path to the saved JSONL file.
    """
    from datasets import load_dataset
    
    raw_dir = Path(project_root) / config["data"]["raw_dir"] / "hf_codereview"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = raw_dir / "hf_codereview.jsonl"
    if output_path.exists():
        logger.info(f"HF dataset already downloaded: {output_path}")
        return str(output_path)
    
    dataset_name = config["data"]["hf_dataset"]
    logger.info(f"Downloading {dataset_name} from HuggingFace...")
    
    ds = load_dataset(dataset_name)
    
    all_records = []
    for split_name in ds:
        split_ds = ds[split_name]
        logger.info(f"Processing HF split '{split_name}': {len(split_ds)} samples")
        
        for sample in tqdm(split_ds, desc=f"HF {split_name}"):
            # Map to our unified format
            # is_negative=True means "No issues found" → label 0
            # is_negative=False means reviewer commented → label 1
            is_negative = sample.get("is_negative", False)
            
            record = {
                "diff_hunk": sample.get("diff_context", "") or sample.get("before_code", ""),
                "comment": sample.get("reviewer_comment", ""),
                "old_code": sample.get("before_code", ""),
                "target": sample.get("after_code", ""),
                "label": 0 if is_negative else 1,
                "source": "hf_codereview",
                # Extra fields for Phase 2 team simulation
                "comment_type": sample.get("comment_type", ""),
                "quality_score": sample.get("quality_score", None),
                "language": sample.get("language", ""),
                "repo_name": sample.get("repo_name", ""),
            }
            all_records.append(record)
    
    # Save as JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(all_records)} HF records to {output_path}")
    return str(output_path)
