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
    Parse CodeReviewer's JSONL files into structured records.
    
    The Zenodo download provides:
    - Code_Refinement/ref-{split}.jsonl
    - Comment_Generation/msg-{split}.jsonl
    """
    data_dir = Path(data_dir)
    records = []
    
    # Check for jsonl files
    jsonl_files = []
    for pattern in [f"ref-{split}.jsonl", f"msg-{split}.jsonl", f"{split}.jsonl"]:
        # Search recursively
        for path in data_dir.rglob(pattern):
            jsonl_files.append(path)
    
    if not jsonl_files:
        logger.warning(
            f"Could not find CodeReviewer {split} jsonl files targeting {data_dir}"
        )
        return []
    
    logger.info(f"Found CodeReviewer {split} files: {[p.name for p in jsonl_files]}")
    
    for file_path in jsonl_files:
        logger.info(f"Parsing {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Extract fields (names might vary slightly between tasks)
                diff_hunk = obj.get("diff_hunk", obj.get("patch", ""))
                comment = obj.get("comment", obj.get("msg", ""))
                old_code = obj.get("old_code", obj.get("old_file", ""))
                target = obj.get("target", obj.get("refinement", ""))
                
                if not diff_hunk or not comment:
                    continue
                
                # Label: 1 if developer acted on comment (non-empty target different from source)
                # In Comment_Generation, 'target' is missing or just the comment.
                # In Code_Refinement, 'target' is the refined code.
                if "refinement" in obj or "target" in obj:
                    has_target = bool(target and target.strip() and target.strip() != old_code.strip())
                    label = 1 if has_target else 0
                else:
                    # If there's no target field at all, we can't be sure it was acted on.
                    # For safety in training reward model, we might skip these or label 0,
                    # but let's assume Comment_Generation samples without a target
                    # might not represent "acted upon" code changes.
                    label = 0
                
                records.append({
                    "diff_hunk": diff_hunk,
                    "comment": comment,
                    "old_code": old_code,
                    "target": target,
                    "label": label,
                    "source": f"codereviewer_{file_path.parent.name}",
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
