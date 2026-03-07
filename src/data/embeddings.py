"""
Embedding precomputation module for RLCR v2.

Precomputes MiniLM embeddings for all comments.
Used in Phase 2 for reward alignment (R3: calibration).
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def precompute_embeddings(
    data_paths: list[str],
    config: dict,
    project_root: str,
    batch_size: Optional[int] = None,
) -> tuple[str, str]:
    """
    Precompute MiniLM embeddings for all comments.
    
    Args:
        data_paths: List of JSONL file paths to embed
        config: Full config dict
        project_root: Project root directory
        batch_size: Override batch size
    
    Returns:
        (embeddings_path, ids_path) - paths to saved .npy and .json files
    """
    from sentence_transformers import SentenceTransformer
    
    emb_config = config["embeddings"]
    emb_dir = Path(project_root) / config["data"]["embeddings_dir"]
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    emb_path = emb_dir / "comment_embeddings.npy"
    ids_path = emb_dir / "comment_ids.json"
    
    if emb_path.exists() and ids_path.exists():
        logger.info(f"Embeddings already exist: {emb_path}")
        return str(emb_path), str(ids_path)
    
    # Collect all comments
    comments = []
    sample_ids = []
    
    for data_path in data_paths:
        if not Path(data_path).exists():
            logger.warning(f"Data file not found: {data_path}")
            continue
        
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        comment = sample.get("comment", "").strip()
                        if comment:
                            comments.append(comment)
                            sample_ids.append({
                                "file": Path(data_path).name,
                                "index": i,
                                "label": sample.get("label", -1),
                            })
                    except json.JSONDecodeError:
                        continue
    
    logger.info(f"Embedding {len(comments)} comments...")
    
    # Load model
    model_name = emb_config["model_name"]
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Compute embeddings in batches
    bs = batch_size or emb_config["batch_size"]
    embeddings = model.encode(
        comments,
        batch_size=bs,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    # Save
    np.save(str(emb_path), embeddings)
    with open(ids_path, "w") as f:
        json.dump(sample_ids, f)
    
    logger.info(f"Saved embeddings: {embeddings.shape} to {emb_path}")
    logger.info(f"Saved IDs: {len(sample_ids)} to {ids_path}")
    
    return str(emb_path), str(ids_path)
