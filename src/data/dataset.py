"""
PyTorch Dataset for reward model training.

Formats inputs as:  [diff_hunk] [SEP] [comment]
Returns: input_ids, attention_mask, label
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class RewardModelDataset(Dataset):
    """
    Dataset for reward model training.
    
    Input format: "{diff_hunk} [SEP] {comment}"
    Label: 0 or 1 (binary: was the comment acted on?)
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_path: Path to JSONL file with {diff_hunk, comment, label}
            tokenizer: HuggingFace tokenizer
            max_length: Max token length for input
            max_samples: Optional limit for debugging
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        self.samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        
        n_pos = sum(1 for s in self.samples if s["label"] == 1)
        logger.info(f"Loaded {len(self.samples)} samples "
                    f"(pos={n_pos}, neg={len(self.samples)-n_pos})")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _format_input(self, sample: dict) -> str:
        """Format (diff_hunk, comment) into model input string."""
        diff = sample.get("diff_hunk", "").strip()
        comment = sample.get("comment", "").strip()
        
        # Truncate diff if very long (pre-tokenization truncation)
        # Leave room for comment and special tokens
        max_diff_chars = self.max_length * 3  # ~3 chars per token rough estimate
        if len(diff) > max_diff_chars:
            diff = diff[:max_diff_chars]
        
        return f"{diff} [SEP] {comment}"
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        text = self._format_input(sample)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.float32),
        }


def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer,
    config: dict,
    max_samples: Optional[int] = None,
) -> tuple:
    """
    Create train and validation DataLoaders.
    
    Returns:
        (train_loader, val_loader)
    """
    rm_config = config["reward_model"]
    max_length = config["data"]["max_input_tokens"]
    batch_size = rm_config["batch_size"]
    
    train_dataset = RewardModelDataset(
        train_path, tokenizer, max_length, max_samples
    )
    val_dataset = RewardModelDataset(
        val_path, tokenizer, max_length, max_samples
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    logger.info(f"Train loader: {len(train_loader)} batches, "
                f"Val loader: {len(val_loader)} batches")
    
    return train_loader, val_loader
