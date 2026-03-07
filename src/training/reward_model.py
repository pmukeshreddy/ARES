"""
Reward Model for RLCR v2 (Phase 1).

Architecture:
    Qwen2.5-Coder (1.5B or 7B) as backbone
    + LoRA adapters (r=32, alpha=64)
    + Binary classification head (mean pooling → Linear → sigmoid)

Purpose:
    Learn f(diff, comment) → quality_score
    "Will a developer act on this comment?"
    
    This model's output becomes the REWARD signal for Phase 2 DAPO.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


class RewardModelHead(nn.Module):
    """Classification head on top of the language model."""
    
    def __init__(self, hidden_size: int, intermediate_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(intermediate_dim, 1)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) - last hidden states from LM
            attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding
        
        Returns:
            logits: (batch, 1) - raw logits (pre-sigmoid)
        """
        # Mean pooling over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # (batch, hidden_size)
        count = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)
        pooled = sum_hidden / count  # (batch, hidden_size)
        
        # Classification
        x = self.dense(pooled)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.classifier(x)  # (batch, 1)
        
        return logits


class RewardModel(nn.Module):
    """
    Complete reward model: Qwen2.5-Coder + LoRA + classification head.
    
    Usage:
        model = RewardModel.from_config(config)
        logits = model(input_ids, attention_mask)  # (batch, 1)
        scores = torch.sigmoid(logits)             # probabilities
    """
    
    def __init__(self, backbone, head: RewardModelHead, tokenizer):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.tokenizer = tokenizer
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        
        Returns:
            logits: (batch, 1) - raw logits before sigmoid
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Use last hidden state
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
        logits = self.head(hidden_states, attention_mask)
        
        return logits
    
    def score(
        self,
        diff: str,
        comment: str,
        device: Optional[str] = None,
    ) -> float:
        """
        Score a single (diff, comment) pair.
        
        Args:
            diff: The code diff/hunk
            comment: The review comment
            device: Device to run on (auto-detected if None)
        
        Returns:
            score: float in [0, 1] - probability comment will be acted on
        """
        if device is None:
            device = next(self.parameters()).device
        
        text = f"{diff} [SEP] {comment}"
        encoding = self.tokenizer(
            text,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                encoding["input_ids"],
                encoding["attention_mask"],
            )
            score = torch.sigmoid(logits).item()
        
        return score
    
    def score_batch(
        self,
        diffs: list[str],
        comments: list[str],
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> list[float]:
        """Score a batch of (diff, comment) pairs."""
        if device is None:
            device = next(self.parameters()).device
        
        texts = [f"{d} [SEP] {c}" for d, c in zip(diffs, comments)]
        
        self.eval()
        scores = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoding = self.tokenizer(
                batch_texts,
                max_length=1024,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            
            with torch.no_grad():
                logits = self.forward(
                    encoding["input_ids"],
                    encoding["attention_mask"],
                )
                batch_scores = torch.sigmoid(logits).squeeze(-1).tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)
        
        return scores
    
    @classmethod
    def from_config(cls, config: dict) -> "RewardModel":
        """
        Build reward model from config dict.
        
        Steps:
            1. Load Qwen2.5-Coder base model
            2. Apply LoRA adapters
            3. Add classification head
        """
        rm_config = config["reward_model"]
        model_name = rm_config["model_name"]
        
        logger.info(f"Loading base model: {model_name}")
        
        # Determine dtype
        if rm_config.get("bf16", False):
            dtype = torch.bfloat16
        elif rm_config.get("fp16", False):
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load base model
        backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            # Don't load to GPU yet, let caller handle device placement
            device_map=None,
        )
        
        # Disable generation head (we only need hidden states)
        backbone.config.use_cache = False
        
        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rm_config["lora_r"],
            lora_alpha=rm_config["lora_alpha"],
            lora_dropout=rm_config["lora_dropout"],
            target_modules=rm_config["lora_target_modules"],
            bias="none",
        )
        
        backbone = get_peft_model(backbone, lora_config)
        backbone.print_trainable_parameters()
        
        # Get hidden size from model config
        hidden_size = backbone.config.hidden_size
        
        # Create classification head
        head = RewardModelHead(
            hidden_size=hidden_size,
            intermediate_dim=rm_config["hidden_dim"],
            dropout=rm_config["dropout"],
        )
        # Head is always float32 for stability
        head = head.float()
        
        model = cls(backbone, head, tokenizer)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,} "
                    f"({trainable_params/total_params*100:.2f}%)")
        
        return model
    
    def save_checkpoint(self, save_dir: str):
        """Save LoRA weights + classification head."""
        from pathlib import Path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapter
        self.backbone.save_pretrained(str(save_dir / "lora_adapter"))
        
        # Save classification head
        torch.save(self.head.state_dict(), str(save_dir / "head.pt"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(save_dir / "tokenizer"))
        
        logger.info(f"Checkpoint saved to {save_dir}")
    
    @classmethod
    def load_checkpoint(cls, save_dir: str, config: dict) -> "RewardModel":
        """Load from saved checkpoint."""
        from pathlib import Path
        from peft import PeftModel
        
        save_dir = Path(save_dir)
        rm_config = config["reward_model"]
        model_name = rm_config["model_name"]
        
        # Determine dtype
        if rm_config.get("bf16", False):
            dtype = torch.bfloat16
        elif rm_config.get("fp16", False):
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(save_dir / "tokenizer"), trust_remote_code=True
        )
        
        # Load base model + LoRA
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=None,
        )
        base_model.config.use_cache = False
        
        backbone = PeftModel.from_pretrained(
            base_model,
            str(save_dir / "lora_adapter"),
        )
        
        # Load classification head
        hidden_size = backbone.config.hidden_size
        head = RewardModelHead(
            hidden_size=hidden_size,
            intermediate_dim=rm_config["hidden_dim"],
            dropout=rm_config["dropout"],
        )
        head.load_state_dict(torch.load(str(save_dir / "head.pt"), map_location="cpu"))
        
        model = cls(backbone, head, tokenizer)
        logger.info(f"Loaded checkpoint from {save_dir}")
        
        return model
