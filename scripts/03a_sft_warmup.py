"""
SFT Warm-Up for Phase 2 DAPO.

Performs a short supervised fine-tuning pass on each team's training data
BEFORE running GRPO. This gives the model a better starting point for RL
by teaching it the basic output format and task structure.

Usage:
    python3 scripts/03a_sft_warmup.py
"""

import os
import sys
import json
import logging
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.reward_model import RewardModel
from src.data.team_dataset import simulate_team_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def create_sft_example(item: dict, tokenizer) -> str:
    """Creates a supervised training example with the ideal completion format."""
    prompt = item["prompt"]
    label = item["label"]
    diff = item.get("diff", "")[:200]
    comment = item.get("comment", "")[:200]
    
    # Create a well-formed ideal completion
    decision = "SURFACE" if label == 1 else "FILTER"
    # Fallback to hardcoded scores only for older datasets lacking rm_score
    score = str(item.get("rm_score", 0.8 if label == 1 else 0.2))
    
    # Create a grounded reasoning trace
    if label == 1:
        reasoning = (
            f"The review comment addresses a specific concern related to the code changes in the diff. "
            f"Based on the team's stated priorities, this type of feedback aligns with what they care about. "
            f"The comment provides actionable insight that the team would benefit from seeing."
        )
    else:
        reasoning = (
            f"The review comment does not align with the team's stated priorities. "
            f"While the comment may have some merit, this team explicitly focuses on different aspects "
            f"of code quality. Filtering this comment would reduce noise for the team."
        )
    
    ideal_completion = f"<think>\n{reasoning}\n</think>\n<score>{score}</score>\n<decision>{decision}</decision>"
    
    # Format with chat template
    messages = [
        {"role": "system", "content": "You are a helpful AI code reviewer."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return formatted_prompt + ideal_completion


def sft_warmup_team(model, tokenizer, team_name: str, train_file: str, device: str, num_epochs: int = 2, lr: float = 5e-6):
    """Runs SFT warm-up for a single team."""
    logger.info(f"\n{'='*50}\nSFT Warm-Up for team: {team_name}\n{'='*50}")
    
    if not os.path.exists(train_file):
        logger.error(f"Train file not found: {train_file}")
        return
    
    dataset = []
    with open(train_file, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    
    # Use up to 30 samples for SFT warm-up (small but effective)
    dataset = dataset[:30]
    logger.info(f"Using {len(dataset)} samples for SFT warm-up")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        random_order = list(range(len(dataset)))
        import random
        random.shuffle(random_order)
        
        for idx in tqdm(random_order, desc=f"SFT Epoch {epoch+1}/{num_epochs}"):
            item = dataset[idx]
            full_text = create_sft_example(item, tokenizer)
            
            # Tokenize
            inputs = tokenizer(full_text, truncation=True, max_length=1536, return_tensors="pt").to(device)
            
            # Standard causal LM loss (predict next token)
            labels = inputs.input_ids.clone()
            
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset)
        logger.info(f"SFT Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
    
    logger.info(f"SFT warm-up complete for {team_name}")


def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    with open(PROJECT_ROOT / "configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    dapo_config = config["dapo"]
    model_name = dapo_config["model_name"]
    teams_dir = PROJECT_ROOT / "data/teams"
    output_dir = PROJECT_ROOT / "checkpoints/sft_warmup"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Ensure Data Exists
    if not teams_dir.exists() or len(list(teams_dir.glob("*"))) == 0:
        logger.info("Simulated team data not found, generating now using Reward Model for labels...")
        
        # Load RM
        rm_path = PROJECT_ROOT / "checkpoints/reward_model/best"
        if not rm_path.exists():
            logger.error(f"Phase 1 RM checkpoint not found at {rm_path}")
            sys.exit(1)
            
        rm_model = RewardModel.load_checkpoint(str(rm_path), config)
        rm_model = rm_model.to(device)
        rm_model.eval()
        
        processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
        base_data = processed_dir / "train.jsonl"
        if (processed_dir / "train_small.jsonl").exists():
            base_data = processed_dir / "train_small.jsonl"
            
        simulate_team_datasets(str(base_data), str(teams_dir), rm_model=rm_model, rm_tokenizer=rm_model.tokenizer)
        
        # Free RM from memory to save VRAM for SFT
        del rm_model
        torch.cuda.empty_cache()
    
    # Load model & tokenizer ONCE
    logger.info(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None
    )
    
    # Apply LoRA (same config as DAPO)
    lora_config = LoraConfig(
        r=dapo_config.get("lora_r", 16),
        lora_alpha=dapo_config.get("lora_alpha", 32),
        target_modules=dapo_config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=dapo_config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    
    # SFT warm-up for each team
    for team_dir in sorted(teams_dir.iterdir()):
        if not team_dir.is_dir():
            continue
        
        team_name = team_dir.name
        train_file = str(team_dir / "train.jsonl")
        
        # Reset LoRA weights for each team
        for name, param in model.named_parameters():
            if "lora" in name:
                torch.nn.init.normal_(param, std=0.01)
        
        sft_warmup_team(model, tokenizer, team_name, train_file, device)
        
        # Save the warm-up checkpoint
        save_path = output_dir / f"sft_warmup_{team_name}"
        model.save_pretrained(str(save_path))
        logger.info(f"Saved SFT warm-up checkpoint to {save_path}")
    
    logger.info("\n" + "="*50 + "\nAll SFT warm-ups complete!\n" + "="*50)


if __name__ == "__main__":
    main()
