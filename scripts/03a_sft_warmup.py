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
    score = item.get("score", 0.5)
    
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
    
    ideal_completion = f"<think>\n{reasoning}\n</think>\n<score>{score:.4f}</score>\n<decision>{decision}</decision>"
    
    # Format with chat template
    messages = [
        {"role": "system", "content": "You are a helpful AI code reviewer."},
        {"role": "user", "content": prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return prompt_text, ideal_completion


def sft_warmup_team(model, tokenizer, team_name: str, threshold: float, full_dataset: list, precomputed_scores: dict, device: str, num_epochs: int = 1, lr: float = 5e-6):
    """Runs SFT warm-up for a single team using a subset of the unlabeled data."""
    logger.info(f"\n{'='*50}\nSFT Warm-Up for team: {team_name}\n{'='*50}")
    
    import random
    import hashlib
    from src.data.team_dataset import generate_prompt
    
    # Use up to 300 samples for SFT warm-up to deeply bake in the score calibration
    raw_dataset = random.sample(full_dataset, min(300, len(full_dataset)))
    
    dataset = []
    for item in raw_dataset:
        diff = item.get("diff", "")
        comment = item.get("comment", "")
        ex_id = item.get("example_id", hashlib.md5(f"{diff}_{comment}".encode('utf-8')).hexdigest())
        
        score = precomputed_scores.get(ex_id, 0.5)
        label = 1 if score >= threshold else 0
        prompt = generate_prompt(diff, comment, team_name)
        
        dataset.append({
            "prompt": prompt,
            "label": label,
            "diff": diff,
            "comment": comment,
            "score": score
        })
        
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
            prompt_text, completion_text = create_sft_example(item, tokenizer)
            full_text = prompt_text + completion_text
            
            # Tokenize full combined text
            inputs = tokenizer(full_text, truncation=True, max_length=1536, return_tensors="pt").to(device)
            
            # Tokenize just the prompt to find where to mask
            prompt_inputs = tokenizer(prompt_text, truncation=True, max_length=1536, return_tensors="pt")
            prompt_len = prompt_inputs.input_ids.shape[1]
            
            # Standard causal LM loss (predict next token)
            labels = inputs.input_ids.clone()
            
            # Mask out the prompt so loss is ONLY computed on the completion!
            labels[0, :prompt_len] = -100
            
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
    output_dir = PROJECT_ROOT / "checkpoints/sft_warmup"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load precomputed scores and unlabeled dataset
    precomputed_scores_path = PROJECT_ROOT / dapo_config.get("precomputed_scores_path", "data/precomputed_rm_scores.json")
    if not precomputed_scores_path.exists():
        logger.error(f"Precomputed scores not found at {precomputed_scores_path}. Run 00b_precompute_scores.py first!")
        sys.exit(1)
        
    with open(precomputed_scores_path, "r") as f:
        precomputed_scores = json.load(f)
        
    unlabeled_path = PROJECT_ROOT / dapo_config.get("unlabeled_data_path", "data/unlabeled_pairs.json")
    if not unlabeled_path.exists():
        logger.error(f"Unlabeled dataset not found at {unlabeled_path}. Run 00b_precompute_scores.py first!")
        sys.exit(1)
        
    full_dataset = []
    with open(unlabeled_path, "r") as f:
        for line in f:
            full_dataset.append(json.loads(line))
    
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
    from src.data.team_dataset import TEAM_PROFILES
    
    for team_name, profile in TEAM_PROFILES.items():
        threshold = profile["rm_threshold"]
        
        # Reset LoRA weights for each team
        for name, param in model.named_parameters():
            if "lora" in name:
                torch.nn.init.normal_(param, std=0.01)
        
        sft_warmup_team(model, tokenizer, team_name, threshold, full_dataset, precomputed_scores, device)
        
        # Save the warm-up checkpoint
        save_path = output_dir / f"sft_warmup_{team_name}"
        model.save_pretrained(str(save_path))
        logger.info(f"Saved SFT warm-up checkpoint to {save_path}")
    
    logger.info("\n" + "="*50 + "\nAll SFT warm-ups complete!\n" + "="*50)


if __name__ == "__main__":
    main()
