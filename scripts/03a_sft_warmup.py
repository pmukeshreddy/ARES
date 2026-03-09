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
    
    # Create a grounded reasoning trace that argues both sides (matches new prompt instruction)
    if label == 1:
        reasoning = (
            f"Let me consider both sides. "
            f"Reasons this might NOT be relevant: the comment could be seen as a minor suggestion "
            f"rather than a critical issue. "
            f"However, reasons this IS relevant: looking at the code changes and the comment's content, "
            f"this touches on a concern that aligns with what the team cares about — it could affect "
            f"correctness, reliability, or the team's stated priorities. "
            f"On balance, this comment provides value the team should see."
        )
    else:
        reasoning = (
            f"Let me consider both sides. "
            f"Reasons this might be relevant: the comment does point out something in the code "
            f"that could be improved. "
            f"However, reasons this is NOT relevant: the specific concern raised doesn't align "
            f"with what this team prioritizes — it falls outside their stated focus areas. "
            f"The team would likely consider this noise rather than actionable feedback. "
            f"On balance, filtering this comment would reduce noise for the team."
        )
    
    ideal_completion = f"<think>\n{reasoning}\n</think>\n<score>{score:.4f}</score>\n<decision>{decision}</decision>"
    
    # Format with chat template
    messages = [
        {"role": "system", "content": "You are a helpful AI code reviewer."},
        {"role": "user", "content": prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return prompt_text, ideal_completion


def sft_warmup_team(model, tokenizer, team_name: str, threshold: float, full_dataset: list, precomputed_scores: dict, device: str, num_epochs: int = 5, lr: float = 2e-6):
    """Runs SFT warm-up for a single team using a subset of the unlabeled data."""
    logger.info(f"\n{'='*50}\nSFT Warm-Up for team: {team_name}\n{'='*50}")
    
    import random
    import hashlib
    from src.data.team_dataset import generate_prompt
    
    # Use up to 300 samples for SFT warm-up to deeply bake in the score calibration
    raw_dataset = random.sample(full_dataset, min(300, len(full_dataset)))
    
    missing_scores = 0
    dataset = []
    for item in raw_dataset:
        diff = item.get("diff", "")
        comment = item.get("comment", "")
        ex_id = item.get("example_id", hashlib.md5(f"{diff}_{comment}".encode('utf-8')).hexdigest())
        
        if ex_id in precomputed_scores:
            score = precomputed_scores[ex_id]
        else:
            score = 0.5
            missing_scores += 1
            
        label = 1 if score >= threshold else 0
        prompt = generate_prompt(diff, comment, team_name)
        
        dataset.append({
            "prompt": prompt,
            "label": label,
            "diff": diff,
            "comment": comment,
            "score": score
        })
        
    if missing_scores > 0:
        logger.warning(f"SFT Warmup: {missing_scores} out of {len(raw_dataset)} samples missing from precomputed scores. (Fallback to 0.5 applied)")
    
    # DEBUG: Show training data composition
    n_surface_train = sum(1 for d in dataset if d["label"] == 1)
    n_filter_train = sum(1 for d in dataset if d["label"] == 0)
    print(f"\n{'='*60}")
    print(f"SFT TRAINING DATA for {team_name}:")
    print(f"  {len(dataset)} samples: {n_surface_train} SURFACE ({100*n_surface_train/len(dataset):.0f}%) / {n_filter_train} FILTER ({100*n_filter_train/len(dataset):.0f}%)")
    print(f"  Threshold: {threshold}")
    # Show a sample of what the ideal completions look like
    for i, item in enumerate(dataset[:2]):
        _, comp = create_sft_example(item, tokenizer)
        label = "SURFACE" if item["label"] == 1 else "FILTER"
        print(f"  Example [{label}]: {comp[:200]}...")
    print(f"{'='*60}\n")
        
    logger.info(f"Using {len(dataset)} samples for SFT warm-up")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    # Prepare eval subset for per-epoch debugging
    eval_subset = random.sample(dataset, min(20, len(dataset)))
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        valid_steps = 0
        nan_steps = 0
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
            
            # Skip if prompt consumes all tokens (empty completion → NaN loss)
            if prompt_len >= inputs.input_ids.shape[1]:
                nan_steps += 1
                continue
            
            # Standard causal LM loss (predict next token)
            labels = inputs.input_ids.clone()
            
            # Mask out the prompt so loss is ONLY computed on the completion!
            labels[0, :prompt_len] = -100
            
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
            loss = outputs.loss
            
            # Skip NaN losses (numerical instability)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_steps += 1
                optimizer.zero_grad()
                continue
            
            loss.backward()
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            valid_steps += 1
        
        avg_loss = total_loss / max(1, valid_steps)
        logger.info(f"SFT Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Valid: {valid_steps} | Skipped (NaN/empty): {nan_steps}")
        
        # ── Per-Epoch Eval ──────────────────────────────────
        model.eval()
        import re
        
        with torch.no_grad():
            # Prepare all prompts
            eval_prompts = []
            eval_gts = []
            for item in eval_subset:
                prompt_text, _ = create_sft_example(item, tokenizer)
                eval_prompts.append(prompt_text)
                eval_gts.append("SURFACE" if item["label"] == 1 else "FILTER")
            
            # Batch tokenize with left-padding for generation
            tokenizer.padding_side = "left"
            batch_inputs = tokenizer(eval_prompts, truncation=True, max_length=1536, 
                                     padding=True, return_tensors="pt").to(device)
            
            # Batch generate
            gen_ids = model.generate(
                batch_inputs.input_ids,
                attention_mask=batch_inputs.attention_mask,
                max_new_tokens=256,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
            tokenizer.padding_side = "right"  # Reset
            
            # Parse results
            correct = 0
            n_surface_pred = 0
            n_filter_pred = 0
            n_invalid = 0
            example_outputs = []
            
            for i in range(len(eval_subset)):
                prompt_len = batch_inputs.input_ids.shape[1]
                gen_text = tokenizer.decode(gen_ids[i][prompt_len:], skip_special_tokens=True)
                
                dec_match = re.search(r'<decision>(.*?)</decision>', gen_text, re.DOTALL)
                pred = dec_match.group(1).strip().upper() if dec_match else None
                gt = eval_gts[i]
                
                if pred == "SURFACE":
                    n_surface_pred += 1
                elif pred == "FILTER":
                    n_filter_pred += 1
                else:
                    n_invalid += 1
                if pred == gt:
                    correct += 1
                if len(example_outputs) < 3:
                    example_outputs.append((gt, pred, gen_text[:250]))
        
        total_eval = len(eval_subset)
        print(f"\n{'='*60}")
        print(f"SFT EVAL after Epoch {epoch+1}/{num_epochs} (loss={avg_loss:.4f}):")
        print(f"  Predictions: {n_surface_pred} SURFACE ({100*n_surface_pred/total_eval:.0f}%) / "
              f"{n_filter_pred} FILTER ({100*n_filter_pred/total_eval:.0f}%) / {n_invalid} invalid")
        print(f"  Accuracy: {correct}/{total_eval} ({100*correct/total_eval:.0f}%)")
        for gt, pred, text in example_outputs:
            match = "✓" if pred == gt else "✗"
            print(f"  [{match}] GT={gt:7s} Pred={pred or 'NONE':7s} | {text}")
        print(f"{'='*60}\n")
        
        model.train()
    
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
    
    # SFT warm-up for selected teams
    from src.data.team_dataset import TEAM_PROFILES
    
    # Parse --teams argument
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", type=str, default=None, help="Comma-separated team names to train (snake_case)")
    args = parser.parse_args()
    
    # Build team list: filter by --teams if provided
    _team_name_lookup = {k.lower().replace('-', '_'): k for k in TEAM_PROFILES}
    if args.teams:
        requested = [t.strip() for t in args.teams.split(",")]
        teams_to_train = {}
        for t in requested:
            canonical = _team_name_lookup.get(t, t)
            if canonical in TEAM_PROFILES:
                teams_to_train[canonical] = TEAM_PROFILES[canonical]
            else:
                logger.warning(f"Unknown team: {t}, skipping.")
        logger.info(f"Training SFT for specified teams: {list(teams_to_train.keys())}")
    else:
        teams_to_train = TEAM_PROFILES
        logger.info(f"Training SFT for all teams: {list(teams_to_train.keys())}")
    
    for team_name, profile in teams_to_train.items():
        threshold = profile["rm_threshold"]
        
        # Reset LoRA weights for each team
        for name, param in model.named_parameters():
            if "lora" in name:
                torch.nn.init.normal_(param, std=0.01)
        
        sft_warmup_team(model, tokenizer, team_name, threshold, full_dataset, precomputed_scores, device)
        
        # Normalize team name to snake_case to match folder-based lookup in 03_train_dapo.py
        normalized_name = team_name.lower().replace("-", "_")
        save_path = output_dir / f"sft_warmup_{normalized_name}"
        model.save_pretrained(str(save_path))
        logger.info(f"Saved SFT warm-up checkpoint to {save_path}")
    
    logger.info("\n" + "="*50 + "\nAll SFT warm-ups complete!\n" + "="*50)


if __name__ == "__main__":
    main()
