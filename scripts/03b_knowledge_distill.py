"""
Phase 3: Knowledge Distillation (KD).

Distills per-team DAPO LoRA adapters into a single unified LoRA adapter.
The student model learns from the teacher (per-team LoRAs) by minimizing
KL divergence on each team's training data.

Usage:
    # Distill all teams
    python scripts/03b_knowledge_distill.py

    # Distill specific teams
    python scripts/03b_knowledge_distill.py --teams pragmatic_shippers thorough_mentors
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S", force=True)
logger = logging.getLogger("kd_main")


def distill(config, teams, project_root):
    """Distill per-team DAPO LoRA adapters into a single unified LoRA."""
    dapo_config = config.get("dapo", {})
    model_name = dapo_config.get("model_name", "Qwen/Qwen2.5-Coder-3B-Instruct")
    
    teams_dir = Path(project_root) / "data" / "teams"
    checkpoints_dir = Path(project_root) / "checkpoints" / "dapo"
    output_dir = Path(project_root) / "checkpoints" / "kd_unified"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # 2. Load base model
    logger.info(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None
    )
    
    # 3. Create student LoRA (fresh adapter on base model)
    lora_config = LoraConfig(
        r=dapo_config.get("lora_r", 16),
        lora_alpha=dapo_config.get("lora_alpha", 32),
        target_modules=dapo_config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=dapo_config.get("lora_dropout", 0.05),
        task_type="CAUSAL_LM"
    )
    student_model = get_peft_model(base_model, lora_config)
    student_model.train()
    logger.info(f"Student model created with {sum(p.numel() for p in student_model.parameters() if p.requires_grad):,} trainable params")
    
    # 4. Load all teacher LoRAs and their training data
    teachers = {}
    for team_name in teams:
        lora_path = checkpoints_dir / f"dapo_lora_{team_name}"
        train_path = teams_dir / team_name / "train.jsonl"
        
        if not lora_path.exists():
            logger.warning(f"No DAPO LoRA found for {team_name} at {lora_path}, skipping.")
            continue
        if not train_path.exists():
            logger.warning(f"No training data for {team_name}, skipping.")
            continue
            
        # Load training data
        with open(train_path) as f:
            train_data = [json.loads(line) for line in f]
        
        teachers[team_name] = {
            "lora_path": str(lora_path),
            "train_data": train_data,
        }
        logger.info(f"Teacher {team_name}: {len(train_data)} train samples, LoRA at {lora_path}")
    
    if not teachers:
        logger.error("No valid teacher LoRAs found. Run DAPO training first.")
        sys.exit(1)
    
    # 5. Pre-generate teacher logits for each team
    logger.info("Generating teacher logits...")
    teacher_cache = {}
    
    # Load a separate teacher model instance
    teacher_base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None
    )
    teacher_peft = None
    
    for team_name, info in teachers.items():
        logger.info(f"Generating teacher logits for {team_name}...")
        
        # Load teacher LoRA
        if teacher_peft is None:
            teacher_peft = PeftModel.from_pretrained(teacher_base, info["lora_path"], adapter_name=team_name)
        else:
            teacher_peft.load_adapter(info["lora_path"], adapter_name=team_name)
        teacher_peft.set_adapter(team_name)
        teacher_peft.eval()
        
        team_logits = []
        team_inputs = []
        
        with torch.no_grad():
            for item in tqdm(info["train_data"], desc=f"Teacher {team_name}"):
                messages = [
                    {"role": "system", "content": "You are a helpful AI code reviewer."},
                    {"role": "user", "content": item["prompt"]}
                ]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(formatted, truncation=True, max_length=1536, return_tensors="pt").to(device)
                
                outputs = teacher_peft(**inputs)
                # Store the logits for the last position (next-token prediction)
                # We keep the full sequence logits for KL on all positions
                team_logits.append(outputs.logits.cpu())
                team_inputs.append(inputs["input_ids"].cpu())
        
        teacher_cache[team_name] = {
            "logits": team_logits,
            "input_ids": team_inputs,
        }
    
    # Free teacher memory
    del teacher_peft, teacher_base
    torch.cuda.empty_cache()
    
    # 6. Distillation training
    kd_lr = config.get("kd", {}).get("learning_rate", 5e-6)
    kd_epochs = config.get("kd", {}).get("epochs", 3)
    kd_temperature = config.get("kd", {}).get("temperature", 2.0)
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=kd_lr)
    
    logger.info(f"Starting KD: {kd_epochs} epochs, lr={kd_lr}, temperature={kd_temperature}")
    
    for epoch in range(kd_epochs):
        total_loss = 0.0
        num_steps = 0
        
        for team_name, cache in teacher_cache.items():
            for idx in tqdm(range(len(cache["logits"])), desc=f"Epoch {epoch+1} - {team_name}"):
                input_ids = cache["input_ids"][idx].to(device)
                teacher_logits = cache["logits"][idx].to(device)
                
                # Forward pass through student
                student_outputs = student_model(input_ids=input_ids)
                student_logits = student_outputs.logits
                
                # KL divergence loss with temperature scaling
                T = kd_temperature
                teacher_probs = F.log_softmax(teacher_logits / T, dim=-1)
                student_log_probs = F.log_softmax(student_logits / T, dim=-1)
                
                # KL(teacher || student)
                loss = F.kl_div(student_log_probs, teacher_probs, log_target=True, reduction="batchmean") * (T * T)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                num_steps += 1
        
        avg_loss = total_loss / max(1, num_steps)
        logger.info(f"Epoch {epoch+1}/{kd_epochs}: Avg KD Loss = {avg_loss:.6f}")
    
    # 7. Save unified student LoRA
    os.makedirs(output_dir, exist_ok=True)
    student_model.save_pretrained(str(output_dir))
    logger.info(f"Unified KD LoRA saved to {output_dir}")
    
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="RLCR v2: Phase 3 Knowledge Distillation")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--teams", nargs="*", default=None, help="Teams to distill (default: all). E.g. --teams pragmatic_shippers")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Discover teams
    teams_dir = project_root / "data" / "teams"
    all_teams = sorted([d.name for d in teams_dir.iterdir() if d.is_dir()])
    
    if args.teams:
        teams = [t for t in args.teams if t in all_teams]
        missing = [t for t in args.teams if t not in all_teams]
        if missing:
            logger.warning(f"Teams not found: {missing}. Available: {all_teams}")
    else:
        teams = all_teams
    
    logger.info(f"Distilling {len(teams)} teams: {', '.join(teams)}")
    
    distill(config, teams, project_root)


if __name__ == "__main__":
    main()
