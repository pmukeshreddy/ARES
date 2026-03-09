"""
Phase 6: Standard Teacher-to-Student Knowledge Distillation.

This script extracts the complex reasoning/formatting capabilities from a massive
DAPO-trained model (14B Teacher) and logically transfers it to a smaller, faster
model (3B Student) using synthetic data generation.

Steps:
1. Boot the Teacher Model (14B) in SGLang.
2. Feed it entirely unlabeled Python code diffs (from GitHub).
3. Record the Teacher's "Gold Standard" <think>...</think> + SURFACE/FILTER outputs.
4. Shut down the Teacher, boot PyTorch.
5. Perform Supervised Fine-Tuning (SFT) on the Student Model (3B) using the Teacher's generated outputs.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import time
from pathlib import Path
from tqdm import tqdm

import yaml
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SGLANG_PORT = 30006
SGLANG_URL = f"http://localhost:{SGLANG_PORT}"

def kill_existing_sglang():
    logger.info("Killing any existing SGLang processes on our port...")
    subprocess.run(f"fuser -k {SGLANG_PORT}/tcp", shell=True, capture_output=True)
    time.sleep(3)

def start_sglang_teacher(teacher_model_path: str, lora_path: str = None):
    """Boot the 14B Teacher in SGLang."""
    kill_existing_sglang()
    
    logger.info(f"Starting Teacher Model ({teacher_model_path}) on port {SGLANG_PORT}...")
    sglang_python = "/opt/sglang_venv/bin/python3"
    
    cmd = [
        sglang_python, "-m", "sglang.launch_server",
        "--model-path", teacher_model_path,
        "--port", str(SGLANG_PORT),
        "--trust-remote-code",
        "--mem-fraction-static", "0.7", # Max memory since we don't have PT loaded yet
        "--dtype", "bfloat16"
    ]
    
    # If the teacher was DAPO trained, attach the LoRA adapters
    if lora_path and os.path.exists(lora_path):
        cmd.extend([
            "--enable-lora",
            "--max-lora-rank", "16",
            "--max-loras-per-batch", "1",
            "--lora-target-modules", "q_proj", "k_proj", "v_proj", "o_proj"
        ])
    
    log_file = open("sglang_teacher.log", "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, text=True)
    
    logger.info("Waiting for SGLang Teacher to boot (up to 120s)...")
    for _ in range(60):
        try:
            resp = requests.get(f"{SGLANG_URL}/health", timeout=2)
            if resp.status_code == 200:
                logger.info("Teacher is fully booted and ready!")
                return process
        except Exception:
            pass
        time.sleep(2)
    
    logger.error("Teacher failed to start within 120s.")
    process.terminate()
    sys.exit(1)

def generate_teacher_data(unlabeled_file: str, tokenizer, max_samples: int = 1000, lora_name: str = None):
    """Feed unlabeled prompts to the Teacher to generate 'Gold Standard' synthetic responses."""
    logger.info("Loading Unlabeled Diff Dataset...")
    
    if not os.path.exists(unlabeled_file):
        logger.error(f"Cannot find unlabeled dataset at {unlabeled_file}")
        sys.exit(1)
        
    dataset = []
    with open(unlabeled_file, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
            
    # Sample a manageable chunk for Distillation limits
    import random
    random.seed(42)
    random.shuffle(dataset)
    dataset = dataset[:max_samples]
    
    logger.info(f"Teacher evaluating {len(dataset)} unlabeled code diffs...")
    
    url = f"{SGLANG_URL}/generate"
    synthetic_data = []
    batch_size = 8
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Teacher Generating Synthetic Knowledge"):
        batch_prompts = [item["prompt"] for item in dataset[i:i+batch_size]]
        formatted_prompts = []
        
        for p in batch_prompts:
            messages = [
                {"role": "system", "content": "You are a helpful AI code reviewer."},
                {"role": "user", "content": p}
            ]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted)
            
        payload = {
            "text": formatted_prompts,
            "sampling_params": {
                "temperature": 0.2, # Low temp for high-confidence teacher data
                "max_new_tokens": 512
            }
        }
        
        if lora_name:
            payload["lora_name"] = lora_name
            
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp_json = resp.json()
            
            for j, response_obj in enumerate(resp_json):
                text = response_obj.get("text", "")
                
                # Only keep structurally valid teacher thoughts
                if "<think>" in text and ("SURFACE" in text or "FILTER" in text):
                    synthetic_data.append({
                        "prompt": batch_prompts[j],
                        "comment": text,
                        "label": 1 if "SURFACE" in text else 0 # Best guess mapping
                    })
        except Exception as e:
            logger.error(f"Teacher Generation Error: {e}")
            
    logger.info(f"Successfully generated {len(synthetic_data)} Gold Standard knowledge samples.")
    
    # Save the synthetic data cache
    cache_path = PROJECT_ROOT / "data" / "teacher_synthetic_cache.jsonl"
    with open(cache_path, "w") as f:
        for item in synthetic_data:
            f.write(json.dumps(item) + "\n")
            
    return synthetic_data

def format_prompt(sample, tokenizer):
    """Format the student SFT training prompts."""
    sys_prompt = "You are a helpful AI code reviewer. Please review the following code diff and provide a constructive comment."
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": sample["prompt"]},
        {"role": "assistant", "content": sample["comment"]}
    ]
    
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

def train_student_model(student_model_name: str, synthetic_data: list, output_dir: str):
    """Run PyTorch SFT to physically embed the Teacher's knowledge into the Student's weights."""
    logger.info(f"\n{'='*50}\nBooting PyTorch for STUDENT Knowledge Distillation\n{'='*50}")
    
    tokenizer = AutoTokenizer.from_pretrained(student_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    logger.info(f"Loading Student Model ({student_model_name}) into VRAM...")
    model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    student_model = get_peft_model(model, lora_config)
    student_model.train()
    
    logger.info("Formatting Student Dataset...")
    raw_dataset = Dataset.from_list(synthetic_data)
    train_dataset = raw_dataset.map(lambda x: format_prompt(x, tokenizer), remove_columns=raw_dataset.column_names)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5.0e-5,
        logging_steps=10,
        warmup_ratio=0.05,
        bf16=True,
        save_strategy="no",
        remove_unused_columns=False
    )
    
    trainer = SFTTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False
    )
    
    logger.info("Initiating Neurological Knowledge Transfer (SFT)...")
    trainer.train()
    
    logger.info(f"Saving new, highly-capable Student LoRA to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

def main():
    parser = argparse.ArgumentParser(description="RLCR: Big Model -> Small Model Knowledge Distillation")
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-Coder-14B-Instruct", help="Massive Model")
    parser.add_argument("--student", default="Qwen/Qwen2.5-Coder-3B-Instruct", help="Tiny Fast Model")
    parser.add_argument("--unlabeled-data", default="data/teams/pragmatic_shippers/train.jsonl", help="Raw coding data diffs")
    parser.add_argument("--teacher-lora", default=None, help="The heavily optimized 14B DAPO LoRA weights")
    args = parser.parse_args()
    
    teacher_lora_path = None
    if args.teacher_lora:
        teacher_lora_path = str(PROJECT_ROOT / "checkpoints" / "dapo" / args.teacher_lora)
        
    sglang_process = None
    synthetic_data = None
    
    # 1. Start the Massive Teacher in SGLang
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    sglang_process = start_sglang_teacher(args.teacher, teacher_lora_path)
    
    try:
        if args.teacher_lora:
            # We must load the teacher LoRA into the running SGLang Engine natively via API
            url = f"{SGLANG_URL}/load_peft_adapter"
            requests.post(url, json={"peft_path": teacher_lora_path, "adapter_name": "teacher_lora"})
            logger.info("Teacher LoRA injected into SGLang.")
            
        # 2. Extract Knowledge into a synthetic dataset
        synthetic_data = generate_teacher_data(
            unlabeled_file=str(PROJECT_ROOT / args.unlabeled_data),
            tokenizer=tokenizer,
            lora_name="teacher_lora" if args.teacher_lora else None
        )
        
    finally:
        logger.info("Teacher's job is done. Shutting down 14B SGLang server to free VRAM for PyTorch...")
        if sglang_process:
            sglang_process.terminate()
            sglang_process.wait(timeout=10)
    
    # 3. Force-feed the Teacher's knowledge into the 3B Student
    if synthetic_data and len(synthetic_data) > 0:
        output_dir = str(PROJECT_ROOT / "checkpoints" / "distilled_student_3B")
        train_student_model(args.student, synthetic_data, output_dir)
        logger.info("\nSUCCESS: Small Model Distillation Complete. The 3B model is now incredibly smart.")
    else:
        logger.error("Failed to generate synthetic knowledge data from the 14B Teacher.")

if __name__ == "__main__":
    main()
