import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.rewards import DAPORewardScales

logger = logging.getLogger(__name__)

class SGLangBridge:
    """Handles communication with the SGLang backend for fast rollouts."""
    def __init__(self, port=30000):
        self.base_url = f"http://localhost:{port}"
        
    def generate(self, prompts: List[str], lora_path: str, n=8, max_tokens=256) -> List[List[str]]:
        """Generates N completions per prompt using SGLang's API."""
        # Note: In a real SGLang production setup we'd use /generate with lora_path 
        # For simplicity in this script, we assume a single prompt batched to N
        # If we have multiple prompts in a batch (e.g., batch_size=4), we send 4x8 requests.
        
        url = f"{self.base_url}/generate"
        results = []
        
        for p in prompts:
            payload = {
                "text": p,
                "sampling_params": {
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "max_new_tokens": max_tokens,
                    "n": n
                },
                "lora_path": lora_path
            }
            try:
                resp = requests.post(url, json=payload).json()
                if "text" in resp: # List if n > 1
                    completions = resp["text"] if isinstance(resp["text"], list) else [resp["text"]]
                    # If the API returned fewer than N (e.g. single string for n=1), pad or handle
                    if len(completions) != n:
                        # Fallback parsing depending on sglang version
                        completions = [c["text"] for c in resp.get("choices", [{"text": ""}])] * n
                    results.append(completions[:n])
                else:
                    results.append([""] * n)
            except Exception as e:
                logger.error(f"SGLang generation failed: {e}")
                results.append([""] * n)
                
        return results

    def reload_lora(self, lora_name: str, lora_path: str):
        """Instructs SGLang to dynamically reload the LoRA adapter."""
        # Depending on the SGLang version, dynamic LoRA via HTTP is typically done via 
        # sending the lora_path in the generate request, which handles caching automatically.
        pass


class DAPOTrainer:
    def __init__(self, config: dict, rm_model, rm_tokenizer):
        self.config = config["dapo"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Base Model & Tokenizer for PyTorch Gradients
        self.model_name = self.config["model_name"]
        
        logger.info(f"Loading base model {self.model_name} for DAPO PyTorch training...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # We load reference model (frozen) to compute base logprobs
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.ref_model.eval()
        
        # We load a trainable model (base + LoRA)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=self.config["lora_r"],
            lora_alpha=self.config["lora_alpha"],
            target_modules=self.config["lora_target_modules"],
            lora_dropout=self.config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Rewards & SGLang Bridge
        self.reward_scales = DAPORewardScales(rm_model, rm_tokenizer, device=self.device)
        self.sglang = SGLangBridge(port=self.config.get("sglang_port", 30000))
        
        # GRPO Params
        self.group_size = self.config["group_size"]  # N=8
        self.clip_ratio = self.config["clip_ratio_high"]
        
    def _get_logprobs(self, model, input_ids, mask=None):
        """Helper to get token-level log probabilities for the generated response."""
        outputs = model(input_ids, attention_mask=mask)
        logits = outputs.logits[:, :-1, :]  # shift
        labels = input_ids[:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather the log prob of the actual token
        token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)
        return token_log_probs

    def train_team(self, team_name: str, train_dataset: list):
        """Trains the DAPO model for a single team."""
        logger.info(f"\n{'='*50}\nStarting DAPO training for team: {team_name}\n{'='*50}")
        
        # Re-initialize specific LoRA for this team (reset weights)
        for name, param in self.model.named_parameters():
            if "lora" in name:
                torch.nn.init.normal_(param, std=0.01)
                
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        
        num_epochs = self.config.get("num_epochs", 1) # ~20-50 samples, usually 1-2 epochs
        batch_size = self.config.get("batch_size", 4)
        
        lora_sync_dir = f"/tmp/lora_dapo_{team_name}"
        os.makedirs(lora_sync_dir, exist_ok=True)
        
        global_step = 0
        total_steps = (len(train_dataset) // batch_size) * num_epochs
        
        self.model.train()
        
        for epoch in range(num_epochs):
            # Shuffle
            np.random.shuffle(train_dataset)
            
            for i in tqdm(range(0, len(train_dataset), batch_size), desc=f"Team {team_name} Epoch {epoch+1}"):
                batch = train_dataset[i:i+batch_size]
                prompts = [item["prompt"] for item in batch]
                diffs = [item["diff"] for item in batch]
                comments = [item["comment"] for item in batch]
                labels = [item["label"] for item in batch]
                
                # 1. Sync LoRA weights to disk for SGLang
                self.model.save_pretrained(lora_sync_dir)
                
                # 2. Rollout 8 samples per prompt using SGLang
                # completions_grouped = List of length batch_size, where each element is a list of N strings
                completions_grouped = self.sglang.generate(
                    prompts=prompts,
                    lora_path=lora_sync_dir,
                    n=self.group_size,
                    max_tokens=self.config.get("max_new_tokens", 256)
                )
                
                # Flatten for processing
                flat_prompts = []
                flat_completions = []
                flat_diffs = []
                flat_comments = []
                flat_labels = []
                
                for b_idx in range(len(batch)):
                    group_comps = completions_grouped[b_idx]
                    flat_prompts.extend([prompts[b_idx]] * self.group_size)
                    flat_completions.extend(group_comps)
                    flat_diffs.extend([diffs[b_idx]] * self.group_size)
                    flat_comments.extend([comments[b_idx]] * self.group_size)
                    flat_labels.extend([labels[b_idx]] * self.group_size)
                
                # 3. Compute Rewards
                rewards, logs = self.reward_scales.compute_total_reward(
                    flat_completions, flat_diffs, flat_comments, flat_labels
                )
                
                # Format rewards into groups and calculate Advantage
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                rewards_grouped = rewards_tensor.view(-1, self.group_size)
                
                # A_i = (R_i - mean(R)) / (std(R) + eps)
                mean_grouped = rewards_grouped.mean(dim=1, keepdim=True)
                std_grouped = rewards_grouped.std(dim=1, keepdim=True)
                advantages = (rewards_grouped - mean_grouped) / (std_grouped + 1e-8)
                flat_advantages = advantages.view(-1)
                
                # 4. Compute GRPO Loss & Update
                # Convert prompts + completions to tensors
                full_texts = [p + c for p, c in zip(flat_prompts, flat_completions)]
                
                inputs = self.tokenizer(full_texts, padding=True, truncation=True, max_length=1536, return_tensors="pt").to(self.device)
                prompt_inputs = self.tokenizer(flat_prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(self.device)
                
                # Mask out prompt tokens so loss is only on generated tokens
                prompt_lens = prompt_inputs.attention_mask.sum(dim=1) - 1 # approximate index
                
                with torch.no_grad():
                    ref_log_probs = self._get_logprobs(self.ref_model, inputs.input_ids, inputs.attention_mask)
                    
                curr_log_probs = self._get_logprobs(self.model, inputs.input_ids, inputs.attention_mask)
                
                # Calculate ratio and clip per token
                loss_total = 0.0
                valid_tokens = 0
                
                for idx in range(len(flat_advantages)):
                    start_idx = prompt_lens[idx]
                    end_idx = inputs.attention_mask[idx].sum() - 1
                    
                    if start_idx >= end_idx:
                        continue 
                        
                    curr_logp = curr_log_probs[idx, start_idx:end_idx]
                    ref_logp = ref_log_probs[idx, start_idx:end_idx]
                    
                    ratio = torch.exp(curr_logp - ref_logp)
                    adv = flat_advantages[idx]
                    
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
                    
                    # Token-level GRPO loss
                    token_loss = -torch.min(surr1, surr2).mean()
                    loss_total += token_loss
                    valid_tokens += 1
                
                if valid_tokens > 0:
                    loss_total = loss_total / valid_tokens
                    loss_total.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                
                global_step += 1
                
                if global_step % 5 == 0:
                    logger.info(
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {loss_total.item():.4f} | "
                        f"Avg R1(RM): {logs['r1']:.2f} | Avg R2(Match): {logs['r2']:.2f} | "
                        f"Total R: {logs['total_reward']:.2f} | Format OK: {logs['valid_format_ratio']*100:.0f}%"
                    )
        
        # Save final team LoRA
        final_out = Path(self.config["output_dir"]) / f"dapo_lora_{team_name}"
        final_out.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(final_out))
        logger.info(f"Team {team_name} training complete. Saved to {final_out}")
        return final_out
