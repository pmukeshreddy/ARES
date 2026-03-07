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
        
    def generate(self, prompts: List[str], lora_path: str, n=8, max_tokens=256, tokenizer=None) -> List[List[str]]:
        """Generates N completions per prompt using SGLang's API."""
        url = f"{self.base_url}/generate"
        results = []
        
        for p in prompts:
            # Format prompt with ChatML template so the Instruct model understands it
            if tokenizer is not None:
                messages = [
                    {"role": "system", "content": "You are a helpful AI code reviewer."},
                    {"role": "user", "content": p}
                ]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted_prompt = f"<|im_start|>system\nYou are a helpful AI code reviewer.<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
                
            payload = {
                "text": formatted_prompt,
                "sampling_params": {
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "max_new_tokens": max_tokens
                    # "n" is ignored by sglang /generate, we loop manually
                },
                "lora_path": lora_path
            }
            
            prompt_completions = []
            for _ in range(n):
                try:
                    resp = requests.post(url, json=payload).json()
                    completion = resp.get("text", "")
                    if isinstance(completion, list): # just in case
                        completion = completion[0] if len(completion) > 0 else ""
                    prompt_completions.append(completion)
                except Exception as e:
                    logger.error(f"SGLang generation failed: {e}")
                    prompt_completions.append("")
                    
            results.append(prompt_completions)
                
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
        ).to(self.device)
        self.ref_model.eval()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        # Turn on Gradient Checkpointing to save massive VRAM during backward passes
        self.model.gradient_checkpointing_enable({"use_reentrant": False})
        
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
        self.reward_scales = DAPORewardScales(rm_model, rm_tokenizer, device=self.device, config=self.config)
        self.sglang = SGLangBridge(port=self.config.get("sglang_port", 30000))
        
        # GRPO / DAPO Params
        self.group_size = self.config["group_size"]  # N=8
        self.clip_ratio_low = self.config.get("clip_ratio_low", 0.2)
        self.clip_ratio_high = self.config.get("clip_ratio_high", 0.28)
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
                
                # 2. Rollout 16 samples per prompt using SGLang (oversampling)
                # completions_grouped = List of length batch_size, where each element is a list of N strings
                oversample_size = self.group_size * 2
                completions_grouped = self.sglang.generate(
                    prompts=prompts,
                    lora_path=lora_sync_dir,
                    n=oversample_size,
                    max_tokens=self.config.get("max_new_tokens", 256),
                    tokenizer=self.tokenizer
                )
                
                # Flatten for processing
                flat_prompts = []
                flat_completions = []
                flat_diffs = []
                flat_comments = []
                flat_labels = []
                
                for b_idx in range(len(batch)):
                    group_comps = completions_grouped[b_idx]
                    flat_prompts.extend([prompts[b_idx]] * oversample_size)
                    flat_completions.extend(group_comps)
                    flat_diffs.extend([diffs[b_idx]] * oversample_size)
                    flat_comments.extend([comments[b_idx]] * oversample_size)
                    flat_labels.extend([labels[b_idx]] * oversample_size)
                
                # 3. Compute Rewards
                rewards, logs = self.reward_scales.compute_total_reward(
                    flat_completions, flat_diffs, flat_comments, flat_labels, self.config
                )
                
                # Format rewards into groups and calculate Advantage
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                rewards_grouped = rewards_tensor.view(-1, oversample_size)
                
                # A_i = (R_i - mean(R)) / (std(R) + eps)
                mean_grouped = rewards_grouped.mean(dim=1, keepdim=True)
                std_grouped = rewards_grouped.std(dim=1, keepdim=True)
                advantages = (rewards_grouped - mean_grouped) / (std_grouped + 1e-8)
                
                # 4. Dynamic Resampling: Keep only top 25% and bottom 25% of advantages
                keep_per_group = self.group_size # e.g. 8
                half_keep = keep_per_group // 2
                
                filtered_prompts = []
                filtered_completions = []
                filtered_advantages = []
                
                for b_idx in range(advantages.size(0)):
                    group_advs = advantages[b_idx]
                    
                    # Sort indices by advantage
                    sorted_indices = torch.argsort(group_advs, descending=True)
                    
                    # Pick highest `half_keep` and lowest `half_keep`
                    top_indices = sorted_indices[:half_keep]
                    bottom_indices = sorted_indices[-half_keep:]
                    
                    selected_indices = torch.cat([top_indices, bottom_indices]).cpu().tolist()
                    
                    for idx in selected_indices:
                        flat_idx = b_idx * oversample_size + idx
                        filtered_prompts.append(flat_prompts[flat_idx])
                        filtered_completions.append(flat_completions[flat_idx])
                        filtered_advantages.append(group_advs[idx].item())
                
                flat_prompts = filtered_prompts
                flat_completions = filtered_completions
                flat_advantages = torch.tensor(filtered_advantages, dtype=torch.float32, device=self.device)
                
                # 5. Compute GRPO Loss & Update on the filtered sequences
                # Convert prompts + completions to tensors
                full_texts = [p + c for p, c in zip(flat_prompts, flat_completions)]
                
                # Micro-batching to prevent OOM on 3B model
                micro_batch_size = 2
                num_microbatches = (len(flat_prompts) + micro_batch_size - 1) // micro_batch_size
                
                loss_total_logging = 0.0
                optimizer.zero_grad()
                
                for mb_idx in range(0, len(flat_prompts), micro_batch_size):
                    mb_full_texts = full_texts[mb_idx:mb_idx+micro_batch_size]
                    mb_prompts = flat_prompts[mb_idx:mb_idx+micro_batch_size]
                    mb_adv = flat_advantages[mb_idx:mb_idx+micro_batch_size]
                    
                    inputs = self.tokenizer(mb_full_texts, padding=True, truncation=True, max_length=1536, return_tensors="pt").to(self.device)
                    prompt_inputs = self.tokenizer(mb_prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(self.device)
                    
                    # Mask out prompt tokens so loss is only on generated tokens
                    # The sum of mask gives the exact index where generation starts (since it's 0-indexed)
                    prompt_lens = prompt_inputs.attention_mask.sum(dim=1)
                    
                    with torch.no_grad():
                        ref_log_probs = self._get_logprobs(self.ref_model, inputs.input_ids, inputs.attention_mask)
                        
                    curr_log_probs = self._get_logprobs(self.model, inputs.input_ids, inputs.attention_mask)
                    
                    # Calculate ratio and clip per token
                    mb_loss = 0.0
                    mb_valid_tokens = 0
                    
                    for idx in range(len(mb_adv)):
                        start_idx = prompt_lens[idx]
                        end_idx = inputs.attention_mask[idx].sum() - 1
                        
                        if start_idx >= end_idx:
                            continue 
                            
                        curr_logp = curr_log_probs[idx, start_idx:end_idx]
                        ref_logp = ref_log_probs[idx, start_idx:end_idx]
                        
                        ratio = torch.exp(curr_logp - ref_logp)
                        adv = mb_adv[idx]
                        
                        surr1 = ratio * adv
                        
                        # DAPO Asymmetric Decoupled Clipping
                        # adv > 0 gets upper clipped. adv < 0 gets lower clipped.
                        ratio_clipped = torch.where(
                            adv > 0,
                            torch.clamp(ratio, max=1.0 + self.clip_ratio_high),
                            torch.clamp(ratio, min=1.0 - self.clip_ratio_low)
                        )
                        surr2 = ratio_clipped * adv
                        
                        # Token-level GRPO loss
                        token_loss = -torch.min(surr1, surr2).mean()
                        mb_loss += token_loss
                        mb_valid_tokens += 1
                    
                    if mb_valid_tokens > 0:
                        mb_loss = mb_loss / mb_valid_tokens
                        # Scale for accumulation
                        (mb_loss / num_microbatches).backward()
                        loss_total_logging += mb_loss.item()
                
                # Step after accumulating all micro-batches
                optimizer.step()
                loss_total = torch.tensor(loss_total_logging / num_microbatches)
                
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
