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

    def load_lora(self, lora_name: str, lora_path: str):
        """Dynamically load/reload a LoRA adapter into SGLang."""
        resp = requests.post(f"{self.base_url}/load_lora_adapter", json={
            "lora_name": lora_name,
            "lora_path": lora_path,
        })
        if resp.status_code != 200:
            logger.error(f"Failed to load LoRA: {resp.text}")
        return resp.json()

    def unload_lora(self, lora_name: str):
        """Unload a LoRA adapter from SGLang."""
        resp = requests.post(f"{self.base_url}/unload_lora_adapter", json={
            "lora_name": lora_name,
        })
        return resp.json()

    def generate(self, prompts: List[str], lora_path: str, n=8, max_tokens=256, tokenizer=None) -> List[List[str]]:
        """Generates N completions per prompt using SGLang's API."""
        url = f"{self.base_url}/generate"
        results = []
        
        for p in prompts:
            # Format prompt with ChatML template
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
                    "temperature": 1.0, # High temp to encourage exploration
                    "top_p": 0.95,
                    "n": n,             # Natively sample N diverse completions!
                    "max_new_tokens": max_tokens
                },
                "lora_path": lora_path
            }
            
            try:
                resp = requests.post(url, json=payload).json()
                
                # In native SGLang, passing n=16 returns 'text' as a list of strings
                completions = resp.get("text", [])
                
                if not isinstance(completions, list):
                    completions = [completions]
                    
                # Pad with empty strings if it failed to return exactly N
                while len(completions) < n:
                    completions.append("")
                
                # Just log the first prompt's first completion for debugging
                if len(results) == 0:
                    logger.info(f"SGLang raw response keys: {list(resp.keys())}, snippet: {str(resp)[:300]}")
                    
                results.append(completions[:n])
                
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
    def __init__(self, config: dict):
        self.config = config["dapo"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load precomputed scores
        import json
        from pathlib import Path
        import os
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        precomputed_scores_path = PROJECT_ROOT / self.config.get("precomputed_scores_path", "data/precomputed_rm_scores.json")
        self.precomputed_scores = {}
        if precomputed_scores_path.exists():
            with open(precomputed_scores_path, "r") as f:
                self.precomputed_scores = json.load(f)
        else:
            logger.warning("Precomputed scores file not found!")
            
        unlabeled_path = PROJECT_ROOT / self.config.get("unlabeled_data_path", "data/unlabeled_pairs.json")
        self.unlabeled_dataset = []
        if unlabeled_path.exists():
            with open(unlabeled_path, "r") as f:
                for line in f:
                    self.unlabeled_dataset.append(json.loads(line))
        else:
            logger.warning("Unlabeled dataset not found!")
        
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
        self.reward_scales = DAPORewardScales(tokenizer=self.tokenizer, precomputed_scores=self.precomputed_scores, device=self.device, config=self.config)
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
        
        # Load SFT warm-up weights into BOTH the training model AND the reference model
        # KL penalty against SFT ref prevents collapse toward all-FILTER
        # because drifting from the 50/50 SFT prior incurs a KL cost
        sft_warmup_path = Path("checkpoints/sft_warmup") / f"sft_warmup_{team_name}"
        if sft_warmup_path.exists():
            logger.info(f"Loading SFT warm-up weights from {sft_warmup_path}")
            import safetensors.torch
            sft_state = safetensors.torch.load_file(str(sft_warmup_path / "adapter_model.safetensors"))
            
            # Load into training model
            model_state = self.model.state_dict()
            for key in sft_state:
                if key in model_state:
                    model_state[key].copy_(sft_state[key])
            logger.info(f"SFT warm-up weights loaded into training model for {team_name}")
            
            # Load SFT as reference model (apply LoRA to ref_model temporarily)
            from peft import PeftModel
            ref_lora = PeftModel.from_pretrained(
                self.ref_model, 
                str(sft_warmup_path),
                is_trainable=False
            )
            ref_lora.eval()
            self._sft_ref_model = ref_lora
            logger.info(f"SFT warm-up loaded as KL reference model for {team_name}")
        else:
            logger.info(f"No SFT warm-up found at {sft_warmup_path}, using random LoRA init")
            for name, param in self.model.named_parameters():
                if "lora" in name:
                    torch.nn.init.normal_(param, std=0.01)
            self._sft_ref_model = None
                
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
            
            import hashlib
            for i in tqdm(range(0, len(train_dataset), batch_size), desc=f"Team {team_name} Epoch {epoch+1}"):
                labeled_batch = train_dataset[i:i+batch_size//2]
                for item in labeled_batch:
                    item["has_label"] = True
                    
                import random
                unlabeled_batch = []
                if len(self.unlabeled_dataset) > 0:
                    samples_needed = batch_size - len(labeled_batch)
                    unlabeled_samples = random.sample(self.unlabeled_dataset, min(samples_needed, len(self.unlabeled_dataset)))
                    for item in unlabeled_samples:
                        # Generate prompt for unlabeled 
                        if "prompt" not in item:
                            from src.data.team_dataset import generate_prompt
                            item["prompt"] = generate_prompt(item["diff"], item["comment"], team_name)
                        item["has_label"] = False
                        item["label"] = 0 # Dummy
                        unlabeled_batch.append(item)
                        
                batch = labeled_batch + unlabeled_batch
                
                prompts = [item["prompt"] for item in batch]
                diffs = [item["diff"] for item in batch]
                comments = [item["comment"] for item in batch]
                labels = [item["label"] for item in batch]
                has_label = [item["has_label"] for item in batch]
                # Default to MD5 hash if example_id is missing to securely join with precomputed scores
                example_ids = [item.get("example_id", hashlib.md5(f"{d}_{c}".encode('utf-8')).hexdigest()) for d, c in zip(diffs, comments, strict=False)]
                
                # 1. Sync LoRA weights to disk for SGLang
                self.model.save_pretrained(lora_sync_dir)
                
                # 2. Dynamically load into SGLang (or reload with updated weights)
                try:
                    self.sglang.unload_lora("active_lora")
                except Exception:
                    pass  # First time, nothing to unload
                self.sglang.load_lora("active_lora", str(lora_sync_dir))
                
                # 3. Rollout 16 samples per prompt using SGLang (oversampling)
                # completions_grouped = List of length batch_size, where each element is a list of N strings
                oversample_size = self.group_size * 2
                completions_grouped = self.sglang.generate(
                    prompts=prompts,
                    lora_path="active_lora",
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
                flat_example_ids = []
                flat_has_label = []
                
                for b_idx in range(len(batch)):
                    group_comps = completions_grouped[b_idx]
                    flat_prompts.extend([prompts[b_idx]] * oversample_size)
                    flat_completions.extend(group_comps)
                    flat_diffs.extend([diffs[b_idx]] * oversample_size)
                    flat_comments.extend([comments[b_idx]] * oversample_size)
                    flat_labels.extend([labels[b_idx]] * oversample_size)
                    flat_example_ids.extend([example_ids[b_idx]] * oversample_size)
                    flat_has_label.extend([has_label[b_idx]] * oversample_size)
                
                # 3. Compute Rewards
                rewards, logs = self.reward_scales.compute_total_reward(
                    flat_completions, flat_diffs, flat_comments, flat_labels, flat_example_ids, flat_has_label, self.config, flat_prompts
                )
                
                # GDPO: Per-Reward Normalization (normalize each R1-R6 independently within each group)
                # Then sum the normalized advantages. This prevents reward signal collapse.
                w_r1, w_r2, w_r3, w_r4, w_r5 = logs["weights"]
                
                r1_tensor = torch.tensor(logs["r1_raw"], dtype=torch.float32, device=self.device).view(-1, oversample_size)
                r2_tensor = torch.tensor(logs["r2_raw"], dtype=torch.float32, device=self.device).view(-1, oversample_size)
                r3_tensor = torch.tensor(logs["r3_raw"], dtype=torch.float32, device=self.device).view(-1, oversample_size)
                r4_tensor = torch.tensor(logs["r4_raw"], dtype=torch.float32, device=self.device).view(-1, oversample_size)
                r5_tensor = torch.tensor(logs["r5_raw"], dtype=torch.float32, device=self.device).view(-1, oversample_size)
                
                def normalize_within_group(t):
                    """Normalize each row (group) independently: (x - mean) / (std + eps)"""
                    m = t.mean(dim=1, keepdim=True)
                    s = t.std(dim=1, keepdim=True)
                    return (t - m) / (s + 1e-8)
                
                # Per-component normalization as requested by user
                adv_r1 = normalize_within_group(r1_tensor) * w_r1
                adv_r2 = normalize_within_group(r2_tensor) * w_r2
                adv_r3 = normalize_within_group(r3_tensor) * w_r3
                adv_r4 = normalize_within_group(r4_tensor) * w_r4
                adv_r5 = normalize_within_group(r5_tensor) * w_r5
                
                advantages = adv_r1 + adv_r2 + adv_r3 + adv_r4 + adv_r5
                
                # Batch-wise normalization for final scaling stability
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Dynamic sampling: replace the top/bottom 25% filtering with true DAPO variance check
                valid_groups = []
                for g_idx in range(advantages.size(0)):
                    # Check if total advantage has zero variance in this group
                    if advantages[g_idx].std() < 1e-5:
                        valid_groups.append(False)
                    else:
                        valid_groups.append(True)
                        
                if sum(valid_groups) == 0:
                    continue
                    
                flat_prompts_filtered = []
                flat_completions_filtered = []
                flat_advantages_filtered = []
                
                for b_idx in range(advantages.size(0)):
                    if not valid_groups[b_idx]:
                        continue
                    group_advs = advantages[b_idx]
                    for idx in range(oversample_size):
                        flat_idx = b_idx * oversample_size + idx
                        flat_prompts_filtered.append(flat_prompts[flat_idx])
                        flat_completions_filtered.append(flat_completions[flat_idx])
                        flat_advantages_filtered.append(group_advs[idx].item())
                
                flat_prompts = flat_prompts_filtered
                flat_completions = flat_completions_filtered
                flat_advantages = torch.tensor(flat_advantages_filtered, dtype=torch.float32, device=self.device)
                
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
                        # Use SFT ref model for KL if available (prevents FILTER collapse)
                        ref_model = self._sft_ref_model if self._sft_ref_model is not None else self.ref_model
                        ref_log_probs = self._get_logprobs(ref_model, inputs.input_ids, inputs.attention_mask)
                        
                    curr_log_probs = self._get_logprobs(self.model, inputs.input_ids, inputs.attention_mask)
                    
                    # Calculate ratio and clip per token
                    mb_loss = 0.0
                    mb_valid_tokens = 0
                    kl_penalty_weight = self.config.get("kl_penalty", 0.10)
                    entropy_bonus_weight = self.config.get("entropy_bonus", 0.03)
                    
                    # Get full logits for entropy computation
                    outputs = self.model(inputs.input_ids, attention_mask=inputs.attention_mask)
                    logits = outputs.logits[:, :-1, :]  # shift for next-token prediction
                    
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
                        ratio_clipped = torch.where(
                            adv > 0,
                            torch.clamp(ratio, max=1.0 + self.clip_ratio_high),
                            torch.clamp(ratio, min=1.0 - self.clip_ratio_low)
                        )
                        surr2 = ratio_clipped * adv
                        
                        # KL divergence estimator
                        kl = torch.exp(ref_logp - curr_logp) - (ref_logp - curr_logp) - 1.0
                        
                        # Token-level entropy of the policy's distribution
                        # H(π) = -Σ p(token) * log(p(token)) at each generated position
                        gen_logits = logits[idx, start_idx:end_idx, :]
                        gen_probs = F.softmax(gen_logits, dim=-1)
                        gen_log_probs = F.log_softmax(gen_logits, dim=-1)
                        token_entropy = -(gen_probs * gen_log_probs).sum(dim=-1)  # H per position
                        
                        # Loss = -min(surr1, surr2) + kl_weight*KL - β*H(π)
                        # Subtracting entropy bonus ENCOURAGES higher entropy (more exploration)
                        token_loss = -torch.min(surr1, surr2).mean() + kl_penalty_weight * kl.mean() - entropy_bonus_weight * token_entropy.mean()
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
