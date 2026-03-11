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
        logger.info(f"Loading LoRA '{lora_name}' from path: {lora_path}")
        resp = requests.post(f"{self.base_url}/load_lora_adapter", json={
            "lora_name": lora_name,
            "lora_path": lora_path,
        })
        logger.info(f"LoRA load response [{resp.status_code}]: {resp.text[:300]}")
        if resp.status_code != 200:
            logger.error(f"Failed to load LoRA: {resp.text}")
        return resp.json()

    def unload_lora(self, lora_name: str):
        """Unload a LoRA adapter from SGLang."""
        resp = requests.post(f"{self.base_url}/unload_lora_adapter", json={
            "lora_name": lora_name,
        })
        return resp.json()

    def generate(self, prompts: List[str], lora_path: str, n=8, max_tokens=256, tokenizer=None, temperature=0.8) -> List[List[str]]:
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
                    "temperature": temperature,
                    "top_p": 0.95,
                    "n": n,             # Natively sample N diverse completions!
                    "max_new_tokens": max_tokens
                },
                "lora_path": lora_path
            }
            
            try:
                if len(results) == 0:
                    logger.info(f"SGLang generate using lora_path='{lora_path}'")
                resp = requests.post(url, json=payload).json()
                
                # Handle both old and new SGLang response formats:
                # Old: {"text": ["completion1", "completion2", ...]}
                # New: [{"text": "completion1"}, {"text": "completion2"}, ...]
                if isinstance(resp, list):
                    completions = [item.get("text", "") if isinstance(item, dict) else str(item) for item in resp]
                elif isinstance(resp, dict):
                    completions = resp.get("text", [])
                    if not isinstance(completions, list):
                        completions = [completions]
                else:
                    completions = []
                    
                # Pad with empty strings if it failed to return exactly N
                while len(completions) < n:
                    completions.append("")
                
                # Just log the first prompt's first completion for debugging
                if len(results) == 0:
                    logger.info(f"SGLang raw response type: {type(resp).__name__}, snippet: {str(resp)[:300]}")
                    
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
            
    # We no longer load a separate reference model. 
    # We will use PEFT's `disable_adapter()` to get reference logprobs to save massive VRAM.
        
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
            self.model.load_adapter(str(sft_warmup_path), "sft_ref")
            logger.info(f"Loaded 'sft_ref' adapter. DAPO will train 'default' adapter.")
            
            # Copy sft_ref weights into default so the default policy starts at the SFT optimal point.
            import safetensors.torch
            sft_state = safetensors.torch.load_file(str(sft_warmup_path / "adapter_model.safetensors"))
            model_state = self.model.state_dict()
            matched = 0
            for sft_key, sft_val in sft_state.items():
                remapped = sft_key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                if remapped in model_state:
                    model_state[remapped].copy_(sft_val)
                    matched += 1
            logger.info(f"SFT warm-up copied to active 'default' adapter. Matched {matched} keys.")
            
        else:
            logger.info(f"No SFT warm-up found at {sft_warmup_path}, using random LoRA init")
            for name, param in self.model.named_parameters():
                if "lora" in name:
                    torch.nn.init.normal_(param, std=0.01)
                    
        self.model.set_adapter("default")
        
        # Compute dataset-level label counts for stable inverse class frequency weighting in R2
        surface_count = sum(1 for item in train_dataset if item.get("label") == 1)
        filter_count = sum(1 for item in train_dataset if item.get("label") == 0)
        self.reward_scales.dataset_label_counts = {"surface": surface_count, "filter": filter_count}
        logger.info(f"Dataset label counts for {team_name}: {surface_count} SURFACE / {filter_count} FILTER (w_surface={( surface_count + filter_count) / (2.0 * max(1, surface_count)):.3f}, w_filter={(surface_count + filter_count) / (2.0 * max(1, filter_count)):.3f})")
                
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        
        max_steps = self.config.get("max_steps", 300)
        lr_schedule = self.config.get("lr_schedule", "cosine")
        num_optimizer_steps = max_steps // self.config.get("grad_accum_steps", 4)
        if lr_schedule == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=max(1, int(num_optimizer_steps * 0.1)),
                num_training_steps=num_optimizer_steps
            )
        else:
            from transformers import get_constant_schedule_with_warmup
            scheduler = get_constant_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=max(1, int(num_optimizer_steps * 0.1))
            )
        batch_size = self.config.get("batch_size", 4)
        
        lora_sync_dir = f"/tmp/lora_dapo_{team_name}"
        os.makedirs(lora_sync_dir, exist_ok=True)
        current_lora_name = None  # Track active LoRA name for SGLang
        lora_sync_interval = self.config.get("lora_sync_interval", 2)
        
        global_step = 0
        best_eval_accuracy = 0.0
        best_checkpoint_path = None
        total_steps = max_steps
        
        self.model.train()
        
        import random
        import hashlib
        
        # ── SFT Baseline Eval ────────────────────────────────
        # Generate completions from the SFT warmup weights to verify
        # balanced starting point before DAPO training begins.
        sft_eval_size = min(20, len(train_dataset))
        sft_eval_batch = random.sample(train_dataset, sft_eval_size)
        sft_eval_prompts = [item["prompt"] for item in sft_eval_batch]
        sft_eval_labels = [item["label"] for item in sft_eval_batch]
        
        # Sync initial SFT weights to SGLang for eval
        sync_path = f"{lora_sync_dir}_step0"
        os.makedirs(sync_path, exist_ok=True)
        self.model.save_pretrained(sync_path)
        sft_lora_name = f"{team_name}_step0"
        self.sglang.load_lora(sft_lora_name, sync_path)
        current_lora_name = sft_lora_name
        
        sft_eval_completions = self.sglang.generate(
            prompts=sft_eval_prompts, lora_path=sft_lora_name,
            n=1, max_tokens=self.config.get("max_new_tokens", 256),
            tokenizer=self.tokenizer,
            temperature=self.config.get("temperature", 0.8)
        )
        
        from src.training.rewards import parse_completion
        sft_decisions = [parse_completion(c[0])["decision"] if c else None for c in sft_eval_completions]
        sft_n_surface = sum(1 for d in sft_decisions if d == "SURFACE")
        sft_n_filter = sum(1 for d in sft_decisions if d == "FILTER")
        sft_n_other = sum(1 for d in sft_decisions if d not in ("SURFACE", "FILTER"))
        gt_surface = sum(1 for l in sft_eval_labels if l == 1)
        gt_filter = sum(1 for l in sft_eval_labels if l == 0)
        
        print(f"\n{'='*60}")
        print(f"SFT BASELINE EVAL ({sft_eval_size} prompts, 1 completion each):")
        print(f"  Ground truth:  {gt_surface} SURFACE / {gt_filter} FILTER")
        print(f"  SFT outputs:   {sft_n_surface} SURFACE ({100*sft_n_surface/sft_eval_size:.0f}%) / "
              f"{sft_n_filter} FILTER ({100*sft_n_filter/sft_eval_size:.0f}%) / {sft_n_other} invalid")
        
        # Show per-sample breakdown
        correct = 0
        for i in range(sft_eval_size):
            gt = "SURFACE" if sft_eval_labels[i] == 1 else "FILTER"
            pred = sft_decisions[i] or "INVALID"
            match = "✓" if pred == gt else "✗"
            if pred == gt:
                correct += 1
            if i < 10:  # Show first 10
                print(f"    [{match}] GT={gt:7s} Pred={pred}")
        if sft_eval_size > 10:
            print(f"    ... ({sft_eval_size - 10} more)")
        print(f"  Accuracy: {correct}/{sft_eval_size} ({100*correct/sft_eval_size:.0f}%)")
        print(f"{'='*60}\n")
        
        for step in tqdm(range(max_steps), desc=f"Team {team_name} Training"):
            
            # 1. Sync LoRA weights to SGLang every N steps (avoids deadlock + faster)
            if step % lora_sync_interval == 0 or current_lora_name is None:
                new_lora_name = f"{team_name}_step{step}"
                if new_lora_name == current_lora_name:
                    pass  # Already loaded (e.g. from SFT baseline eval)
                else:
                    sync_path = f"{lora_sync_dir}_step{step}"
                    os.makedirs(sync_path, exist_ok=True)
                    self.model.save_pretrained(sync_path)
                    self.sglang.load_lora(new_lora_name, sync_path)
                current_lora_name = new_lora_name
                
                # Clean up old sync dirs to avoid filling /tmp
                if step > 0:
                    old_path = f"{lora_sync_dir}_step{step - lora_sync_interval}"
                    if os.path.exists(old_path):
                        shutil.rmtree(old_path, ignore_errors=True)
            
            # ── DAPO Dynamic Sampling ──────────────────────────────────
            # Instead of skipping zero-variance groups (losing gradient signal),
            # accumulate valid groups by resampling new prompts until we reach
            # the target batch size. This is the actual DAPO paper algorithm.
            
            oversample_size = self.group_size  # Was *2, now 1x → frees budget for more prompts
            # ── DIAG-3: Log oversample size ──────────────────────
            if step == 0:
                logger.info(f"  DIAG-3 group_size={self.group_size}, oversample_size={oversample_size}")
            max_resample_times = self.config.get("max_resample_times", 3)
            target_valid_groups = batch_size  # Need this many groups with variance
            
            accumulated_prompts = []      # List of flat prompts (each repeated oversample_size times)
            accumulated_completions = []  # List of flat completions
            accumulated_advantages = []   # List of flat advantages
            used_prompt_hashes = set()    # Track which prompts we've already tried
            
            for resample_round in range(max_resample_times + 1):
                # STRATIFIED SAMPLING: always draw equal SURFACE + FILTER.
                # Random sampling from 44%S/56%F dataset gives FILTER-heavy
                # batches ~80% of the time, causing the gradient to consistently
                # push toward more FILTER. Stratified ensures balanced gradient.
                surface_pool = [x for x in train_dataset if x["label"] == 1]
                filter_pool = [x for x in train_dataset if x["label"] == 0]
                
                n_per_class = batch_size // 2
                n_surface = min(n_per_class, len(surface_pool))
                n_filter = min(n_per_class, len(filter_pool))
                
                batch = random.sample(surface_pool, n_surface) + random.sample(filter_pool, n_filter)
                random.shuffle(batch)  # Shuffle so order doesn't matter
                for item in batch:
                    item["has_label"] = True
                
                prompts = [item["prompt"] for item in batch]
                diffs = [item["diff"] for item in batch]
                comments = [item["comment"] for item in batch]
                labels = [item["label"] for item in batch]
                has_label = [item["has_label"] for item in batch]
                example_ids = [item.get("example_id", hashlib.md5(f"{d}_{c}".encode('utf-8')).hexdigest()) for d, c in zip(diffs, comments, strict=False)]
                
                # 2. Rollout N completions per prompt using SGLang
                completions_grouped = self.sglang.generate(
                    prompts=prompts,
                    lora_path=current_lora_name,
                    n=oversample_size,
                    max_tokens=self.config.get("max_new_tokens", 512),
                    tokenizer=self.tokenizer,
                    temperature=self.config.get("temperature", 0.8)
                )
                
                # Flatten for reward computation
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
                
                # Count decisions across ALL completions
                from src.training.rewards import parse_completion
                all_decisions = [parse_completion(c)["decision"] for c in flat_completions]
                
                # Calculate metrics for Address Rate
                n_surface_pred = 0
                n_filter_pred = 0
                n_invalid_gen = 0
                true_positives = 0
                false_positives = 0
                
                for pred, gt in zip(all_decisions, flat_labels):
                    if pred == "SURFACE":
                        n_surface_pred += 1
                        if gt == 1:
                            true_positives += 1
                        else:
                            false_positives += 1
                    elif pred == "FILTER":
                        n_filter_pred += 1
                    else:
                        n_invalid_gen += 1
                
                total_gen = len(all_decisions)
                
                # 3. Compute Rewards
                rewards, logs = self.reward_scales.compute_total_reward(
                    flat_completions, flat_diffs, flat_comments, flat_labels, flat_example_ids, flat_has_label, self.config, flat_prompts
                )
                
                # GDPO: Per-Reward Normalization
                w_r1, w_r2, w_r3, w_r4, w_r5 = logs["weights"]
                
                r1_tensor = torch.tensor(logs["r1_raw"], dtype=torch.float32, device=self.device).view(-1, oversample_size)
                r2_tensor = torch.tensor(logs["r2_raw"], dtype=torch.float32, device=self.device).view(-1, oversample_size)
                r3_tensor = torch.tensor(logs["r3_raw"], dtype=torch.float32, device=self.device).view(-1, oversample_size)
                r4_tensor = torch.tensor(logs["r4_raw"], dtype=torch.float32, device=self.device).view(-1, oversample_size)
                r5_tensor = torch.tensor(logs["r5_raw"], dtype=torch.float32, device=self.device).view(-1, oversample_size)
                
                def normalize_within_group(t):
                    """Dr.GRPO: subtract mean only, no std division."""
                    m = t.mean(dim=1, keepdim=True)
                    return t - m
                
                adv_r1 = normalize_within_group(r1_tensor) * w_r1
                adv_r2 = normalize_within_group(r2_tensor) * w_r2
                adv_r3 = normalize_within_group(r3_tensor) * w_r3
                adv_r4 = normalize_within_group(r4_tensor) * w_r4
                adv_r5 = normalize_within_group(r5_tensor) * w_r5
                
                advantages = adv_r1 + adv_r2 + adv_r3 + adv_r4 + adv_r5
                
                # ── DIAG-1: GDPO scale mismatch ──────────────────────
                # If components have very different scales after mean-sub,
                # one component dominates the advantage signal.
                if step % 5 == 0:
                    for comp_name, comp_adv in [("R1", adv_r1), ("R2", adv_r2), ("R3", adv_r3), ("R4", adv_r4), ("R5", adv_r5)]:
                        c_std = comp_adv.std().item()
                        c_absmax = comp_adv.abs().max().item()
                        logger.info(f"  DIAG-1 {comp_name} after GDPO: std={c_std:.4f}, absmax={c_absmax:.4f}")
                    total_std = advantages.std().item()
                    r2_contribution = (adv_r2.abs().sum() / max(1e-8, advantages.abs().sum())).item()
                    logger.info(f"  DIAG-1 Total adv std={total_std:.4f}, R2 contributes {r2_contribution*100:.1f}% of |advantage|")
                
                # Accumulate valid groups
                n_zero_var = 0
                n_valid_round = 0
                for g_idx in range(advantages.size(0)):
                    g_std = advantages[g_idx].std().item()
                    
                    if g_std < 1e-5:
                        n_zero_var += 1
                        # Skip zero-variance groups entirely — no gradient signal
                        # Punishing confident-correct predictions was counterproductive
                        if step % 5 == 0:
                            zv_label = flat_labels[g_idx * oversample_size]
                            zv_decisions = [parse_completion(flat_completions[g_idx*oversample_size+i])['decision'] for i in range(oversample_size)]
                            zv_majority = max(set(zv_decisions), key=zv_decisions.count) if zv_decisions else None
                            logger.info(f"  DIAG-2 ZeroVar group {g_idx}: label={zv_label}, majority={zv_majority} → SKIPPED")
                        continue  # Don't add to accumulated buffers
                    else:
                        n_valid_round += 1
                        group_advs = advantages[g_idx]
                        
                    for idx in range(oversample_size):
                        flat_idx = g_idx * oversample_size + idx
                        accumulated_prompts.append(flat_prompts[flat_idx])
                        accumulated_completions.append(flat_completions[flat_idx])
                        accumulated_advantages.append(group_advs[idx].item())
                
                if step % 5 == 0 and n_zero_var > 0:
                    logger.info(f"  DIAG-2 SUMMARY: {n_zero_var} zero-var groups skipped")
                        
                # 3. After advantage computation, before the zero-variance check - see what the model is actually working with:
                if step % 5 == 0:
                    for g_idx in range(advantages.size(0)):
                        g = advantages[g_idx]
                        g_r2 = adv_r2[g_idx]
                        logger.info(
                            f"  DEBUG Group {g_idx}: adv std={g.std().item():.4f}, "
                            f"min={g.min().item():.3f}, max={g.max().item():.3f}, "
                            f"r2_adv std={g_r2.std().item():.4f}, "
                            f"label={flat_labels[g_idx*oversample_size]}, "
                            f"decisions={[parse_completion(flat_completions[g_idx*oversample_size+i])['decision'] for i in range(oversample_size)]}"
                        )
                
                n_valid = len(accumulated_advantages) // oversample_size
                if n_valid >= target_valid_groups:
                    break
                
                if resample_round < max_resample_times:
                    logger.info(f"  Step {step}: resampling ({n_valid}/{target_valid_groups} valid groups)")
            
            # If we still have nothing after all resample rounds, skip this step
            if len(accumulated_advantages) == 0:
                logger.warning(f"  Step {step}: no valid groups, skipping.")
                continue
            
            # Normalize accumulated advantages across all valid groups
            flat_prompts = accumulated_prompts
            flat_completions = accumulated_completions
            flat_advantages = torch.tensor(accumulated_advantages, dtype=torch.float32, device=self.device)
            
            # Compute S:F ratio & Address Rate for this step
            sf_ratio = n_surface_pred / max(1, n_filter_pred)
            sf_pct = 100 * n_surface_pred / max(1, total_gen)
            ff_pct = 100 * n_filter_pred / max(1, total_gen)
            inv_pct = 100 * n_invalid_gen / max(1, total_gen)
            
            address_rate = (true_positives / max(1, (true_positives + false_positives))) * 100
            
            # Soft normalization: scale to unit std without subtracting mean.
            # This preserves GDPO per-component weighting while keeping advantage
            # magnitudes large enough to produce meaningful gradients.
            adv_std = flat_advantages.std() + 1e-8
            # Always normalize to unit variance — standard GRPO behavior
            # Previous conditional amplification (only when std<0.5) was inconsistent
            if step % 5 == 0:
                logger.info(f"  DIAG-5 Advantage std={adv_std.item():.4f}, normalizing to unit variance")
            flat_advantages = flat_advantages / adv_std
            
            # 4. After group normalization, before the loss loop - see final advantage distribution:
            logger.info(
                f"  DEBUG Advantages: n={len(flat_advantages)}, "
                f"mean={flat_advantages.mean().item():.4f}, std={flat_advantages.std().item():.4f}, "
                f"min={flat_advantages.min().item():.3f}, max={flat_advantages.max().item():.3f}, "
                f"n_positive={( flat_advantages > 0).sum().item()}, n_negative={(flat_advantages < 0).sum().item()}, "
                f"n_zero_var_groups={n_zero_var}"
            )
            
            # 5. Compute GRPO Loss & Update on the filtered sequences
            # Convert prompts + completions to tensors
            full_texts = [p + c for p, c in zip(flat_prompts, flat_completions)]
            
            # Micro-batching to prevent OOM on 3B model
            micro_batch_size = 2
            num_microbatches = (len(flat_prompts) + micro_batch_size - 1) // micro_batch_size
            
            loss_total_logging = 0.0
            grad_accum_steps = self.config.get("grad_accum_steps", 4)
            if step % grad_accum_steps == 0:
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
                
                # Calculate logprobs using SFT reference adapter
                with torch.no_grad():
                    if "sft_ref" in self.model.peft_config:
                        self.model.set_adapter("sft_ref")
                        ref_log_probs = self._get_logprobs(self.model, inputs.input_ids, inputs.attention_mask)
                        self.model.set_adapter("default")
                    else:
                        with self.model.disable_adapter():
                            ref_log_probs = self._get_logprobs(self.model, inputs.input_ids, inputs.attention_mask)
                    
                # Single forward pass: get logits for both log_probs AND entropy
                outputs = self.model(inputs.input_ids, attention_mask=inputs.attention_mask)
                logits = outputs.logits[:, :-1, :]  # shift for next-token prediction
                labels = inputs.input_ids[:, 1:]
                all_log_probs = F.log_softmax(logits, dim=-1)
                curr_log_probs = torch.gather(all_log_probs, 2, labels.unsqueeze(2)).squeeze(2)
                
                # Verify KL is sane:
                with torch.no_grad():
                    kl_check = (curr_log_probs - ref_log_probs).mean().item()
                    ratio_check = torch.exp(curr_log_probs - ref_log_probs).mean().item()
                    if mb_idx == 0:
                        logger.info(f"  DEBUG KL: mean(curr-ref)={kl_check:.4f}, mean_ratio={ratio_check:.4f}")
                
                # Calculate ratio and clip per token
                mb_loss = 0.0
                mb_valid_tokens = 0
                kl_penalty_weight = self.config.get("kl_penalty", 0.10)
                entropy_bonus_weight = self.config.get("entropy_bonus", 0.03)
                
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
                    
                    # HIGH-ENTROPY TOKEN MASK: Keep top 80% of tokens
                    # while dropping the 20% most certain (lowest entropy) boilerplate tokens.
                    rho_keep = 0.8
                    n_tokens = token_entropy.size(0)
                    k = max(1, int(n_tokens * rho_keep))
                    entropy_threshold = torch.topk(token_entropy, k).values[-1]
                    entropy_mask = (token_entropy >= entropy_threshold).float()
                    
                    # ── Detect decision token region (used for mask exemption + boost) ──
                    decision_boost = self.config.get("decision_token_boost", 5.0)
                    token_weight = torch.ones_like(token_entropy)
                    gen_token_ids = inputs.input_ids[idx, start_idx+1:end_idx+1]
                    gen_tokens_list = self.tokenizer.convert_ids_to_tokens(gen_token_ids.tolist())
                    gen_text = self.tokenizer.decode(gen_token_ids.tolist())
                    dec_start_char = gen_text.find('<decision>')
                    dec_end_char = gen_text.find('</decision>')
                    
                    n_exempted = 0
                    if dec_start_char >= 0 and dec_end_char >= 0:
                        dec_end_char += len('</decision>')
                        chars_so_far = 0
                        for t_i, tok in enumerate(gen_tokens_list):
                            if t_i >= len(token_weight):
                                break
                            tok_text = self.tokenizer.convert_tokens_to_string([tok])
                            tok_start = chars_so_far
                            tok_end = chars_so_far + len(tok_text)
                            if tok_end > dec_start_char and tok_start < dec_end_char:
                                token_weight[t_i] = decision_boost
                                entropy_mask[t_i] = 1.0  # Exempt from masking!
                                n_exempted += 1
                            chars_so_far = tok_end
                    
                    # ── DIAG-4/6: Decision token diagnostics ──────────
                    if step % 5 == 0 and mb_idx == 0 and idx == 0:
                        n_boosted = (token_weight > 1.0).sum().item()
                        logger.info(
                            f"  DIAG-4/6 Decision tokens: {n_boosted} boosted {decision_boost}x, "
                            f"{n_exempted} exempted from entropy mask, "
                            f"threshold={entropy_threshold.item():.3f}, seq_mean_entropy={token_entropy.mean().item():.3f}"
                        )
                    
                    # Apply mask and token weight to surrogate loss
                    surr_min = torch.min(surr1, surr2)
                    weighted_surr = -(surr_min * entropy_mask * token_weight).sum() / max(1.0, (entropy_mask * token_weight).sum())
                    
                    # KL and entropy bonus only on masked tokens (no boost needed)
                    masked_kl = (kl * entropy_mask).sum() / max(1.0, entropy_mask.sum())
                    masked_entropy = (token_entropy * entropy_mask).sum() / max(1.0, entropy_mask.sum())
                    
                    token_loss = weighted_surr + kl_penalty_weight * masked_kl - entropy_bonus_weight * masked_entropy
                    mb_loss += token_loss
                    mb_valid_tokens += 1
                    
                    # 5. Inside the per-token loss, log the actual loss components once per step:
                    if mb_idx == 0 and idx == 0:
                        logger.info(
                            f"  DEBUG Loss components: surr={weighted_surr.item():.4f}, "
                            f"kl={kl_penalty_weight * masked_kl.item():.4f}, "
                            f"entropy={entropy_bonus_weight * masked_entropy.item():.4f}, "
                            f"total={token_loss.item():.4f}, "
                            f"mean_ratio={ratio.mean().item():.4f}, "
                            f"clipped_frac={(( ratio > 1.0 + self.clip_ratio_high) | (ratio < 1.0 - self.clip_ratio_low)).float().mean().item():.2f}"
                        )
                
                if mb_valid_tokens > 0:
                    mb_loss = mb_loss / mb_valid_tokens
                    # Scale for accumulation
                    (mb_loss / num_microbatches).backward()
                    loss_total_logging += mb_loss.item()
            
            # Only step every grad_accum_steps to average over more prompts
            if (step + 1) % grad_accum_steps == 0 or step == max_steps - 1:
                # Log grad norm before stepping
                grads_norm = 0.0
                n_frozen = 0
                n_active = 0
                for name, param in self.model.named_parameters():
                    if "lora" in name:
                        if param.grad is not None:
                            grads_norm += param.grad.data.norm(2).item() ** 2
                            n_active += 1
                        else:
                            n_frozen += 1
                grads_norm = grads_norm ** 0.5
                logger.info(f"  DEBUG Pre-Step: grad_norm={grads_norm:.4f}, active_lora_tensors={n_active}, frozen_lora_tensors={n_frozen}, accum={grad_accum_steps} batches")
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            
            loss_total = torch.tensor(loss_total_logging / num_microbatches)
            
            global_step += 1
            
            if global_step % 1 == 0:
                logger.info(
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {loss_total.item():.4f} | "
                    f"S:F={sf_ratio:.2f} ({sf_pct:.0f}%S/{ff_pct:.0f}%F/{inv_pct:.0f}%inv) | "
                    f"AddrRate: {address_rate:.0f}% | "
                    f"R2(Match): {logs['r2']:.2f} | "
                    f"Total R: {logs['total_reward']:.2f} | Format: {logs['valid_format_ratio']*100:.0f}%"
                )
            
            # ── Mid-Training Evaluation ──────────────────────────
            eval_at_steps = self.config.get("eval_at_steps", [15, 30])
            if isinstance(eval_at_steps, int):
                eval_at_steps = [eval_at_steps]
            
            if global_step in eval_at_steps:
                logger.info(f"\n{'='*60}")
                logger.info(f"MID-TRAINING EVAL at step {global_step}")
                logger.info(f"{'='*60}")
                
                # Use test set if available, else train set
                test_file = Path(f"data/teams/{team_name}/test.jsonl")
                if test_file.exists():
                    eval_data = [json.loads(l) for l in open(test_file)]
                else:
                    eval_data = train_dataset
                
                # Regenerate prompts from current template (same as 03_train_dapo.py)
                from src.data.team_dataset import generate_prompt
                for item in eval_data:
                    if "diff" in item and "comment" in item:
                        item["prompt"] = generate_prompt(item["diff"], item["comment"], team_name)
                
                eval_size = min(20, len(eval_data))
                eval_batch = random.sample(eval_data, eval_size)
                eval_prompts = [item["prompt"] for item in eval_batch]
                eval_labels = [item["label"] for item in eval_batch]
                
                num_votes = 8
                eval_completions = self.sglang.generate(
                    prompts=eval_prompts, lora_path=current_lora_name,
                    n=num_votes, max_tokens=self.config.get("max_new_tokens", 256),
                    tokenizer=self.tokenizer,
                    temperature=self.config.get("temperature", 0.8)
                )
                
                # Majority-vote metrics (matches production eval)
                mv_correct = 0
                mv_tp = 0
                mv_fp = 0
                mv_fn = 0
                mv_tn = 0
                per_prompt_results = []
                
                gt_surface = sum(1 for l in eval_labels if l == 1)
                gt_filter = sum(1 for l in eval_labels if l == 0)
                logger.info(f"  GT distribution: {gt_surface} SURFACE / {gt_filter} FILTER")
                
                for i in range(eval_size):
                    votes = []
                    for comp in eval_completions[i]:
                        parsed = parse_completion(comp)
                        dec = parsed["decision"]
                        if dec in ("SURFACE", "FILTER"):
                            votes.append(dec)
                    
                    gt_label = eval_labels[i]
                    gt = "SURFACE" if gt_label == 1 else "FILTER"
                    
                    if not votes:
                        per_prompt_results.append(f"[?] GT={gt:7s} Majority=INVALID (0/0S)")
                        continue
                    
                    n_surf = sum(1 for v in votes if v == "SURFACE")
                    majority = "SURFACE" if n_surf > len(votes) / 2 else "FILTER"
                    majority_label = 1 if majority == "SURFACE" else 0
                    
                    if gt_label == majority_label:
                        mv_correct += 1
                    if gt_label == 1 and majority_label == 1:
                        mv_tp += 1
                    elif gt_label == 0 and majority_label == 1:
                        mv_fp += 1
                    elif gt_label == 1 and majority_label == 0:
                        mv_fn += 1
                    else:
                        mv_tn += 1
                    
                    mark = "✓" if gt_label == majority_label else "✗"
                    per_prompt_results.append(f"[{mark}] GT={gt:7s} Majority={majority} ({n_surf}/{len(votes)}S)")
                
                mv_total = mv_tp + mv_fp + mv_fn + mv_tn
                mv_acc = mv_correct / max(1, mv_total)
                mv_addr = mv_tp / max(1, mv_tp + mv_fp) * 100
                
                logger.info(f"  Majority-vote accuracy: {mv_correct}/{mv_total} ({mv_acc*100:.0f}%)")
                logger.info(f"  Majority-vote Address Rate: {mv_addr:.0f}%")
                logger.info(f"  TP={mv_tp} FP={mv_fp} FN={mv_fn} TN={mv_tn}")
                for r in per_prompt_results[:10]:
                    logger.info(f"    {r}")
                if len(per_prompt_results) > 10:
                    logger.info(f"    ... ({len(per_prompt_results) - 10} more)")
                logger.info(f"{'='*60}")
                
                # Best checkpoint tracking — use majority-vote address rate
                if mv_addr > best_eval_accuracy:
                    best_eval_accuracy = mv_addr
                    best_cp = Path(self.config["output_dir"]) / f"dapo_lora_{team_name}_best"
                    best_cp.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save_pretrained(str(best_cp))
                    best_checkpoint_path = best_cp
                    logger.info(f"  NEW BEST checkpoint saved (addr_rate={mv_addr:.0f}%) → {best_cp}")
        
        # Save final team LoRA — use best checkpoint if available
        final_out = Path(self.config["output_dir"]) / f"dapo_lora_{team_name}"
        final_out.parent.mkdir(parents=True, exist_ok=True)
        if best_checkpoint_path and best_checkpoint_path.exists():
            import shutil as _shutil
            if final_out.exists():
                _shutil.rmtree(final_out)
            _shutil.copytree(best_checkpoint_path, final_out)
            logger.info(f"Team {team_name} training complete. Using BEST checkpoint (acc={best_eval_accuracy*100:.0f}%) → {final_out}")
        else:
            self.model.save_pretrained(str(final_out))
            logger.info(f"Team {team_name} training complete. Saved final step to {final_out}")
        return final_out
