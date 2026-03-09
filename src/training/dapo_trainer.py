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
            # PEFT uses ".default." in key names for named adapters, but save_pretrained
            # saves without it. Remap: lora_A.weight -> lora_A.default.weight
            model_state = self.model.state_dict()
            matched = 0
            for sft_key, sft_val in sft_state.items():
                # Try direct match first
                if sft_key in model_state:
                    model_state[sft_key].copy_(sft_val)
                    matched += 1
                else:
                    # Remap: insert ".default" before ".weight" in lora key
                    remapped = sft_key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                    if remapped in model_state:
                        model_state[remapped].copy_(sft_val)
                        matched += 1
            logger.info(f"SFT warm-up: matched {matched}/{len(sft_state)} keys into training model for {team_name}")
            
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
        # Compute dataset-level label counts for stable inverse class frequency weighting in R2
        surface_count = sum(1 for item in train_dataset if item.get("label") == 1)
        filter_count = sum(1 for item in train_dataset if item.get("label") == 0)
        self.reward_scales.dataset_label_counts = {"surface": surface_count, "filter": filter_count}
        logger.info(f"Dataset label counts for {team_name}: {surface_count} SURFACE / {filter_count} FILTER (w_surface={( surface_count + filter_count) / (2.0 * max(1, surface_count)):.3f}, w_filter={(surface_count + filter_count) / (2.0 * max(1, filter_count)):.3f})")
                
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        
        max_steps = self.config.get("max_steps", 300)
        batch_size = self.config.get("batch_size", 4)
        
        lora_sync_dir = f"/tmp/lora_dapo_{team_name}"
        os.makedirs(lora_sync_dir, exist_ok=True)
        current_lora_name = None  # Track active LoRA name for SGLang
        lora_sync_interval = self.config.get("lora_sync_interval", 2)
        
        global_step = 0
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
            tokenizer=self.tokenizer
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
                sync_path = f"{lora_sync_dir}_step{step}"
                os.makedirs(sync_path, exist_ok=True)
                self.model.save_pretrained(sync_path)
                new_lora_name = f"{team_name}_step{step}"
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
            
            oversample_size = self.group_size * 2
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
                
                # DEBUG: Show batch composition
                n_surface_gt = sum(1 for l in labels if l == 1)
                n_filter_gt = sum(1 for l in labels if l == 0)
                print(f"\n{'='*60}")
                print(f"DEBUG STEP {step} (resample round {resample_round}):")
                print(f"  Batch: {len(batch)} all labeled")
                print(f"  Ground truth: {n_surface_gt} SURFACE / {n_filter_gt} FILTER")
                
                # 2. Rollout N completions per prompt using SGLang
                completions_grouped = self.sglang.generate(
                    prompts=prompts,
                    lora_path=current_lora_name,
                    n=oversample_size,
                    max_tokens=self.config.get("max_new_tokens", 256),
                    tokenizer=self.tokenizer
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
                
                # DEBUG: Count decisions across ALL completions in this batch
                from src.training.rewards import parse_completion
                all_decisions = [parse_completion(c)["decision"] for c in flat_completions]
                n_surface_gen = sum(1 for d in all_decisions if d == "SURFACE")
                n_filter_gen = sum(1 for d in all_decisions if d == "FILTER")
                n_invalid_gen = sum(1 for d in all_decisions if d not in ("SURFACE", "FILTER"))
                total_gen = len(all_decisions)
                print(f"  Model outputs ({total_gen} completions): {n_surface_gen} SURFACE ({100*n_surface_gen/max(1,total_gen):.0f}%) / {n_filter_gen} FILTER ({100*n_filter_gen/max(1,total_gen):.0f}%) / {n_invalid_gen} invalid")
                
                # DEBUG: Per-group breakdown
                for g_idx in range(len(batch)):
                    group_start = g_idx * oversample_size
                    group_end = group_start + oversample_size
                    group_decisions = all_decisions[group_start:group_end]
                    g_surface = sum(1 for d in group_decisions if d == "SURFACE")
                    g_filter = sum(1 for d in group_decisions if d == "FILTER")
                    gt_label = "SURFACE" if labels[g_idx] == 1 else "FILTER"
                    has_lbl = "labeled" if has_label[g_idx] else "unlabeled"
                    print(f"    Group {g_idx} [{has_lbl}, GT={gt_label}]: {g_surface}S/{g_filter}F out of {oversample_size}")
                
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
                    """Dr.GRPO: subtract mean only, no std division.
                    Dividing by std over-weights groups where the model is already
                    confident (low variance), reinforcing the existing bias."""
                    m = t.mean(dim=1, keepdim=True)
                    return t - m
                
                adv_r1 = normalize_within_group(r1_tensor) * w_r1
                adv_r2 = normalize_within_group(r2_tensor) * w_r2
                adv_r3 = normalize_within_group(r3_tensor) * w_r3
                adv_r4 = normalize_within_group(r4_tensor) * w_r4
                adv_r5 = normalize_within_group(r5_tensor) * w_r5
                
                advantages = adv_r1 + adv_r2 + adv_r3 + adv_r4 + adv_r5
                
                # DEBUG: Per-group advantage analysis
                print(f"  Advantage analysis per group:")
                n_zero_var = 0
                n_valid_round = 0
                for g_idx in range(advantages.size(0)):
                    g_std = advantages[g_idx].std().item()
                    g_mean = advantages[g_idx].mean().item()
                    g_min = advantages[g_idx].min().item()
                    g_max = advantages[g_idx].max().item()
                    is_valid = g_std >= 1e-5
                    status = "VALID" if is_valid else "ZERO-VAR (skipped)"
                    print(f"    Group {g_idx}: std={g_std:.6f} mean={g_mean:.4f} range=[{g_min:.4f}, {g_max:.4f}] -> {status}")
                    
                    if not is_valid:
                        n_zero_var += 1
                        continue
                    
                    n_valid_round += 1
                    group_advs = advantages[g_idx]
                    for idx in range(oversample_size):
                        flat_idx = g_idx * oversample_size + idx
                        accumulated_prompts.append(flat_prompts[flat_idx])
                        accumulated_completions.append(flat_completions[flat_idx])
                        accumulated_advantages.append(group_advs[idx].item())
                
                print(f"  Round {resample_round}: {n_valid_round} valid / {n_zero_var} zero-var groups")
                
                n_valid = len(accumulated_advantages) // oversample_size
                if n_valid >= target_valid_groups:
                    print(f"  Accumulated {n_valid}/{target_valid_groups} valid groups. Proceeding.")
                    break
                
                if resample_round < max_resample_times:
                    print(f"  Dynamic resample round {resample_round+1}: {n_valid}/{target_valid_groups} valid groups, resampling...")
            
            # If we still have nothing after all resample rounds, skip this step
            if len(accumulated_advantages) == 0:
                logger.warning(f"  Step {step}: no valid groups after {max_resample_times+1} rounds, skipping.")
                continue
            
            # Normalize accumulated advantages across all valid groups
            flat_prompts = accumulated_prompts
            flat_completions = accumulated_completions
            flat_advantages = torch.tensor(accumulated_advantages, dtype=torch.float32, device=self.device)
            
            # DEBUG: Show advantage distribution before normalization
            from src.training.rewards import parse_completion as _pc
            acc_decisions = [_pc(c)["decision"] for c in flat_completions]
            surface_advs = [a for a, d in zip(accumulated_advantages, acc_decisions) if d == "SURFACE"]
            filter_advs = [a for a, d in zip(accumulated_advantages, acc_decisions) if d == "FILTER"]
            print(f"  FINAL accumulated: {len(flat_completions)} completions from {len(flat_completions)//oversample_size} valid groups")
            print(f"    SURFACE completions: {len(surface_advs)}, mean_adv={sum(surface_advs)/max(1,len(surface_advs)):.4f}")
            print(f"    FILTER completions:  {len(filter_advs)}, mean_adv={sum(filter_advs)/max(1,len(filter_advs)):.4f}")
            if len(surface_advs) > 0 and len(filter_advs) > 0:
                print(f"    → SURFACE gets {'HIGHER' if sum(surface_advs)/len(surface_advs) > sum(filter_advs)/len(filter_advs) else 'LOWER'} advantage (gradient pushes toward {'more SURFACE' if sum(surface_advs)/len(surface_advs) > sum(filter_advs)/len(filter_advs) else 'more FILTER'})")
            print(f"{'='*60}\n")
            
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
            
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
