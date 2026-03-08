import re
import torch
import logging

logger = logging.getLogger(__name__)

def parse_completion(completion: str) -> dict:
    """Parses the generated completion to extract think, score, and decision tags."""
    parsed = {
        "think": None,
        "score": None,
        "decision": None,
        "format_score": -1.0 # Default garbage
    }
    
    # Extract think
    think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
    if think_match:
        parsed["think"] = think_match.group(1).strip()
        
    # Extract score
    score_match = re.search(r'<score>(.*?)</score>', completion, re.DOTALL)
    if score_match:
        try:
            val = float(score_match.group(1).strip())
            parsed["score"] = max(0.0, min(1.0, val))
        except ValueError:
            pass
            
    # Extract decision
    decision_match = re.search(r'<decision>(.*?)</decision>', completion, re.DOTALL)
    if decision_match:
        dec = decision_match.group(1).strip().upper()
        if dec in ["SURFACE", "FILTER"]:
            parsed["decision"] = dec
            
    # Calculate format score
    parts_found = sum(1 for v in [parsed["think"], parsed["score"], parsed["decision"]] if v is not None)
    
    if parts_found == 3:
        parsed["format_score"] = 1.0
    elif parts_found > 0:
        parsed["format_score"] = 0.5
        
    return parsed


class DAPORewardScales:
    """Computes the 4 components of the DAPO reward."""
    
    def __init__(self, rm_model=None, tokenizer=None, device="cuda", config=None):
        self.rm_model = rm_model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
    def compute_r1_reasoning_quality(self, parsed_completions: list, diffs: list, comments: list, prompts: list) -> list:
        """
        R1: Reasoning Quality (Rule-based Evaluator).
        Decision-aware: relaxes groundedness for SURFACE (focuses on 'why it matters')
        vs FILTER (which should reference the diff to justify ignoring).
        """
        rewards = []
        for p, diff, comment, prompt in zip(parsed_completions, diffs, comments, prompts):
            think = p.get("think")
            if not think:
                rewards.append(-1.0)
                continue
                
            think_lower = think.lower()
            prompt_lower = prompt.lower()
            
            # 1. Length check
            if len(think) < 50:
                rewards.append(-0.5)
                continue
                
            # 2. Prompt Parroting Check (Bigram overlap)
            def get_bigrams(text):
                words = text.split()
                return set(zip(words[:-1], words[1:]))
                
            think_bigrams = get_bigrams(think_lower)
            prompt_bigrams = get_bigrams(prompt_lower)
            if len(think_bigrams) > 0:
                parroting_ratio = len(think_bigrams.intersection(prompt_bigrams)) / len(think_bigrams)
                if parroting_ratio > 0.6:
                    # Echoing the system prompt instead of reasoning
                    rewards.append(-1.0)
                    continue
                
            # 3. Context groundedness (must overlap with DIFF specifically)
            # Decision-aware: SURFACE reasoning focuses on 'why it matters' and may not
            # quote the diff verbatim, so we use a lower threshold
            import re
            diff_words = set(re.findall(r'\b[a-z]{5,}\b', diff.lower()[:1500])) # Truncated diff
            comment_words = set(re.findall(r'\b[a-z]{5,}\b', comment.lower()))
            context_words = diff_words | comment_words  # Allow overlap with either diff OR comment
            think_words = set(re.findall(r'\b[a-z]{5,}\b', think_lower))
            overlap_context = len(context_words.intersection(think_words))
            
            decision = p.get("decision")
            min_overlap = 1 if decision == "SURFACE" else 3
            
            if overlap_context < min_overlap:
                # Hallucinated reasoning, doesn't actually discuss the code
                rewards.append(-0.5)
                continue
                
            # 3. Repetition check (unique words / total words)
            words = think_lower.split()
            if len(set(words)) / max(1, len(words)) < 0.4:
                # Highly repetitive loop
                rewards.append(-1.0)
                continue
                
            # Continuous reward based on groundedness (capped at 10 overlapping significant words)
            overlap_score = min(1.0, overlap_context / 10.0)
            
            # Continuous reward based on length (sweet spot up to 150 chars, R5 penalizes > 400)
            len_score = min(1.0, len(think) / 150.0)
            
            # Final R1 score is a blend of groundedness and sufficient length
            r = (overlap_score * 0.7) + (len_score * 0.3)
            rewards.append(r)
            
        return rewards

    def compute_r2_outcome_match(self, decisions: list, ground_truth_labels: list) -> list:
        """
        R2: Outcome Match (weight 0.35)
        SURFACE + label=1 → +1.0
        FILTER  + label=0 → +1.0
        SURFACE + label=0 → -1.0 (noise, trust killer)
        FILTER  + label=1 → -1.0 (missed, recoverable)
        """
        rewards = []
        for dec, label in zip(decisions, ground_truth_labels):
            if dec == "SURFACE":
                r = 1.0 if label == 1 else -1.0
            elif dec == "FILTER":
                r = 1.0 if label == 0 else -1.0
            else:
                r = -1.0 # Invalid decision
            rewards.append(r)
        return rewards

    def _get_rm_scores(self, diffs: list, comments: list) -> list:
        """Helper to get actual scores from the Phase 1 reward model."""
        if self.rm_model is None:
            return [0.5] * len(diffs)
            
        rm_scores = []
        rm_batch_size = 8
        self.rm_model.eval()
        with torch.no_grad():
            for i in range(0, len(diffs), rm_batch_size):
                b_diffs = diffs[i:i+rm_batch_size]
                b_comments = comments[i:i+rm_batch_size]
                prompts = [f"{d[:2048]} [SEP] {c[:512]}" for d, c in zip(b_diffs, b_comments)]
                inputs = self.tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(self.device)
                outputs = self.rm_model(inputs["input_ids"], inputs["attention_mask"])
                probs = torch.sigmoid(outputs).squeeze(-1).cpu().tolist()
                if not isinstance(probs, list): probs = [probs]
                rm_scores.extend(probs)
        return rm_scores

    def compute_r3_calibration(self, model_scores: list, ground_truth_labels: list) -> list:
        """
        R3: Calibration (weight 0.05)
        Compares model's <score> against the ground truth label (0 or 1).
        This avoids the bias from the reward model's naturally low outputs.
        """
        rewards = []
        for m_score, label in zip(model_scores, ground_truth_labels):
            if m_score is not None:
                target = float(label)  # 0.0 or 1.0
                diff = abs(m_score - target)
                r = 1.0 - (2.0 * diff)  # Perfect match = +1.0, max error = -1.0
            else:
                r = -1.0
            rewards.append(r)
        return rewards

    def compute_r4_format(self, format_scores: list) -> list:
        """
        R4: Format (weight 0.15)
        Passed straight through from parser logic.
        """
        return format_scores
        
    def compute_r5_overlong_penalty(self, completions: list) -> list:
        """
        R5: Overlong Penalty (weight 0.20)
        DAPO specific feature to penalize bloated reasoning traces.
        If length > 400 characters, apply progressive negative penalty.
        Max penalty of -1.0 at 1000 characters.
        """
        rewards = []
        for text in completions:
            length = len(text)
            if length < 400:
                rewards.append(0.0)
            elif length > 1000:
                rewards.append(-1.0)
            else:
                # Linearly interpolate between 0.0 and -1.0
                penalty = -((length - 400) / 600.0)
                rewards.append(penalty)
        return rewards
    
    def compute_r6_exploration(self, decisions: list) -> list:
        """
        R6: Exploration Bonus (not weighted — applied directly in GDPO).
        When FILTER dominates the batch, reward SURFACE decisions and penalize FILTER
        to create intra-group variance for GDPO gradient signal.
        Without this, all-FILTER groups produce zero R2 variance → no learning.
        """
        surface_count = sum(1 for d in decisions if d == "SURFACE")
        surface_ratio = surface_count / max(1, len(decisions))
        
        rewards = []
        for dec in decisions:
            if surface_ratio < 0.30:
                # Heavily dominated by FILTER — boost exploration
                if dec == "SURFACE":
                    rewards.append(2.0)
                else:
                    rewards.append(-0.5)
            elif surface_ratio > 0.70:
                # Heavily dominated by SURFACE — boost FILTER exploration
                if dec == "FILTER":
                    rewards.append(2.0)
                else:
                    rewards.append(-0.5)
            else:
                # Balanced — no exploration bonus needed
                rewards.append(0.0)
        return rewards
        
    def compute_total_reward(self, 
                             completions: list, 
                             diffs: list, 
                             comments: list, 
                             labels: list,
                             config: dict = None,
                             prompts: list = None) -> dict:
        """
        Computes total reward for a batch of completions.
        Returns total_rewards list, per-component reward lists, and a dict of component averages for logging.
        """
        batch_size = len(completions)
        
        # Parse all completions
        parsed = [parse_completion(c) for c in completions]
        
        # Extract individual components
        decisions = [p["decision"] for p in parsed]
        m_scores = [p["score"] for p in parsed]
        format_scores = [p["format_score"] for p in parsed]
        
        # R1: Rule-based reasoning quality
        r1_scaled = self.compute_r1_reasoning_quality(parsed, diffs, comments, prompts)
        
        # R2
        r2 = self.compute_r2_outcome_match(decisions, labels)
        
        # R3: Calibration against ground truth labels (not reward model)
        r3 = self.compute_r3_calibration(m_scores, labels)
        
        # R4
        r4 = self.compute_r4_format(format_scores)
        
        # R5 (DAPO Overlong Penalty)
        r5 = self.compute_r5_overlong_penalty(completions)
        
        # R6: Exploration bonus (creates variance when one decision dominates)
        r6 = self.compute_r6_exploration(decisions)
        
        # Get weights
        if config is not None:
            w_r1 = config.get("r1_weight", 0.20)
            w_r2 = config.get("r2_weight", 0.35)
            w_r3 = config.get("r3_weight", 0.15)
            w_r4 = config.get("r4_weight", 0.15)
            w_r5 = config.get("r5_weight", 0.15)
        else:
            w_r1, w_r2, w_r3, w_r4, w_r5 = 0.20, 0.35, 0.15, 0.15, 0.15
        
        # Total (weighted sum, used for logging only; GDPO normalizes per-component in trainer)
        total_rewards = []
        for i in range(batch_size):
            total = (w_r1 * r1_scaled[i]) + (w_r2 * r2[i]) + (w_r3 * r3[i]) + (w_r4 * r4[i]) + (w_r5 * r5[i]) + r6[i]
            total_rewards.append(total)
            
        print("\n" + "="*50)
        print(f"DEBUG - Sample Completion:\n{completions[0]}\n")
        print(f"DEBUG - Parsed Data: Decision={decisions[0]}, Score={m_scores[0]}, FormatOK={format_scores[0]}")
        print(f"DEBUG - Rewards: R1={r1_scaled[0]:.2f}, R2={r2[0]:.2f}, R3={r3[0]:.2f}, R4={r4[0]:.2f}, R5={r5[0]:.2f}, R6={r6[0]:.2f} | Total={total_rewards[0]:.2f}")
        print("="*50 + "\n")
            
        logs = {
            "r1": sum(r1_scaled) / batch_size,
            "r2": sum(r2) / batch_size,
            "r3": sum(r3) / batch_size,
            "r4": sum(r4) / batch_size,
            "r5_penalty": sum(r5) / batch_size,
            "r6_explore": sum(r6) / batch_size,
            "total_reward": sum(total_rewards) / batch_size,
            "valid_format_ratio": sum(1 for p in format_scores if p == 1.0) / batch_size,
            # Per-component raw lists for GDPO normalization in trainer
            "r1_raw": r1_scaled,
            "r2_raw": r2,
            "r3_raw": r3,
            "r4_raw": r4,
            "r5_raw": r5,
            "r6_raw": r6,
            "weights": [w_r1, w_r2, w_r3, w_r4, w_r5],
        }
        
        return total_rewards, logs
