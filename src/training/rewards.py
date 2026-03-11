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
        
    # Extract score (also match <scores> which the model sometimes generates)
    score_match = re.search(r'<scores?>(.*?)</scores?>', completion, re.DOTALL)
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
    """Computes the 6 components of the DAPO reward."""
    
    def __init__(self, tokenizer=None, precomputed_scores=None, device="cuda", config=None, dataset_label_counts=None):
        self.tokenizer = tokenizer
        self.precomputed_scores = precomputed_scores or {}
        self.device = device
        self.config = config or {}
        # Precomputed dataset-level label counts for stable inverse frequency weighting
        # Expected: {"surface": N, "filter": M}
        self.dataset_label_counts = dataset_label_counts or {}
        
    def compute_r1_reasoning_quality(self, parsed_completions: list, diffs: list, prompts: list) -> list:
        """
        R1: Reasoning Quality (Verifiable Sub-Task Checks).
        1. Contains code identifier from the diff.
        2. Contains causal connective.
        3. Score agrees with the decision.
        """
        rewards = []
        for p, diff, prompt in zip(parsed_completions, diffs, prompts):
            think = p.get("think")
            if not think:
                rewards.append(-1.0)
                continue
                
            think_lower = think.lower()
            decision = p.get("decision")
            parsed_score = p.get("score")
            
            # Check 1: Code identifier from diff (added/removed lines)
            code_lines = [line[1:] for line in diff.split('\n') if line.startswith(('+', '-')) and not line.startswith(('+++', '---'))]
            code_text = " ".join(code_lines).lower()
            code_tokens = set(re.findall(r'\b[a-zA-Z_]\w{2,}\b', code_text))
            stopwords = {"the", "and", "this", "that", "with", "for", "def", "class", "return", "import", "from", "if", "else", "elif", "try", "except"}
            code_identifiers = code_tokens - stopwords
            
            has_ident = any(ident in think_lower for ident in code_identifiers)
            
            # Check 2: Causal connective
            has_causal = any(c in think_lower for c in ["because", "since", "so", "therefore"])
            
            # Check 3: Score agrees with decision
            score_agrees = False
            if parsed_score is not None:
                if parsed_score > 0.5 and decision == "SURFACE":
                    score_agrees = True
                elif parsed_score < 0.5 and decision == "FILTER":
                    score_agrees = True
                    
            checks_passed = sum([has_ident, has_causal, score_agrees])
            if checks_passed == 3:
                rewards.append(1.0)
            elif checks_passed == 2:
                rewards.append(0.5)
            elif checks_passed == 1:
                rewards.append(0.0)
            else:
                rewards.append(-1.0)
                
        return rewards

    def compute_r2_outcome_match(self, decisions: list, ground_truth_labels: list, has_label: list, example_ids: list, team_names: list) -> list:
        """
        R2: Outcome Match (Non-Linear Margin Reward)
        Applies concave rewards for True Positives/Negatives and convex penalties for False Positives/Negatives.
        Reduces penalties inside a margin of ambiguity.
        """
        from src.data.team_dataset import TEAM_PROFILES
        import math
        
        # Dataset-level counts for stable weighting (w)
        ds_surface = self.dataset_label_counts.get("surface", 1)
        ds_filter = self.dataset_label_counts.get("filter", 1)
        ds_total = ds_surface + ds_filter
        
        w1_base = ds_total / (2.0 * max(1, ds_surface))
        w0_base = ds_total / (2.0 * max(1, ds_filter))
        
        # Hyperparameters
        alpha = self.config.get("r2_fp_alpha", 1.5)
        beta = self.config.get("r2_fn_beta", 2.5)
        p = self.config.get("r2_reward_power", 0.5)
        q = self.config.get("r2_penalty_power", 2.0)
        r_power = self.config.get("r2_fn_penalty_power", 0.5)
        m = self.config.get("r2_margin_width", 0.10)
        delta = self.config.get("r2_margin_delta", 0.5)
        
        def f(x):
            return x ** p if x > 0 else 0.0
            
        def g_fp(x):
            return abs(x) ** q
            
        def g_fn(x):
            return abs(x) ** r_power
            
        rewards = []
        for dec, label, h, ex_id, team_name in zip(decisions, ground_truth_labels, has_label, example_ids, team_names):
            if not h:
                rewards.append(0.0)
                continue
                
            rm_score = self.precomputed_scores.get(ex_id)
            if rm_score is None:
                # Fallback to binary distance if score is missing
                rm_score = float(label)
                
            team_threshold = TEAM_PROFILES.get(team_name, {}).get("rm_threshold", 0.5)
            
            # Determine if in ambiguity margin
            in_margin = abs(rm_score - team_threshold) <= m
            penalty_multiplier = delta if in_margin else 1.0
            
            # Item weight based on ground truth class frequency
            w = w1_base if label == 1 else w0_base
            
            if dec == "SURFACE":
                dist = rm_score - team_threshold
                if dist >= 0:
                    r = w * f(dist)  # True Positive
                else:
                    r = -1.0 * w * alpha * g_fp(dist) * penalty_multiplier  # False Positive
                    
            elif dec == "FILTER":
                dist = team_threshold - rm_score
                if dist >= 0:
                    r = w * f(dist)  # True Negative
                else:
                    r = -1.0 * w * beta * g_fn(dist) * penalty_multiplier  # False Negative
                    
            else:
                r = 0.0  # Invalid decision — R4 handles format penalty
                
            rewards.append(r)
        return rewards

    def compute_r3_score_calibration(self, m_scores: list, example_ids: list) -> list:
        """
        R3: Score Calibration
        Compares DAPO model's `<score>` against the Phase 1 RM precomputed score.
        Reward = 1.0 - (2.0 * |dapo_score - rm_score|)
        """
        rewards = []
        for m_score, ex_id in zip(m_scores, example_ids):
            target_score = self.precomputed_scores.get(ex_id)
            if m_score is None or target_score is None:
                rewards.append(0.0)  # Missing score — don't penalize for data pipeline issue
            else:
                try:
                    score_val = float(m_score)
                    diff = abs(score_val - target_score)
                    # Perfect match = 1.0, max disconnect (1.0 off) = -1.0
                    r = 1.0 - (2.0 * diff)
                    rewards.append(max(-1.0, min(1.0, r)))
                except ValueError:
                    rewards.append(-1.0)
                    
        return rewards

    def compute_r4_format(self, format_scores: list) -> list:
        """
        R4: Format (weight 0.15)
        Passed straight through from parser logic.
        """
        return format_scores
        
    def compute_r5_overlong_penalty(self, completions: list, decisions: list) -> list:
        """
        R5: Overlong Penalty (weight 0.20)
        Token-based progressive negative penalty for bloated completions.
        Decision-aware thresholds: FILTER should be brief, SURFACE can be longer.
        """
        max_new_tokens = self.config.get("max_new_tokens", 256)
        
        rewards = []
        for text, dec in zip(completions, decisions):
            token_count = len(self.tokenizer.encode(text, add_special_tokens=False))
            
            # Decision-aware thresholds
            if dec == "FILTER":
                threshold = int(0.55 * max_new_tokens)  # ~140 tokens — enough to write the <decision> tag
            else:
                threshold = int(0.8 * max_new_tokens)  # ~200 tokens
                
            if token_count <= threshold:
                rewards.append(0.0)
            elif token_count >= max_new_tokens:
                rewards.append(-1.0)
            else:
                penalty = -((token_count - threshold) / (max_new_tokens - threshold))
                rewards.append(penalty)
        return rewards
    

        
    def compute_total_reward(self, 
                             completions: list, 
                             diffs: list, 
                             comments: list, 
                             labels: list,
                             example_ids: list,
                             has_label: list,
                             config: dict = None,
                             prompts: list = None,
                             team_names: list = None) -> dict:
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
        r1_scaled = self.compute_r1_reasoning_quality(parsed, diffs, prompts)
        
        # R2: Outcome Match (Continuous Margin Reward)
        if team_names is None:
            # Fallback for old/test code
            team_names = ["pragmatic_shippers"] * len(decisions)
        r2 = self.compute_r2_outcome_match(decisions, labels, has_label, example_ids, team_names)
        
        # R3: Score Calibration (calibrates <score> output to precomputed target)
        r3 = self.compute_r3_score_calibration(m_scores, example_ids)
        
        # R4
        r4 = self.compute_r4_format(format_scores)
        
        # R5 (DAPO Overlong Penalty)
        r5 = self.compute_r5_overlong_penalty(completions, decisions)
        
        # Get weights
        if config is not None:
            w_r1 = config.get("r1_weight", 0.10)
            w_r2 = config.get("r2_weight", 0.60)
            w_r3 = config.get("r3_weight", 0.10)
            w_r4 = config.get("r4_weight", 0.10)
            w_r5 = config.get("r5_weight", 0.10)
        else:
            w_r1, w_r2, w_r3, w_r4, w_r5 = 0.10, 0.60, 0.10, 0.10, 0.10
        
        # Total (weighted sum, used for logging only; GDPO normalizes per-component in trainer)
        total_rewards = []
        for i in range(batch_size):
            total = (w_r1 * r1_scaled[i]) + (w_r2 * r2[i]) + (w_r3 * r3[i]) + (w_r4 * r4[i]) + (w_r5 * r5[i])
            total_rewards.append(total)
            
        logs = {
            "r1": sum(r1_scaled) / batch_size,
            "r2": sum(r2) / batch_size,
            "r3": sum(r3) / batch_size,
            "r4": sum(r4) / batch_size,
            "r5_penalty": sum(r5) / batch_size,
            "total_reward": sum(total_rewards) / batch_size,
            "valid_format_ratio": sum(1 for p in format_scores if p == 1.0) / batch_size,
            # Per-component raw lists for GDPO normalization in trainer
            "r1_raw": r1_scaled,
            "r2_raw": r2,
            "r3_raw": r3,
            "r4_raw": r4,
            "r5_raw": r5,
            "weights": [w_r1, w_r2, w_r3, w_r4, w_r5],
        }
        
        return total_rewards, logs
