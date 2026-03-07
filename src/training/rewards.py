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
    
    def __init__(self, rm_model=None, tokenizer=None, device="cuda"):
        self.rm_model = rm_model
        self.tokenizer = tokenizer
        self.device = device
        
    def compute_r1_reward_model(self, diffs: list, comments: list) -> list:
        """
        R1: Phase 1 model scores the (diff, comment) pair. Scaled to [-1, 1].
        Weight: 0.35
        """
        if self.rm_model is None:
            # If RM isn't loaded (e.g. testing), return 0s
            return [0.0] * len(diffs)
            
        rm_scores = []
        
        # Batching for Phase 1 model
        self.rm_model.eval()
        with torch.no_grad():
            inputs = []
            for d, c in zip(diffs, comments):
                # Truncate diff gently if needed
                text = f"{d[:2048]} [SEP] {c[:512]}"
                inputs.append(text)
                
            # Micro-batching to avoid OOM when evaluating 64 samples at once
            micro_batch_size = 8
            for i in range(0, len(inputs), micro_batch_size):
                batch_inputs = inputs[i:i+micro_batch_size]
                
                tokenized = self.tokenizer(
                    batch_inputs,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt"
                ).to(self.device)
                
                # The Phase 1 model outputs logits
                logits = self.rm_model(tokenized.input_ids, tokenized.attention_mask).squeeze(-1)
                
                # Handle scalar case if batch size is 1
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                    
                probs = torch.sigmoid(logits).cpu().tolist()
                
                # Scale [0, 1] to [-1, 1]
                batch_scores = [(p * 2.0) - 1.0 for p in probs]
                rm_scores.extend(batch_scores)
                
        return rm_scores

    def compute_r2_outcome_match(self, decisions: list, ground_truth_labels: list) -> list:
        """
        R2: Outcome Match (weight 0.35)
        SURFACE + label=1 → +1.0
        FILTER  + label=0 → +1.0
        SURFACE + label=0 → -2.0 (noise, trust killer)
        FILTER  + label=1 → -0.5 (missed, recoverable)
        """
        rewards = []
        for dec, label in zip(decisions, ground_truth_labels):
            if dec == "SURFACE":
                r = 1.0 if label == 1 else -2.0
            elif dec == "FILTER":
                r = 1.0 if label == 0 else -0.5
            else:
                r = -1.0 # Invalid decision
            rewards.append(r)
        return rewards

    def compute_r3_calibration(self, model_scores: list, rm_scores_0_1: list) -> list:
        """
        R3: Calibration (weight 0.15)
        |model_score - reward_model_score| < 0.2 → +1.0
        else → -1.0
        """
        rewards = []
        for m_score, rm_score in zip(model_scores, rm_scores_0_1):
            if m_score is not None and rm_score is not None:
                diff = abs(m_score - rm_score)
                r = 1.0 if diff < 0.2 else -1.0
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
        If length > 600 characters (~150 tokens), apply progressive negative penalty.
        Max penalty of -1.0 at 1600 characters.
        """
        rewards = []
        for text in completions:
            length = len(text)
            if length < 600:
                rewards.append(0.0)
            elif length > 1600:
                rewards.append(-1.0)
            else:
                # Linearly interpolate between 0.0 and -1.0
                penalty = -((length - 600) / 1000.0)
                rewards.append(penalty)
        return rewards
        
    def compute_total_reward(self, 
                             completions: list, 
                             diffs: list, 
                             comments: list, 
                             labels: list,
                             config: dict = None) -> dict:
        """
        Computes total reward for a batch of completions.
        Returns total_rewards list and a dict of component averages for logging.
        """
        batch_size = len(completions)
        
        # Parse all completions
        parsed = [parse_completion(c) for c in completions]
        
        # Extract individual components
        decisions = [p["decision"] for p in parsed]
        m_scores = [p["score"] for p in parsed]
        format_scores = [p["format_score"] for p in parsed]
        
        # R1 (RM scaled -1 to 1) 
        # For R3, we need the 0 to 1 version, so we convert back
        r1_scaled = self.compute_r1_reward_model(diffs, comments)
        rm_scores_0_1 = [(s + 1.0) / 2.0 for s in r1_scaled]
        
        # R2
        r2 = self.compute_r2_outcome_match(decisions, labels)
        
        # R3
        r3 = self.compute_r3_calibration(m_scores, rm_scores_0_1)
        
        # R4
        r4 = self.compute_r4_format(format_scores)
        
        # R5 (DAPO Overlong Penalty)
        r5 = self.compute_r5_overlong_penalty(completions)
        
        # Total
        total_rewards = []
        for i in range(batch_size):
            if config is not None:
                w_r1 = config.get("r1_weight", 0.30)
                w_r2 = config.get("r2_weight", 0.30)
                w_r3 = config.get("r3_weight", 0.10)
                w_r4 = config.get("r4_weight", 0.10)
                w_r5 = config.get("r5_weight", 0.20)
            else:
                # Fallback weights
                w_r1, w_r2, w_r3, w_r4, w_r5 = 0.30, 0.30, 0.10, 0.10, 0.20
            
            total = (w_r1 * r1_scaled[i]) + (w_r2 * r2[i]) + (w_r3 * r3[i]) + (w_r4 * r4[i]) + (w_r5 * r5[i])
            total_rewards.append(total)
            
        logs = {
            "r1": sum(r1_scaled) / batch_size,
            "r2": sum(r2) / batch_size,
            "r3": sum(r3) / batch_size,
            "r4": sum(r4) / batch_size,
            "r5_penalty": sum(r5) / batch_size,
            "total_reward": sum(total_rewards) / batch_size,
            "valid_format_ratio": sum(1 for p in format_scores if p == 1.0) / batch_size
        }
        
        return total_rewards, logs
