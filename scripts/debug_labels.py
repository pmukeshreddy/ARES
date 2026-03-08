"""
Debug script: Examine team training data quality.
Run on server: python3 scripts/debug_labels.py
"""
import json
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

teams_dir = Path("data/teams")

if not teams_dir.exists():
    print("ERROR: data/teams/ not found. Run team_dataset.py first.")
    sys.exit(1)

for team_dir in sorted(teams_dir.iterdir()):
    if not team_dir.is_dir():
        continue
    
    team_name = team_dir.name
    train_file = team_dir / "train.jsonl"
    if not train_file.exists():
        continue
    
    samples = []
    with open(train_file) as f:
        for line in f:
            samples.append(json.loads(line))
    
    surface_count = sum(1 for s in samples if s["label"] == 1)
    filter_count = sum(1 for s in samples if s["label"] == 0)
    
    print(f"\n{'='*70}")
    print(f"Team: {team_name} | Total: {len(samples)} | SURFACE: {surface_count} | FILTER: {filter_count}")
    print(f"{'='*70}")
    
    # Show 3 SURFACE and 3 FILTER examples
    surface_examples = [s for s in samples if s["label"] == 1][:3]
    filter_examples = [s for s in samples if s["label"] == 0][:3]
    
    print("\n--- SURFACE (label=1) examples ---")
    for i, s in enumerate(surface_examples):
        print(f"\n  [{i+1}] Comment: {s['comment'][:200]}")
        print(f"      Label reason: keyword matched + original author acted on it")
    
    print("\n--- FILTER (label=0) examples ---")
    for i, s in enumerate(filter_examples):
        print(f"\n  [{i+1}] Comment: {s['comment'][:200]}")

# Also check: what does the model actually SEE?
print(f"\n\n{'='*70}")
print("SAMPLE PROMPT (what the model sees):")
print(f"{'='*70}")
sample = samples[0]
print(sample["prompt"][:800])
print("...")

# Check if the reward model would agree with the labels
print(f"\n\n{'='*70}")
print("CHECKING REWARD MODEL AGREEMENT WITH LABELS")
print(f"{'='*70}")

try:
    import torch
    import yaml
    from src.training.reward_model import RewardModel
    
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    
    rm_config = config["reward_model"]
    from transformers import AutoTokenizer
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_config["model_name"], trust_remote_code=True)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    
    rm_model = RewardModel(rm_config)
    checkpoint_path = Path("checkpoints/reward_model/best")
    if checkpoint_path.exists():
        rm_model.load(str(checkpoint_path))
        print("Reward model loaded!")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rm_model.to(device)
        rm_model.eval()
        
        # Check RM scores for all teams
        for team_dir in sorted(teams_dir.iterdir()):
            if not team_dir.is_dir():
                continue
            
            team_name = team_dir.name
            train_file = team_dir / "train.jsonl"
            samples = []
            with open(train_file) as f:
                for line in f:
                    samples.append(json.loads(line))
            
            agree = 0
            disagree = 0
            rm_surface = 0
            rm_filter = 0
            
            for s in samples:
                diff = s["diff"][:2048]
                comment = s["comment"][:512]
                prompt = f"{diff} [SEP] {comment}"
                inputs = rm_tokenizer(prompt, truncation=True, max_length=1024, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    output = rm_model(inputs["input_ids"], inputs["attention_mask"])
                    rm_score = torch.sigmoid(output).item()
                
                rm_decision = 1 if rm_score > 0.5 else 0
                if rm_score > 0.5:
                    rm_surface += 1
                else:
                    rm_filter += 1
                    
                if rm_decision == s["label"]:
                    agree += 1
                else:
                    disagree += 1
            
            total = agree + disagree
            print(f"\n  {team_name}: RM agrees with {agree}/{total} labels ({100*agree/total:.0f}%)")
            print(f"    RM says SURFACE: {rm_surface}, FILTER: {rm_filter}")
            print(f"    Labels say SURFACE: {sum(1 for s in samples if s['label']==1)}, FILTER: {sum(1 for s in samples if s['label']==0)}")
    else:
        print("No reward model checkpoint found!")
except Exception as e:
    print(f"Could not load RM: {e}")
    import traceback
    traceback.print_exc()
