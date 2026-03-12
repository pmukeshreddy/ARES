import subprocess
import os
import yaml
import time
from pathlib import Path

def run_tuning():
    print("🚀 Starting DAPO Hyperparameter Tuning...\n")
    
    # Backup original config
    original_config_path = "configs/default.yaml"
    backup_path = "configs/default_tuning_backup.yaml"
    if os.path.exists(original_config_path):
        import shutil
        shutil.copy(original_config_path, backup_path)
    
    # Define our grid
    configurations = [
        {"name": "Config_A_Strict_Balance", "alpha": 2.0, "beta": 1.0, "majority": 5},
        {"name": "Config_B_Super_Majority_Precision", "alpha": 2.0, "beta": 1.0, "majority": 6},
        {"name": "Config_C_Standard_High_Address", "alpha": 1.5, "beta": 1.0, "majority": 6},
    ]
    
    results_file = "tuning_results.csv"
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("Config_Name,Alpha,Beta,Majority_Threshold,Eval_Address_Rate,Eval_Accuracy\n")

    try:
        for config in configurations:
            print(f"\n{'='*70}")
            print(f"🧪 TESTING CONFIGURATION: {config['name']}")
            print(f"Alpha: {config['alpha']}, Beta: {config['beta']}, Majority Threshold: {config['majority']}")
            print(f"{'='*70}\n")
            
            # 1. Update config.yaml with new alpha/beta
            with open(original_config_path, "r") as f:
                yaml_data = yaml.safe_load(f)
                
            yaml_data["dapo"]["r2_fp_alpha"] = config["alpha"]
            yaml_data["dapo"]["r2_fn_beta"] = config["beta"]
            
            with open(original_config_path, "w") as f:
                yaml.dump(yaml_data, f)
                
            # 2. Update 04_evaluate_dapo.py and dapo_trainer.py with new majority threshold
            # Updating dapo_trainer.py
            trainer_path = "src/training/dapo_trainer.py"
            with open(trainer_path, "r") as f:
                trainer_code = f.read()
            import re
            trainer_code = re.sub(
                r'majority = "SURFACE" if n_surf >= \d+', 
                f'majority = "SURFACE" if n_surf >= {config["majority"]}', 
                trainer_code
            )
            with open(trainer_path, "w") as f:
                f.write(trainer_code)
                
            # Updating 04_evaluate_dapo.py
            eval_path = "scripts/04_evaluate_dapo.py"
            with open(eval_path, "r") as f:
                eval_code = f.read()
            eval_code = re.sub(
                r'majority = 1 if n_s >= \d+', 
                f'majority = 1 if n_s >= {config["majority"]}', 
                eval_code
            )
            with open(eval_path, "w") as f:
                f.write(eval_code)
                
            # 3. Run Training Script (Wait for it to finish)
            print("⏳ Running DAPO Training (50 steps)...")
            subprocess.run(["python", "scripts/03_train_dapo.py", "--teams", "pragmatic_shippers"], check=True)
            
            # 4. Run Evaluation Script to get final metrics
            print("\n⏳ Running Final Evaluation...")
            eval_proc = subprocess.run(
                ["python", "scripts/04_evaluate_dapo.py", "--teams", "pragmatic_shippers", "--max-samples", "250"],
                capture_output=True, text=True, check=True
            )
            
            # Parse output for Address Rate and Accuracy
            output = eval_proc.stdout
            address_rate = "N/A"
            accuracy = "N/A"
            for line in output.split('\n'):
                if "Address Rate:" in line and "TP=" in line:
                    parts = line.split("Address Rate:")
                    if len(parts) > 1:
                        address_rate = parts[1].split("%")[0].strip()
                if "Majority Vote Accuracy:" in line:
                    parts = line.split("Majority Vote Accuracy:")
                    if len(parts) > 1:
                        accuracy = parts[1].split("(")[1].split("%")[0].strip()
                        
            print(f"✅ RESULTS: Address Rate {address_rate}%, Accuracy {accuracy}%")
            
            # Save to CSV
            with open(results_file, "a") as f:
                f.write(f"{config['name']},{config['alpha']},{config['beta']},{config['majority']},{address_rate},{accuracy}\n")
                
            print(f"Results appended to {results_file}")
            
    finally:
        # Restore backups
        print("\n🔧 Restoring original configurations...")
        if os.path.exists(backup_path):
            import shutil
            shutil.copy(backup_path, original_config_path)
            os.remove(backup_path)
            
        print("Done tuning.")

if __name__ == "__main__":
    run_tuning()
