import os
import subprocess
import time
from pathlib import Path

def run_training_pipeline():
    """
    Automated Training Pipeline.
    Generates datasets and runs the ADAP pipeline to train the RL Agent and ML models.
    """
    print("=== ADAP Automated Training Pipeline ===")
    
    # 1. Generate E-commerce Data
    print("\n[1/3] Generating E-commerce Dataset...")
    generator_script = Path(__file__).parent / "large_scale" / "ecommerce_generator.py"
    if generator_script.exists():
        subprocess.run(["python", str(generator_script)], check=True)
    else:
        # Fallback if run from training dir
        generator_script = Path(__file__).parent / "ecommerce_generator.py"
        if generator_script.exists():
            subprocess.run(["python", str(generator_script)], check=True)
        else:
            print("ERROR: ecommerce_generator.py not found.")

    # 2. Generate Real Banking Data (if not already there)
    banking_data_script = Path(__file__).parent.parent / "generate_real_banking.py"
    if banking_data_script.exists():
        print("\n[2/3] Generating Banking Dataset...")
        subprocess.run(["python", str(banking_data_script), "--rows", "10000"], check=False)

    # 3. Process Datasets to train RL & ML Models
    print("\n[3/3] Running Pipeline on Datasets to Train Agent...")
    datasets = [
        ("data/raw/ecommerce_churn.csv", "is_churned"),
        ("data/raw/demo_mock_bank_data.csv", "churn_flag")
    ]
    
    run_script = Path(__file__).parent.parent / "run.py"
    
    for file_path, target_col in datasets:
        if not os.path.exists(file_path):
            print(f"Skipping {file_path} (not found)")
            continue
            
        print(f"\n---> Training on {file_path} with target = {target_col}")
        try:
            cmd = ["python", str(run_script), "--source-type", "file", "--source-uri", file_path, "--target", target_col]
            subprocess.run(cmd, check=True)
            print(f"Successfully processed {file_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed processing {file_path}: {e}")
            
        time.sleep(2) # brief pause between runs

    print("\n=== Training Completed ===")

if __name__ == "__main__":
    run_training_pipeline()
