
import os
import subprocess
import sys
import nltk

def run_command(cmd):
    print(f"[EXEC] {cmd}")
    subprocess.check_call(cmd, shell=True)

def check_nltk_resources():
    print("--- Checking NLTK resources ---")
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("NLTK resources downloaded.")
    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")

def main():
    base_dir = os.getcwd()
    sigext_repo = os.path.join(base_dir, "SigExt")
    experiments_dir = os.path.join(base_dir, "experiments")
    
    # Ensure NLTK data is available for SigExt
    check_nltk_resources()

    # Paths for experiments
    dataset_dir = os.path.join(experiments_dir, "multi_news_dataset")
    model_dir = os.path.join(experiments_dir, "multi_news_extractor_model")
    output_dir = os.path.join(experiments_dir, "multi_news_dataset_with_keyphrase")
    
    # 1. Clone SigExt
    if not os.path.exists(sigext_repo):
        print("--- Cloning SigExt ---")
        run_command("git clone https://github.com/amazon-science/SigExt.git")
    else:
        print("SigExt repo already exists.")

    # 2. Patch prepare_data.py (Fix for datasets library)
    prepare_script = os.path.join(sigext_repo, "src", "prepare_data.py")
    with open(prepare_script, "r") as f:
        content = f.read()
    
    if "trust_remote_code=True" not in content:
        print("--- Patching prepare_data.py ---")
        content = content.replace(
            "dataset = load_dataset(*DATASET_NAME_MAPPER[dataset])",
            "dataset = load_dataset(*DATASET_NAME_MAPPER[dataset], trust_remote_code=True)"
        )
        with open(prepare_script, "w") as f:
            f.write(content)
        print("Patch applied.")
    
    # 4. Prepare Data
    if not os.path.exists(dataset_dir):
        print("--- Running Data Preparation ---")
        run_command(f"python3 {sigext_repo}/src/prepare_data.py --dataset multi_news --output_dir {dataset_dir}")

    # 5. Train Model
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        print("--- Training SigExt Model ---")
        run_command(f"python3 {sigext_repo}/src/train_longformer_extractor_context.py --dataset_dir {dataset_dir} --checkpoint_dir {model_dir}")
    
    # 6. Inference
    if not os.path.exists(output_dir):
        print("--- Running Inference ---")
        run_command(f"python3 {sigext_repo}/src/inference_longformer_extractor.py --dataset_dir {dataset_dir} --checkpoint_dir {model_dir} --output_dir {output_dir}")
    else:
        print(f"Output directory {output_dir} already exists. Skipping inference.")

    print("
=== Setup Complete ===")
    print(f"Data generated in: {output_dir}")

if __name__ == "__main__":
    main()
