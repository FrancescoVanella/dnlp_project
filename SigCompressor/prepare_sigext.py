import os
import subprocess
import sys
import nltk

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def download_nltk_resources():
    """
    Download NLTK resources if not present
    """
    resources = ['stopwords', 'punkt', 'punkt_tab']
    for r in resources:
        try:
            nltk.data.find(f'tokenizers/{r}')
        except LookupError:
            print(f"Downloading NLTK resource: {r}")
            nltk.download(r, quiet=True)

def main():
    download_nltk_resources()
    BASE_DIR = os.getcwd()
    SIGEXT_REPO_DIR = os.path.join(BASE_DIR, "SigExt_repo")
    DATASET_DIR = os.path.join(BASE_DIR, "cnn_dataset")
    MODEL_DIR = os.path.join(BASE_DIR, "cnn_extractor_model")
    OUTPUT_DIR = os.path.join(BASE_DIR, "cnn_dataset_with_keyphrase")
    
    # 1. Clone
    if not os.path.exists(SIGEXT_REPO_DIR):
        print("Cloning SigExt repository")
        run_command(f"git clone https://github.com/amazon-science/SigExt.git {SIGEXT_REPO_DIR}")
    else:
        print("SigExt repository already exists.")

    sigext_src = os.path.join(SIGEXT_REPO_DIR, "src")
    
    # 2. Prepare Data
    print("--- Preparing Data ---")
    # Ensure output dir exists
    if not os.path.exists(DATASET_DIR):
        run_command(f"python3 {os.path.join(sigext_src, 'prepare_data.py')} --dataset cnn --output_dir {DATASET_DIR}/")
    else:
        print(f"Dataset dir {DATASET_DIR} already exists")

    # 3. Train
    print("--- Training SigExt ---")
    if not os.path.exists(MODEL_DIR):
        run_command(f"python3 {os.path.join(sigext_src, 'train_longformer_extractor_context.py')} --dataset_dir {DATASET_DIR}/ --checkpoint_dir {MODEL_DIR}/")
    else:
        print(f"Model dir {MODEL_DIR} already exists.")

    # 4. Inference
    print("--- Running SigExt ---")
    if not os.path.exists(OUTPUT_DIR):
        run_command(f"python3 {os.path.join(sigext_src, 'inference_longformer_extractor.py')} --dataset_dir {DATASET_DIR}/ --checkpoint_dir {MODEL_DIR}/ --output_dir {OUTPUT_DIR}/")
    else:
        print(f"Output dir {OUTPUT_DIR} already exists.")

    print(f"\nPreparation complete. Data are available in {OUTPUT_DIR}")
    print(f"Run: python main.py --input_file {os.path.join(OUTPUT_DIR, 'test.jsonl')}")

if __name__ == "__main__":
    main()
