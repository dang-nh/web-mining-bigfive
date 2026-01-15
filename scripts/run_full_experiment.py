
import subprocess
import sys
from pathlib import Path

# Config
RESULTS_DIR = "results_new"
LANGUAGES = ["en", "es", "it", "nl"]
PYTHON = sys.executable

def run_command(cmd, description):
    print(f"=== {description} ===")
    print(f"Command: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Success: {description}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}: {e}\n")

def main():
    root_dir = Path(__file__).parent.parent
    results_path = root_dir / RESULTS_DIR
    results_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Full Experiment Suite in {results_path}...\n")
    
    # 1. TF-IDF for all languages
    for lang in LANGUAGES:
        cmd = f"{PYTHON} scripts/train_eval_baseline_tfidf.py --lang {lang} --results_dir {RESULTS_DIR}"
        run_command(cmd, f"TF-IDF Baseline ({lang})")
        
    # 2. XLM-RoBERTa (Chunked) for all languages
    # Using existing data (en, es, it, nl)
    # Model: xlm-roberta-base
    # Suffix: {lang}_xlm
    for lang in LANGUAGES:
        # Note: --model_name xlm-roberta-base forces usage of XLM-R even for English
        # Using data_lang=lang ensuring correct data loading
        # Suffix: {lang}_xlm creates 'metrics_transformer_{lang}_xlm.csv'
        output_lang = f"{lang}_xlm"
        cmd = f"CUDA_VISIBLE_DEVICES=6 {PYTHON} scripts/train_eval_transformer.py --lang {output_lang} --data_lang {lang} --model_name xlm-roberta-base --epochs 50 --batch_size 4 --early_stopping 10 --results_dir {RESULTS_DIR}"
        run_command(cmd, f"XLM-RoBERTa Chunked ({lang})")

    # 3. Twitter-RoBERTa (Chunked) for English
    # Model: cardiffnlp/twitter-roberta-base (default for en)
    # Suffix: en_twitter
    cmd = f"CUDA_VISIBLE_DEVICES=7 {PYTHON} scripts/train_eval_transformer.py --lang en_twitter --data_lang en --epochs 50 --batch_size 4 --early_stopping 10 --results_dir {RESULTS_DIR}"
    run_command(cmd, "Twitter-RoBERTa Chunked (en)")
    
    print("Full Experiment Suite Completed.")

if __name__ == "__main__":
    main()
