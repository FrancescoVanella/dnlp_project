import os
import json
import argparse
import pandas as pd
import jsonlines
from bert_score import BERTScorer
import torch

from src.utils import seed_everything, download_nltk_resources
from src.benchmark import BenchmarkRunner
from src.compressors import SigExtSentenceCompressor, SigExtPhraseCompressor
from src.pipeline import run_compression, run_generation, run_evaluation, run_full_text_benchmark

def read_jsonl(file_path):
    docs = []
    with jsonlines.open(file_path) as reader:
        for doc in reader:
            docs.append(doc)
    return docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to jsonl file with sigext extracted data")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--openai_key", type=str, default=None)
    
    # CLI Arguments for configuration
    parser.add_argument("--compressor", type=str, default="all", choices=["sentence", "phrase", "all"], help="Type of compression to run")
    parser.add_argument("--ratios", type=float, nargs="+", default=None, help="List of keep ratios (e.g. 0.2 0.3)")
    parser.add_argument("--llm", type=str, default="gpt", choices=["gpt", "mistral", "all"], help="LLM to use for summarization")

    args = parser.parse_args()

    seed_everything(42)
    download_nltk_resources()
    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading data")
    test_docs = read_jsonl(args.input_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_scorer = BERTScorer(lang="en", device=device, rescale_with_baseline=True)
    
    runner = BenchmarkRunner(openai_api_key=args.openai_key)

    # Select Models based on CLI args
    models = []
    if args.llm in ["mistral", "all"]:
        models.append(("mistralai/Mistral-7B-Instruct-v0.3", "local"))
    if args.llm in ["gpt", "all"]:
        models.append(("gpt-3.5-turbo", "api"))
    
    if not models:
        print("No models selected. Exiting.")
        return

    # Default ratios if not provided
    default_ratios_sent = [0.18, 0.22, 0.3]
    default_ratios_phrase = [0.3, 0.4, 0.6]
    
    # --- SENTENCE LEVEL ---
    if args.compressor in ["sentence", "all"]:
        print("\n--- SENTENCE LEVEL ---")
        keep_ratios_sent = args.ratios if args.ratios else default_ratios_sent
        
        compressor_sent = SigExtSentenceCompressor()
        compressed_data_sent, ratio_chars_sent = run_compression(test_docs, keep_ratios_sent, compressor_sent)
        predictions_sent = run_generation(test_docs, compressed_data_sent, keep_ratios_sent, models, runner)
        
        with open(os.path.join(args.results_dir, "sentence_summaries.json"), "w") as f:
            json.dump(predictions_sent, f)
            
        metrics_sent = run_evaluation(test_docs, predictions_sent, bert_scorer)
        
        for key in metrics_sent:
            if "_ratio_" in key:
                r = float(key.split("_ratio_")[-1])
                if r in ratio_chars_sent: metrics_sent[key]['compression_rate'] = ratio_chars_sent[r]

        df_sent = pd.DataFrame(metrics_sent).T
        df_sent.to_csv(os.path.join(args.results_dir, "sentence_metrics.csv"))
        print(df_sent)

    # --- PHRASE LEVEL ---
    if args.compressor in ["phrase", "all"]:
        print("\n--- PHRASE LEVEL ---")
        keep_ratios_phrase = args.ratios if args.ratios else default_ratios_phrase
        
        compressor_phrase = SigExtPhraseCompressor()
        compressed_data_phrase, ratio_chars_phrase = run_compression(test_docs, keep_ratios_phrase, compressor_phrase)
        predictions_phrase = run_generation(test_docs, compressed_data_phrase, keep_ratios_phrase, models, runner)
        
        with open(os.path.join(args.results_dir, "phrase_summaries.json"), "w") as f:
            json.dump(predictions_phrase, f)

        metrics_phrase = run_evaluation(test_docs, predictions_phrase, bert_scorer)
        
        for key in metrics_phrase:
            if "_ratio_" in key:
                r = float(key.split("_ratio_")[-1])
                if r in ratio_chars_phrase: metrics_phrase[key]['compression_rate'] = ratio_chars_phrase[r]

        df_phrase = pd.DataFrame(metrics_phrase).T
        df_phrase.to_csv(os.path.join(args.results_dir, "phrase_metrics.csv"))
        print(df_phrase)

    # --- FULL TEXT ---
    # Always run full text baseline unless explicitly disabled (feature for future?)
    print("\n--- FULL TEXT ---")
    full_text_results = run_full_text_benchmark(test_docs, models, runner, bert_scorer=bert_scorer)
    df_full = pd.DataFrame(full_text_results).T
    df_full.to_csv(os.path.join(args.results_dir, "full_text_metrics.csv"))
    print(df_full)

if __name__ == "__main__":
    main()
