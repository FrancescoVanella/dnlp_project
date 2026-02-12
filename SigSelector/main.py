
import os
import sys
import json
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import torch
import nltk
from transformers import logging as transformers_logging

from src.selector import DocumentSelector
from src.benchmark import BenchmarkRunner

def check_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"Warning: NLTK download failed: {e}")

def main():
    # --- CLI Arguments ---
    parser = argparse.ArgumentParser(description="SigSelector Benchmark")
    parser.add_argument("--strategies", nargs="+", default=["all"], 
                        help="Strategies to run. Options: '1', '2' (integers for Top-K), 'full' (Full Text), or 'all' (runs 1, 2, and full).")
    parser.add_argument("--models", nargs="+", default=["gpt", "llama"], choices=["gpt", "llama"], 
                        help="Models to run (gpt, llama)")
    parser.add_argument("--limit", type=int, default=200, 
                        help="Limit number of documents to process (useful for quick testing)")
    args = parser.parse_args()
    
    # Parse strategies
    active_strategies = []
    if "all" in args.strategies:
        active_strategies = ["full", "1", "2"]
    else:
        active_strategies = args.strategies

    print("--- Starting SigSelector Benchmark ---")
    print(f"Config: strategies={active_strategies}, models={args.models}, limit={args.limit}")
    
    transformers_logging.set_verbosity_error()
    check_nltk_resources()
    
    # Setup
    openai_key = os.environ.get("OPENAI_API_KEY")
    # We also groq mostly to speed up the generation phase of Llama
    groq_key = os.environ.get("GROQ_API_KEY") 

    if "gpt" in args.models and not openai_key:
        print("WARNING: OPENAI_API_KEY not set but GPT requested.")
    if "llama" in args.models and not groq_key:
        print("WARNING: GROQ_API_KEY not set but Llama requested.")
    
    runner = BenchmarkRunner(openai_api_key=openai_key, groq_api_key=groq_key)
    selector = DocumentSelector()
    
    # Load data
    data_path = "experiments/multi_news_dataset_with_keyphrase/test.jsonl"
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Please run 'prepare_sigext.py' first.")
        return

    with open(data_path, 'r') as f:
        test_docs = [json.loads(line) for line in f]
    
    subset_docs = test_docs[:args.limit]
    results = []

    # Execution
    print(f"Processing {len(subset_docs)} documents")
    for i, doc in enumerate(tqdm(subset_docs)):
        try:
            full_text = doc['raw_input']
            entry = {
                "id": i,
                "reference": doc['raw_output'],
                "len_full": len(full_text)
            }
            
            # Prepare inputs based on strategies
            inputs = {}
            
            # Prepare full text if requested
            if "full" in active_strategies:
                inputs["full"] = full_text
            
            # Prepare Top-K texts
            for s in active_strategies:
                if s.isdigit():
                    k = int(s)
                    text, _ = selector.select_top_k(doc, k=k)
                    inputs[f"top{k}"] = text
                    entry[f"len_top{k}"] = len(text)

            # Run models
            models_map = []
            if "gpt" in args.models: models_map.append(("gpt", runner.get_summary_gpt))
            if "llama" in args.models: models_map.append(("llama", runner.get_summary_groq))

            for model_name, func in models_map:
                for strat_name, input_text in inputs.items():
                    # key example: gpt_full, gpt_top1
                    entry[f"{model_name}_{strat_name}"] = func(input_text)

            results.append(entry)

        except Exception as e:
            print(f"Error on {i}: {e}")

    # Save generations
    df_res = pd.DataFrame(results)
    df_res.to_csv("sigselector_generations.csv", index=False)
    print("Generations saved to sigselector_generations.csv")

    # Compute metrics
    print("Computing ROUGE and BERTScore")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device="cuda" if torch.cuda.is_available() else "cpu")
    
    metrics_list = []
    
    # Build list of columns to evaluate
    # Strategies are dynamic now, so we verify what exists in df_res
    refs = df_res['reference'].tolist()

    # Identify columns that look like 'model_strategy'
    for col in df_res.columns:
        if col in ['id', 'reference'] or col.startswith('len_'):
            continue
        
        # Parse model and strategy 
        parts = col.split('_')
        if len(parts) < 2: continue
        model_label = parts[0].upper()
        strat_label = parts[1].capitalize()
        
        label = f"{model_label} {strat_label}"
        preds = df_res[col].fillna("").tolist()
        
        # ROUGE
        r1, r2, rl = [], [], []
        for r, p in zip(refs, preds):
            if not p.strip(): 
                r1.append(0); r2.append(0); rl.append(0)
                continue
            s = scorer.score(r, p)
            r1.append(s['rouge1'].fmeasure * 100)
            r2.append(s['rouge2'].fmeasure * 100)
            rl.append(s['rougeL'].fmeasure * 100)
            
        # BERTScore
        P, R, F1 = bert_scorer.score(preds, refs, verbose=False)
        b_scores = [f.item() * 100 for f in F1]
        
        metrics_list.append({
            "Strategy": label,
            "ROUGE-1": np.mean(r1),
            "ROUGE-2": np.mean(r2),
            "ROUGE-L": np.mean(rl),
            "BERTScore": np.mean(b_scores)
        })

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv("final_benchmark_metrics.csv", index=False)
    
    print("
=== Final Metrics ===")
    print(df_metrics.round(2).to_string(index=False))

if __name__ == "__main__":
    main()
