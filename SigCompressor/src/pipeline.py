import numpy as np
from tqdm import tqdm
import gc
import torch
from .metrics import compute_detailed_metrics

def run_compression(test_docs, keep_ratios, compressor):
  """
  Compress texts, from this compressed versions generate the summaries and compute metrics
  """
  compressed_test_data = {}
  ratio_chars = {}

  print("=== COMPRESSION PHASE ===")
  for ratio in keep_ratios:
    compressed_txt_list = []
    compressed_lengths = []
    original_lengths = []

    for sample in test_docs:
        sigext_data = {
            'raw_input': sample['raw_input'],
            'trunc_input_phrases': sample['trunc_input_phrases'],
            'input_kw_model': sample['input_kw_model']
        }
        compressed_txt = compressor.compress(sigext_data, keep_ratio=ratio)
        compressed_txt_list.append(compressed_txt)
        compressed_test_data[ratio] = compressed_txt_list
        compressed_lengths.append(len(compressed_txt))
        original_lengths.append(len(sample['raw_input']))

    ratio_chars[ratio] = 1 - np.mean(compressed_lengths) / np.mean(original_lengths)

  print("=== COMPRESSION DONE ===")
  return compressed_test_data, ratio_chars

def run_generation(test_docs, compressed_test_data, keep_ratios, models, runner):
  """
  From the compressed versions generate the summaries and compute metrics
  """
  print("=== GENERATION PHASE ===")
  all_predictions = {}
  metrics_dict = {}
  for model_name, mode in models:
    print(f"=== {model_name} ===")

    model = None
    tokenizer = None

    if mode == "local":
      tokenizer, model = runner.load_model(model_name)

    for ratio in keep_ratios:
      preds_batch = []
      inference_data_batch = []

      for compressed_sample, sample in tqdm(list(zip(compressed_test_data[ratio], test_docs))):
        summary = ""
        if mode == "local":
          summary = runner.get_summary_local(model, tokenizer, model_name, compressed_sample)
        else:
          summary = runner.get_summary_gpt(compressed_sample)
        preds_batch.append(summary)
      key = f"{model_name}_ratio_{ratio}"
      all_predictions[key] = preds_batch

    if mode == "local":
      del model
      del tokenizer
      torch.cuda.empty_cache()
      gc.collect()
  return all_predictions

def run_evaluation(test_docs, all_predictions, bert_scorer):
  print("\n=== 3. EVALUATION PHASE ===")
  metrics_dict = {}
  ground_truths = [d['raw_output'] for d in test_docs]
  for key, preds in all_predictions.items():
    inference_data = [{"raw_output": gt} for gt in ground_truths]
    metrics = compute_detailed_metrics(inference_data, preds, bert_scorer)
    metrics_dict[key] = metrics

  return metrics_dict

def run_full_text_benchmark(test_docs, models, runner, bert_scorer=None):
  print("=== STARTING FULL TEXT BENCHMARK ===")
  metrics_results = {}

  for model_name, mode in models:
    print(f"\n=== Evaluating Model: {model_name} ===")

    model = None
    tokenizer = None

    if mode == "local":
      tokenizer, model = runner.load_model(model_name)

    preds_batch = []
    inference_data_batch = []

    for sample in tqdm(test_docs, desc=f"{model_name} Full"):
      full_text = sample['raw_input']
      summary = ""

      if mode == "local":
        if model is not None:
          summary = runner.get_summary_local(model, tokenizer, model_name, full_text)
        else:
          print("Error: Model not loaded correctly.")
      else:
        # API-based models (GPT, etc.)
        summary = runner.get_summary_gpt(full_text)

      preds_batch.append(summary)
      inference_data_batch.append({"raw_output": sample['raw_output']})

    key_name = f"{model_name}_full_text"

    print(f"Computing metrics for {key_name}")
    metrics = compute_detailed_metrics(inference_data_batch, preds_batch, bert_scorer=bert_scorer)
    metrics_results[key_name] = metrics

    # Cleanup for local models to save VRAM
    if mode == "local":
      print(f"Unloading {model_name}")
      model = None
      tokenizer = None
      torch.cuda.empty_cache()
      gc.collect()

  return metrics_results
