import numpy as np
from collections import defaultdict
from rouge_score import rouge_scorer
import sacrebleu
from nltk.tokenize import sent_tokenize, word_tokenize

def postprocess_text(preds, labels):
  preds = [pred.strip() for pred in preds]
  labels = [label.strip() for label in labels]

  preds_rouge = ["\n".join(sent_tokenize(pred)) for pred in preds]
  labels_rouge = ["\n".join(sent_tokenize(label)) for label in labels]
  return preds_rouge, labels_rouge, preds, labels

def compute_detailed_metrics(inference_data, preds, bert_scorer=None):
  """
  Compute ROUGE, BLEU, and BERTScore.
  """
  scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
  labels = [item["raw_output"] for item in inference_data]

  decoded_preds_rouge, decoded_labels_rouge, decoded_preds, decoded_labels = postprocess_text(preds, labels)

  # --- ROUGE ---
  result_element = defaultdict(list)
  for pred, label in zip(decoded_preds_rouge, decoded_labels_rouge):
    score = scorer.score(target=label, prediction=pred)
    for metric_name, value in score.items():
        result_element[f"{metric_name}p"].append(value.precision)
        result_element[f"{metric_name}r"].append(value.recall)
        result_element[f"{metric_name}f"].append(value.fmeasure)

  result = {}
  for metric_name, values in result_element.items():
    result[metric_name] = np.mean(values)

  result = {k: round(v * 100, 4) for k, v in result.items()}

  # --- BLEU ---
  bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
  result["bleu"] = round(bleu.score, 4)

  # --- BERTSCORE ---
  if bert_scorer:
    P, R, F1 = bert_scorer.score(
        decoded_preds,
        decoded_labels,
        verbose=False,
        batch_size=16
        )
    result["bertscore_f1"] = round(F1.mean().item() * 100, 4)

  # Average length of generated summaries
  prediction_lens = [len(word_tokenize(pred)) for pred in preds]
  result["gen_len"] = np.mean(prediction_lens)

  return result
