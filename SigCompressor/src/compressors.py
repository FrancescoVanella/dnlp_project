from nltk.tokenize import PunktSentenceTokenizer

class SigExtSentenceCompressor:
  """
  Sentence level compressor
  """
  def __init__(self):
    # Tokenizer produces coordinates of phrases
    self.tokenizer = PunktSentenceTokenizer()

  def compress(self, data, keep_ratio=0.3):

    raw_text = data['raw_input']

    # Map phrases to scores
    phrase_scores_map = {}
    for item in data['input_kw_model']:
      # 'kw_index' is the position in 'trunc_input_phrases' array
      phrase_scores_map[item['kw_index']] = item['score']

    # [(start_char_index, score), ...]
    phrases_extracted = []
    for idx, p_info in enumerate(data['trunc_input_phrases']):
      if idx in phrase_scores_map:
        phrases_extracted.append({
          'start_char': p_info['index'],
          'score': phrase_scores_map[idx],
          'text': p_info['phrase']
        })

    # (start, end) for each sentence
    sentence_spans = list(self.tokenizer.span_tokenize(raw_text))

    sentences_data = []

    # Compute a default low score for phrases without score if any
    all_scores = [p['score'] for p in phrases_extracted]
    min_score = min(all_scores) if all_scores else -100.0
    default_score = min_score - 1.0

    for i, (sent_start, sent_end) in enumerate(sentence_spans):
      # Estract text from sentence
      sent_text = raw_text[sent_start:sent_end]

      #----------------
      # We look for which phrases fall in each sentence
      contained_scores = []
      for p in phrases_extracted:
        if sent_start <= p['start_char'] < sent_end:
          contained_scores.append(p['score'])

      # Maxpool of phrase scores in each sentence
      final_sent_score = max(contained_scores) if contained_scores else default_score

      sentences_data.append({
        'id': i,
        'text': sent_text,
        'score': final_sent_score,
        'num_keywords': len(contained_scores)
      })

    # Ordering by decreasing score
    sentences_sorted_by_importance = sorted(sentences_data, key=lambda x: x['score'], reverse=True)

    # Cutoff
    num_keep = int(len(sentences_data) * keep_ratio)
    if num_keep < 1: num_keep = 1

    kept_sentences = sentences_sorted_by_importance[:num_keep]

    # Ordering by position in original text
    kept_sentences_ordered = sorted(kept_sentences, key=lambda x: x['id'])

    compressed_text = " ".join([s['text'] for s in kept_sentences_ordered])

    return compressed_text

class SigExtPhraseCompressor:
  def __init__(self):
    pass

  def compress(self, data, keep_ratio=0.3):
    """
    Phrase level compressor
    """
    # Data estraction
    phrases_list = data['trunc_input_phrases']
    scores_list = data['input_kw_model']
    total_phrases = len(phrases_list)
    if total_phrases == 0:
      return ""

    # Map score to phrase
    candidates = []
    seen_indices = set() # Avoid duplicates

    for item in scores_list:
      idx = item['kw_index'] # Positional index in phrases_list
      score = item['score']

      # Security check
      if 0 <= idx < total_phrases:
        phrase_obj = phrases_list[idx]

        candidate = {
            'text': phrase_obj['phrase'],
            'char_index': phrase_obj['index'],
            'score': score
        }

        if idx not in seen_indices:
          candidates.append(candidate)
          seen_indices.add(idx)

    # Order by score in decreasing order
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # Compute number of phrases to keep
    num_to_keep = max(1, int(total_phrases * keep_ratio))

    # Take the top phrases based on num_to_keep
    selected_candidates = candidates[:num_to_keep]

    # Recostruction of test based on position in the test
    selected_candidates.sort(key=lambda x: x['char_index'])

    compressed_text = " ".join([c['text'] for c in selected_candidates])

    return compressed_text
