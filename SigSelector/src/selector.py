
class DocumentSelector:
    """
    Selects the most relevant source documents from a Multi-News input
    based on aggregated SigExt phrase scores.
    """
    def __init__(self, delimiter="|||||"):
        self.delimiter = delimiter

    def score_documents(self, data):
        full_text = data.get('trunc_input', '')
        phrases_info = data.get('trunc_input_phrases', [])
        scores_info = data.get('input_kw_model', [])

        doc_texts = full_text.split(self.delimiter)
        doc_spans = []
        current_idx = 0
        delimiter_len = len(self.delimiter)

        for i, text in enumerate(doc_texts):
            start = current_idx
            end = current_idx + len(text)
            doc_spans.append({
                'id': i,
                'text': text,
                'start': start,
                'end': end,
                'score': 0.0,
                'num_phrases': 0
            })
            current_idx = end + delimiter_len

        score_map = {item['kw_index']: item['score'] for item in scores_info}

        for idx, p_info in enumerate(phrases_info):
            if idx not in score_map:
                continue
            p_score = score_map[idx]
            p_start = p_info['index']

            for doc in doc_spans:
                if doc['start'] <= p_start < doc['end']:
                    doc['score'] += p_score
                    doc['num_phrases'] += 1
                    break

        return doc_spans

    def select_top_k(self, data, k=2):
        scored_docs = self.score_documents(data)
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        selected = scored_docs[:k]
        final_text = "\n".join([d['text'].strip() for d in selected])
        
        return final_text, scored_docs
