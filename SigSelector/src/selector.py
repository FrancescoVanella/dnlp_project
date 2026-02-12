import numpy as np

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
                'num_scored_phrases': 0,
                'total_phrases': 0 
            })
            current_idx = end + delimiter_len

        score_map = {item['kw_index']: item['score'] for item in scores_info}

        for idx, p_info in enumerate(phrases_info):
            p_start = p_info['index']
            
            # Identify which document this phrase belongs to
            target_doc = None
            for doc in doc_spans:
                if doc['start'] <= p_start < doc['end']:
                    target_doc = doc
                    break
            
            if target_doc:
                # Increment total phrases count for this doc
                target_doc['total_phrases'] += 1
                
                # If the phrase has a score, add it
                if idx in score_map:
                    p_score = score_map[idx]
                    # Convert log-prob to probability 
                    term_score = np.exp(p_score)
                    target_doc['score'] += term_score
                    target_doc['num_scored_phrases'] += 1

        return doc_spans

    def select_top_k(self, data, k=2):
        scored_docs = self.score_documents(data)
        
        # Normalize scores by the total number of phrases in each document
        for doc in scored_docs:
            if doc['total_phrases'] > 0:
                doc['score'] = doc['score'] / doc['total_phrases']
            else:
                doc['score'] = 0.0
                
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        selected = scored_docs[:k]
        final_text = "\n".join([d['text'].strip() for d in selected])
        return final_text, scored_docs
