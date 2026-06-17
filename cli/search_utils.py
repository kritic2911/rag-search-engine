BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR = "/home/kc/rag-search-engine/cache/"
MODEL_PATH = CACHE_DIR + "model_weights"
SCORE_PRECISION = 4

def format_search_result(doc_id, title, document, score, metadata=None):
    return {
      "id": doc_id,
      "title": title,
      "document": document[:100],
      "score": round(float(score), SCORE_PRECISION),
      "metadata": metadata or {}
    }
