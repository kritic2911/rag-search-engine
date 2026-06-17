import json
import re
from pathlib import Path
from typing import TypedDict
import numpy as np
import torch
from search_utils import CACHE_DIR, MODEL_PATH, format_search_result
from sentence_transformers import SentenceTransformer

if torch.cuda.is_available():
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("GPU not found. Falling back to CPU.")


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embeddings = None
        self.documents = None
        self.document_map = {}

        model_path = Path(MODEL_PATH)
        if not model_path.is_dir():
            self.model = SentenceTransformer(model_name, device=device)
            self.model.save(MODEL_PATH)
        else:
            self.model = SentenceTransformer(MODEL_PATH, device=device)

    def generate_embedding(self, text: str) -> np.ndarray:
        if len(text) == 0 or len(text.strip()) == 0:
            raise ValueError("Empty text input")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        movie_list = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_string = f"{doc['title']}: {doc['description']}"
            movie_list.append(doc_string)
        self.embeddings = self.model.encode(movie_list, show_progress_bar=True)
        np.save(CACHE_DIR + "movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        file_path = Path(CACHE_DIR + "movie_embeddings.npy")
        if file_path.is_file():
            self.embeddings = np.load(file_path, allow_pickle=True)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        else:
            return self.build_embeddings(documents)

    def search(self, query: str, limit: int):
        if len(self.embeddings) == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        similarity_list = []

        for document in self.documents:
            doc_id = document["id"] - 1
            cosine_score = cosine_similarity(query_embedding, self.embeddings[doc_id])
            similarity_list.append((cosine_score, document))
        similarity_list.sort(key=lambda a: a[0], reverse=True)

        result = []
        for score, document in similarity_list[:limit]:
            result.append(
                {
                    "score": score,
                    "title": document["title"],
                    "description": document["description"],
                }
            )
        return result


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int
    chunk: str


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        # self.documents = None
        # self.document_map = {}

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        chunks = []
        chunk_metadata = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            if len(doc["description"]) > 0:
                result = semantic_chunk_query(doc["description"], 4, 1)
                chunks.extend(result)
                chunk_len = len(result)
                for ind, chunk in enumerate(result):
                    chunk_metadata.append(
                        {
                            "movie_idx": doc["id"],
                            "chunk_idx": ind,
                            "total_chunks": chunk_len,
                            "chunk": chunk[:20] + "..." + chunk[-20:],
                        }
                    )
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        np.save(CACHE_DIR + "chunk_embeddings.npy", self.chunk_embeddings)
        with open(CACHE_DIR + "chunk_metadata.json", "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(chunks)}, f, indent=2
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        chunk_path = Path(CACHE_DIR + "chunk_embeddings.npy")
        metadata_path = Path(CACHE_DIR + "chunk_metadata.json")

        if chunk_path.is_file() and metadata_path.is_file():
            self.chunk_embeddings = np.load(chunk_path, allow_pickle=True)
            with open(CACHE_DIR + "chunk_metadata.json", "r") as f:
                self.chunk_metadata = json.load(f)['chunks']
            if len(self.chunk_embeddings) == len(documents):
                return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        embedding = self.generate_embedding(query)
        chunk_score = []
        movie_score = {}
        for chunk, metadata in zip(self.chunk_embeddings, self.chunk_metadata):
            score = cosine_similarity(chunk, embedding)
            chunk_score.append({
                'chunk_idx': metadata['chunk_idx'],
                'movie_idx': metadata['movie_idx'],
                'score': score
            })
            if metadata['movie_idx'] not in movie_score.keys():
                movie_score[metadata['movie_idx']] = score
            else:
                movie_score[metadata['movie_idx']] = max(score, movie_score[metadata['movie_idx']])
            
        #     if metadata['movie_idx'] not in movie_score.keys():
        #         movie_score[metadata['movie_idx']] = [score]
        #     else:
        #         movie_score[metadata['movie_idx']].append(score)

        # for k, v in movie_score.items():
        #     movie_score[k] = sum(v) / len(v)
        
        sorted_movie_score = sorted(movie_score.items(), key=lambda item: item[1], reverse=True)[:limit]
        result = [
            format_search_result(
                res[0],
                self.document_map[res[0]]['title'],      
                self.document_map[res[0]]['description'], 
                res[1]
            )
            for res in sorted_movie_score
        ]
        return result

def chunk_query(query: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    query_split = query.split()
    n = len(query_split)
    total_chars = len(query)
    result = []
    i = 0
    while i < n:
        result.append(" ".join(query_split[i : i + chunk_size]))
        i += chunk_size
        if i >= overlap:
            i -= overlap
    return result


def semantic_chunk_query(query: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    query.strip()
    if not query:
        return []
    query_split = re.split(r"(?<=[.!?])\s+", query)
    if len(query_split) == 1 and query_split[-1] not in ".?!":
        return [query]
    total_chars = len(query)
    n = len(query_split)
    result = []
    i = 0
    while i < n:
        q = (' '.join(query_split[i : i + max_chunk_size])).strip()
        i += max_chunk_size
        if q:
            result.append(q)
            if i >= overlap:
                i -= overlap
    return result


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 * norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def verify_model():
    semanticSearch = SemanticSearch()
    MODEL = semanticSearch.model
    MAX_LENGTH = MODEL.max_seq_length
    print(f"Model loaded: {MODEL}\nMax sequence length: {MAX_LENGTH}")


def embed_text(text):
    semanticSearch = SemanticSearch()
    embedding = semanticSearch.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def load_movies():
    file_path = "/home/kc/rag-search-engine/data/movies.json"
    with open(file_path, "r") as f:
        movies = json.load(f)
    return movies["movies"]


def verify_embeddings():
    semanticSearch = SemanticSearch()
    documents = load_movies()
    embeddings = semanticSearch.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def search(query: str, limit: int = 5):
    semanticSearch = SemanticSearch()
    documents = load_movies()
    embeddings = semanticSearch.load_or_create_embeddings(documents)
    result = semanticSearch.search(query, limit)
    for i, res in enumerate(result, 1):
        print(
            f"{i}. {res['title']} (score: {res['score']:.4f})\n {res['description'][:100]}..."
        )


def embed_chunks():
    documents = load_movies()
    chunkedSemanticSearch = ChunkedSemanticSearch()
    embeddings = chunkedSemanticSearch.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked(query: str, limit: int = 5):
    documents = load_movies()
    chunkedSemanticSearch = ChunkedSemanticSearch()
    embeddings = chunkedSemanticSearch.load_or_create_chunk_embeddings(documents)
    result = chunkedSemanticSearch.search_chunks(query, limit)
    for i, res in enumerate(result, 1):
        print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['document']}...")