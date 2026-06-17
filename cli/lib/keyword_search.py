from collections import Counter
import json
import math
from nltk.stem import PorterStemmer
import os
import pickle
import string
from pathlib import Path
from search_utils import BM25_B, BM25_K1, CACHE_DIR

class InvertedIndex:
    def __init__(self):
        self.index = {}  # str: [int] => token: doc_ids
        self.docmap = {}  # int: str => doc_id: doc_title
        self.term_frequencies = {}  # int: Counter() => doc_id: term_freq_count in a document
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        # self.stopwords = stop_words or []
        # self.movie_list = movie_list or []

    def __add_document(self, doc_id: str, doc_title: str, text: str):
        self.docmap[doc_id] = doc_title
        tokenized_text = tokenize(text)
        doc_counter = Counter()
        total_tokens = len(tokenized_text)
        self.doc_lengths[doc_id] = total_tokens
        for tok in tokenized_text:
            doc_counter[tok] += 1
            if tok not in self.index.keys():
                self.index[tok] = set([doc_id])
            else:
                self.index[tok].add(doc_id)
        self.term_frequencies[doc_id] = doc_counter or {}

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index[term])

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.index[term])
        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf = self.get_tf(doc_id, term)
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component

    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_idf * bm25_tf

    def bm25_search(self, query, limit):
        tokenized_query = tokenize(query)
        scores = {}
        for token in tokenized_query:
            for doc in self.index[token]:
                if doc in scores.keys():
                    scores[doc] += self.bm25(doc, token)
                else:
                    scores[doc] = self.bm25(doc, token)
        top_docs = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit])
        return top_docs

    def __get_avg_doc_length(self) -> float:
        total_doc_len = 0
        total_docs = len(self.docmap)
        for i in range(1, total_docs + 1):
            total_doc_len += self.doc_lengths[i]
        if total_docs:    
            return total_doc_len / total_docs
        return 0.0
    
    def build(self) -> tuple:
        for m in movie_list:
            input_text = f"{m['title']} {m['description']}"
            doc_id = m["id"]
            doc_title = m["title"]
            self.__add_document(doc_id, doc_title, input_text)
        return self.index, self.docmap

    def save(self):
        dir_path = Path("/home/kc/rag-search-engine/cache")
        dir_path.mkdir(parents=True, exist_ok=True)
        with open( CACHE_DIR + "index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open(CACHE_DIR + "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open(CACHE_DIR + "term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(CACHE_DIR + "doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open(CACHE_DIR + "index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open(CACHE_DIR + "docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open(CACHE_DIR + "term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(CACHE_DIR + "doc_lengths.pkl", "rb") as f:
            self.doc_lengths = pickle.load(f)


def load_movie(file_path: str) -> list[dict]:
    with open(file_path, "r") as f:
        movies = json.load(f)
    return movies["movies"]


def preprocess_text(text: str) -> str:
    text = text.lower()
    remove_chars = string.punctuation
    rem_table = str.maketrans("", "", remove_chars)
    text = text.translate(rem_table)
    return text


def tokenize(ip_text: str) -> list[str]:
    stemmer = PorterStemmer()
    text = preprocess_text(ip_text)
    text = text.split()
    text = [q for q in text if q and q not in stop_words]
    text = [stemmer.stem(token) for token in text]
    return text


def search_by_title(ip_query: str) -> set:
    ans = set()
    query = tokenize(ip_query)
    for token in query:
        if token in invInd.index.keys():
            for doc in invInd.get_documents(token):
                ans.add(doc)
            if len(ans) > 4:
                break
    return set(sorted(ans)[:5])
    
movie_list = load_movie("/home/kc/rag-search-engine/data/movies.json")

with open("/home/kc/rag-search-engine/data/stopwords.txt", "r") as f:
    stop_words = f.read()
stop_words = [preprocess_text(s) for s in stop_words.splitlines()]
    
invInd = InvertedIndex()