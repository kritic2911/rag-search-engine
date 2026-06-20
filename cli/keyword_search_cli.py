#!/usr/bin/env python3
import argparse
from lib.keyword_search import tokenize, search_by_title, invInd
import math
from search_utils import BM25_B, BM25_K1

def tokenize_single_term(ip_term: str) -> str:
    term = tokenize(ip_term)
    if len(term) != 1:
        raise Exception("Tokenizer did not return a single token")
    return term[0]

def call_bm25_idf(ip_term: str) -> float:
    invInd.load()
    term = tokenize_single_term(ip_term)
    return float(invInd.get_bm25_idf(term))
    
def call_bm25_tf(doc_id: int, ip_term: str, k1=BM25_K1, b=BM25_B) -> float:
    invInd.load()
    term = tokenize_single_term(ip_term)
    saturated_tf = invInd.get_bm25_tf(doc_id, term, k1, b)
    return saturated_tf

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the Inverted Index")

    tf_parser = subparsers.add_parser("tf", help="Print term frequencies")
    tf_parser.add_argument("document_id", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="Term for calculating frequency")

    idf_parser = subparsers.add_parser("idf", help="Inverse Document Frequency")
    idf_parser.add_argument("term", type=str, help="Term for calculating IDF")

    tfidf_parser = subparsers.add_parser("tfidf", help="TF-IDF")
    tfidf_parser.add_argument("document_id", type=int, help="Document id")
    tfidf_parser.add_argument("term", type=str, help="Term to calculate TF-IDF for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="bm25 IDF")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
      "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("--k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("--b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results to be displayed")
    
    args = parser.parse_args()

    match args.command:
        case "search":
            invInd.load()
            print(f"Searching for: {args.query}")
            result = sorted(search_by_title(args.query))
            for i in range(len(result)):
                print(f"{i + 1}. {result[i]} {invInd.docmap[result[i]]}")
            pass
            try:
                invInd.load()
            except FileNotFoundError:
                print("Unable to load indices")
                exit()
            pass
        case "build":
            invInd.build()
            invInd.save()
            pass
        case "tf":
            invInd.load()
            doc_id = args.document_id
            term = args.term
            term = tokenize_single_term(term)
            print(invInd.get_tf(doc_id, term))
            pass
        case "idf":
            invInd.load()
            term = args.term
            term = tokenize_single_term(term)
            total_doc_count = len(invInd.docmap)
            doc_freq = len(invInd.index[term])
            idf = math.log((total_doc_count + 1) / (doc_freq + 1))
            # idf = inv_doc_frequency(term, invInd)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            pass
        case "tfidf":
            invInd.load()
            doc_id = args.document_id
            term = args.term
            term = tokenize_single_term(term)
            tf = invInd.get_tf(doc_id, term)
            total_doc_count = len(invInd.docmap)
            doc_freq = len(invInd.index[term])
            idf = math.log((total_doc_count + 1) / (doc_freq + 1))
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{doc_id}': {tf_idf:.2f}")
            pass
        case "bm25idf":
            invInd.load()
            term = args.term
            bm25idf = call_bm25_idf(term)
            print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")
            pass
        case "bm25tf":
            invInd.load()
            docid = args.doc_id
            term = args.term
            k1 = args.k1
            b = args.b
            bm25tf = call_bm25_tf(docid, term, k1, b)
            print(f"BM25 TF score of '{term}' in document '{docid}': {bm25tf:.2f}")
            pass
        case "bm25search":
            invInd.load()
            query = args.query
            limit = args.limit
            results = invInd.bm25_search(query, limit)
            for i, res in enumerate(results, 1):
                doc_id = res['doc_id']
                score = res['score']
                doc_name = invInd.docmap[doc_id]
                print(f"{i}. ({doc_id}) {doc_name} - Score: {score: .2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    
    main()
