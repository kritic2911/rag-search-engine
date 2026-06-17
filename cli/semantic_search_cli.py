import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, search, chunk_query, semantic_chunk_query, embed_chunks

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Searcher CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Model verification")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate Text embedding")
    embed_text_parser.add_argument("text", type=str, help="Text to get embedding for")

    verify_embed_parser = subparsers.add_parser("verify_embeddings", help="verify embedding")

    search_parser = subparsers.add_parser("search", help="seaarch for a movie")
    search_parser.add_argument("query", type=str, help="movie title to search for")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of items to be displayed")

    chunk_parser = subparsers.add_parser("chunk", help="fixed sized document chunking")
    chunk_parser.add_argument("query", type=str, help="text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=200, help="size of a chunk")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="overlap in chunk")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="semantic chunk document")
    semantic_chunk_parser.add_argument("query", type=str, help="text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=4, help="size of chunk (sentences)")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="overlap in chunk(sentences)")

    embed_chunk_parser = subparsers.add_parser("embed_chunks", help="load movie documents and build chunk embeddings")
    
    args = parser.parse_args()
    
    match args.command:
        case "verify":
            verify_model()
            pass
        case "embed_text":
            text = args.text
            embed_text(text)
            pass
        case "verify_embeddings":
            verify_embeddings()
            pass
        case "search":
            query = args.query
            lim = args.limit
            search(query, lim)
            pass
        case "chunk":
            query = args.query
            chunk_size = args.chunk_size
            overlap = args.overlap
            result, total_chars = chunk_query(query, chunk_size, overlap)
            print(f"Chunking {total_chars} characters")
            for i, res in enumerate(result, 1):
                print(f"{i}. {res}")

            pass
        case "semantic_chunk":
            query = args.query
            max_chunk_size = args.max_chunk_size
            overlap = args.overlap
            result, total_chars = semantic_chunk_query(query, max_chunk_size, overlap)
            print(f"Semantically chunking {total_chars} characters")
            for i, res in enumerate(result, 1):
                print(f"{i}. {res}")
        case "embed_chunks":
            embed_chunks()
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
  