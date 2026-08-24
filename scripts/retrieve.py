import argparse
from pathlib import Path

# We will import libraries inside main to prevent import errors if dependencies aren't loaded yet.

def main():
    parser = argparse.ArgumentParser(description="Corpus Vector Retriever CLI (Phase 3b)")
    parser.add_argument("--query", required=True, help="Gujarati search query string")
    parser.add_argument("--db-dir", default="../data/vector_db", help="Folder containing the Chroma database")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--type", choices=["all", "chunk", "knowledge"], default="all", help="Filter retrieval by record type")
    args = parser.parse_args()
    
    db_dir = Path(args.db_dir).resolve()
    if not db_dir.exists():
        print(f"Error: Chroma database not found at {db_dir}. Run indexing first.")
        return
        
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError as e:
        print(f"Error: Required libraries not installed. Details: {e}")
        return
        
    # Load model
    model_name = "intfloat/multilingual-e5-base"
    model = SentenceTransformer(model_name)
    
    # Load Chroma
    chroma_client = chromadb.PersistentClient(path=str(db_dir))
    try:
        collection = chroma_client.get_collection(name="author_voice_collection")
    except Exception:
        print("Error: Collection 'author_voice_collection' not found in database. Run index_corpus.py first.")
        return
        
    # Prep query vector (E5 requires prefixing search query with "query: ")
    query_text = f"query: {args.query}"
    query_vector = model.encode([query_text], convert_to_numpy=True).tolist()[0]
    
    # Set up metadata filters if requested
    where_filter = {}
    if args.type != "all":
        where_filter = {"type": args.type}
        
    # Query database
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=args.top_k,
        where=where_filter if where_filter else None
    )
    
    # Print results
    print(f"\n🔍 Semantic Search Results for: '{args.query}'")
    print(f"======================================================================")
    
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    if not ids:
        print("No matching results found.")
        return
        
    for i in range(len(ids)):
        doc_id = ids[i]
        doc_text = docs[i]
        meta = metadatas[i]
        dist = distances[i]
        
        # Remove E5 prefix "passage: " from document text for clean display
        clean_doc_text = doc_text
        if doc_text.startswith("passage: "):
            clean_doc_text = doc_text[len("passage: "):]
            
        record_type = meta.get("type", "unknown").upper()
        book_title = meta.get("book", "Unknown Book")
        page_str = f"p. {meta.get('page_start', '?')}"
        if meta.get('page_start') != meta.get('page_end'):
            page_str += f"-{meta.get('page_end', '?')}"
            
        print(f"\n[{i+1}] {record_type} | {book_title} | {page_str} | ID: {meta.get('chunk_id')} | Distance: {dist:.4f}")
        print(f"----------------------------------------------------------------------")
        print(clean_doc_text)
        print(f"======================================================================")

if __name__ == "__main__":
    main()
