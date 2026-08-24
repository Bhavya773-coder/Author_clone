import os
import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm

# We will import sentence_transformers and chromadb inside main to allow 
# running this script's initialization checks after dependencies are installed.

def parse_yaml_frontmatter(content):
    """Simple parser to read YAML metadata from markdown files without external dependencies."""
    if not content.startswith("---"):
        return {}, content
        
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
        
    yaml_text = parts[1]
    body_text = parts[2]
    
    metadata = {}
    for line in yaml_text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            metadata[k.strip()] = v.strip().strip('"').strip("'")
            
    return metadata, body_text

def main():
    parser = argparse.ArgumentParser(description="Corpus Vector Indexer (Phase 3a)")
    parser.add_argument("--cleaned-dir", default="../data/cleaned", help="Cleaned corpus folder")
    parser.add_argument("--knowledge-dir", default="../data/knowledge", help="Structured knowledge folder")
    parser.add_argument("--db-dir", default="../data/vector_db", help="Folder to save local Chroma database")
    args = parser.parse_args()
    
    cleaned_dir = Path(args.cleaned_dir).resolve()
    knowledge_dir = Path(args.knowledge_dir).resolve()
    db_dir = Path(args.db_dir).resolve()
    
    # Import RAG libraries
    print("Initializing local NLP and Database libraries...")
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError as e:
        print(f"Error: Required libraries not installed. Run 'pip install -r requirements.txt' first. Details: {e}")
        return
        
    # Check for manifest
    manifest_path = cleaned_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest.json not found in {cleaned_dir}. Run cleaning first.")
        return
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    processed_books = manifest.get("books", {})
    
    # Gather clean text chunks
    print("Loading cleaned text chunks...")
    chunks_to_index = []
    for book_slug in processed_books.keys():
        chunks_file = cleaned_dir / book_slug / "chunks.jsonl"
        if not chunks_file.exists():
            continue
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks_to_index.append(json.loads(line))
                    
    print(f"Loaded {len(chunks_to_index)} clean chunks.")
    
    # Gather structured knowledge base records
    print("Loading structured knowledge units...")
    knowledge_to_index = []
    for book_slug in processed_books.keys():
        book_k_dir = knowledge_dir / book_slug
        if not book_k_dir.exists():
            continue
        for md_file in book_k_dir.glob("*.md"):
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            metadata, body_text = parse_yaml_frontmatter(content)
            if metadata.get("status") == "empty":
                continue # Skip empty placeholder records
            
            knowledge_to_index.append({
                "chunk_id": metadata.get("chunk_id", md_file.stem),
                "book": metadata.get("book", book_slug),
                "book_slug": book_slug,
                "page_start": int(metadata.get("page_start", 1)),
                "page_end": int(metadata.get("page_end", 1)),
                "text": body_text.strip()
            })
            
    print(f"Loaded {len(knowledge_to_index)} structured knowledge units.")
    
    if not chunks_to_index and not knowledge_to_index:
        print("Error: No documents found to index.")
        return
        
    # Load SentenceTransformer model
    # Model: intfloat/multilingual-e5-base (1.1 GB, great for Indic/Gujarati script)
    model_name = "intfloat/multilingual-e5-base"
    print(f"Loading embedding model '{model_name}' (this may take a minute on first run)...")
    model = SentenceTransformer(model_name)
    
    # Initialize Persistent Chroma DB
    print(f"Connecting to persistent Chroma DB at {db_dir}...")
    db_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(db_dir))
    
    # Delete old collection if it exists to allow full clean re-indexing
    try:
        chroma_client.delete_collection(name="author_voice_collection")
        print("Cleared previous vector index collection.")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(name="author_voice_collection")
    
    # Process text chunks
    docs = []
    ids = []
    metadatas = []
    
    print("Preparing documents for indexing...")
    for chunk in chunks_to_index:
        # Multilingual-E5 models require prefixing passage documents with "passage: "
        docs.append(f"passage: {chunk['text']}")
        ids.append(f"chunk_{chunk['chunk_id']}")
        metadatas.append({
            "chunk_id": chunk["chunk_id"],
            "book": chunk["book"],
            "book_slug": chunk["book_slug"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "type": "chunk"
        })
        
    # Process knowledge units
    for k_unit in knowledge_to_index:
        docs.append(f"passage: {k_unit['text']}")
        ids.append(f"knowledge_{k_unit['chunk_id']}")
        metadatas.append({
            "chunk_id": k_unit["chunk_id"],
            "book": k_unit["book"],
            "book_slug": k_unit["book_slug"],
            "page_start": k_unit["page_start"],
            "page_end": k_unit["page_end"],
            "type": "knowledge"
        })
        
    # Compute embeddings
    print(f"Encoding {len(docs)} documents into vector embeddings...")
    embeddings = model.encode(
        docs, 
        batch_size=64, 
        show_progress_bar=True, 
        convert_to_numpy=True
    ).tolist()
    
    # Batch add to Chroma DB (Chroma handles batching automatically but let's push in chunks of 500 to be safe)
    batch_size = 500
    print(f"Saving vector embeddings into Chroma DB (collections: 'author_voice_collection')...")
    for offset in range(0, len(docs), batch_size):
        limit = offset + batch_size
        collection.add(
            embeddings=embeddings[offset:limit],
            documents=docs[offset:limit],
            metadatas=metadatas[offset:limit],
            ids=ids[offset:limit]
        )
        
    print("\n=========================================")
    print("Vector DB Indexing Complete!")
    print(f"Persistent Storage: {db_dir}")
    print(f"Clean Chunks Indexed: {len(chunks_to_index)}")
    print(f"Knowledge Units Indexed: {len(knowledge_to_index)}")
    print(f"Total Vectors Stored: {collection.count()}")
    print("=========================================")

if __name__ == "__main__":
    main()
