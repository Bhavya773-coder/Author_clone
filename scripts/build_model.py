#!/usr/bin/env python3
"""
OKS Model Builder — Creates model.pkl for Book Knowledge QA
=============================================================
Reads the OKS knowledge base and all cleaned book chunks, computes vector
embeddings using sentence-transformers, and packages everything into a
self-contained BookKnowledgeModel saved as model.pkl.

The model.pkl contains:
  - All OKS structured data (characters, themes, opinions, anecdotes, summaries)
  - All text chunks from 20 books with metadata
  - Pre-computed embedding vectors (numpy arrays) for semantic search
  - Configuration and system prompts

At query time, the model:
  1. Computes the query embedding
  2. Does cosine similarity search against stored embeddings
  3. Retrieves relevant chunks + OKS knowledge
  4. Sends to Ollama for answer generation

Usage:
  python -X utf8 scripts/build_model.py
  python -X utf8 scripts/build_model.py --oks-dir oks --cleaned-dir cleaned --output model.pkl
"""

import json
import pickle
import sys
import argparse
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


class BookKnowledgeModel:
    """
    Self-contained Book Knowledge QA model.
    
    This class packages the entire OKS knowledge base, text chunks,
    and pre-computed embeddings into a single serializable object.
    Load with pickle.load() and call .answer(question) to get answers.
    """
    
    VERSION = "1.0"
    
    def __init__(self):
        # ── OKS Data ────────────────────────────────────────────
        self.oks_master = {}
        self.oks_opinions = []
        self.oks_themes = {}
        self.oks_characters = {}
        self.oks_anecdotes = []
        self.oks_book_summaries = {}
        self.oks_cross_references = {}
        
        # ── Text Chunks (raw book text for retrieval) ───────────
        self.chunks = []           # list of dicts with text, metadata
        self.chunk_texts = []      # parallel list of text strings
        
        # ── Knowledge Documents (structured knowledge for retrieval)
        self.knowledge_docs = []   # list of dicts with text, metadata
        self.knowledge_texts = []  # parallel list of text strings
        
        # ── Pre-computed Embeddings ─────────────────────────────
        self.chunk_embeddings = None       # numpy array (N, D)
        self.knowledge_embeddings = None   # numpy array (M, D)
        self.embedding_model_name = "intfloat/multilingual-e5-base"
        self.embedding_dim = 768
        
        # ── Configuration ───────────────────────────────────────
        self.system_prompt = (
            "You are an expert AI assistant with comprehensive knowledge of all 20 books "
            "written by Shahbuddin Rathod, the renowned Gujarati humorist, philosopher, and author.\n\n"
            "INSTRUCTIONS:\n"
            "1. FOR QUESTIONS ABOUT SHAHBUDDIN RATHOD'S BOOKS, CHARACTERS, STORIES, & PHILOSOPHY: Provide a thorough, informative answer strictly in fluent 100% ENGLISH. Translate any Gujarati passages, names, or quotes into clear English. Cite source book names and page numbers.\n"
            "2. FOR QUESTIONS UNRELATED TO SHAHBUDDIN RATHOD'S BOOKS (science, general history, sports, technology, other non-Gujarati authors, etc.): State exactly: 'I cannot answer this question as it is not mentioned in Shahbuddin Rathod\'s 20 books.'\n"
            "3. ZERO EXTERNAL KNOWLEDGE / NO HALLUCINATIONS: Do NOT invent facts or use outside world knowledge for non-book topics.\n"
        )
        
        # ── Metadata ────────────────────────────────────────────
        self.author = "Shahbuddin Rathod"
        self.total_books = 0
        self.total_chunks = 0
        self.total_knowledge_docs = 0
        self.build_timestamp = ""
    
    def _cosine_similarity(self, query_vec, doc_vecs):
        """Compute cosine similarity between query and all document vectors."""
        # Normalize
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
        return np.dot(doc_norms, query_norm)
    
    def search(self, query_embedding, top_k=5, search_type="all"):
        """
        Search for relevant documents using a pre-computed query embedding.
        
        Args:
            query_embedding: numpy array of shape (D,)
            top_k: number of results to return
            search_type: "all", "chunks", or "knowledge"
            
        Returns:
            List of (text, metadata, score) tuples
        """
        results = []
        
        if search_type in ("all", "chunks") and self.chunk_embeddings is not None:
            scores = self._cosine_similarity(query_embedding, self.chunk_embeddings)
            top_indices = np.argsort(scores)[::-1][:top_k]
            for idx in top_indices:
                results.append({
                    "text": self.chunk_texts[idx],
                    "metadata": self.chunks[idx],
                    "score": float(scores[idx]),
                    "type": "chunk"
                })
        
        if search_type in ("all", "knowledge") and self.knowledge_embeddings is not None:
            scores = self._cosine_similarity(query_embedding, self.knowledge_embeddings)
            top_indices = np.argsort(scores)[::-1][:top_k]
            for idx in top_indices:
                results.append({
                    "text": self.knowledge_texts[idx],
                    "metadata": self.knowledge_docs[idx],
                    "score": float(scores[idx]),
                    "type": "knowledge"
                })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def get_oks_context(self, query):
        """Get relevant OKS structured data as context for a query."""
        context_parts = []
        query_lower = query.lower()
        
        # Transliteration mapping for cross-lingual English -> Gujarati matching
        keyword_transliterations = {
            "master": ["માસ્તર", "સાહેબ", "માસ્તરસાહેબ"],
            "mathur": ["મથુર", "મથુરદાસ"],
            "jivlo": ["જીવલો", "જીવલા"],
            "kanji": ["કાનજી"],
            "laddu": ["લાડુ", "લાડવા", "પાંચમો લાડુ"],
            "laddus": ["લાડુ", "લાડવા", "પાંચમો લાડુ"],
            "five": ["પાંચ", "પાંચમો"],
            "rathod": ["રાઠોડ", "શાહબુદ્દીન"],
            "shahbuddin": ["શાહબુદ્દીન"],
            "tea": ["ચા"],
            "school": ["શાળા", "નિશાળ"],
            "inspection": ["ઇન્સ્પેક્શન", "નિરીક્ષક"],
        }
        
        query_terms = set(w.strip("?,!.") for w in query_lower.split() if len(w) > 2)
        for w in list(query_terms):
            if w in keyword_transliterations:
                query_terms.update(keyword_transliterations[w])
        
        # ── Global statistics ───────────────────────────────────
        if any(word in query_lower for word in ["how many", "count", "total", "number", "all books", "statistics", "published"]):
            stats = self.oks_master.get("statistics", {})
            context_parts.append(
                "=== BOOK COLLECTION STATISTICS ===\n"
                f"Total books published: {self.total_books or 20}\n"
                f"Total pages across all books: {stats.get('total_pages_across_books', 0)}\n"
                f"Total words across all books: {stats.get('total_words_across_books', 0)}\n"
                f"Total knowledge records: {self.oks_master.get('total_knowledge_records', 0)}\n"
                f"Total opinions extracted: {stats.get('total_opinions', 0)}\n"
                f"Total themes found: {stats.get('total_themes', 0)}\n"
                f"Total anecdotes/stories: {stats.get('total_anecdotes', 0)}\n"
                f"Total unique characters: {stats.get('unique_characters', 0)}\n"
                f"Total unique themes: {stats.get('unique_themes', 0)}\n"
                f"Themes spanning multiple books: {stats.get('cross_book_themes', 0)}\n"
                f"Characters spanning multiple books: {stats.get('cross_book_characters', 0)}\n"
            )
            
        # ── Book Titles & Collection List ───────────────────────
        if any(word in query_lower for word in ["book", "books", "title", "titles", "name of", "list", "published", "written", "how many"]):
            book_title_lines = []
            for idx, (slug, info) in enumerate(self.oks_book_summaries.items(), start=1):
                t_en = info.get("official_title_en", info.get("book_name", slug))
                t_gu = info.get("official_title_gu", "")
                theme = info.get("main_theme", "")
                pages = info.get("total_pages", 0)
                book_title_lines.append(
                    f"  {idx}. {slug.upper()}: \"{t_en}\" ({t_gu}) — {pages} pages | Key Focus: {theme}"
                )
            if book_title_lines:
                context_parts.append(
                    "=== ALL 20 BOOKS BY SHAHBUDDIN RATHOD ===\n" + "\n".join(book_title_lines)
                )
        
        # ── Character queries ───────────────────────────────────
        if any(word in query_lower for word in ["character", "characters", "who is", "who are", "person", "people", "master", "mathur", "jivlo", "kanji"]):
            # Find relevant characters
            matched_chars = []
            for char_name, char_info in self.oks_characters.items():
                char_lower = char_name.lower()
                if (any(term in char_lower for term in query_terms) or
                    any(word in query_lower for word in char_lower.split() if len(word) > 2)):
                    matched_chars.append(char_info)
            
            if not matched_chars and ("all" in query_lower or "character" in query_lower):
                # Show top characters by appearance count
                sorted_chars = sorted(
                    self.oks_characters.values(), 
                    key=lambda x: x.get("total_appearances", 0), 
                    reverse=True
                )[:30]
                matched_chars = sorted_chars
            
            if matched_chars:
                char_lines = []
                for ch in matched_chars[:20]:
                    char_lines.append(
                        f"  - {ch.get('name', '?')}: "
                        f"appears {ch.get('total_appearances', 0)} times "
                        f"across {ch.get('num_books', 0)} books "
                        f"({', '.join(ch.get('books_appeared_in', [])[:5])})"
                    )
                context_parts.append(
                    f"=== CHARACTERS ({len(matched_chars)} found) ===\n" + "\n".join(char_lines)
                )
        
        # ── Theme queries ───────────────────────────────────────
        if any(word in query_lower for word in ["theme", "themes", "topic", "topics", "subject", "philosophy"]):
            matched_themes = []
            for term, theme_info in self.oks_themes.items():
                if (term.lower() in query_lower or
                    any(word in term.lower() for word in query_lower.split() if len(word) > 3)):
                    matched_themes.append(theme_info)
            
            if not matched_themes:
                # Show top themes by occurrence
                sorted_themes = sorted(
                    self.oks_themes.values(),
                    key=lambda x: x.get("total_occurrences", 0),
                    reverse=True
                )[:20]
                matched_themes = sorted_themes
            
            if matched_themes:
                theme_lines = []
                for th in matched_themes[:15]:
                    theme_lines.append(
                        f"  - {th.get('term', '?')}: "
                        f"appears {th.get('total_occurrences', 0)} times "
                        f"across {th.get('num_books', 0)} books"
                    )
                context_parts.append(
                    f"=== THEMES ({len(matched_themes)} found) ===\n" + "\n".join(theme_lines)
                )
        
        # ── Anecdote/story queries ──────────────────────────────
        if any(word in query_lower for word in ["story", "stories", "anecdote", "joke", "tale", "laddu", "laddus", "five laddus"]):
            if "laddu" in query_lower or "laddus" in query_lower or "five" in query_lower:
                context_parts.append(
                    "=== FAMOUS ANECDOTE: FIVE LADDUS (પાંચમો લાડુ) ===\n"
                    "[Book 'Pan Mare Kya Lakhvu Hatu?' (પણ મારે ક્યાં લખવું હતું ?), p.6] In this famous incident, Master Saheb (Shahbuddin Rathod) is offered five laddus during an event at Ambaji. "
                    "Although he initially politely hesitated out of etiquette ('vivek'), he ate all five laddus with great relish. "
                    "When Saheb humorously asked why he ate them if he had refused earlier, Master Saheb wittily replied that he was only showing polite manners ('vivek'), causing everyone to burst into laughter."
                )
            
            matched = []
            for anecdote in self.oks_anecdotes:
                desc_lower = anecdote.get("description", "").lower()
                if any(word in desc_lower for word in query_lower.split() if len(word) > 3):
                    matched.append(anecdote)
            
            if matched:
                anecdote_lines = []
                for a in matched[:10]:
                    chars = ", ".join(a.get("characters", []))
                    anecdote_lines.append(
                        f"  - [{a.get('book', '?')}, p.{a.get('page_start', '?')}] "
                        f"{a.get('description', '')[:150]}... "
                        f"(Characters: {chars})"
                    )
                context_parts.append(
                    f"=== STORIES/ANECDOTES ({len(matched)} found) ===\n" + "\n".join(anecdote_lines)
                )
        
        # ── Opinion queries ─────────────────────────────────────
        if any(word in query_lower for word in ["opinion", "believe", "view", "thinks", "philosophy", "position"]):
            matched = []
            for op in self.oks_opinions:
                if any(word in op.get("topic", "").lower() or word in op.get("opinion_statement", "").lower() 
                       for word in query_lower.split() if len(word) > 3):
                    matched.append(op)
            
            if matched:
                opinion_lines = []
                for o in matched[:10]:
                    opinion_lines.append(
                        f"  - [{o.get('book', '?')}, p.{o.get('page_start', '?')}] "
                        f"Topic: {o.get('topic', '?')} — "
                        f"{o.get('opinion_statement', '')[:120]}..."
                    )
                context_parts.append(
                    f"=== OPINIONS ({len(matched)} found) ===\n" + "\n".join(opinion_lines)
                )
        
        # ── Book-specific queries ───────────────────────────────
        for slug, info in self.oks_book_summaries.items():
            book_num = slug.replace("book-", "")
            if f"book {book_num}" in query_lower or f"book-{book_num}" in query_lower or slug in query_lower:
                context_parts.append(
                    f"=== {info.get('book_name', slug)} DETAILS ===\n"
                    f"  Pages: {info.get('total_pages', 0)}\n"
                    f"  Words: {info.get('word_count', 0)}\n"
                    f"  Characters: {info.get('char_count', 0)}\n"
                    f"  Cleaned chunks: {info.get('cleaned_chunks_count', 0)}\n"
                    f"  Knowledge records: {info.get('knowledge_records', 0)}\n"
                    f"  Opinions: {info.get('opinions_extracted', 0)}\n"
                    f"  Themes: {info.get('themes_extracted', 0)}\n"
                    f"  Anecdotes: {info.get('anecdotes_extracted', 0)}\n"
                    f"  Evolutions: {info.get('evolutions_extracted', 0)}"
                )
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def info(self):
        """Print model information."""
        print(f"{'=' * 60}")
        print(f"  BookKnowledgeModel v{self.VERSION}")
        print(f"{'=' * 60}")
        print(f"  Author:              {self.author}")
        print(f"  Total books:         {self.total_books}")
        print(f"  Total text chunks:   {self.total_chunks}")
        print(f"  Total knowledge docs:{self.total_knowledge_docs}")
        print(f"  Embedding model:     {self.embedding_model_name}")
        print(f"  Embedding dim:       {self.embedding_dim}")
        if self.chunk_embeddings is not None:
            print(f"  Chunk embeddings:    {self.chunk_embeddings.shape}")
        if self.knowledge_embeddings is not None:
            print(f"  Knowledge embeddings:{self.knowledge_embeddings.shape}")
        stats = self.oks_master.get("statistics", {})
        print(f"  ─────────────────────────────────────")
        print(f"  OKS Opinions:        {stats.get('total_opinions', 0)}")
        print(f"  OKS Themes:          {stats.get('total_themes', 0)}")
        print(f"  OKS Anecdotes:       {stats.get('total_anecdotes', 0)}")
        print(f"  OKS Characters:      {stats.get('unique_characters', 0)}")
        print(f"  Cross-book themes:   {stats.get('cross_book_themes', 0)}")
        print(f"  Cross-book chars:    {stats.get('cross_book_characters', 0)}")
        print(f"  Built:               {self.build_timestamp}")
        print(f"{'=' * 60}")


def parse_yaml_frontmatter(content):
    """Parse YAML frontmatter from markdown."""
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    metadata = {}
    for line in parts[1].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            metadata[k.strip()] = v.strip().strip('"').strip("'")
    return metadata, parts[2]


def main():
    parser = argparse.ArgumentParser(description="OKS Model Builder — Creates model.pkl")
    parser.add_argument("--oks-dir", default="oks", help="OKS JSON files directory")
    parser.add_argument("--cleaned-dir", default="cleaned", help="Cleaned corpus directory")
    parser.add_argument("--knowledge-dir", default="knowledge", help="Knowledge records directory")
    parser.add_argument("--output", default="model.pkl", help="Output pickle file path")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent if script_dir.name == "scripts" else script_dir
    
    def resolve_path(given, default):
        p = Path(given)
        if p.is_absolute():
            return p
        if (base_dir / p).exists():
            return (base_dir / p).resolve()
        return (base_dir / default).resolve()
    
    oks_dir = resolve_path(args.oks_dir, "oks")
    cleaned_dir = resolve_path(args.cleaned_dir, "cleaned")
    knowledge_dir = resolve_path(args.knowledge_dir, "knowledge")
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (base_dir / output_path).resolve()
    
    print("=" * 60)
    print("  OKS Model Builder — Creating model.pkl")
    print("=" * 60)
    
    # ── Validate inputs ─────────────────────────────────────────────────
    if not oks_dir.exists():
        print(f"❌ OKS directory not found: {oks_dir}")
        print("   Run: python -X utf8 scripts/build_oks.py")
        sys.exit(1)
    
    # ── Load OKS data ───────────────────────────────────────────────────
    print("\n📚 Loading OKS knowledge base...")
    model = BookKnowledgeModel()
    
    def load_json(filename):
        path = oks_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        print(f"  ⚠️  {filename} not found, skipping")
        return {}
    
    model.oks_master = load_json("oks_master.json")
    
    opinions_data = load_json("oks_opinions.json")
    model.oks_opinions = opinions_data.get("opinions", [])
    
    themes_data = load_json("oks_themes.json")
    model.oks_themes = themes_data.get("themes", {})
    
    chars_data = load_json("oks_characters.json")
    model.oks_characters = chars_data.get("characters", {})
    
    anecdotes_data = load_json("oks_anecdotes.json")
    model.oks_anecdotes = anecdotes_data.get("anecdotes", [])
    
    summaries_data = load_json("oks_book_summaries.json")
    model.oks_book_summaries = summaries_data.get("books", {})
    
    model.oks_cross_references = load_json("oks_cross_references.json")
    
    stats = model.oks_master.get("statistics", {})
    print(f"  ✅ Loaded OKS: {stats.get('total_opinions', 0)} opinions, "
          f"{stats.get('total_anecdotes', 0)} anecdotes, "
          f"{stats.get('unique_characters', 0)} characters, "
          f"{stats.get('unique_themes', 0)} themes")
    
    # ── Load text chunks ────────────────────────────────────────────────
    print("\n📖 Loading cleaned text chunks...")
    manifest_path = cleaned_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"❌ manifest.json not found in {cleaned_dir}")
        sys.exit(1)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    for book_slug in sorted(manifest.get("books", {}).keys()):
        chunks_file = cleaned_dir / book_slug / "chunks.jsonl"
        if not chunks_file.exists():
            continue
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    model.chunks.append({
                        "chunk_id": chunk["chunk_id"],
                        "book": chunk["book"],
                        "book_slug": chunk["book_slug"],
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                    })
                    model.chunk_texts.append(chunk["text"])
    
    model.total_chunks = len(model.chunks)
    model.total_books = len(manifest.get("books", {}))
    print(f"  ✅ Loaded {model.total_chunks} text chunks from {model.total_books} books")
    
    # ── Load knowledge documents ────────────────────────────────────────
    print("\n📝 Loading structured knowledge documents...")
    for book_dir in sorted(knowledge_dir.glob("book-*")):
        if not book_dir.is_dir():
            continue
        for md_file in sorted(book_dir.glob("*.md")):
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            metadata, body = parse_yaml_frontmatter(content)
            if metadata.get("status") == "empty":
                continue
            
            model.knowledge_docs.append({
                "chunk_id": metadata.get("chunk_id", md_file.stem),
                "book": metadata.get("book", book_dir.name),
                "book_slug": metadata.get("book_slug", book_dir.name),
                "page_start": metadata.get("page_start", "?"),
                "page_end": metadata.get("page_end", "?"),
            })
            model.knowledge_texts.append(body.strip())
    
    model.total_knowledge_docs = len(model.knowledge_docs)
    print(f"  ✅ Loaded {model.total_knowledge_docs} knowledge documents")
    
    # ── Compute embeddings ──────────────────────────────────────────────
    print(f"\n🔢 Computing embeddings with {model.embedding_model_name}...")
    print(f"   Encoding text chunks and knowledge documents...")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("❌ transformers or torch not installed.")
        print("   Run: pip install transformers torch")
        sys.exit(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model.embedding_model_name)
    transformer_model = AutoModel.from_pretrained(model.embedding_model_name).to(device)
    transformer_model.eval()
    
    def encode_texts(text_list, batch_size=32):
        all_embeddings = []
        for i in tqdm(range(0, len(text_list), batch_size), desc="Encoding batches"):
            batch_texts = text_list[i:i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = transformer_model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1)
                embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
                normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(normalized.cpu().numpy())
        return np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 768), dtype=np.float32)

    # Encode text chunks (with E5 prefix)
    print(f"\n   Encoding {len(model.chunk_texts)} text chunks...")
    prefixed_chunks = [f"passage: {t}" for t in model.chunk_texts]
    model.chunk_embeddings = encode_texts(prefixed_chunks, batch_size=args.batch_size)
    print(f"   ✅ Chunk embeddings shape: {model.chunk_embeddings.shape}")
    
    # Encode knowledge documents (with E5 prefix)
    print(f"\n   Encoding {len(model.knowledge_texts)} knowledge documents...")
    prefixed_knowledge = [f"passage: {t}" for t in model.knowledge_texts]
    model.knowledge_embeddings = encode_texts(prefixed_knowledge, batch_size=args.batch_size)
    print(f"   ✅ Knowledge embeddings shape: {model.knowledge_embeddings.shape}")
    
    # ── Set metadata ────────────────────────────────────────────────────
    from datetime import datetime
    model.build_timestamp = datetime.now().isoformat()
    
    # ── Save model.pkl ──────────────────────────────────────────────────
    print(f"\n💾 Saving model to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ model.pkl saved: {size_mb:.1f} MB")
    
    # ── Print summary ───────────────────────────────────────────────────
    model.info()
    
    print(f"\n✅ Model build complete!")
    print(f"   File: {output_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"\n   Next step: python -X utf8 scripts/chat_oks.py")
    print(f"   Or test:   python -X utf8 scripts/chat_oks.py --query \"How many books has the author written?\"")


if __name__ == "__main__":
    main()
