import os
import json
import urllib.request
import argparse
from pathlib import Path
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

SYSTEM_PROMPT = """You are an expert literary researcher analyzing the writings of Shahbuddin Rathod, a prominent Gujarati author, humorist, and philosopher.
Analyze the provided Gujarati text chunk and extract structured knowledge.

CRITICAL REQUIREMENT:
- All values in the JSON (topic, opinion_statement, author_rationale, term_or_theme, contextual_meaning, description, key_characters) MUST be written in the Gujarati language using the Gujarati script.
- Do NOT output any English words in the JSON values, except for proper names that cannot be translated, or if the author uses English terms in his original text.
- Maintain the exact tone, perspective, and voice of the author in your extractions.

You MUST respond strictly in valid JSON format matching the schema below. Do not include any other text, markdown block formatting, or commentary.

JSON Schema:
{
  "opinions": [
    {
      "topic": "topic name in Gujarati",
      "opinion_statement": "detailed opinion/belief statement in Gujarati",
      "author_rationale": "author's rationale or reasons in Gujarati",
      "confidence": "high" or "medium" or "low"
    }
  ],
  "themes_and_terminology": [
    {
      "term_or_theme": "term or theme in Gujarati",
      "contextual_meaning": "what this term/theme means in the author's context in Gujarati"
    }
  ],
  "anecdotes": [
    {
      "description": "description of the story/anecdote/joke in Gujarati",
      "key_characters": ["character 1 in Gujarati", "character 2 in Gujarati"]
    }
  ],
  "contradictions_or_evolutions": [
    {
      "description": "description of the view evolution/contradiction in Gujarati"
    }
  ]
}
"""

def query_ollama(prompt):
    """Sends a request to the local Ollama API and returns the parsed JSON response."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            OLLAMA_URL, 
            data=data, 
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            res_body = json.loads(response.read().decode('utf-8'))
            response_text = res_body.get("response", "")
            return json.loads(response_text)
    except Exception as e:
        print(f"\nError querying Ollama: {e}")
        return None

def write_knowledge_file(output_path, chunk, knowledge_data):
    """Writes the extracted structured knowledge to a Markdown file with YAML frontmatter."""
    yaml_header = f"""---
book: "{chunk['book']}"
book_slug: "{chunk['book_slug']}"
chunk_id: "{chunk['chunk_id']}"
page_start: {chunk['page_start']}
page_end: {chunk['page_end']}
model: "{MODEL_NAME}"
---

# Structured Knowledge Base Record ({chunk['chunk_id']})

## 📝 Source Text Context
> {chunk['text'][:200]}...

---

"""
    content_parts = []
    
    # 1. Opinions
    opinions = knowledge_data.get("opinions", [])
    if isinstance(opinions, list) and opinions:
        content_parts.append("## 💭 Stated Opinions & Beliefs")
        for idx, op in enumerate(opinions, 1):
            if isinstance(op, dict):
                content_parts.append(f"### {idx}. {op.get('topic', 'Opinion')}")
                content_parts.append(f"- **Opinion Statement**: {op.get('opinion_statement', '')}")
                content_parts.append(f"- **Rationale**: {op.get('author_rationale', '')}")
                content_parts.append(f"- **Confidence**: `{op.get('confidence', 'medium')}`")
            elif isinstance(op, str):
                content_parts.append(f"### {idx}. Opinion")
                content_parts.append(f"- **Opinion Statement**: {op}")
            content_parts.append("")
            
    # 2. Themes & Terminology
    themes = knowledge_data.get("themes_and_terminology", [])
    if isinstance(themes, list) and themes:
        content_parts.append("## 🏷️ Recurring Themes & Terminology")
        for t in themes:
            if isinstance(t, dict):
                content_parts.append(f"- **{t.get('term_or_theme', '')}**: {t.get('contextual_meaning', '')}")
            elif isinstance(t, str):
                content_parts.append(f"- {t}")
        content_parts.append("")
        
    # 3. Anecdotes
    anecdotes = knowledge_data.get("anecdotes", [])
    if isinstance(anecdotes, list) and anecdotes:
        content_parts.append("## 📖 Stories & Anecdotes")
        for a in anecdotes:
            if isinstance(a, dict):
                content_parts.append(f"- **Anecdote/Joke**: {a.get('description', '')}")
                chars = a.get("key_characters", [])
                if isinstance(chars, list) and chars:
                    content_parts.append(f"  - *Characters*: {', '.join(chars)}")
            elif isinstance(a, str):
                content_parts.append(f"- **Anecdote/Joke**: {a}")
        content_parts.append("")
        
    # 4. Contradictions/Evolutions
    evolutions = knowledge_data.get("contradictions_or_evolutions", [])
    if isinstance(evolutions, list) and evolutions:
        content_parts.append("## 🔄 View Evolutions & Contradictions")
        for ev in evolutions:
            if isinstance(ev, dict):
                content_parts.append(f"- {ev.get('description', '')}")
            elif isinstance(ev, str):
                content_parts.append(f"- {ev}")
        content_parts.append("")
        
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(yaml_header + "\n".join(content_parts))

def main():
    parser = argparse.ArgumentParser(description="Structured Knowledge Base Extractor (Phase 2a)")
    parser.add_argument("--cleaned-dir", default="./data/cleaned", help="Directory with cleaned books")
    parser.add_argument("--knowledge-dir", default="./data/knowledge", help="Directory to save knowledge records")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to process (for testing)")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing knowledge records")
    args = parser.parse_args()
    
    cleaned_dir = Path(args.cleaned_dir).resolve()
    knowledge_dir = Path(args.knowledge_dir).resolve()
    
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = cleaned_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest.json not found in {cleaned_dir}. Run Phase 1.5 first.")
        return
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    processed_books = manifest.get("books", {})
    
    # Gather all chunks to process
    all_chunks = []
    for book_slug, book_info in processed_books.items():
        chunks_path = cleaned_dir / book_slug / "chunks.jsonl"
        if not chunks_path.exists():
            continue
            
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_chunks.append(json.loads(line))
                    
    print(f"Loaded {len(all_chunks)} chunks from cleaned corpus.")
    
    # Filter already processed chunks
    chunks_to_process = []
    for chunk in all_chunks:
        book_slug = chunk["book_slug"]
        chunk_id = chunk["chunk_id"]
        out_dir = knowledge_dir / book_slug
        out_path = out_dir / f"{chunk_id}.md"
        
        if out_path.exists() and not args.force:
            continue
        chunks_to_process.append((chunk, out_path))
        
    print(f"Chunks pending extraction: {len(chunks_to_process)}")
    
    if args.limit:
        chunks_to_process = chunks_to_process[:args.limit]
        print(f"Testing with limit={args.limit} chunks.")
        
    if not chunks_to_process:
        print("No new chunks to process. Use --force to re-process.")
        return
        
    success_count = 0
    
    # Run loop
    for chunk, out_path in tqdm(chunks_to_process, desc="Extracting knowledge"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare query prompt
        prompt = f"Book: {chunk['book']}\nChunk ID: {chunk['chunk_id']}\nText:\n{chunk['text']}"
        
        # Query Ollama
        knowledge_data = query_ollama(prompt)
        
        if knowledge_data:
            # Check if we actually extracted anything meaningful
            has_data = any(knowledge_data.values())
            if has_data:
                write_knowledge_file(out_path, chunk, knowledge_data)
                success_count += 1
            else:
                # Save empty placeholders to avoid re-querying skipped chunks
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(f"---\nbook: \"{chunk['book']}\"\nchunk_id: \"{chunk['chunk_id']}\"\nstatus: \"empty\"\n---\n")
                success_count += 1
                
    print(f"\nExtraction complete. Successfully wrote {success_count} structured knowledge records.")

if __name__ == "__main__":
    main()
