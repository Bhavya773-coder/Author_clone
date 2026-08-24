import os
import json
import re
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

SYSTEM_PROMPT = """You are preparing training data for a conversational AI model designed to speak in the distinct voice, style, and philosophy of Shahbuddin Rathod, the famous Gujarati humorist.
Using the provided structured opinions/stories, generate a Gujarati user query (asking a question, seeking advice, or prompting about a topic) and two responses:

1. CHOSEN (Shahbuddin Rathod style): Authentic Gujarati response.
   CRITICAL STYLE RULES:
   - Use VERY SHORT, punchy sentences (maximum 6-10 words per sentence). Do NOT write long compound sentences. Split clauses using periods (`.`).
   - Include dialogue markers or conversational quotes (e.g. 'મેં કહ્યું:', 'મેં પૂછ્યું:').
   - Include at least one rhetorical question (using `?`) or conversational exclamations.
   - Infuse dry, philosophical humor and personal character references (like Master Saheb).

2. REJECTED (Generic/Flat style): Flat, dry, academic, formal Gujarati response with longer sentences.

You MUST respond strictly in valid JSON format matching the schema below. Do not include any other text, markdown formatting, or explanations.

JSON Schema:
{
  "prompt": "Gujarati user question/prompt",
  "chosen": "Rathod-style answer in Gujarati matching all the rules above",
  "rejected": "Flat/generic answer in Gujarati"
}
"""

def parse_yaml_frontmatter(content):
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

def get_ngrams(text, n=10):
    """Generate n-grams from text by stripping punctuation and splitting on whitespace."""
    words = re.findall(r'\w+', text.lower())
    if len(words) < n:
        return set()
    return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))

def has_ngram_overlap(chosen_text, source_text, n=10):
    """Check if the chosen response has an n-gram overlap of n or more words with the source text."""
    chosen_ngrams = get_ngrams(chosen_text, n)
    source_ngrams = get_ngrams(source_text, n)
    overlap = chosen_ngrams.intersection(source_ngrams)
    return len(overlap) > 0, list(overlap)

def query_ollama(prompt, system_prompt, timeout=120):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 512
        }
    }
    
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            raw_response = res_body.get("response", "").strip()
            
            # Clean up potential markdown formatting block if the model outputs it
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]
            raw_response = raw_response.strip()
            
            return json.loads(raw_response)
    except Exception as e:
        # Return error representation
        return {"error": str(e), "raw": raw_response if 'raw_response' in locals() else ""}

def main():
    parser = argparse.ArgumentParser(description="Fine-Tuning Data Construction (Phase 4)")
    parser.add_argument("--cleaned-dir", default="../data/cleaned", help="Cleaned corpus folder")
    parser.add_argument("--knowledge-dir", default="../data/knowledge", help="Structured knowledge folder")
    parser.add_argument("--output-dir", default="../data/tuning", help="Tuning output folder")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of generated pairs (0 for unlimited)")
    parser.add_argument("--force", action="store_true", help="Force overwrite old generated data")
    args = parser.parse_args()
    
    cleaned_dir = Path(args.cleaned_dir).resolve()
    knowledge_dir = Path(args.knowledge_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_file = output_dir / "sft_data.jsonl"
    dpo_file = output_dir / "dpo_data.jsonl"
    
    # Load manifest to get all books
    manifest_path = cleaned_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest.json not found in {cleaned_dir}")
        return
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    processed_books = manifest.get("books", {})
    
    # Load raw book chunks to perform overlap checking
    print("Loading original clean text chunks for overlap verification...")
    source_chunks = {}
    for book_slug in processed_books.keys():
        chunks_file = cleaned_dir / book_slug / "chunks.jsonl"
        if not chunks_file.exists():
            continue
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    source_chunks[chunk["chunk_id"]] = chunk["text"]
    print(f"Loaded {len(source_chunks)} source chunks.")
    
    # Gather structured knowledge base records
    print("Loading structured knowledge records...")
    knowledge_records = []
    for book_slug in processed_books.keys():
        book_k_dir = knowledge_dir / book_slug
        if not book_k_dir.exists():
            continue
        for md_file in book_k_dir.glob("*.md"):
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            metadata, body_text = parse_yaml_frontmatter(content)
            if metadata.get("status") == "empty":
                continue
            
            chunk_id = metadata.get("chunk_id", md_file.stem)
            knowledge_records.append({
                "chunk_id": chunk_id,
                "book": metadata.get("book", book_slug),
                "content": body_text.strip()
            })
    print(f"Loaded {len(knowledge_records)} structured knowledge records.")
    
    # Filter out records if limit is set
    if args.limit > 0:
        knowledge_records = knowledge_records[:args.limit]
        print(f"Limiting execution to first {args.limit} records.")
        
    # Check if files exist and build resume set of already-processed chunk_ids
    processed_chunk_ids = set()
    write_mode = "w"
    if not args.force and sft_file.exists() and dpo_file.exists():
        print("Tuning data files already exist. Reading completed chunk IDs to enable resume...")
        with open(sft_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cid = rec.get("chunk_id")
                    if cid:
                        processed_chunk_ids.add(cid)
        write_mode = "a"
        print(f"Resuming pipeline. Skipping {len(processed_chunk_ids)} already generated chunk IDs.")
    elif args.force:
        print("--force flag set. Overwriting any existing tuning data.")
        
    # Open files for writing
    sft_f = open(sft_file, write_mode, encoding="utf-8")
    dpo_f = open(dpo_file, write_mode, encoding="utf-8")
    
    success_count = 0
    overlap_skipped = 0
    error_count = 0
    
    print(f"\nStarting GPU-accelerated training data generation...")
    if processed_chunk_ids:
        print(f"Will skip {len(processed_chunk_ids)} already-done chunks and process the remaining {len(knowledge_records) - len(processed_chunk_ids)}.")
    for record in tqdm(knowledge_records):
        chunk_id = record["chunk_id"]
        
        # RESUME: skip chunks already successfully written in a prior run
        if chunk_id in processed_chunk_ids:
            continue
            
        source_text = source_chunks.get(chunk_id, "")
        
        # Format generation prompt
        generation_prompt = f"""Structured Knowledge Record:
{record['content']}

Please generate the prompt, chosen, and rejected responses according to the instructions."""
        
        # Query Ollama
        result = query_ollama(generation_prompt, SYSTEM_PROMPT)
        
        if "error" in result:
            error_count += 1
            continue
            
        prompt_text = result.get("prompt", "").strip()
        chosen_text = result.get("chosen", "").strip()
        rejected_text = result.get("rejected", "").strip()
        
        if not prompt_text or not chosen_text or not rejected_text:
            error_count += 1
            continue
            
        # Overlap Check — prevent verbatim source leakage into training data
        if source_text:
            has_overlap, overlaps = has_ngram_overlap(chosen_text, source_text, n=10)
            if has_overlap:
                overlap_skipped += 1
                continue
                
        # Write SFT record (includes chunk_id so resume works on next run)
        sft_f.write(json.dumps({
            "chunk_id": chunk_id,
            "instruction": prompt_text,
            "output": chosen_text
        }, ensure_ascii=False) + "\n")
        
        # Write DPO record
        dpo_f.write(json.dumps({
            "chunk_id": chunk_id,
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text
        }, ensure_ascii=False) + "\n")
        
        sft_f.flush()
        dpo_f.flush()
        success_count += 1
        # Add to set so we don't re-process if loop is extended in future
        processed_chunk_ids.add(chunk_id)
        
    sft_f.close()
    dpo_f.close()
    
    print("\n=========================================")
    print("Fine-Tuning Data Generation Complete!")
    print(f"Output Directory: {output_dir}")
    print(f"Successfully Generated: {success_count} pairs")
    print(f"Skipped due to verbatim overlap: {overlap_skipped}")
    print(f"Errors/Timeouts: {error_count}")
    print("=========================================")

if __name__ == "__main__":
    main()
