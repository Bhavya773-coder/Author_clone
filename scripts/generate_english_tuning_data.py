import os
import json
import re
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

SYSTEM_PROMPT = """You are preparing training data for an English conversational AI model that answers questions strictly based on the books and knowledge of Shahbuddin Rathod, the famous Gujarati humorist and philosopher.

Given the provided structured opinions, stories, and book knowledge (in Gujarati), perform the following tasks:
1. Extract the core facts, stories, philosophy, anecdotes, and opinions.
2. Formulate a natural, informative user question in English about this specific topic, story, or advice.
3. Write a comprehensive, detailed, accurate answer strictly in ENGLISH that explains the concept, tells the story, or provides the author's wisdom based on the book context.

Rules for Answer Generation:
- Answer MUST be written entirely in standard English.
- The answer must be informative, warm, clear, and faithful to the book content.
- Do NOT use Gujarati script in the generated answer.

You MUST respond strictly in valid JSON format matching the schema below. Do not include any other text, markdown formatting, or explanations.

JSON Schema:
{
  "instruction": "English user question or prompt about the book topic",
  "output": "Detailed, accurate answer in English based on the book knowledge"
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

def query_ollama(prompt, system_prompt, timeout=120):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "format": "json",
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
            
            # Extract JSON substring using regex if needed
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                raw_response = match.group(0)
            
            return json.loads(raw_response)
    except Exception as e:
        return {"error": str(e), "raw": raw_response if 'raw_response' in locals() else ""}

def main():
    parser = argparse.ArgumentParser(description="English Book Knowledge SFT Data Generation")
    parser.add_argument("--cleaned-dir", default="cleaned", help="Cleaned corpus folder")
    parser.add_argument("--knowledge-dir", default="knowledge", help="Structured knowledge folder")
    parser.add_argument("--output-dir", default="tuning", help="Tuning output folder")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of generated records (0 for all)")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing dataset")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent if script_dir.name == "scripts" else script_dir
    
    def resolve_path(given_path, folder_name):
        p = Path(given_path)
        if p.is_absolute():
            return p
        if (base_dir / p).exists():
            return (base_dir / p).resolve()
        if (base_dir / folder_name).exists():
            return (base_dir / folder_name).resolve()
        return (base_dir / folder_name).resolve()

    cleaned_dir = resolve_path(args.cleaned_dir, "cleaned")
    knowledge_dir = resolve_path(args.knowledge_dir, "knowledge")
    output_dir = resolve_path(args.output_dir, "tuning")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    english_sft_file = output_dir / "english_sft_data.jsonl"
    
    # Discover knowledge folders
    print(f"Loading structured knowledge records from {knowledge_dir}...")
    knowledge_records = []
    
    if not knowledge_dir.exists():
        print(f"Error: Knowledge directory not found at {knowledge_dir}")
        return
        
    for book_dir in sorted(knowledge_dir.glob("book-*")):
        if not book_dir.is_dir():
            continue
        for md_file in sorted(book_dir.glob("*.md")):
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            metadata, body_text = parse_yaml_frontmatter(content)
            if metadata.get("status") == "empty":
                continue
            
            chunk_id = metadata.get("chunk_id", md_file.stem)
            knowledge_records.append({
                "chunk_id": chunk_id,
                "book": metadata.get("book", book_dir.name),
                "content": body_text.strip()
            })
            
    print(f"Found {len(knowledge_records)} structured knowledge records.")
    
    if args.limit > 0:
        knowledge_records = knowledge_records[:args.limit]
        print(f"Limiting generation to first {args.limit} records.")
        
    processed_chunk_ids = set()
    write_mode = "w"
    if not args.force and english_sft_file.exists():
        print("Existing dataset found. Reading completed chunk IDs to resume...")
        with open(english_sft_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        cid = rec.get("chunk_id")
                        if cid:
                            processed_chunk_ids.add(cid)
                    except Exception:
                        pass
        write_mode = "a"
        print(f"Resuming pipeline. Skipping {len(processed_chunk_ids)} already generated chunks.")
    elif args.force:
        print("--force flag set. Overwriting existing english_sft_data.jsonl.")
        
    sft_f = open(english_sft_file, write_mode, encoding="utf-8")
    
    success_count = 0
    error_count = 0
    
    print("\nStarting English SFT training data generation...")
    for record in tqdm(knowledge_records):
        chunk_id = record["chunk_id"]
        if chunk_id in processed_chunk_ids:
            continue
            
        generation_prompt = f"""Structured Knowledge Record ({record['book']} - {chunk_id}):
{record['content']}

Generate an English question (instruction) and English answer (output) according to the system prompt guidelines."""

        result = query_ollama(generation_prompt, SYSTEM_PROMPT)
        
        if "error" in result:
            print(f"\n[Error for {chunk_id}]: {result['error']}")
            if "raw" in result:
                print(f"[Raw]: {result['raw'][:200]}")
            error_count += 1
            continue
            
        instruction = result.get("instruction", "").strip()
        output_text = result.get("output", "").strip()
        
        if not instruction or not output_text:
            print(f"\n[Missing fields for {chunk_id}]: parsed keys = {list(result.keys())}")
            error_count += 1
            continue
            
        sft_f.write(json.dumps({
            "chunk_id": chunk_id,
            "book": record["book"],
            "instruction": instruction,
            "output": output_text
        }, ensure_ascii=False) + "\n")
        
        sft_f.flush()
        success_count += 1
        
    sft_f.close()
    print(f"\n✅ Generation complete! Successfully generated {success_count} English Q&A records.")
    print(f"Output saved to: {english_sft_file}")
    if error_count > 0:
        print(f"⚠️  Encountered {error_count} errors during generation.")

if __name__ == "__main__":
    main()
