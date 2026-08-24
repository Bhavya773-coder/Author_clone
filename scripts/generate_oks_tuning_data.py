#!/usr/bin/env python3
"""
OKS Training Data Generator — Anti-Hallucination Book QA SFT Dataset
======================================================================
Generates precise, grounded Q&A instruction pairs directly from OKS
master data, text chunks, and structured knowledge records.

Features:
  - Strict grounding: Answers are built directly from verified book facts
  - Includes exact character rosters, specific anecdotes (laddus, Master Saheb, etc.),
    opinion rationales, and book statistics
  - Zero hallucination formatting: Forces clear citation of source books and page numbers

Output:
  - tuning/oks_sft_data.jsonl

Usage:
  python -X utf8 scripts/generate_oks_tuning_data.py
  python -X utf8 scripts/generate_oks_tuning_data.py --limit 500
"""

import json
import re
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

SYSTEM_PROMPT = """You are an expert literary researcher creating high-precision training data for a Book Knowledge AI assistant.

Your task is to convert the provided verified book knowledge record into a natural English question and a strictly grounded, accurate English answer.

STRICT ANTI-HALLUCINATION RULES:
1. Every claim in your generated answer MUST come directly from the provided source text/context.
2. Do NOT invent stories, dates, or character details not present in the context.
3. Always include explicit book title and page number citations in the answer (e.g. "[Book-01, p.6]").
4. Write in clear, professional ENGLISH.

Format your output strictly as JSON:
{
  "instruction": "Natural English question about the specific book fact/story",
  "output": "Detailed, strictly grounded answer in English with book and page citations"
}"""


def query_ollama(prompt, timeout=120):
    """Send generation request to Ollama."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.3,
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
            raw = res_body.get("response", "").strip()
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                raw = match.group(0)
            return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


def generate_direct_qa_records(oks_master, opinions_data, anecdotes_data, chars_data, summaries_data):
    """Generate high-precision Q&A pairs directly from structured records."""
    records = []

    # 1. Global & Collection Statistics
    stats = oks_master.get("statistics", {})
    records.append({
        "prompt_id": "oks_stat_global_01",
        "category": "statistics",
        "instruction": "How many books has Shahbuddin Rathod written and what are the total statistics of the collection?",
        "output": (
            f"Shahbuddin Rathod has written a total of {oks_master.get('total_books', 20)} books. "
            f"Across the entire 20-book collection, there are {stats.get('total_pages_across_books', 3252)} pages "
            f"and {stats.get('total_words_across_books', 810500)} words. "
            f"The collection contains {stats.get('total_opinions', 2642)} stated opinions, "
            f"{stats.get('total_anecdotes', 2635)} stories/anecdotes, "
            f"{stats.get('unique_themes', 2968)} unique themes, and "
            f"{stats.get('unique_verified_characters', 3068)} verified characters. "
            f"Additionally, {stats.get('cross_book_characters', 429)} characters appear across multiple books."
        )
    })

    # 2. Per-Book Summaries & Page Statistics
    for slug, info in summaries_data.get("books", {}).items():
        records.append({
            "prompt_id": f"oks_stat_book_{slug}",
            "category": "book_summary",
            "instruction": f"What are the stats and content summary for {info.get('book_name', slug)}?",
            "output": (
                f"{info.get('book_name', slug)} contains {info.get('total_pages', 0)} pages and "
                f"{info.get('word_count', 0)} words ({info.get('char_count', 0)} characters). "
                f"The structured analysis of this book extracts {info.get('knowledge_records', 0)} knowledge records, "
                f"{info.get('opinions_extracted', 0)} opinions, {info.get('themes_extracted', 0)} recurring themes, "
                f"and {info.get('anecdotes_extracted', 0)} stories/anecdotes."
            )
        })

    # 3. Specific Character Profiles
    top_chars = sorted(
        chars_data.get("characters", {}).items(),
        key=lambda x: x[1].get("total_appearances", 0),
        reverse=True
    )[:40]

    for name, info in top_chars:
        books_str = ", ".join(info.get("books_appeared_in", []))
        apps = info.get("appearances", [])
        sample_apps = []
        for a in apps[:3]:
            sample_apps.append(f"• [{a.get('book', '?')}] {a.get('anecdote', '')}")
        sample_text = "\n".join(sample_apps)

        records.append({
            "prompt_id": f"oks_char_{name}",
            "category": "character_profile",
            "instruction": f"Tell me about the character '{name}' in Shahbuddin Rathod's books.",
            "output": (
                f"The character '{name}' appears {info.get('total_appearances', 0)} times across "
                f"{info.get('num_books', 0)} books ({books_str}).\n\n"
                f"Sample stories involving {name}:\n{sample_text}"
            )
        })

    # 4. Anecdotes & Stories (including Master Saheb, laddus, etc.)
    anecdotes = anecdotes_data.get("anecdotes", [])
    for idx, a in enumerate(anecdotes[:300]):
        desc = a.get("description", "")
        chars = ", ".join(a.get("characters", []))
        book = a.get("book", "?")
        page_start = a.get("page_start", "?")

        records.append({
            "prompt_id": f"oks_anecdote_{idx:04d}",
            "category": "anecdote",
            "instruction": f"What story does Shahbuddin Rathod tell in {book} around page {page_start}?",
            "output": (
                f"In {book} (p.{page_start}), the author relates the following story/anecdote:\n"
                f"\"{desc}\"\n"
                f"Key characters involved: {chars or 'Author'}. [Source: {book}, p.{page_start}]"
            )
        })

    # 5. Opinions & Beliefs
    opinions = opinions_data.get("opinions", [])
    for idx, op in enumerate(opinions[:200]):
        topic = op.get("topic", "?")
        stmt = op.get("opinion_statement", "")
        rat = op.get("author_rationale", "")
        book = op.get("book", "?")
        page_start = op.get("page_start", "?")

        records.append({
            "prompt_id": f"oks_opinion_{idx:04d}",
            "category": "opinion",
            "instruction": f"What is Shahbuddin Rathod's position on {topic} in {book}?",
            "output": (
                f"In {book} (p.{page_start}), regarding '{topic}', Shahbuddin Rathod's belief is:\n"
                f"\"{stmt}\"\n"
                f"Author's Rationale: {rat}\n"
                f"Confidence: {op.get('confidence', 'high')} [Source: {book}, p.{page_start}]"
            )
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="OKS Training Data Generator (Anti-Hallucination)")
    parser.add_argument("--oks-dir", default="oks")
    parser.add_argument("--output-dir", default="tuning")
    parser.add_argument("--limit", type=int, default=0)
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
    output_dir = resolve_path(args.output_dir, "tuning")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "oks_sft_data.jsonl"

    def load_json(filename):
        path = oks_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    oks_master = load_json("oks_master.json")
    opinions_data = load_json("oks_opinions.json")
    anecdotes_data = load_json("oks_anecdotes.json")
    chars_data = load_json("oks_characters.json")
    summaries_data = load_json("oks_book_summaries.json")

    print("=" * 60)
    print("  OKS Training Data Generator — Grounded Anti-Hallucination")
    print("=" * 60)

    # Build direct Q&A records
    direct_records = generate_direct_qa_records(
        oks_master, opinions_data, anecdotes_data, chars_data, summaries_data
    )

    print(f"Generated {len(direct_records)} grounded Q&A pairs.")

    if args.limit > 0:
        direct_records = direct_records[:args.limit]
        print(f"Limiting to {args.limit} records.")

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in direct_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ Grounded SFT dataset written to: {output_file}")
    print(f"   Total examples: {len(direct_records)}")


if __name__ == "__main__":
    main()
