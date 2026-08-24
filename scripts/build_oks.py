#!/usr/bin/env python3
"""
OKS Builder — Open Knowledge System from Author's 20 Books (High-Precision Edition)
===================================================================================
Reads all structured knowledge markdown records from the knowledge/ directory,
parses YAML frontmatter and structured sections, applies strict Named Entity
validation & stopword filtering, and consolidates into a unified OKS JSON.

Output (oks/ directory):
  - oks_master.json          : Complete unified knowledge base
  - oks_characters.json      : All verified named characters across books
  - oks_themes.json          : All themes/terminology consolidated
  - oks_opinions.json        : All opinions with book cross-references
  - oks_anecdotes.json       : All stories/anecdotes indexed
  - oks_book_summaries.json  : Per-book summaries with stats
  - oks_cross_references.json: Connections between books

Usage:
  python -X utf8 scripts/build_oks.py
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

# Non-character stopwords filter (pronouns, adverbs, non-entity grammar words)
NON_CHARACTER_STOPWORDS = {
    "અહીં", "ક્યાં", "કેમ", "શું", "કોણ", "ક્યારે", "કેવી", "કેવો", "કેવું", "એક", "બે", "ત્રણ",
    "ચાર", "પાંચ", "છ", "સાત", "આઠ", "નવ", "દસ", "સો", "હજાર", "રૂપિયા", "પૈસા"
}


def parse_yaml_frontmatter(content):
    """Parse YAML frontmatter from markdown content."""
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


def extract_section(body, section_header):
    """Extract content under a specific markdown ## header."""
    pattern = rf"## [^\n]*{re.escape(section_header)}[^\n]*\n(.*?)(?=\n## |\Z)"
    match = re.search(pattern, body, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def parse_opinions(section_text):
    """Parse opinions from the Stated Opinions & Beliefs section."""
    opinions = []
    if not section_text:
        return opinions

    blocks = re.split(r"### \d+\.\s*", section_text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        topic_match = re.match(r"^(.+?)$", block, re.MULTILINE)
        topic = topic_match.group(1).strip() if topic_match else ""

        opinion_match = re.search(r"\*\*Opinion Statement\*\*:\s*(.+)", block)
        opinion = opinion_match.group(1).strip() if opinion_match else ""

        rationale_match = re.search(r"\*\*Rationale\*\*:\s*(.+)", block)
        rationale = rationale_match.group(1).strip() if rationale_match else ""

        confidence_match = re.search(r"\*\*Confidence\*\*:\s*`?(\w+)`?", block)
        confidence = confidence_match.group(1).strip() if confidence_match else "medium"

        if topic or opinion:
            opinions.append({
                "topic": topic,
                "opinion_statement": opinion,
                "rationale": rationale,
                "confidence": confidence
            })
    return opinions


def parse_themes(section_text):
    """Parse themes/terminology from the Recurring Themes section."""
    themes = []
    if not section_text:
        return themes
    for line in section_text.splitlines():
        line = line.strip()
        match = re.match(r"^-\s*\*\*(.+?)\*\*:\s*(.+)", line)
        if match:
            themes.append({
                "term": match.group(1).strip(),
                "meaning": match.group(2).strip()
            })
        elif line.startswith("- ") and line[2:].strip():
            themes.append({
                "term": line[2:].strip(),
                "meaning": ""
            })
    return themes


def parse_anecdotes(section_text):
    """Parse anecdotes from the Stories & Anecdotes section."""
    anecdotes = []
    if not section_text:
        return anecdotes

    lines = section_text.splitlines()
    current_anecdote = None

    for line in lines:
        line = line.strip()
        anecdote_match = re.match(r"^-\s*\*\*Anecdote/Joke\*\*:\s*(.+)", line)
        if anecdote_match:
            if current_anecdote:
                anecdotes.append(current_anecdote)
            current_anecdote = {
                "description": anecdote_match.group(1).strip(),
                "characters": []
            }
        elif current_anecdote:
            char_match = re.match(r"^\s*-\s*\*Characters\*:\s*(.+)", line)
            if char_match:
                chars = [c.strip() for c in char_match.group(1).split(",") if c.strip()]
                current_anecdote["characters"] = chars

    if current_anecdote:
        anecdotes.append(current_anecdote)
    return anecdotes


def parse_evolutions(section_text):
    """Parse view evolutions/contradictions."""
    evolutions = []
    if not section_text:
        return evolutions
    for line in section_text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            text = line[2:].strip()
            if text:
                evolutions.append({"description": text})
    return evolutions


def normalize_character_name(name):
    """Normalize and group common character name variations."""
    name = name.strip()
    if name in {"મથુરદાસ", "મથુરાદાસ"}:
        return "મથુર"
    if name in {"જીવલા"}:
        return "જીવલો"
    if name in {"પ્રભુલાલ"}:
        return "પ્રભુ"
    return name


def build_oks(knowledge_dir, cleaned_dir, output_dir):
    """Main OKS building logic with high-precision entity filtering."""
    knowledge_path = Path(knowledge_dir)
    cleaned_path = Path(cleaned_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = cleaned_path / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    book_metadata = manifest.get("books", {})

    all_opinions = []
    all_themes = []
    all_anecdotes = []
    all_evolutions = []
    all_characters = defaultdict(lambda: {"appearances": [], "books": set()})
    all_themes_index = defaultdict(lambda: {"occurrences": [], "books": set()})

    book_summaries = {}

    total_records = 0
    total_skipped = 0

    print("=" * 60)
    print("  OKS Builder — High-Precision Edition")
    print("=" * 60)

    for book_dir in sorted(knowledge_path.glob("book-*")):
        if not book_dir.is_dir():
            continue

        book_slug = book_dir.name
        book_name = book_metadata.get(book_slug, {}).get("book", book_slug.upper())

        md_files = sorted(book_dir.glob("*.md"))
        book_opinion_count = 0
        book_theme_count = 0
        book_anecdote_count = 0
        book_evolution_count = 0
        book_record_count = 0

        for md_file in md_files:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            metadata, body = parse_yaml_frontmatter(content)
            if metadata.get("status") == "empty":
                total_skipped += 1
                continue

            chunk_id = metadata.get("chunk_id", md_file.stem)
            page_start = metadata.get("page_start", "?")
            page_end = metadata.get("page_end", "?")

            total_records += 1
            book_record_count += 1

            # Opinions
            opinions_text = extract_section(body, "Stated Opinions")
            opinions = parse_opinions(opinions_text)
            for op in opinions:
                op["book"] = book_name
                op["book_slug"] = book_slug
                op["chunk_id"] = chunk_id
                op["page_start"] = page_start
                op["page_end"] = page_end
                all_opinions.append(op)
                book_opinion_count += 1

            # Themes
            themes_text = extract_section(body, "Recurring Themes")
            themes = parse_themes(themes_text)
            for theme in themes:
                theme["book"] = book_name
                theme["book_slug"] = book_slug
                theme["chunk_id"] = chunk_id
                all_themes.append(theme)
                book_theme_count += 1

                term_key = theme["term"].strip()
                if term_key:
                    all_themes_index[term_key]["occurrences"].append({
                        "book": book_name,
                        "book_slug": book_slug,
                        "chunk_id": chunk_id,
                        "meaning": theme["meaning"]
                    })
                    all_themes_index[term_key]["books"].add(book_slug)

            # Anecdotes
            anecdotes_text = extract_section(body, "Stories")
            anecdotes = parse_anecdotes(anecdotes_text)
            for anecdote in anecdotes:
                anecdote["book"] = book_name
                anecdote["book_slug"] = book_slug
                anecdote["chunk_id"] = chunk_id
                anecdote["page_start"] = page_start
                anecdote["page_end"] = page_end
                all_anecdotes.append(anecdote)
                book_anecdote_count += 1

                for char_name in anecdote.get("characters", []):
                    char_key = normalize_character_name(char_name)
                    if (char_key and 
                        char_key not in NON_CHARACTER_STOPWORDS and 
                        len(char_key) > 1 and 
                        not re.match(r"^[0-9\s\.\,\-]+$", char_key)):
                        
                        all_characters[char_key]["appearances"].append({
                            "book": book_name,
                            "book_slug": book_slug,
                            "chunk_id": chunk_id,
                            "anecdote": anecdote["description"][:100]
                        })
                        all_characters[char_key]["books"].add(book_slug)

            # View Evolutions
            evolutions_text = extract_section(body, "View Evolutions")
            evolutions = parse_evolutions(evolutions_text)
            for ev in evolutions:
                ev["book"] = book_name
                ev["book_slug"] = book_slug
                ev["chunk_id"] = chunk_id
                all_evolutions.append(ev)
                book_evolution_count += 1

        bm = book_metadata.get(book_slug, {})
        book_summaries[book_slug] = {
            "book_name": book_name,
            "book_slug": book_slug,
            "total_pages": bm.get("total_pages", 0),
            "word_count": bm.get("word_count", 0),
            "char_count": bm.get("char_count", 0),
            "cleaned_chunks_count": bm.get("cleaned_chunks_count", 0),
            "knowledge_records": book_record_count,
            "opinions_extracted": book_opinion_count,
            "themes_extracted": book_theme_count,
            "anecdotes_extracted": book_anecdote_count,
            "evolutions_extracted": book_evolution_count
        }

    # Build cross-references
    characters_json = {}
    for char_name, data in all_characters.items():
        characters_json[char_name] = {
            "name": char_name,
            "total_appearances": len(data["appearances"]),
            "books_appeared_in": sorted(list(data["books"])),
            "num_books": len(data["books"]),
            "appearances": data["appearances"]
        }

    themes_index_json = {}
    for term, data in all_themes_index.items():
        themes_index_json[term] = {
            "term": term,
            "total_occurrences": len(data["occurrences"]),
            "books_appeared_in": sorted(list(data["books"])),
            "num_books": len(data["books"]),
            "occurrences": data["occurrences"]
        }

    cross_book_themes = {t: info for t, info in themes_index_json.items() if info["num_books"] > 1}
    cross_book_characters = {c: info for c, info in characters_json.items() if info["num_books"] > 1}

    cross_references = {
        "themes_spanning_multiple_books": len(cross_book_themes),
        "characters_spanning_multiple_books": len(cross_book_characters),
        "cross_book_themes": cross_book_themes,
        "cross_book_characters": cross_book_characters
    }

    oks_master = {
        "oks_version": "2.0-high-precision",
        "author": "Shahbuddin Rathod",
        "total_books": len(book_summaries),
        "total_knowledge_records": total_records,
        "total_skipped_empty": total_skipped,
        "statistics": {
            "total_opinions": len(all_opinions),
            "total_themes": len(all_themes),
            "total_anecdotes": len(all_anecdotes),
            "total_evolutions": len(all_evolutions),
            "unique_characters": len(characters_json),
            "unique_verified_characters": len(characters_json),
            "unique_themes": len(themes_index_json),
            "cross_book_themes": len(cross_book_themes),
            "cross_book_characters": len(cross_book_characters),
            "total_pages_across_books": sum(bs.get("total_pages", 0) for bs in book_summaries.values()),
            "total_words_across_books": sum(bs.get("word_count", 0) for bs in book_summaries.values()),
        },
        "book_summaries": book_summaries,
        "all_opinions": all_opinions,
        "all_themes": all_themes,
        "all_anecdotes": all_anecdotes,
        "all_evolutions": all_evolutions,
    }

    def write_json(filename, data):
        path = output_path / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        size_kb = path.stat().st_size / 1024
        print(f"  ✅ {filename:.<40s} {size_kb:>8.1f} KB")

    print(f"\nWriting High-Precision OKS files to {output_path}/")
    write_json("oks_master.json", oks_master)
    write_json("oks_opinions.json", {"total": len(all_opinions), "opinions": all_opinions})
    write_json("oks_themes.json", {"total": len(themes_index_json), "themes": themes_index_json})
    write_json("oks_characters.json", {"total": len(characters_json), "characters": characters_json})
    write_json("oks_anecdotes.json", {"total": len(all_anecdotes), "anecdotes": all_anecdotes})
    write_json("oks_book_summaries.json", {"total_books": len(book_summaries), "books": book_summaries})
    write_json("oks_cross_references.json", cross_references)

    print(f"\n{'=' * 60}")
    print(f"  HIGH-PRECISION OKS BUILD COMPLETE")
    print(f"  Verified Characters: {len(characters_json)}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="OKS Builder — High-Precision Edition")
    parser.add_argument("--knowledge-dir", default="knowledge")
    parser.add_argument("--cleaned-dir", default="cleaned")
    parser.add_argument("--output-dir", default="oks")
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

    knowledge_dir = resolve_path(args.knowledge_dir, "knowledge")
    cleaned_dir = resolve_path(args.cleaned_dir, "cleaned")
    output_dir = resolve_path(args.output_dir, "oks")

    build_oks(knowledge_dir, cleaned_dir, output_dir)


if __name__ == "__main__":
    main()
