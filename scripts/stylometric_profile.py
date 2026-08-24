import os
import json
import re
import argparse
from pathlib import Path
import numpy as np

# Standard sentence delimiters in Gujarati (including the traditional danda '।')
SENTENCE_DELIMITERS = re.compile(r'[.!?;।\n]+')

# Common Gujarati openers and stylistic filler words to track
TARGET_OPENERS = [
    "એક", "પરંતુ", "અલબત્ત", "હું", "મેં", "અમે", "ત્યારે", "હવે", "ખરેખર", "વાસ્તવમાં", "આમ", "જોકે"
]

def analyze_style(text):
    """Computes comprehensive stylometric metrics for a given text."""
    # Strip whitespace
    text_clean = text.strip()
    if not text_clean:
        return {}
        
    char_count = len(text_clean)
    
    # 1. Paragraph Analysis
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraph_count = len(paragraphs)
    paragraph_lengths = [len(p.split()) for p in paragraphs]
    
    # 2. Sentence Analysis
    # Clean sentences and drop empty fragments
    sentences = [s.strip() for s in SENTENCE_DELIMITERS.split(text) if s.strip()]
    sentence_count = len(sentences)
    sentence_lengths = [len(s.split()) for s in sentences]
    
    # 3. Word and Vocabulary Analysis
    # Remove punctuation for word tracking
    words_raw = re.sub(r'[^\w\s\u0a80-\u0aff]', '', text).split()
    words = [w.strip() for w in words_raw if w.strip()]
    word_count = len(words)
    
    unique_words = set(words)
    vocab_size = len(unique_words)
    
    # Lexical Diversity
    ttr = vocab_size / word_count if word_count > 0 else 0
    rttr = vocab_size / np.sqrt(word_count) if word_count > 0 else 0
    
    # 4. Punctuation Densities (per 10,000 words)
    exclamations = text.count("!")
    questions = text.count("?")
    quotes = sum(text.count(q) for q in ['"', "'", "“", "”", "‘", "’"])
    parentheses = text.count("(") + text.count(")")
    
    scale_factor = 10000.0 / word_count if word_count > 0 else 0
    
    exclamation_density = exclamations * scale_factor
    question_density = questions * scale_factor
    quotes_density = quotes * scale_factor
    parentheses_density = parentheses * scale_factor
    
    # 5. Sentence Openers Analysis
    opener_counts = {op: 0 for op in TARGET_OPENERS}
    opener_counts["other"] = 0
    
    for s in sentences:
        s_words = s.split()
        if s_words:
            first_word = s_words[0].strip()
            # Remove punctuation from first word
            first_word_clean = re.sub(r'[^\w\u0a80-\u0aff]', '', first_word)
            if first_word_clean in opener_counts:
                opener_counts[first_word_clean] += 1
            else:
                opener_counts["other"] += 1
                
    opener_ratios = {}
    if sentence_count > 0:
        for op, count in opener_counts.items():
            opener_ratios[op] = count / sentence_count
            
    return {
        "word_count": word_count,
        "char_count": char_count,
        "paragraph_count": paragraph_count,
        "sentence_count": sentence_count,
        "vocab_size": vocab_size,
        
        "lexical_diversity": {
            "type_token_ratio": ttr,
            "root_ttr": rttr
        },
        "paragraph_distribution": {
            "mean": float(np.mean(paragraph_lengths)) if paragraph_lengths else 0,
            "median": float(np.median(paragraph_lengths)) if paragraph_lengths else 0,
            "std": float(np.std(paragraph_lengths)) if paragraph_lengths else 0
        },
        "sentence_distribution": {
            "mean": float(np.mean(sentence_lengths)) if sentence_lengths else 0,
            "median": float(np.median(sentence_lengths)) if sentence_lengths else 0,
            "std": float(np.std(sentence_lengths)) if sentence_lengths else 0
        },
        "punctuation_density_per_10k_words": {
            "exclamations": exclamation_density,
            "questions": question_density,
            "quotes": quotes_density,
            "parentheses": parentheses_density
        },
        "sentence_opener_ratios": opener_ratios
    }

def main():
    parser = argparse.ArgumentParser(description="Author Stylometric Profiler (Phase 2b)")
    parser.add_argument("--cleaned-dir", default="./data/cleaned", help="Directory containing cleaned books")
    parser.add_argument("--output-json", default="./data/style_profile.json", help="Path to save quantitative metrics")
    parser.add_argument("--output-report", default="./data/style_report.md", help="Path to save human-readable report")
    args = parser.parse_args()
    
    cleaned_dir = Path(args.cleaned_dir).resolve()
    output_json = Path(args.output_json).resolve()
    output_report = Path(args.output_report).resolve()
    
    # Load manifest
    manifest_path = cleaned_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest.json not found in {cleaned_dir}. Run Phase 1.5 first.")
        return
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    books = manifest.get("books", {})
    
    style_profiles = {}
    
    print(f"Profiling {len(books)} cleaned books...")
    
    for book_slug, book_info in books.items():
        book_title = book_info.get("book", book_slug)
        text_path = cleaned_dir / book_slug / "full_text.txt"
        
        if not text_path.exists():
            continue
            
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        print(f"- Analyzing style of: {book_title}")
        style_profiles[book_slug] = {
            "title": book_title,
            "metrics": analyze_style(text)
        }
        
    # Compute corpus-wide averages
    corpus_metrics = {}
    if style_profiles:
        keys_to_avg = ["word_count", "sentence_count", "paragraph_count", "vocab_size"]
        for key in keys_to_avg:
            corpus_metrics[f"avg_{key}"] = float(np.mean([b["metrics"][key] for b in style_profiles.values()]))
            
        corpus_metrics["avg_sentence_len"] = float(np.mean([b["metrics"]["sentence_distribution"]["mean"] for b in style_profiles.values()]))
        corpus_metrics["avg_paragraph_len"] = float(np.mean([b["metrics"]["paragraph_distribution"]["mean"] for b in style_profiles.values()]))
        corpus_metrics["avg_ttr"] = float(np.mean([b["metrics"]["lexical_diversity"]["type_token_ratio"] for b in style_profiles.values()]))
        
        # Punctuation averages
        punc_keys = ["exclamations", "questions", "quotes", "parentheses"]
        corpus_metrics["avg_punctuation_density_per_10k"] = {
            k: float(np.mean([b["metrics"]["punctuation_density_per_10k_words"][k] for b in style_profiles.values()]))
            for k in punc_keys
        }
        
        # Opener averages
        opener_keys = TARGET_OPENERS + ["other"]
        corpus_metrics["avg_sentence_opener_ratios"] = {
            k: float(np.mean([b["metrics"]["sentence_opener_ratios"][k] for b in style_profiles.values()]))
            for k in opener_keys
        }
        
    output_data = {
        "corpus_averages": corpus_metrics,
        "books": style_profiles
    }
    
    # Save JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        
    # Generate Markdown Report
    report_lines = [
        "# Shahbuddin Rathod - Corpus Stylometric Fingerprint Report",
        "",
        "This report summarizes the quantitative style markers of Shahbuddin Rathod's writings across all 20 cleaned books.",
        "",
        "## 📈 Corpus-Wide Stylometric Averages",
        f"- **Average Words per Book**: `{corpus_metrics.get('avg_word_count', 0):,.1f}`",
        f"- **Average Sentences per Book**: `{corpus_metrics.get('avg_sentence_count', 0):,.1f}`",
        f"- **Average Paragraphs per Book**: `{corpus_metrics.get('avg_paragraph_count', 0):,.1f}`",
        f"- **Average Vocabulary Size**: `{corpus_metrics.get('avg_vocab_size', 0):,.1f}` unique words per book",
        f"- **Average Type-Token Ratio (Lexical Diversity)**: `{corpus_metrics.get('avg_ttr', 0):.2%}`",
        "",
        "### ✍️ Typical Sentence & Paragraph Lengths",
        f"- **Mean Words per Sentence**: `{corpus_metrics.get('avg_sentence_len', 0):.2f}` words",
        f"- **Mean Words per Paragraph**: `{corpus_metrics.get('avg_paragraph_len', 0):.2f}` words",
        "",
        "### ❓ Punctuation Habits (Density per 10,000 words)",
        f"- **Exclamation Marks (`!`)**: `{corpus_metrics.get('avg_punctuation_density_per_10k', {}).get('exclamations', 0):.1f}`",
        f"- **Question Marks (`?`)**: `{corpus_metrics.get('avg_punctuation_density_per_10k', {}).get('questions', 0):.1f}`",
        f"- **Quotes (Dialogue / Speech)**: `{corpus_metrics.get('avg_punctuation_density_per_10k', {}).get('quotes', 0):.1f}`",
        f"- **Parentheses (`()`)**: `{corpus_metrics.get('avg_punctuation_density_per_10k', {}).get('parentheses', 0):.1f}`",
        "",
        "### 🔀 Favorite Sentence Openers (Ratios)",
    ]
    
    sorted_openers = sorted(
        [(k, v) for k, v in corpus_metrics.get("avg_sentence_opener_ratios", {}).items() if k != "other"],
        key=lambda x: x[1],
        reverse=True
    )
    
    for opener, ratio in sorted_openers:
        report_lines.append(f"- **\"{opener}\"** starts `{ratio:.2%}` of sentences.")
        
    report_lines.extend([
        "",
        "## 📚 Individual Book Fingerprints",
        "| Book Slug | Word Count | Avg Sentence Len | Avg Paragraph Len | Lexical TTR | Exclamations/10k | Questions/10k |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])
    
    for b_slug, b_data in style_profiles.items():
        m = b_data["metrics"]
        report_lines.append(
            f"| {b_slug} | {m['word_count']:,} | {m['sentence_distribution']['mean']:.2f} | "
            f"{m['paragraph_distribution']['mean']:.2f} | {m['lexical_diversity']['type_token_ratio']:.2%} | "
            f"{m['punctuation_density_per_10k_words']['exclamations']:.1f} | {m['punctuation_density_per_10k_words']['questions']:.1f} |"
        )
        
    with open(output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"Stylometric profiling complete. JSON saved to {output_json}, Report saved to {output_report}.")

if __name__ == "__main__":
    main()
