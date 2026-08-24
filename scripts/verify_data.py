import os
import json
import re
import argparse
from pathlib import Path

# Target stylometric metrics (computed in Phase 2 from the 20 raw books)
TARGET_METRICS = {
    "mean_sentence_length": 8.95,
    "quote_density_per_10k": 401.2,
    "exclamation_density_per_10k": 33.7,
    "question_density_per_10k": 88.8,
    "vocabulary_richness_ttr": 28.97
}

def clean_and_tokenize_words(text):
    # Strip punctuation and tokenize
    words = re.findall(r'\w+', text.lower())
    return words

def split_sentences(text):
    # Split on common Gujarati sentence ending marks
    sentences = re.split(r'[.!?|।\n]+', text)
    return [s.strip() for s in sentences if s.strip()]

def calculate_metrics(texts):
    total_words = 0
    total_sentences = 0
    total_quotes = 0
    total_exclamations = 0
    total_questions = 0
    unique_words = set()
    
    for text in texts:
        words = clean_and_tokenize_words(text)
        sentences = split_sentences(text)
        
        total_words += len(words)
        total_sentences += len(sentences)
        unique_words.update(words)
        
        # Count punctuation
        total_quotes += len(re.findall(r'["\'“”‘’:]', text))
        total_exclamations += text.count("!")
        total_questions += text.count("?")
        
    if total_words == 0:
        return {}
        
    mean_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
    quote_density = (total_quotes / total_words) * 10000
    exclamation_density = (total_exclamations / total_words) * 10000
    question_density = (total_questions / total_words) * 10000
    ttr = (len(unique_words) / total_words) * 100 if total_words > 0 else 0
    
    return {
        "mean_sentence_length": round(mean_sentence_length, 2),
        "quote_density_per_10k": round(quote_density, 2),
        "exclamation_density_per_10k": round(exclamation_density, 2),
        "question_density_per_10k": round(question_density, 2),
        "vocabulary_richness_ttr": round(ttr, 2),
        "total_words": total_words,
        "total_sentences": total_sentences
    }

def main():
    parser = argparse.ArgumentParser(description="Fine-Tuning Dataset Verification (Phase 4b)")
    parser.add_argument("--tuning-dir", default="../data/tuning", help="Tuning data folder")
    args = parser.parse_args()
    
    tuning_dir = Path(args.tuning_dir).resolve()
    sft_file = tuning_dir / "sft_data.jsonl"
    
    if not sft_file.exists():
        print(f"Error: Fine-tuning data file not found at {sft_file}. Run generation first.")
        return
        
    print(f"Reading SFT training data from {sft_file}...")
    chosen_texts = []
    with open(sft_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # SFT format has 'output' which corresponds to the 'chosen' (Rathod-styled) text
                chosen_texts.append(record.get("output", ""))
                
    print(f"Loaded {len(chosen_texts)} training examples.")
    print("Calculating stylometric metrics of generated responses...")
    
    gen_metrics = calculate_metrics(chosen_texts)
    
    print("\n📊 STYLOMETRIC MATCH COMPARISON (Target vs Generated)")
    print("======================================================================")
    print(f"{'Metric':<30} | {'Target Profile':<16} | {'Generated SFT':<15} | {'Diff':<10}")
    print("----------------------------------------------------------------------")
    
    for key, target_val in TARGET_METRICS.items():
        gen_val = gen_metrics.get(key, 0.0)
        diff = round(gen_val - target_val, 2)
        diff_str = f"+{diff}" if diff > 0 else f"{diff}"
        
        # Check alignment
        status = "✅ Match"
        if key == "mean_sentence_length" and abs(diff) > 2.0:
            status = "⚠️ Sentence too long"
        elif key == "vocabulary_richness_ttr" and abs(diff) > 10.0:
            status = "⚠️ Vocab skew"
            
        print(f"{key:<30} | {target_val:<16} | {gen_val:<15} | {diff_str:<10} {status}")
        
    print("======================================================================")
    print(f"Total Words Analyzed: {gen_metrics.get('total_words')}")
    print(f"Total Sentences Analyzed: {gen_metrics.get('total_sentences')}")
    print("======================================================================")

if __name__ == "__main__":
    main()
