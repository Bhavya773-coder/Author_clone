import os
import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm

# Common Tesseract OCR errors in Gujarati to standardize
REPLACEMENTS = {
    "ભાસ્તરસાહેબ": "માસ્તરસાહેબ",
    "ભાસ્તરસાહેબે": "માસ્તરસાહેબે",
    "ભાસ્તર": "માસ્તર",
    "રત્રે": "રાત્રે",
    "હાસ્થે": "હાસ્યે",
    "સથસ્યા": "સમસ્યા",
    "સશ્જનો": "સજ્જનો",
    "સજજ્નો": "સજ્જનો",
    "જ%ન્‍્યડિવરન": "જન્મદિવસ",
    "શુભપસંઝે": "શુભપ્રસંગે",
    "રેમાંચેલ્ાં": "રોમાંચક",
    "૧1દ્વસજઠતતઊના": "ઉદ્ઘાટનથી",
    "11દ્વસજઠતતઊના": "ઉદ્ઘાટનથી",
}

# Keywords to detect copyright/publisher credits on early pages
COPYRIGHT_KEYWORDS = [
    "પ્રકાશક", "આવૃત્તિ", "ISBN", "કિંમત", "મુદ્રક", 
    "ટાઇપસેટિંગ", "વેબસાઇટ", "ઈમેઈલ", "ફોન", "સૌજન્ય", 
    "published by", "r. r. sheth", "pravin prakashan", 
    "લાભ ચેમ્બર્સ", "d Dwarkesh", "rrsheth"
]

def slugify(text):
    """Simple slugify implementation matching extract.py"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text

def is_gujarati_char(char):
    """Check if a character is in the Gujarati Unicode range."""
    return '\u0a80' <= char <= '\u0aff'

def clean_text_line(line):
    """Apply typo standardizations to a line."""
    for typo, correction in REPLACEMENTS.items():
        line = re.sub(re.escape(typo), correction, line)
    return line

def is_junk_line(line):
    """
    Evaluate if a line is OCR noise / junk:
    1. Contains very low Gujarati character ratio.
    2. Consists only of digits, punctuation, and symbols.
    3. Contains too high ratio of English letters.
    """
    stripped = line.strip()
    if not stripped:
        return False
        
    total_chars = len(stripped)
    guj_chars = sum(1 for c in stripped if is_gujarati_char(c))
    eng_chars = sum(1 for c in stripped if c.isalpha() and c.isascii())
    
    # 1. Gujarati characters should be dominant in a Gujarati text line
    if total_chars > 5:
        guj_ratio = guj_chars / total_chars
        if guj_ratio < 0.35:  # Less than 35% Gujarati characters is likely metadata or junk
            return True
            
    # 2. Contains mostly English letters in a Gujarati context
    total_alphas = guj_chars + eng_chars
    if total_alphas > 0 and (eng_chars / total_alphas) > 0.40:
        if any(kw in stripped.lower() for kw in ["http", "www", ".com", "@"]):
            return True
        if guj_chars == 0 and total_chars > 5:
            return True
            
    # 3. OCR remnants consisting only of symbols, punctuation, and digits (both English and Gujarati digits)
    if total_chars > 3 and re.match(r'^[-\s_@#*¢\[\]\(\):./\\+=\d%&!?\x00-\x1f૦૧૨૩૪૫૬૭૮૯]+$', stripped):
        return True
        
    return False

def clean_book_paragraphs(chunks_data):
    """
    Clean paragraphs by filtering out publisher metadata (pages 1-10) and junk lines.
    Returns cleaned paragraphs and skipped chunks metadata.
    """
    cleaned_paragraphs = []
    skipped_metadata_lines = 0
    skipped_junk_lines = 0
    
    # Group paragraphs/lines by page number context
    for chunk in chunks_data:
        page_start = chunk.get("page_start", 1)
        page_end = chunk.get("page_end", 1)
        text = chunk.get("text", "")
        lines = text.splitlines()
        
        # Check 1: Copyright metadata on early pages (pages 1 to 10)
        # If the entire chunk text contains copyright metadata keywords, discard the whole chunk
        if page_start <= 10:
            if any(keyword in text.lower() for keyword in COPYRIGHT_KEYWORDS):
                skipped_metadata_lines += len(lines)
                continue
                
        cleaned_chunk_lines = []
        for line in lines:
            # Check 2: General junk lines (any page)
            if is_junk_line(line):
                skipped_junk_lines += 1
                continue
                
            # Clean and keep
            cleaned_line = clean_text_line(line)
            if cleaned_line.strip():
                cleaned_chunk_lines.append(cleaned_line)
                
        # Re-group into paragraphs
        if cleaned_chunk_lines:
            paragraph = " ".join(cleaned_chunk_lines)
            # Minimize multiple spaces
            paragraph = re.sub(r'\s+', ' ', paragraph).strip()
            cleaned_paragraphs.append({
                "text": paragraph,
                "page_start": page_start,
                "page_end": page_end
            })
            
    return cleaned_paragraphs, skipped_metadata_lines, skipped_junk_lines

def rebuild_chunks(paragraphs, book_title, book_slug):
    """
    Rebuild chunks targeting 200-500 words with ~10% overlap.
    """
    chunks = []
    chunk_idx = 1
    
    i = 0
    n = len(paragraphs)
    
    while i < n:
        chunk_paras = []
        chunk_words = 0
        page_start = paragraphs[i]["page_start"]
        page_end = paragraphs[i]["page_end"]
        
        j = i
        # Gather paragraphs up to word limit (approx. 400 words)
        while j < n:
            p_text = paragraphs[j]["text"]
            p_words = len(p_text.split())
            
            # If adding this paragraph exceeds limit and we already have content, stop
            if chunk_words + p_words > 400 and chunk_paras:
                break
                
            chunk_paras.append(paragraphs[j])
            chunk_words += p_words
            page_end = paragraphs[j]["page_end"]
            j += 1
            
        # Join paragraph text
        chunk_text = "\n\n".join(p["text"] for p in chunk_paras)
        
        chunks.append({
            "chunk_id": f"{book_slug}_cleaned_{chunk_idx:04d}",
            "book": book_title,
            "book_slug": book_slug,
            "page_start": page_start,
            "page_end": page_end,
            "text": chunk_text,
            "char_count": len(chunk_text),
            "word_count": chunk_words
        })
        chunk_idx += 1
        
        # Calculate overlap target (~10% of the chunk words)
        overlap_target = max(15, int(chunk_words * 0.10))
        overlap_words = 0
        overlap_count = 0
        
        for k in range(j - 1, i - 1, -1):
            p_text = paragraphs[k]["text"]
            p_words = len(p_text.split())
            if overlap_words + p_words > overlap_target:
                break
            overlap_words += p_words
            overlap_count += 1
            
        # Next starting index
        next_i = j - overlap_count
        if next_i <= i:
            next_i = j  # Force progress if overlap would cause a loop
            
        i = next_i
        
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Author Corpus Cleaning Pipeline (Phase 1.5)")
    parser.add_argument("--processed-dir", default="./data/processed", help="Directory containing raw processed books")
    parser.add_argument("--cleaned-dir", default="./data/cleaned", help="Directory to save cleaned outputs")
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir).resolve()
    cleaned_dir = Path(args.cleaned_dir).resolve()
    
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest.json not found in {processed_dir}. Run Phase 1 first.")
        return
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    processed_books = manifest.get("books", {})
    if not processed_books:
        print("No processed books found in manifest.json.")
        return
        
    print(f"Found {len(processed_books)} books to clean.")
    
    cleaned_books_stats = {}
    total_skipped_metadata = 0
    total_skipped_junk = 0
    
    for book_slug, book_info in tqdm(processed_books.items(), desc="Cleaning books"):
        book_title = book_info.get("book", book_slug)
        book_processed_dir = processed_dir / book_slug
        chunks_path = book_processed_dir / "chunks.jsonl"
        
        if not chunks_path.exists():
            print(f"Warning: chunks.jsonl not found for '{book_title}' in {book_processed_dir}. Skipping.")
            continue
            
        # Read chunks
        chunks_data = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks_data.append(json.loads(line))
                    
        # Clean paragraphs
        cleaned_paras, skipped_metadata, skipped_junk = clean_book_paragraphs(chunks_data)
        total_skipped_metadata += skipped_metadata
        total_skipped_junk += skipped_junk
        
        if not cleaned_paras:
            print(f"Warning: Cleaning resulted in empty text for '{book_title}'. Skipping output.")
            continue
            
        # Rebuild full text
        full_cleaned_text = "\n\n".join(p["text"] for p in cleaned_paras)
        
        # Re-chunk
        cleaned_chunks = rebuild_chunks(cleaned_paras, book_title, book_slug)
        
        # Save output
        book_cleaned_dir = cleaned_dir / book_slug
        book_cleaned_dir.mkdir(parents=True, exist_ok=True)
        
        # Write full text
        with open(book_cleaned_dir / "full_text.txt", "w", encoding="utf-8") as f:
            f.write(full_cleaned_text)
            
        # Write chunks
        with open(book_cleaned_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in cleaned_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                
        cleaned_books_stats[book_slug] = {
            "book": book_title,
            "book_slug": book_slug,
            "total_pages": book_info.get("total_pages", 0),
            "original_chunks_count": len(chunks_data),
            "cleaned_chunks_count": len(cleaned_chunks),
            "char_count": len(full_cleaned_text),
            "word_count": sum(c["word_count"] for c in cleaned_chunks),
            "skipped_metadata_lines": skipped_metadata,
            "skipped_junk_lines": skipped_junk,
            "status": "cleaned"
        }
        
    # Write cleaned manifest
    cleaned_manifest = {
        "total_books_cleaned": len(cleaned_books_stats),
        "total_skipped_metadata_lines": total_skipped_metadata,
        "total_skipped_junk_lines": total_skipped_junk,
        "total_chunks_generated": sum(b["cleaned_chunks_count"] for b in cleaned_books_stats.values()),
        "books": cleaned_books_stats
    }
    
    with open(cleaned_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_manifest, f, indent=2)
        
    print("\n=========================================")
    print("Corpus Cleaning Complete!")
    print(f"Total Books Cleaned: {cleaned_manifest['total_books_cleaned']}")
    print(f"Total Skipped Metadata Lines (ISBN/Credits): {cleaned_manifest['total_skipped_metadata_lines']}")
    print(f"Total Skipped Junk/OCR Noise Lines: {cleaned_manifest['total_skipped_junk_lines']}")
    print(f"Total Clean Chunks Generated: {cleaned_manifest['total_chunks_generated']}")
    print("=========================================")

if __name__ == "__main__":
    main()
