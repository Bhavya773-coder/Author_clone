import os
import sys
import json
import re
import logging
import argparse
import io
from pathlib import Path
from collections import Counter
from PIL import Image
from tqdm import tqdm
from slugify import slugify

# Try imports, handle missing dependencies gracefully
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import pdf2image
except ImportError:
    pdf2image = None

# Configure logging
logger = logging.getLogger("extraction_pipeline")

def setup_logging(log_file_path):
    """Set up logging to both console and a file."""
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
        
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def check_ocr_available(tesseract_path=None):
    """
    Check what OCR engines are available.
    Returns:
      'tesseract' if PyTesseract is configured/found
      'easyocr' if EasyOCR is installed
      None if no OCR engines are available
    """
    # 1. Check Tesseract
    if pytesseract is not None:
        default_windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")
        ]

        if tesseract_path:
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info(f"Using custom Tesseract path: {tesseract_path}")
                return "tesseract"
                
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR is available. Version: {version}")
            return "tesseract"
        except Exception:
            pass

        for path in default_windows_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                try:
                    version = pytesseract.get_tesseract_version()
                    logger.info(f"Found Tesseract at default path: {path}. Version: {version}")
                    return "tesseract"
                except Exception:
                    pass

    # 2. Check EasyOCR as fallback
    if easyocr is not None:
        logger.info("EasyOCR is available as fallback.")
        return "easyocr"
        
    logger.warning("No OCR engines (Tesseract or EasyOCR) are available.")
    return None

def is_page_garbled_or_empty(text):
    """
    Determine if the extracted text layer is empty, extremely short, or garbled.
    """
    text_stripped = text.strip()
    if not text_stripped:
        return True
    
    # Cover pages, title pages, or blank filler pages
    if len(text_stripped) < 40:
        return True
        
    # Check if text contains spaces (otherwise it's a single run of characters)
    words = text_stripped.split()
    if len(words) < 5 and len(text_stripped) > 100:
        return True
        
    # Calculate ratio of readable standard characters (alphanumeric, spaces, common punctuation)
    readable = sum(1 for c in text_stripped if c.isalnum() or c.isspace() or c in ".,!?;:()[]\"'-$%/\\&*+=")
    ratio = readable / len(text_stripped)
    if ratio < 0.75:
        return True
        
    return False

# Global easyocr reader instance
_easyocr_reader = None

def extract_page_ocr(pdf_path, page_num, fitz_page, engine, lang="guj+eng"):
    """
    Render a page to image and perform OCR using the specified engine ('tesseract' or 'easyocr').
    """
    global _easyocr_reader
    img = None
    
    # Try pdf2image first
    if pdf2image is not None:
        try:
            images = pdf2image.convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=150)
            if images:
                img = images[0]
        except Exception as e:
            logger.debug(f"pdf2image failed (likely missing poppler): {e}. Falling back to PyMuPDF rendering.")

    # Fallback to PyMuPDF rendering
    if img is None:
        try:
            pix = fitz_page.get_pixmap(dpi=150)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        except Exception as e:
            raise RuntimeError(f"Failed to render page {page_num} to image using PyMuPDF: {e}")

    # Perform OCR
    if engine == "tesseract":
        try:
            ocr_text = pytesseract.image_to_string(img, lang=lang)
            return ocr_text
        except Exception as e:
            raise RuntimeError(f"Tesseract OCR failed on page {page_num} with lang {lang}: {e}")
    elif engine == "easyocr":
        if _easyocr_reader is None:
            logger.info(f"Initializing EasyOCR Reader for lang {lang}...")
            easyocr_langs = []
            if 'guj' in lang:
                easyocr_langs.append('gu')
            if 'eng' in lang:
                easyocr_langs.append('en')
            if not easyocr_langs:
                easyocr_langs = ['en']
            _easyocr_reader = easyocr.Reader(easyocr_langs, gpu=False) # CPU-only fallback
            
        try:
            # Convert PIL Image back to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            results = _easyocr_reader.readtext(img_bytes, detail=0)
            ocr_text = "\n".join(results)
            return ocr_text
        except Exception as e:
            raise RuntimeError(f"EasyOCR failed on page {page_num}: {e}")
            
    raise ValueError(f"Unknown OCR engine: {engine}")

def detect_headers_footers(pages_lines):
    """
    Collect first 2 and last 2 non-empty lines from all pages to identify
    running headers and footers that appear on more than 20% of the pages.
    """
    header_candidates = []
    footer_candidates = []
    
    for lines in pages_lines:
        if not lines:
            continue
        non_empty = [l.strip() for l in lines if l.strip()]
        
        # Candidate headers (first 2 lines)
        if len(non_empty) >= 1:
            header_candidates.append(non_empty[0])
        if len(non_empty) >= 2:
            header_candidates.append(non_empty[1])
            
        # Candidate footers (last 2 lines)
        if len(non_empty) >= 1:
            footer_candidates.append(non_empty[-1])
        if len(non_empty) >= 2:
            footer_candidates.append(non_empty[-2])

    num_pages = len(pages_lines)
    min_freq = max(2, int(num_pages * 0.20)) # Must appear on at least 2 pages and >= 20% of pages
    
    header_counts = Counter(header_candidates)
    footer_counts = Counter(footer_candidates)
    
    # Exclude very short strings, pure numbers, or blank lines
    headers_to_strip = {
        line for line, count in header_counts.items() 
        if count >= min_freq and len(line) > 3 and not re.match(r'^\s*\d+\s*$', line)
    }
    footers_to_strip = {
        line for line, count in footer_counts.items() 
        if count >= min_freq and len(line) > 3 and not re.match(r'^\s*\d+\s*$', line)
    }
    
    return headers_to_strip, footers_to_strip

def clean_lines(lines, headers_to_strip, footers_to_strip):
    """
    Cleans individual lines and filters out detected headers, footers, and page numbers.
    """
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        # Filter headers and footers
        if stripped in headers_to_strip or stripped in footers_to_strip:
            continue
            
        # Filter page numbers (e.g., "12", "- 12 -", "Page 12")
        if re.match(r'^\s*\d+\s*$', stripped):
            continue
        if re.match(r'^\s*-\s*\d+\s*-\s*$', stripped):
            continue
        if re.match(r'^\s*page\s+\d+\s*$', stripped, re.IGNORECASE):
            continue
            
        # Normalize quotes and dashes
        line_normalized = line.replace('“', '"').replace('”', '"')
        line_normalized = line_normalized.replace('‘', "'").replace('’', "'")
        line_normalized = line_normalized.replace('—', ' - ').replace('–', ' - ')
        
        cleaned_lines.append(line_normalized)
        
    return cleaned_lines

def normalize_text(text):
    """
    Post-processing clean-up of hyphenated splits and whitespace.
    """
    # Fix hyphenated words split across line breaks (e.g. "exam-\nple" -> "example")
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Normalize spacing
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize paragraphs (ensure no more than 2 consecutive newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def build_chunks(paragraphs, book_title, book_slug):
    """
    Build semantic paragraph-based chunks targeting 200-500 tokens (approx. 150-400 words)
    with ~10% word overlap.
    """
    chunks = []
    i = 0
    n = len(paragraphs)
    chunk_idx = 1
    
    while i < n:
        chunk_paras = []
        chunk_words = 0
        
        # Group paragraphs until we reach the word count target (~350 words)
        j = i
        while j < n:
            para = paragraphs[j]
            # Word count approximation
            para_words = len(para["text"].split())
            para["word_count"] = para_words
            
            # Target ~350 words, maximum 400 words (approx 500 tokens)
            if chunk_words > 0 and chunk_words + para_words > 350:
                break
                
            chunk_paras.append(para)
            chunk_words += para_words
            j += 1
            
        # Handle cases where a single paragraph is extremely long
        if j == i:
            para = paragraphs[i]
            para["word_count"] = len(para["text"].split())
            chunk_paras.append(para)
            chunk_words = para["word_count"]
            j += 1
            
        # Create chunk entry
        page_start = min(p["page_num"] for p in chunk_paras)
        page_end = max(p["page_num"] for p in chunk_paras)
        methods = [p["method"] for p in chunk_paras]
        extraction_method = "ocr" if "ocr" in methods else "text_layer"
        
        chunk_text = "\n\n".join(p["text"] for p in chunk_paras)
        
        chunks.append({
            "chunk_id": f"{book_slug}_{chunk_idx:04d}",
            "book": book_title,
            "book_slug": book_slug,
            "page_start": page_start,
            "page_end": page_end,
            "text": chunk_text,
            "extraction_method": extraction_method,
            "char_count": len(chunk_text)
        })
        chunk_idx += 1
        
        # Calculate overlap target (~10% of the chunk words)
        overlap_target = max(15, int(chunk_words * 0.10))
        overlap_words = 0
        overlap_count = 0
        
        for p in reversed(chunk_paras):
            if overlap_words + p["word_count"] > overlap_target:
                break
            overlap_words += p["word_count"]
            overlap_count += 1
            
        # Next starting index
        next_i = j - overlap_count
        if next_i <= i:
            next_i = j  # Force progress if overlap would cause a loop
            
        i = next_i
        
    return chunks

def process_pdf(pdf_path, output_base_dir, ocr_engine, lang):
    """
    Processes a single PDF book: extracts text (text_layer with OCR fallback),
    cleans headers/footers, normalizes text, chunks it, and saves outputs.
    Returns processing statistics or raises an exception.
    """
    pdf_path = Path(pdf_path)
    book_title = pdf_path.stem
    book_slug = slugify(book_title)
    
    logger.info(f"Starting processing for: '{book_title}' (slug: {book_slug})")
    
    if fitz is None:
        raise ImportError("pymupdf (fitz) is not installed, which is required for PDF parsing.")
        
    doc = fitz.open(str(pdf_path))
    num_pages = len(doc)
    logger.info(f"PDF loaded successfully. Total pages: {num_pages}")
    
    raw_pages_lines = []
    pages_extraction_method = []
    ocr_pages = []
    
    # Phase 1: Text extraction
    for page_num in range(num_pages):
        page = doc[page_num]
        
        # Try native text layer
        text = page.get_text("text")
        
        # Determine if we need OCR
        if is_page_garbled_or_empty(text):
            if ocr_engine:
                logger.info(f"Page {page_num + 1}/{num_pages}: Text layer missing/garbled. Running {ocr_engine} fallback...")
                try:
                    ocr_text = extract_page_ocr(str(pdf_path), page_num, page, ocr_engine, lang)
                    # Verify if OCR actually got text
                    if ocr_text.strip():
                        text = ocr_text
                        pages_extraction_method.append("ocr")
                        ocr_pages.append(page_num + 1)
                    else:
                        logger.warning(f"Page {page_num + 1}: OCR returned empty text. Using empty text layer.")
                        pages_extraction_method.append("text_layer")
                except Exception as e:
                    logger.error(f"Page {page_num + 1}: OCR failed with error: {e}. Falling back to empty text layer.")
                    pages_extraction_method.append("text_layer")
            else:
                logger.warning(f"Page {page_num + 1}/{num_pages}: Text layer missing/garbled, but no OCR engine is available. Skipping OCR fallback.")
                pages_extraction_method.append("text_layer")
        else:
            pages_extraction_method.append("text_layer")
            
        # Split into lines
        raw_pages_lines.append(text.splitlines())
        
    doc.close()
    
    # Phase 2: Detect running headers/footers
    headers_to_strip, footers_to_strip = detect_headers_footers(raw_pages_lines)
    logger.info(f"Detected {len(headers_to_strip)} running headers and {len(footers_to_strip)} running footers.")
    if headers_to_strip:
        logger.debug(f"Headers to strip: {headers_to_strip}")
    if footers_to_strip:
        logger.debug(f"Footers to strip: {footers_to_strip}")
        
    # Phase 3: Clean lines & structure paragraphs
    paragraphs = []
    full_cleaned_text_blocks = []
    
    for page_idx, lines in enumerate(raw_pages_lines):
        page_num = page_idx + 1
        method = pages_extraction_method[page_idx]
        
        cleaned_lines = clean_lines(lines, headers_to_strip, footers_to_strip)
        page_text = "\n".join(cleaned_lines)
        page_text_normalized = normalize_text(page_text)
        
        if not page_text_normalized:
            continue
            
        full_cleaned_text_blocks.append(page_text_normalized)
        
        # Split page text into paragraphs for chunking
        page_paras = [p.strip() for p in page_text_normalized.split('\n\n') if p.strip()]
        for para in page_paras:
            paragraphs.append({
                "text": para,
                "page_num": page_num,
                "method": method
            })
            
    full_cleaned_text = "\n\n".join(full_cleaned_text_blocks)
    
    # Phase 4: Chunking
    chunks = build_chunks(paragraphs, book_title, book_slug)
    logger.info(f"Generated {len(chunks)} semantic chunks for '{book_title}'.")
    
    # Phase 5: Write outputs
    book_output_dir = Path(output_base_dir) / book_slug
    book_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write full text
    full_text_path = book_output_dir / "full_text.txt"
    with open(full_text_path, "w", encoding="utf-8") as f:
        f.write(full_cleaned_text)
        
    # Write chunks.jsonl
    chunks_path = book_output_dir / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    logger.info(f"Outputs written to {book_output_dir}")
    
    return {
        "book": book_title,
        "book_slug": book_slug,
        "total_pages": num_pages,
        "ocr_pages_count": len(ocr_pages),
        "ocr_pages_list": ocr_pages,
        "chunks_count": len(chunks),
        "status": "success"
    }

def main():
    parser = argparse.ArgumentParser(description="Author Corpus Extraction Pipeline (Phase 1)")
    parser.add_argument("--raw-dir", default="./data/raw_books", help="Directory containing raw PDF books")
    parser.add_argument("--output-dir", default="./data/processed", help="Directory to save processed outputs")
    parser.add_argument("--log-file", default="./logs/extraction.log", help="Path to write extraction log file")
    parser.add_argument("--force", action="store_true", help="Reprocess books even if outputs already exist")
    parser.add_argument("--tesseract-path", help="Direct path to the tesseract.exe binary (Windows)")
    parser.add_argument("--lang", default="guj+eng", help="OCR language codes (e.g. 'guj+eng' for Tesseract)")
    args = parser.parse_args()
    
    # Set up paths relative to current script/run directory
    raw_dir = Path(args.raw_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    log_file = Path(args.log_file).resolve()
    
    setup_logging(log_file)
    logger.info("=========================================")
    logger.info("Starting Author Corpus Extraction Pipeline (Dual OCR Version)")
    logger.info(f"Raw directory: {raw_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Language: {args.lang}")
    logger.info("=========================================")
    
    # 1. Setup Directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Check OCR availability
    ocr_engine = check_ocr_available(args.tesseract_path)
    if ocr_engine:
        logger.info(f"Active OCR engine: {ocr_engine}")
    else:
        logger.warning("No OCR fallback engine is available. Pages requiring OCR will be skipped.")
    
    # 3. Find PDFs
    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {raw_dir}.")
        logger.warning("Please drop your 20 PDF books into the raw directory and rerun the script.")
        print(f"\n[ACTION REQUIRED] Please drop your PDF books in: {raw_dir}\n")
        
        # Write an empty manifest to initialize
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            manifest = {
                "total_books_processed": 0,
                "total_pages_processed": 0,
                "total_ocr_fallback_pages": 0,
                "total_chunks_generated": 0,
                "books": {},
                "failed_files": {}
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        return
        
    logger.info(f"Found {len(pdf_files)} PDF books to process.")
    
    # Load manifest if it exists
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "total_books_processed": 0,
        "total_pages_processed": 0,
        "total_ocr_fallback_pages": 0,
        "total_chunks_generated": 0,
        "books": {},
        "failed_files": {}
    }
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing manifest.json: {e}. Starting fresh.")
            
    processed_books = manifest.get("books", {})
    failed_files = manifest.get("failed_files", {})
    
    # Keep track of this run's changes to save
    run_books_stats = {}
    run_failed_files = {}
    
    for pdf_path in tqdm(pdf_files, desc="Processing books"):
        book_title = pdf_path.stem
        book_slug = slugify(book_title)
        
        # Check idempotency
        full_text_path = output_dir / book_slug / "full_text.txt"
        chunks_path = output_dir / book_slug / "chunks.jsonl"
        
        if full_text_path.exists() and chunks_path.exists() and not args.force and book_slug in processed_books:
            logger.info(f"Skipping '{book_title}' - outputs already exist (use --force to reprocess).")
            continue
            
        try:
            stats = process_pdf(pdf_path, output_dir, ocr_engine, args.lang)
            run_books_stats[book_slug] = stats
            # Remove from failed if it succeeded now
            if book_slug in failed_files:
                del failed_files[book_slug]
        except Exception as e:
            logger.exception(f"Failed to process '{book_title}': {e}")
            run_failed_files[book_slug] = {
                "file": pdf_path.name,
                "error": str(e)
            }
            
    # Update manifest
    # Update successful books
    for slug, stats in run_books_stats.items():
        processed_books[slug] = stats
    # Update failed files
    for slug, info in run_failed_files.items():
        failed_files[slug] = info
        
    # Re-calculate totals across all successfully processed books
    total_pages = sum(stats["total_pages"] for stats in processed_books.values())
    total_ocr = sum(stats["ocr_pages_count"] for stats in processed_books.values())
    total_chunks = sum(stats["chunks_count"] for stats in processed_books.values())
    
    manifest["total_books_processed"] = len(processed_books)
    manifest["total_pages_processed"] = total_pages
    manifest["total_ocr_fallback_pages"] = total_ocr
    manifest["total_chunks_generated"] = total_chunks
    manifest["books"] = processed_books
    manifest["failed_files"] = failed_files
    
    # Save manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    logger.info("=========================================")
    logger.info("Pipeline Execution Complete!")
    logger.info(f"Total Books: {manifest['total_books_processed']}")
    logger.info(f"Total Pages: {manifest['total_pages_processed']}")
    logger.info(f"Total OCR Fallback Pages: {manifest['total_ocr_fallback_pages']}")
    logger.info(f"Total Chunks Generated: {manifest['total_chunks_generated']}")
    if manifest["failed_files"]:
        logger.warning(f"Failed files count: {len(manifest['failed_files'])}")
    logger.info("=========================================")

if __name__ == "__main__":
    main()
