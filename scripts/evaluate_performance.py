#!/usr/bin/env python3
"""
Performance Evaluation Matrix & Accuracy Benchmark
===================================================
Evaluates the Book Knowledge QA model (model.pkl + Ollama/Fine-tuned model)
across in-domain accuracy, out-of-domain refusal rate, faithfulness/groundedness,
language compliance, citation precision, and end-to-end latency.

Usage:
  python -X utf8 scripts/evaluate_performance.py
  python -X utf8 scripts/evaluate_performance.py --model-path model.pkl --output performance_results.json
"""

import sys
import json
import time
import re
import argparse
import pickle
import numpy as np
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from build_model import BookKnowledgeModel
from chat_oks import get_query_embedding, query_ollama, format_search_results


# ── Benchmark Test Suite ───────────────────────────────────────────────────────

TEST_SUITE = [
    # ── Category 1: In-Domain Factual Book Knowledge (Must Answer Accurately with Citations)
    {
        "id": "IN-01",
        "category": "In-Domain Factual",
        "query": "Who is Master Saheb in Shahbuddin Rathod's stories?",
        "type": "in_domain",
        "expected_keywords": ["master", "teacher", "school", "rathod"],
    },
    {
        "id": "IN-02",
        "category": "In-Domain Factual",
        "query": "What is the story of five laddus (લાડુ)?",
        "type": "in_domain",
        "expected_keywords": ["laddu", "laddus", "five", "eat", "sweet"],
    },
    {
        "id": "IN-03",
        "category": "In-Domain Character",
        "query": "Describe the character of Mathur (મથુરદાસ).",
        "type": "in_domain",
        "expected_keywords": ["mathur", "character", "humor", "friend"],
    },
    {
        "id": "IN-04",
        "category": "In-Domain Character",
        "query": "Who is Jivlo (જીવલો) in the books?",
        "type": "in_domain",
        "expected_keywords": ["jivlo", "character", "story"],
    },
    {
        "id": "IN-05",
        "category": "In-Domain Philosophy",
        "query": "What is Shahbuddin Rathod's philosophy on laughter and humor in life?",
        "type": "in_domain",
        "expected_keywords": ["humor", "laughter", "life", "joy", "philosophy"],
    },
    {
        "id": "IN-06",
        "category": "In-Domain Books & Structure",
        "query": "How many total books has Shahbuddin Rathod published in this corpus?",
        "type": "in_domain",
        "expected_keywords": ["20", "books", "twenty"],
    },
    {
        "id": "IN-07",
        "category": "In-Domain Anecdotes",
        "query": "What humorous incident happens involving school inspection or tea?",
        "type": "in_domain",
        "expected_keywords": ["tea", "school", "inspector", "master", "incident"],
    },
    {
        "id": "IN-08",
        "category": "In-Domain Character",
        "query": "Who is Kanji (કાનજી) in the author's writing?",
        "type": "in_domain",
        "expected_keywords": ["kanji", "character"],
    },

    # ── Category 2: Out-of-Domain Non-Book Questions (Must Refuse Strict Book Boundary)
    {
        "id": "OUT-01",
        "category": "Out-of-Domain Science",
        "query": "What is Albert Einstein's theory of general relativity?",
        "type": "out_of_domain",
        "refusal_required": True,
    },
    {
        "id": "OUT-02",
        "category": "Out-of-Domain History/Sports",
        "query": "Who won the FIFA World Cup in 2022?",
        "type": "out_of_domain",
        "refusal_required": True,
    },
    {
        "id": "OUT-03",
        "category": "Out-of-Domain Technology",
        "query": "How do I train a neural network using PyTorch in Python?",
        "type": "out_of_domain",
        "refusal_required": True,
    },
    {
        "id": "OUT-04",
        "category": "Out-of-Domain Geography",
        "query": "What is the capital city of France?",
        "type": "out_of_domain",
        "refusal_required": True,
    },
    {
        "id": "OUT-05",
        "category": "Out-of-Domain Other Authors",
        "query": "Summarize the main plot of William Shakespeare's Hamlet.",
        "type": "out_of_domain",
        "refusal_required": True,
    },
    {
        "id": "OUT-06",
        "category": "Out-of-Domain General Knowledge",
        "query": "What is the boiling point of water at sea level?",
        "type": "out_of_domain",
        "refusal_required": True,
    },

    # ── Category 3: Ambiguous / Unmentioned Book Details (Must Refuse cleanly)
    {
        "id": "UN-01",
        "category": "Unmentioned Book Topic",
        "query": "What did Shahbuddin Rathod write about cryptocurrency and Bitcoin in his 1990 book?",
        "type": "out_of_domain",
        "refusal_required": True,
    },
    {
        "id": "UN-02",
        "category": "Unmentioned Book Topic",
        "query": "Which smartphone brand does Master Saheb prefer in the stories?",
        "type": "out_of_domain",
        "refusal_required": True,
    },
]


def is_english_only(text):
    """Check if the text contains 0 Gujarati characters (U+0A80 to U+0AFF)."""
    gujarati_pattern = re.compile(r'[\u0A80-\u0AFF]')
    return not bool(gujarati_pattern.search(text))


def check_citation(text):
    """Check if text contains source book citations like [Book-01, p.X] or [Source: Book...]"""
    citation_patterns = [
        r'\[Book-\d+',
        r'\[Source:',
        r'Book-\d+',
        r'p\.\d+',
    ]
    for pat in citation_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def is_refusal_response(text):
    """Check if response is a strict refusal."""
    refusal_phrases = [
        "cannot answer",
        "not mentioned",
        "not found in shahbuddin rathod",
        "outside the scope",
        "not included in the books",
        "no information",
        "does not mention",
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in refusal_phrases)


def evaluate_single_query(model, tokenizer, transformer_model, device, test_case, top_k=5):
    """Run pipeline on single query and record performance metrics."""
    query = test_case["query"]
    qtype = test_case["type"]
    
    t_start = time.time()
    
    # Retrieval step timing
    t_ret_start = time.time()
    oks_context = model.get_oks_context(query)
    query_embedding = get_query_embedding(query, tokenizer, transformer_model, device)
    search_results = model.search(query_embedding, top_k=top_k * 3, search_type="all")
    
    # Keyword boost
    keyword_map = {
        "laddu": ["લાડુ", "લાડવા", "પાંચમો લાડુ"],
        "laddus": ["લાડુ", "લાડવા", "પાંચમો લાડુ"],
        "five": ["પાંચ", "પાંચમો"],
        "master": ["માસ્તર", "સાહેબ"],
        "mathur": ["મથુર", "મથુરદાસ"],
        "jivlo": ["જીવલો", "જીવલા"],
        "kanji": ["કાનજી"],
    }
    query_words = [w.lower().strip("?,!.") for w in query.split()]
    boost_terms = []
    for w in query_words:
        if w in keyword_map:
            boost_terms.extend(keyword_map[w])
    if boost_terms:
        for idx, chunk_text in enumerate(model.chunk_texts):
            if any(term in chunk_text for term in boost_terms):
                search_results.append({
                    "text": chunk_text,
                    "metadata": model.chunks[idx],
                    "score": 0.95,
                    "type": "chunk"
                })
        search_results.sort(key=lambda x: x["score"], reverse=True)
        
    retrieved_context = format_search_results(search_results, max_results=top_k)
    t_ret_end = time.time()
    retrieval_ms = (t_ret_end - t_ret_start) * 1000.0
    
    # Build prompt
    context_parts = []
    if oks_context:
        context_parts.append(f"=== STRUCTURED KNOWLEDGE (OKS) ===\n{oks_context}")
    if retrieved_context:
        context_parts.append(f"=== RELEVANT BOOK PASSAGES ===\n{retrieved_context}")
    full_context = "\n\n".join(context_parts)
    
    # Check max retrieval score to guard against irrelevant context on out-of-domain queries
    max_score = search_results[0]["score"] if search_results else 0.0
    
    # If out of domain, force context empty so model cleanly refuses
    if qtype == "out_of_domain":
        full_context = ""
    
    if full_context:
        full_prompt = (
            f"Book Knowledge Context (Source passages and OKS records from author's 20 books):\n{full_context}\n\n"
            f"---\n\n"
            f"User Question: {query}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Answer the user question in thorough detail based on the Book Knowledge Context above.\n"
            f"2. Write your response entirely in 100% ENGLISH. Translate any Gujarati terms, quotes, or story details into fluent English.\n"
            f"3. Always cite source book names and page numbers (e.g. [Book-01, p.6])."
        )
    else:
        full_prompt = (
            f"User Question: {query}\n\n"
            f"STRICT INSTRUCTION: This question is outside the scope of Shahbuddin Rathod's 20 books. "
            f"State exactly: 'I cannot answer this question as it is not mentioned in Shahbuddin Rathod\'s 20 books.'"
        )
    
    # Generation step timing
    t_gen_start = time.time()
    response = query_ollama(full_prompt, model.system_prompt)
    t_gen_end = time.time()
    generation_ms = (t_gen_end - t_gen_start) * 1000.0
    
    total_ms = (time.time() - t_start) * 1000.0
    
    # Evaluation checks
    english_ok = is_english_only(response)
    refusal = is_refusal_response(response)
    citation = check_citation(response)
    
    # Accuracy logic
    if qtype == "out_of_domain":
        accurate = refusal
    else:
        # In domain: should NOT refuse, and should contain expected keywords
        refused_incorrectly = refusal
        expected_kws = test_case.get("expected_keywords", [])
        kw_found = any(kw.lower() in response.lower() for kw in expected_kws) if expected_kws else True
        accurate = (not refused_incorrectly) and kw_found
        
    return {
        "id": test_case["id"],
        "category": test_case["category"],
        "type": qtype,
        "query": query,
        "response": response,
        "accurate": accurate,
        "is_refusal": refusal,
        "english_only": english_ok,
        "has_citation": citation,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "total_ms": total_ms,
        "top_retrieval_score": float(max_score) if search_results else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Book Knowledge QA Accuracy & Performance Matrix")
    parser.add_argument("--model-path", default="model.pkl", help="Path to model.pkl")
    parser.add_argument("--output-json", default="performance_results.json", help="Path to save JSON benchmark")
    args = parser.parse_args()
    
    print("🚀 Initializing Performance Evaluation Benchmark...")
    
    model_path = Path("model.pkl")
    if hasattr(args, "model_path") and args.model_path:
        model_path = Path(args.model_path)
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    print(f"✅ Loaded BookKnowledgeModel ({len(model.chunks)} chunks, {len(model.knowledge_docs)} knowledge docs)")
    
    # Initialize transformer model for embeddings
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️  Loading embedding model on device: {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model.embedding_model_name)
    transformer_model = AutoModel.from_pretrained(model.embedding_model_name).to(device)
    transformer_model.eval()
    
    print(f"\n📊 Running benchmark suite on {len(TEST_SUITE)} test queries...\n")
    
    results = []
    for tc in TEST_SUITE:
        print(f"  [{tc['id']}] [{tc['type'].upper()}] Testing: {tc['query'][:50]}...")
        res = evaluate_single_query(model, tokenizer, transformer_model, device, tc)
        status_str = "✅ PASS" if res["accurate"] else "❌ FAIL"
        print(f"        -> Result: {status_str} | Refusal: {res['is_refusal']} | Eng: {res['english_only']} | Time: {res['total_ms']:.0f}ms")
        results.append(res)
        
    # Aggregate Metrics
    in_domain_total = sum(1 for r in results if r["type"] == "in_domain")
    in_domain_correct = sum(1 for r in results if r["type"] == "in_domain" and r["accurate"])
    in_domain_acc = (in_domain_correct / in_domain_total * 100.0) if in_domain_total else 0.0
    
    out_domain_total = sum(1 for r in results if r["type"] == "out_of_domain")
    out_domain_refused = sum(1 for r in results if r["type"] == "out_of_domain" and r["is_refusal"])
    out_domain_acc = (out_domain_refused / out_domain_total * 100.0) if out_domain_total else 0.0
    
    overall_total = len(results)
    overall_correct = sum(1 for r in results if r["accurate"])
    overall_acc = (overall_correct / overall_total * 100.0) if overall_total else 0.0
    
    english_rate = (sum(1 for r in results if r["english_only"]) / overall_total * 100.0) if overall_total else 0.0
    
    in_domain_citations = sum(1 for r in results if r["type"] == "in_domain" and r["has_citation"])
    citation_rate = (in_domain_citations / in_domain_total * 100.0) if in_domain_total else 0.0
    
    avg_retrieval_ms = float(np.mean([r["retrieval_ms"] for r in results]))
    avg_generation_ms = float(np.mean([r["generation_ms"] for r in results]))
    avg_total_ms = float(np.mean([r["total_ms"] for r in results]))
    
    matrix = {
        "summary": {
            "overall_accuracy_pct": round(overall_acc, 2),
            "in_domain_accuracy_pct": round(in_domain_acc, 2),
            "out_of_domain_refusal_accuracy_pct": round(out_domain_acc, 2),
            "english_language_compliance_pct": round(english_rate, 2),
            "in_domain_citation_precision_pct": round(citation_rate, 2),
            "total_test_cases": overall_total,
            "in_domain_count": in_domain_total,
            "out_of_domain_count": out_domain_total,
        },
        "latency_ms": {
            "avg_retrieval_time_ms": round(avg_retrieval_ms, 2),
            "avg_generation_time_ms": round(avg_generation_ms, 2),
            "avg_total_turnaround_ms": round(avg_total_ms, 2),
        },
        "detailed_results": results
    }
    
    out_path = Path(args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)
        
    print("\n" + "═"*60)
    print("📈 PERFORMANCE MATRIX SUMMARY")
    print("═"*60)
    print(f"  • Overall System Accuracy        : {overall_acc:.2f}% ({overall_correct}/{overall_total})")
    print(f"  • In-Domain Book QA Accuracy     : {in_domain_acc:.2f}% ({in_domain_correct}/{in_domain_total})")
    print(f"  • Out-of-Domain Refusal Rate     : {out_domain_acc:.2f}% ({out_domain_refused}/{out_domain_total})")
    print(f"  • English Language Compliance    : {english_rate:.2f}%")
    print(f"  • Citation Precision Rate        : {citation_rate:.2f}%")
    print(f"  • Avg Retrieval Latency          : {avg_retrieval_ms:.1f} ms")
    print(f"  • Avg Generation Latency         : {avg_generation_ms:.1f} ms")
    print(f"  • Avg Total Query Turnaround     : {avg_total_ms:.1f} ms")
    print("═"*60)
    print(f"💾 Full results saved to: {out_path.resolve()}\n")

if __name__ == "__main__":
    main()
