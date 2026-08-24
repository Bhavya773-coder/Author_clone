#!/usr/bin/env python3
"""
Phase 6 (English): Book Knowledge Chat & QA — Inference with Cross-Lingual RAG
=============================================================================
Loads the fine-tuned English LoRA adapter on top of the base model (or Ollama fallback),
retrieves relevant book context from ChromaDB for each query, and generates an accurate,
detailed response in ENGLISH.

Usage:
  # Interactive English chat
  python -X utf8 scripts/chat_english.py --adapter-dir ../models/book-qa-english-v1

  # Single English query
  python -X utf8 scripts/chat_english.py --query "What is Master Saheb's philosophy on life?"

  # Test via Ollama backend (before GPU training)
  python -X utf8 scripts/chat_english.py --use-ollama --query "Tell me the story about the five laddus."
"""

import sys
import argparse
import json
import urllib.request
from pathlib import Path


SYSTEM_PROMPT = (
    "You are an intelligent, articulate AI assistant with deep knowledge of the 20 books "
    "written by Shahbuddin Rathod, the renowned humorist and philosopher.\n\n"
    "STRICT GROUNDING & BOUNDARY RULES:\n"
    "1. ANSWER ONLY FROM THE BOOKS: Answer questions strictly using ONLY information from the provided book context.\n"
    "2. MANDATORY REFUSAL FOR OUT-OF-BOUNDS TOPICS: If a question is not about Shahbuddin Rathod's books, life, characters, anecdotes, philosophy, or stories—OR if the provided context does NOT contain enough information—state exactly: 'I cannot answer this question as it is not mentioned in Shahbuddin Rathod\'s 20 books.'\n"
    "3. ZERO EXTERNAL KNOWLEDGE / NO HALLUCINATIONS: Do NOT use outside general world knowledge or speculative claims.\n"
    "4. 100% ENGLISH LANGUAGE RULE: Always write your entire answer in clear, well-structured ENGLISH. Do not output Gujarati text.\n"
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"


# ── Ollama fallback ───────────────────────────────────────────────────────
def query_ollama(prompt: str, context: str) -> str:
    if context:
        full_prompt = (
            f"Source Book Knowledge / Passages:\n{context}\n\n"
            f"---\n\n"
            f"User Question: {prompt}\n\n"
            f"STRICT INSTRUCTIONS:\n"
            f"1. Answer ONLY using facts present in the Source Book Knowledge above.\n"
            f"2. If the user question is outside the scope of the books or context, state exactly: 'I cannot answer this question as it is not mentioned in Shahbuddin Rathod\'s 20 books.'\n"
            f"3. Provide a thorough, complete answer strictly in ENGLISH based on the source material."
        )
    else:
        full_prompt = (
            f"User Question: {prompt}\n\n"
            f"STRICT INSTRUCTION: No relevant passages were found for this query in the 20 books. "
            f"State exactly: 'I cannot answer this question as it is not mentioned in Shahbuddin Rathod\'s 20 books.'"
        )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 512},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "").strip()
    except Exception as e:
        return f"[Ollama Error]: {e}"


# ── RAG Retrieval ─────────────────────────────────────────────────────────
def retrieve_context(query: str, db_dir: Path, top_k: int = 3) -> str:
    """Retrieve relevant book chunks from ChromaDB using multilingual embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError:
        print("⚠️  sentence-transformers / chromadb not installed — skipping RAG.")
        return ""

    if not db_dir.exists():
        print(f"⚠️  Vector DB not found at {db_dir} — skipping RAG.")
        return ""

    model = SentenceTransformer("intfloat/multilingual-e5-base")
    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        collection = client.get_collection("author_voice_collection")
    except Exception:
        print("⚠️  ChromaDB collection 'author_voice_collection' not found — skipping RAG.")
        return ""

    # E5 format query embedding
    query_vec = model.encode([f"query: {query}"], convert_to_numpy=True).tolist()[0]
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)

    docs = results.get("documents", [[]])[0]
    if not docs:
        return ""

    clean_docs = []
    for d in docs:
        text = d[len("passage: "):] if d.startswith("passage: ") else d
        clean_docs.append(text)

    return "\n---\n".join(clean_docs)


# ── Fine-Tuned Model Inference ────────────────────────────────────────────
def load_fine_tuned_model(adapter_dir: Path, base_model: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
    except ImportError as e:
        print(f"❌ Missing library: {e}")
        sys.exit(1)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    print(f"Loading LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(model, tokenizer, query: str, context: str, max_new_tokens: int = 450) -> str:
    import torch

    if context:
        user_msg = (
            f"Source Book Knowledge:\n{context}\n\n"
            f"---\n\n"
            f"Question: {query}\n\n"
            f"Answer in clear, complete ENGLISH grounded in the source book knowledge."
        )
    else:
        user_msg = query

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6 (English) — Book Knowledge QA Chat"
    )
    parser.add_argument(
        "--adapter-dir",
        default="../models/book-qa-english-v1",
        help="Path to saved English LoRA adapter",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model used during fine-tuning",
    )
    parser.add_argument(
        "--db-dir",
        default="../data/vector_db",
        help="ChromaDB vector database directory",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Single query (non-interactive mode)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of RAG passages to retrieve",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG context retrieval",
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use Ollama instead of local fine-tuned weights",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent if script_dir.name == "scripts" else script_dir

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.is_absolute():
        adapter_dir = (base_dir / adapter_dir).resolve()

    db_dir = Path(args.db_dir)
    if not db_dir.is_absolute():
        db_dir = (base_dir / db_dir).resolve()
        
    # Check alternate vector db path at root if not found
    if not db_dir.exists() and (base_dir / "vector_db").exists():
        db_dir = (base_dir / "vector_db").resolve()

    use_ollama = args.use_ollama
    if not use_ollama and not adapter_dir.exists():
        print(f"⚠️  English LoRA adapter not found at: {adapter_dir}")
        print("   Falling back to Ollama backend for testing.")
        print("   To fine-tune model: python -X utf8 scripts/train_english.py\n")
        use_ollama = True

    model, tokenizer = None, None
    if not use_ollama:
        model, tokenizer = load_fine_tuned_model(adapter_dir, args.base_model)
        print(f"\n✅ English Book QA Model loaded. Ready for queries!\n")
    else:
        print("\n🔄 Using Ollama backend (qwen2.5:3b) with English instructions.\n")

    def answer(query: str):
        print(f"\n{'─'*60}")
        print(f"❓ Question: {query}")
        print(f"{'─'*60}")

        context = ""
        if not args.no_rag:
            context = retrieve_context(query, db_dir, top_k=args.top_k)
            if context:
                print(f"📚 [RAG: Retrieved {args.top_k} book passages]")

        if use_ollama:
            response = query_ollama(query, context)
        else:
            response = generate_response(model, tokenizer, query, context)

        print(f"\n📖 Book QA Model (English):\n{response}")
        print(f"{'─'*60}")
        return response

    if args.query:
        answer(args.query)
    else:
        print("=" * 60)
        print(" English Book QA Chat (Type 'exit' or 'quit' to stop)")
        print("=" * 60)
        while True:
            try:
                user_input = input("\nEnter English question: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break
                answer(user_input)
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    main()
