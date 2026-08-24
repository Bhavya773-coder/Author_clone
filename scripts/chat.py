#!/usr/bin/env python3
"""
Phase 6: Author Voice Chat — Inference with RAG
================================================
Loads the fine-tuned LoRA adapter on top of the base model, retrieves
relevant context from ChromaDB for each query, and generates a response
in Shahbuddin Rathod's authentic voice.

Usage:
  # Interactive chat
  python -X utf8 scripts/chat.py --adapter-dir ../models/rathod-voice-v1

  # Single query
  python -X utf8 scripts/chat.py --query "જીવનમાં ખુશી ક્યાં મળે?"

  # Without RAG (pure model response)
  python -X utf8 scripts/chat.py --no-rag

  # Use Ollama instead of loaded model (quick test before fine-tuning)
  python -X utf8 scripts/chat.py --use-ollama
"""

import sys
import argparse
import json
import urllib.request
from pathlib import Path


SYSTEM_PROMPT_EN = (
    "You are Shahbuddin Rathod — the renowned Gujarati humorist, author, and philosopher. "
    "Respond in your authentic voice: concise, witty, warm, with subtle humor, wisdom, and anecdotes from your books. "
    "CRITICAL REQUIREMENT: Always write your entire answer in clear, well-structured ENGLISH. "
    "Do not output Gujarati text. Provide rich details and explain concepts warmly and clearly."
)

SYSTEM_PROMPT_GU = (
    "તમે શાહ‍બુદ્દીન રાઠોડ છો — ગુજરાતના પ્રસિદ્ધ હાસ્યયોગ અને ફિલસૂફ. "
    "ટૂંકા, ચોક્કસ, સૂક્ષ્મ હ્યુમર અને ગુજરાતી ઉક્તિઓ-દ્રષ્ટાંતો સાથે જ જવાબ દો. "
    "ક્યારેય ઔપચારિક ન બોલો."
)

SYSTEM_PROMPT = SYSTEM_PROMPT_EN

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"


# ── Ollama fallback (no fine-tuning needed) ───────────────────────────────
def query_ollama(prompt: str, context: str, system_prompt: str = SYSTEM_PROMPT_EN, lang: str = "en") -> str:
    full_prompt = ""
    if context:
        if lang == "gu":
            full_prompt = f"સ્ત્રોત સામગ્રી:\n{context}\n\n---\n\nપ્રશ્ન: {prompt}"
        else:
            full_prompt = f"Source Material / Passages:\n{context}\n\n---\n\nQuestion: {prompt}\n\nPlease provide a clear, complete response strictly in ENGLISH."
    else:
        full_prompt = prompt

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": 0.75, "top_p": 0.9, "num_predict": 400},
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


# ── RAG retrieval ─────────────────────────────────────────────────────────
def retrieve_context(query: str, db_dir: Path, top_k: int = 3) -> str:
    """Retrieve relevant context chunks from ChromaDB."""
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
        print("⚠️  ChromaDB collection not found — skipping RAG.")
        return ""

    query_vec = model.encode([f"query: {query}"], convert_to_numpy=True).tolist()[0]
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)

    docs = results.get("documents", [[]])[0]
    if not docs:
        return ""

    # Strip E5 prefix and join
    clean = []
    for d in docs:
        clean.append(d[len("passage: "):] if d.startswith("passage: ") else d)
    return "\n---\n".join(clean)


# ── Fine-tuned model inference ────────────────────────────────────────────
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


def generate_response(model, tokenizer, query: str, context: str, system_prompt: str = SYSTEM_PROMPT_EN, lang: str = "en", max_new_tokens: int = 300) -> str:
    import torch

    if context:
        if lang == "gu":
            user_msg = f"સ્ત્રોત:\n{context}\n\n---\n\n{query}"
        else:
            user_msg = f"Source Material:\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer in ENGLISH."
    else:
        user_msg = query

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    # Use tokenizer's apply_chat_template if available (Qwen2.5 has it)
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.75,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    # Decode only newly generated tokens (strip the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 6 — Rathod Author Voice Chat")
    parser.add_argument("--adapter-dir", default="../models/rathod-voice-v1",
                        help="Path to saved LoRA adapter (from train.py)")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model used during fine-tuning")
    parser.add_argument("--db-dir", default="../data/vector_db",
                        help="ChromaDB vector database directory")
    parser.add_argument("--query", default=None,
                        help="Single query (non-interactive mode)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of RAG context chunks to retrieve")
    parser.add_argument("--no-rag", action="store_true",
                        help="Disable RAG context retrieval")
    parser.add_argument("--use-ollama", action="store_true",
                        help="Use Ollama instead of fine-tuned model (for quick testing)")
    parser.add_argument("--lang", choices=["en", "gu"], default="en",
                        help="Language for responses: 'en' (English, default) or 'gu' (Gujarati)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent if script_dir.name == "scripts" else script_dir

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.is_absolute():
        adapter_dir = (base_dir / adapter_dir).resolve()

    db_dir = Path(args.db_dir)
    if not db_dir.is_absolute():
        db_dir = (base_dir / db_dir).resolve()
    if not db_dir.exists() and (base_dir / "vector_db").exists():
        db_dir = base_dir / "vector_db"
    elif not db_dir.exists() and (base_dir / "data" / "vector_db").exists():
        db_dir = base_dir / "data" / "vector_db"

    # Select language prompt
    lang = args.lang.lower()
    system_prompt = SYSTEM_PROMPT_EN if lang == "en" else SYSTEM_PROMPT_GU

    # ── Choose backend ───────────────────────────────────────────────────
    use_ollama = args.use_ollama
    if not use_ollama and not adapter_dir.exists():
        print(f"⚠️  Fine-tuned adapter not found at: {adapter_dir}")
        print("   Falling back to Ollama (qwen2.5:3b) for testing.")
        print("   To fine-tune: run  python -X utf8 scripts/train.py\n")
        use_ollama = True

    model, tokenizer = None, None
    if not use_ollama:
        model, tokenizer = load_fine_tuned_model(adapter_dir, args.base_model)
        print(f"\n✅ Fine-tuned model loaded. Ready to chat! ({'English' if lang == 'en' else 'Gujarati'})\n")
    else:
        print(f"\n🔄 Using Ollama backend (qwen2.5:3b) - Language: {'English' if lang == 'en' else 'Gujarati'}.\n")

    def answer(query: str):
        print(f"\n{'─'*60}")
        print(f"🗣  You: {query}")
        print(f"{'─'*60}")

        # RAG
        context = ""
        if not args.no_rag:
            context = retrieve_context(query, db_dir, top_k=args.top_k)
            if context:
                print(f"📚 [RAG: retrieved {args.top_k} context passages]")

        # Generate
        if use_ollama:
            response = query_ollama(query, context, system_prompt=system_prompt, lang=lang)
        else:
            response = generate_response(model, tokenizer, query, context, system_prompt=system_prompt, lang=lang)

        print(f"\n🖊  Rathod: {response}")
        print(f"{'─'*60}")
        return response

    # ── Single query or interactive ──────────────────────────────────────
    if args.query:
        answer(args.query)
    else:
        print("=" * 60)
        print("  Shahbuddin Rathod — Author Voice Chat")
        print("  Type 'exit' or press Ctrl+C to quit")
        print("=" * 60)
        try:
            while True:
                query = input("\n📝 You: ").strip()
                if not query:
                    continue
                if query.lower() in {"exit", "quit", "bye", "q"}:
                    print("\nআল্লা হাফিઝ!\n")
                    break
                answer(query)
        except KeyboardInterrupt:
            print("\n\nChatting stopped.")


if __name__ == "__main__":
    main()
