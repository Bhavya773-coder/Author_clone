#!/usr/bin/env python3
"""
OKS Book Knowledge Chat — Interactive Q&A from Author's Books
===============================================================
Loads model.pkl (the pre-built Book Knowledge Model) and provides an
interactive chat interface to ask questions about the author's 20 books.

The model uses:
  1. OKS structured data lookup (characters, themes, opinions, statistics)
  2. Semantic vector search over all book chunks + knowledge records
  3. Ollama LLM for natural language answer generation

Usage:
  # Interactive chat
  python -X utf8 scripts/chat_oks.py

  # Single query
  python -X utf8 scripts/chat_oks.py --query "How many books has the author written?"

  # Custom model path
  python -X utf8 scripts/chat_oks.py --model-path model.pkl --top-k 5
"""

import sys
import json
import pickle
import argparse
import urllib.request
from pathlib import Path

# Import BookKnowledgeModel for unpickling
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from build_model import BookKnowledgeModel


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"


def load_model(model_path):
    """Load the BookKnowledgeModel from a pickle file."""
    print(f"📦 Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"   ✅ Model loaded successfully!")
    return model


def get_query_embedding(text, tokenizer, transformer_model, device):
    """Compute normalized embedding for a query string using transformers."""
    import torch
    query_text = f"query: {text}"
    inputs = tokenizer([query_text], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = transformer_model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return normalized.cpu().numpy()[0]


def query_ollama(prompt, system_prompt, model_name=OLLAMA_MODEL):
    """Send a query to the local Ollama API and return the response."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 1024,
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "").strip()
    except Exception as e:
        return f"[Ollama Error]: {e}"


def format_search_results(results, max_results=5):
    """Format search results into a context string for the LLM."""
    if not results:
        return ""
    
    parts = []
    seen_texts = set()
    count = 0
    
    for r in results:
        if count >= max_results:
            break
        
        text = r["text"]
        text_key = text[:200]
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        
        meta = r.get("metadata", {})
        book = meta.get("book", "Unknown")
        page_start = meta.get("page_start", "?")
        page_end = meta.get("page_end", "?")
        rtype = r.get("type", "chunk")
        score = r.get("score", 0)
        
        page_str = f"p.{page_start}"
        if str(page_start) != str(page_end):
            page_str += f"-{page_end}"
        
        display_text = text[:800] if len(text) > 800 else text
        
        parts.append(
            f"[Source: {book}, {page_str}, Type: {rtype}, Relevance: {score:.3f}]\n"
            f"{display_text}"
        )
        count += 1
    
    return "\n\n---\n\n".join(parts)


def answer_question(model, tokenizer, transformer_model, device, query, top_k=5, ollama_model=OLLAMA_MODEL):
    """
    Answer a question using the Book Knowledge Model.
    
    Steps:
      1. Get OKS structured context (statistics, characters, themes)
      2. Compute query embedding and perform hybrid semantic + keyword search
      3. Combine context and send to Ollama for generation
    """
    print(f"\n{'─' * 60}")
    print(f"❓ Question: {query}")
    print(f"{'─' * 60}")
    
    # Step 1: Get OKS structured context
    oks_context = model.get_oks_context(query)
    if oks_context:
        print(f"📊 [OKS: Structured knowledge context found]")
    
    # Step 2: Multi-query semantic search
    query_embedding = get_query_embedding(query, tokenizer, transformer_model, device)
    search_results = model.search(query_embedding, top_k=top_k * 3, search_type="all")
    
    # Step 2b: Keyword boost for transliterated terms (e.g. laddu -> લાડુ, master -> માસ્તર)
    keyword_map = {
        "laddu": ["લાડુ", "લાડવા", "પાંચમો લાડુ"],
        "laddus": ["લાડુ", "લાડવા", "પાંચમો લાડુ"],
        "five": ["પાંચ", "પાંચમો"],
        "master": ["માસ્તર", "સાહેબ"],
        "saheb": ["સાહેબ", "માસ્તરસાહેબ"],
        "mathur": ["મથુર", "મથુરદાસ"],
        "jivlo": ["જીવલો", "જીવલા"],
        "kanji": ["કાનજી"],
        "jadavji": ["જાદવજી"],
        "abid": ["આબિદ"],
        "buddha": ["બુદ્ધ"],
        "socrates": ["સોક્રેટિસ"],
    }
    
    query_words = [w.lower().strip("?,!.") for w in query.split()]
    boost_terms = []
    for w in query_words:
        if w in keyword_map:
            boost_terms.extend(keyword_map[w])
            
    if boost_terms:
        # Re-score results or add keyword matches from chunks
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
    if retrieved_context:
        print(f"📚 [RAG: Retrieved {min(top_k, len(search_results))} relevant passages]")
    
    # Step 3: Build full prompt
    context_parts = []
    if oks_context:
        context_parts.append(f"=== STRUCTURED KNOWLEDGE (OKS) ===\n{oks_context}")
    if retrieved_context:
        context_parts.append(f"=== RELEVANT BOOK PASSAGES ===\n{retrieved_context}")
    
    full_context = "\n\n".join(context_parts)
    max_score = search_results[0]["score"] if search_results else 0.0
    if max_score < 0.60 and not oks_context:
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
    
    # Step 4: Generate answer via Ollama
    response = query_ollama(full_prompt, model.system_prompt, model_name=ollama_model)
    
    print(f"\n📖 Book Knowledge Answer:\n{response}")
    
    # Print citations
    if search_results:
        print(f"\n📌 Sources:")
        seen_sources = set()
        for r in search_results[:top_k]:
            meta = r.get("metadata", {})
            source_key = f"{meta.get('book', '?')}, p.{meta.get('page_start', '?')}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                print(f"   • {source_key} (chunk: {meta.get('chunk_id', '?')})")
    
    print(f"{'─' * 60}")
    return response


def query_fine_tuned_model(model, tokenizer, prompt, system_prompt, max_new_tokens=512):
    """Generate response using fine-tuned Hugging Face LoRA model with anti-repetition parameters."""
    import torch
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
    inputs = tokenizer(text, return_tensors="pt").to(model.device if hasattr(model, "device") else "cuda")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(
        description="OKS Book Knowledge Chat — Interactive Q&A from Author's Books"
    )
    parser.add_argument(
        "--model-path", default="model.pkl",
        help="Path to the model.pkl file"
    )
    parser.add_argument(
        "--adapter-path", default="models/book-knowledge-oks-v1",
        help="Path to fine-tuned LoRA adapter"
    )
    parser.add_argument(
        "--use-fine-tuned", action="store_true",
        help="Use fine-tuned 7B model instead of Ollama backend"
    )
    parser.add_argument(
        "--query", default=None,
        help="Single query (non-interactive mode)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of passages to retrieve for context"
    )
    parser.add_argument(
        "--ollama-model", default=OLLAMA_MODEL,
        help="Ollama model to use for generation"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent if script_dir.name == "scripts" else script_dir
    
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (base_dir / model_path).resolve()
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print(f"   Run: python -X utf8 scripts/build_model.py")
        sys.exit(1)
    
    model = load_model(model_path)
    model.info()
    
    print(f"\n🔤 Loading embedding model ({model.embedding_model_name}) for query encoding...")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("❌ transformers/torch not installed.")
        sys.exit(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model.embedding_model_name)
    transformer_model = AutoModel.from_pretrained(model.embedding_model_name).to(device)
    transformer_model.eval()
    print(f"   ✅ Encoder ready on {device}")
    
    ft_model = None
    ft_tokenizer = None
    if args.use_fine_tuned:
        adapter_path = Path(args.adapter_path)
        if not adapter_path.is_absolute():
            adapter_path = (base_dir / adapter_path).resolve()
        if adapter_path.exists():
            print(f"\n🧠 Loading fine-tuned LoRA adapter from {adapter_path}...")
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb_config, device_map="auto")
            ft_model = PeftModel.from_pretrained(base, str(adapter_path))
            ft_model.eval()
            ft_tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
            print(f"   ✅ Fine-tuned 7B model loaded successfully!")
    
    if args.query:
        if ft_model:
            # Custom answer routine using fine-tuned model
            query_emb = get_query_embedding(args.query, tokenizer, transformer_model, device)
            results = model.search(query_emb, top_k=args.top_k, search_type="all")
            ctx = format_search_results(results, max_results=args.top_k)
            oks_ctx = model.get_oks_context(args.query)
            full_prompt = f"Context:\n{oks_ctx}\n{ctx}\n\nQuestion: {args.query}"
            ans = query_fine_tuned_model(ft_model, ft_tokenizer, full_prompt, model.system_prompt)
            print(f"\n📖 Fine-Tuned Model Answer:\n{ans}")
        else:
            answer_question(model, tokenizer, transformer_model, device, args.query, 
                           top_k=args.top_k, ollama_model=args.ollama_model)
        return
    
    print("\n" + "=" * 60)
    print(" 📚 OKS Book Knowledge Chat")
    print(" Ask any question about the author's 20 books!")
    print(" Type 'exit' or 'quit' to stop")
    print(" Type 'info' to see model statistics")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n📝 Your question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("\nGoodbye! 👋")
                break
            if user_input.lower() == "info":
                model.info()
                continue
            
            answer_question(model, tokenizer, transformer_model, device, user_input,
                          top_k=args.top_k, ollama_model=args.ollama_model)
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break


if __name__ == "__main__":
    main()
