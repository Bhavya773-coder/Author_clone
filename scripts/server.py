#!/usr/bin/env python3
"""
Voice-Enabled Book Knowledge Chatbot API Server
=================================================
Serves the web chat UI and provides an HTTP API endpoint (/api/chat)
connected to BookKnowledgeModel (model.pkl) and local Ollama backend.

Usage:
  python -X utf8 scripts/server.py
  python -X utf8 scripts/server.py --port 8000
"""

import sys
import json
import pickle
import urllib.request
import argparse
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

# Import BookKnowledgeModel for unpickling
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from build_model import BookKnowledgeModel

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

# Global references
MODEL = None
TOKENIZER = None
TRANSFORMER_MODEL = None
DEVICE = "cpu"


def load_all_resources(model_path="model.pkl"):
    global MODEL, TOKENIZER, TRANSFORMER_MODEL, DEVICE
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    print(f"📦 Loading BookKnowledgeModel from {model_path}...")
    with open(model_path, "rb") as f:
        MODEL = pickle.load(f)
    print("   ✅ Model loaded successfully!")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️  Loading embedding encoder ({MODEL.embedding_model_name}) on {DEVICE}...")
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL.embedding_model_name)
    TRANSFORMER_MODEL = AutoModel.from_pretrained(MODEL.embedding_model_name).to(DEVICE)
    TRANSFORMER_MODEL.eval()
    print("   ✅ Embedding encoder ready!")


def get_query_embedding(text):
    import torch
    query_text = f"query: {text}"
    inputs = TOKENIZER([query_text], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = TRANSFORMER_MODEL(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return normalized.cpu().numpy()[0]


BOOK_TITLE_MAP = {
    "book-01": "Pan Mare Kya Lakhvu Hatu? (પણ મારે ક્યાં લખવું હતું ?)",
    "book-02": "Show Must Go On (શો મસ્ટ ગો ઓન)",
    "book-03": "Lakh Rupiya Ni Vaat (લાખ રૂપિયાની વાત)",
    "book-04": "Devu To Mard Kare (દેવું તો મર્દ કરે)",
    "book-05": "Aashiyana (આશિયાના)",
    "book-06": "Maro Ghededo Kyay Dekhay Chhe? (મારો ગધેડો ક્યાંય દેખાય છે?)",
    "book-07": "Vah Dost Vah (વાહ દોસ્ત વાહ)",
    "book-08": "Chintan Kanikao Ane Haasya (ચિંતનકણિકાઓ અને હાસ્ય)",
    "book-09": "Thangadh Ni Haasya Yatra (થાનગઢની હાસ્ય યાત્રા)",
    "book-10": "Jaju Safir (ઝજુ સાફિર)",
    "book-11": "Murkhai Ane Gyaan (મૂર્ખાઈ અને જ્ઞાન)",
    "book-12": "Vaatne Val Chadavini Kala (વાતને વળ ચડાવીને કહેવાની કળા)",
    "book-13": "Viral Vyaktimatta (વિરલ વ્યક્તિમત્તા)",
    "book-14": "Mahanubhavo Na Mantavyo (મહાનુભાવોનાં મંતવ્યો)",
    "book-15": "Haasya Rasik Prasango (હાસ્યરસિક પ્રસંગો)",
    "book-16": "Master Saheb Na Kissa (માસ્તર સાહેબના કિસ્સા)",
    "book-17": "Sahitya Chintan (સાહિત્ય ચિંતન)",
    "book-18": "Garibai Mathi Pragatelu Haasya (ગરિબીમાંથી પ્રગટેલું હાસ્ય)",
    "book-19": "Galkanu Phool (ગલકાનું ફૂલ)",
    "book-20": "Be Bol (બે બોલ)",
}

def get_book_title(book_raw):
    if not book_raw:
        return "Unknown"
    slug = str(book_raw).lower().strip()
    return BOOK_TITLE_MAP.get(slug, str(book_raw))


def format_search_results(results, max_results=5):
    if not results:
        return ""
    parts = []
    seen = set()
    count = 0
    for r in results:
        if count >= max_results:
            break
        text = r["text"]
        key = text[:200]
        if key in seen:
            continue
        seen.add(key)
        
        meta = r.get("metadata", {})
        book_slug = meta.get("book", "Unknown")
        book_title = get_book_title(book_slug)
        p_start = meta.get("page_start", "?")
        p_end = meta.get("page_end", "?")
        page_str = f"p.{p_start}" if str(p_start) == str(p_end) else f"p.{p_start}-{p_end}"
        parts.append(f"[Source: Book '{book_title}', {page_str}]\n{text[:800]}")
        count += 1
    return "\n\n---\n\n".join(parts)


def answer_query(query, top_k=5):
    oks_context = MODEL.get_oks_context(query)
    q_vec = get_query_embedding(query)
    search_results = MODEL.search(q_vec, top_k=top_k * 3, search_type="all")
    
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
        for idx, chunk_text in enumerate(MODEL.chunk_texts):
            if any(term in chunk_text for term in boost_terms):
                search_results.append({
                    "text": chunk_text,
                    "metadata": MODEL.chunks[idx],
                    "score": 0.95,
                    "type": "chunk"
                })
        search_results.sort(key=lambda x: x["score"], reverse=True)
        
    retrieved_context = format_search_results(search_results, max_results=top_k)
    
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
            f"1. Answer the user question in thorough, clear detail based on the Book Knowledge Context above.\n"
            f"2. Write your response entirely in 100% fluent ENGLISH. Translate any Gujarati terms or quotes into English.\n"
            f"3. Always cite official book titles instead of internal IDs (e.g. state 'in the book \"Pan Mare Kya Lakhvu Hatu?\" (p.6)' or '[\"Pan Mare Kya Lakhvu Hatu?\", p.6]' instead of generic terms like Book-01)."
        )
    else:
        full_prompt = (
            f"User Question: {query}\n\n"
            f"STRICT INSTRUCTION: This question is outside the scope of Shahbuddin Rathod's 20 books. "
            f"State exactly: 'I cannot answer this question as it is not mentioned in Shahbuddin Rathod\'s 20 books.'"
        )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "system": MODEL.system_prompt,
        "stream": False,
        "options": {"temperature": 0.4, "top_p": 0.9, "num_predict": 2048}
    }
    
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            response_text = body.get("response", "").strip()
    except Exception as e:
        response_text = f"[Ollama Connection Error]: {e}"
        
    sources = []
    seen = set()
    for r in search_results[:top_k]:
        meta = r.get("metadata", {})
        book_slug = meta.get("book", "Unknown")
        book_title = get_book_title(book_slug)
        p_start = meta.get("page_start", "?")
        key = f"{book_slug}_p{p_start}"
        if key not in seen:
            seen.add(key)
            sources.append({"book": book_title, "page": f"p.{p_start}", "score": round(r.get("score", 0), 3)})
            
    return response_text, sources


def generate_neural_tts_audio(text, voice="en-US-ChristopherNeural"):
    """Generate ultra-realistic neural speech MP3 audio using edge-tts."""
    import asyncio
    import edge_tts
    import re
    
    # Clean markdown formatting and citations before speaking
    clean = re.sub(r'\[Book-\d+,\s*p\.\d+[^\]]*\]', '', text)
    clean = re.sub(r'[\*\#\_`]', '', clean).strip()
    if not clean:
        clean = "No text to speak."

    async def _amain():
        communicate = edge_tts.Communicate(clean, voice)
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        return bytes(audio_data)

    try:
        return asyncio.run(_amain())
    except Exception as e:
        print(f"⚠️ Neural TTS Error: {e}")
        return None


class RequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        web_dir = Path(__file__).parent.parent / "web"
        super().__init__(*args, directory=str(web_dir), **kwargs)

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
                query = data.get("query", "").strip()
                if not query:
                    self._send_json({"error": "Empty query"}, status=400)
                    return
                    
                response_text, sources = answer_query(query)
                self._send_json({
                    "response": response_text,
                    "sources": sources
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        elif self.path == "/api/tts":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
                text = data.get("text", "").strip()
                voice = data.get("voice", "en-US-ChristopherNeural")
                if not text:
                    self._send_json({"error": "Empty text"}, status=400)
                    return
                audio_bytes = generate_neural_tts_audio(text, voice)
                if audio_bytes:
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/mpeg")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(audio_bytes)))
                    self.end_headers()
                    self.wfile.write(audio_bytes)
                else:
                    self._send_json({"error": "TTS synthesis failed"}, status=500)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        else:
            self.send_error(404, "Not Found")

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Voice-Enabled Book QA Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--model-path", default="model.pkl", help="Path to model.pkl")
    args = parser.parse_args()

    load_all_resources(args.model_path)
    
    server_address = ("", args.port)
    httpd = ThreadingHTTPServer(server_address, RequestHandler)
    print(f"\n🚀 Server running at http://localhost:{args.port}")
    print(f"💬 Web Chat UI: http://localhost:{args.port}/")
    print(f"🎙️  Voice Speech-to-Text & Text-to-Speech active!")
    print("Press Ctrl+C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    main()
