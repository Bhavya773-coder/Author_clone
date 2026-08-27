#!/usr/bin/env python3
"""
Voice & Talking-Avatar Enabled Book Knowledge Chatbot API Server
=================================================================
Serves the web chat UI and provides HTTP API endpoints:
- /api/chat: Answer queries via BookKnowledgeModel (model.pkl) + Ollama
- /api/tts: Text-to-Speech synthesis (Edge-TTS / Piper / Fallback)
- /api/avatar/speak: Trigger talking avatar video render job
- /api/avatar/status/<job_id>: Check avatar job status
- /api/avatar/video/<hash>: Serve cached MP4 avatar video

Usage:
  python -X utf8 scripts/server.py
  python -X utf8 scripts/server.py --port 8000
"""

import os
import sys
import json
import pickle
import shutil
import threading
import urllib.request
import argparse
import logging
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

# Add scripts directory to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from build_model import BookKnowledgeModel
from tts_engine import clean_text_for_speech
from audio_pipeline import synthesize_speech_once
from avatar_engine import (
    enqueue_avatar_job,
    get_avatar_job_status,
    select_avatar_engine,
    AVATAR_OUTPUT_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Server")

# Environment Configuration
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "edge-tts")
AVATAR_ENGINE_PREF = os.environ.get("AVATAR_ENGINE", "sadtalker")
AVATAR_PORTRAIT = os.environ.get("AVATAR_PORTRAIT", str(PROJECT_ROOT / "web" / "avatar_character.jpg"))

# Global Model References
MODEL = None
TOKENIZER = None
TRANSFORMER_MODEL = None
DEVICE = "cpu"

# Gujarati-first answering: Rathod's humor only lands in the original language.
# This overrides the English-only system prompt baked into the pickled model.
SYSTEM_PROMPT_GU_FIRST = (
    "You are Shahbuddin Rathod — the renowned Gujarati humorist, philosopher, and author, "
    "with deep knowledge of all 20 of your books.\n\n"
    "INSTRUCTIONS:\n"
    "1. FOR QUESTIONS ABOUT YOUR BOOKS, CHARACTERS, STORIES, & PHILOSOPHY: Answer thoroughly in natural, "
    "fluent GUJARATI (Gujarati script), in your authentic warm, witty voice. Keep every joke, punchline, "
    "proverb, and quote in its ORIGINAL Gujarati wording exactly as it appears in the source passages — "
    "never translate the humor itself into English. Cite source book names and page numbers.\n"
    "2. FOR QUESTIONS UNRELATED TO YOUR BOOKS: State exactly: "
    "'આ પ્રશ્નનો જવાબ શાહબુદ્દીન રાઠોડની 20 પુસ્તકોમાં નથી. (This question is not covered in Shahbuddin Rathod's 20 books.)'\n"
    "3. ZERO EXTERNAL KNOWLEDGE / NO HALLUCINATIONS: Do NOT invent facts or use outside knowledge for non-book topics.\n"
)


def run_startup_validation(model_path="model.pkl"):
    """Run comprehensive startup validation with clear diagnostic instructions."""
    print("=" * 70)
    print("🔍 AUTHOR AI VOICE & AVATAR SERVER — STARTUP VALIDATION")
    print("=" * 70)

    # 1. Validate model.pkl
    m_path = Path(model_path)
    if not m_path.exists():
        print(f"❌ [model.pkl] Missing at {m_path.resolve()}")
        print("   👉 Fix: Run `python scripts/build_model.py` to build the knowledge model.")
    else:
        print(f"✅ [model.pkl] Found at {m_path.name}")

    # 2. Validate Ollama Backend
    try:
        req = urllib.request.Request(
            OLLAMA_URL.replace("/generate", "/version"),
            headers={"User-Agent": "AuthorAIServer"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            print(f"✅ [Ollama] Service active at {OLLAMA_URL}")
    except Exception as e:
        print(f"⚠️ [Ollama] Could not connect to Ollama at {OLLAMA_URL} ({e})")
        print("   👉 Fix: Start Ollama by running `ollama serve` in a terminal.")

    # 3. Validate Default Portrait Image
    p_path = Path(AVATAR_PORTRAIT)
    if not p_path.exists():
        print(f"⚠️ [Avatar Portrait] Missing image at {p_path}")
        print("   👉 Fix: Place a default portrait JPG image at `web/avatar_character.jpg`.")
    else:
        print(f"✅ [Avatar Portrait] Found at {p_path.name}")

    # 4. Validate TTS Provider
    try:
        import edge_tts
        print(f"✅ [TTS Provider] Edge-TTS available (Provider: {TTS_PROVIDER})")
    except ImportError:
        print(f"⚠️ [TTS Provider] `edge-tts` not installed. Falling back to browser SpeechSynthesis.")
        print("   👉 Fix: Run `pip install edge-tts`.")

    # 5. Validate Avatar Rendering Engine
    engine_obj = select_avatar_engine(AVATAR_ENGINE_PREF)
    if engine_obj.name == "SadTalkerEngine":
        print(f"✅ [Avatar Engine] Real Photorealistic Video Mode ACTIVE ({engine_obj.name})")
    elif engine_obj.name == "Wav2LipEngine":
        print(f"✅ [Avatar Engine] Real Wav2Lip Video Mode ACTIVE ({engine_obj.name})")
    else:
        print(f"ℹ️ [Avatar Engine] Active Engine: {engine_obj.name} (Web Audio Lip-Sync Fallback)")
        print("   👉 Note: To enable real photorealistic video generation, install SadTalker or Wav2Lip")
        print("            and set SADTALKER_PATH environment variable.")

    print("=" * 70 + "\n")


def load_all_resources(model_path="model.pkl"):
    global MODEL, TOKENIZER, TRANSFORMER_MODEL, DEVICE

    run_startup_validation(model_path)

    if Path(model_path).exists():
        print(f"📦 Unpickling BookKnowledgeModel from {model_path}...")
        with open(model_path, "rb") as f:
            MODEL = pickle.load(f)
        print("   ✅ Book Knowledge Index ready!")

        import torch
        from transformers import AutoTokenizer, AutoModel

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️  Loading embedding encoder ({MODEL.embedding_model_name}) on {DEVICE}...")
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL.embedding_model_name)
        TRANSFORMER_MODEL = AutoModel.from_pretrained(MODEL.embedding_model_name).to(DEVICE)
        TRANSFORMER_MODEL.eval()
        print("   ✅ Embedding encoder ready!\n")


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
    if MODEL is None or TRANSFORMER_MODEL is None:
        return "Book knowledge model is not loaded. Please build model.pkl first.", []

    oks_context = MODEL.get_oks_context(query)
    q_vec = get_query_embedding(query)
    search_results = MODEL.search(q_vec, top_k=top_k * 3, search_type="all")
    
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
            f"2. Write your response in natural, fluent GUJARATI (Gujarati script). The humor lives in the original language: keep jokes, punchlines, proverbs, and quotes EXACTLY as written in the passages — never translate the humor into English.\n"
            f"3. Always cite official book titles instead of internal IDs (e.g. state 'in the book \"Pan Mare Kya Lakhvu Hatu?\" (p.6)' or '[\"Pan Mare Kya Lakhvu Hatu?\", p.6]' instead of generic terms like Book-01)."
        )
    else:
        full_prompt = (
            f"User Question: {query}\n\n"
            f"STRICT INSTRUCTION: This question is outside the scope of Shahbuddin Rathod's 20 books. "
            f"State exactly: 'આ પ્રશ્નનો જવાબ શાહબુદ્દીન રાઠોડની 20 પુસ્તકોમાં નથી. (This question is not covered in Shahbuddin Rathod\'s 20 books.)'"
        )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "system": SYSTEM_PROMPT_GU_FIRST,
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
        logger.error(f"Ollama request error: {e}")
        response_text = f"[Ollama Connection Error]: Could not reach Ollama model '{OLLAMA_MODEL}' at {OLLAMA_URL}. Ensure `ollama serve` is running."
        
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


class RequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        web_dir = PROJECT_ROOT / "web"
        super().__init__(*args, directory=str(web_dir), **kwargs)

    def do_GET(self):
        clean_path = self.path.split('?')[0].rstrip('/')
        # Route avatar job status check: GET /api/avatar/status/<job_id>
        if clean_path.startswith("/api/avatar/status"):
            job_id = clean_path.replace("/api/avatar/status", "").lstrip('/').strip()
            status_data = get_avatar_job_status(job_id)
            self._send_json(status_data)
            return

        # Route avatar video serving: GET /api/avatar/video/<job_hash>
        elif clean_path.startswith("/api/avatar/video"):
            job_hash = clean_path.replace("/api/avatar/video", "").lstrip('/').strip()
            video_file = AVATAR_OUTPUT_DIR / f"{job_hash}.mp4"
            if video_file.exists():
                self._send_file(video_file, "video/mp4")
            else:
                self._send_json({"error": "Avatar video file not found"}, status=404)
            return

        # Serve static web files
        super().do_GET()

    def do_POST(self):
        clean_path = self.path.split('?')[0].rstrip('/')
        if clean_path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
                query = data.get("query", "").strip()
                if not query:
                    self._send_json({"error": "Empty query"}, status=400)
                    return
                    
                response_text, sources = answer_query(query)
                clean_text = clean_text_for_speech(response_text)
                voice = data.get("voice", "en-US-AriaNeural")

                # Pre-warm the single-generation TTS cache in the background so
                # the answer text returns immediately, while the browser's
                # /api/tts request (and any Replay) hits warm cache instead of
                # generating speech twice.
                threading.Thread(
                    target=lambda: synthesize_speech_once(clean_text, voice),
                    daemon=True, name="tts-prewarm"
                ).start()

                # Trigger avatar render job asynchronously. The engines reuse
                # the cached audio via the cache key, so TTS runs exactly once.
                job_id = enqueue_avatar_job(
                    text=response_text,
                    voice=voice,
                    portrait_path=data.get("portrait"),
                    engine=data.get("engine"),
                )

                self._send_json({
                    "response": response_text,
                    "sources": sources,
                    "speak_text": clean_text,
                    "avatar_job_id": job_id
                })
            except Exception as e:
                logger.error(f"/api/chat endpoint error: {e}")
                self._send_json({"error": str(e)}, status=500)

        elif clean_path == "/api/tts":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
                text = data.get("text", "").strip()
                voice = data.get("voice", "en-US-AriaNeural")
                provider = data.get("provider", TTS_PROVIDER)
                if not text:
                    self._send_json({"error": "Empty text"}, status=400)
                    return

                # Single-generation cache: identical (text, voice) replays never re-run TTS.
                audio_bytes, mime = synthesize_speech_once(text, voice, provider)
                if audio_bytes:
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(audio_bytes)))
                    self.end_headers()
                    self.wfile.write(audio_bytes)
                else:
                    self._send_json({"status": "fallback", "provider": mime, "message": "Using browser speechSynthesis"}, status=200)
            except Exception as e:
                logger.error(f"/api/tts endpoint error: {e}")
                self._send_json({"error": str(e)}, status=500)

        elif clean_path == "/api/avatar/speak":

            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
                text = data.get("text", "").strip()
                voice = data.get("voice", "en-US-AriaNeural")
                portrait = data.get("portrait")
                engine = data.get("engine", AVATAR_ENGINE_PREF)

                if not text:
                    self._send_json({"error": "Empty text"}, status=400)
                    return

                job_id = enqueue_avatar_job(text=text, voice=voice, portrait_path=portrait, engine=engine)
                status_info = get_avatar_job_status(job_id)
                self._send_json({
                    "job_id": job_id,
                    "status": status_info.get("status"),
                    "engine": status_info.get("engine")
                })
            except Exception as e:
                logger.error(f"/api/avatar/speak endpoint error: {e}")
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

    def _send_file(self, file_path: Path, content_type: str):
        try:
            stat = file_path.stat()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(stat.st_size))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with open(file_path, "rb") as f:
                shutil.copyfileobj(f, self.wfile)
        except Exception as e:
            logger.error(f"Error serving file {file_path}: {e}")
            self.send_error(500, "Internal Server Error")


def main():
    parser = argparse.ArgumentParser(description="Voice & Avatar Enabled Book QA Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--model-path", default="model.pkl", help="Path to model.pkl")
    args = parser.parse_args()

    load_all_resources(args.model_path)
    
    server_address = ("", args.port)
    httpd = ThreadingHTTPServer(server_address, RequestHandler)
    print(f"🚀 Server running at http://localhost:{args.port}")
    print(f"💬 Web Chat UI: http://localhost:{args.port}/")
    print(f"🎙️  Voice Speech Synthesis & Talking Avatar pipeline ready!")
    print("Press Ctrl+C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    main()
