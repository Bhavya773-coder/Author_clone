# Author voice model — PRD & TRD
### "Digital twin" of an author, built from their published books

Version 0.1 — Draft for review

---

## Part 1 — Product Requirements Document (PRD)

### 1. Executive summary

The goal is to build a system that can generate new answers, essays, and opinions **in the voice and thinking style of a specific author**, grounded in the ~20 books they've written — not a search tool that quotes the books back, but a model that has absorbed how the author thinks, argues, and phrases things, and can apply that to questions the author never explicitly answered in print.

This is fundamentally different from a Q&A bot over a PDF library. A retrieval bot answers "what did the author say about X" — this system answers "what would the author say about Y," including topics not directly covered in the source material.

### 2. Problem statement

Two failure modes to avoid:

1. **Retrieval-only systems** (a chatbot with RAG over the books) sound like a librarian quoting the author — accurate, but stiff, and unable to generalize beyond what's literally in the text.
2. **Fine-tuning-only systems** on a small corpus (20 books is a small dataset by LLM standards — likely 1.5–4M tokens total) risk two things: overfitting to memorized phrases (verbatim leakage — a real legal and ethical problem, see §6), and "style without substance" — sounding right but saying things the author never would have believed.

The right answer is neither alone. It's a **hybrid**: a knowledge layer that captures *what the author believes and knows* (facts, opinions, recurring arguments, terminology), a retrieval layer that grounds answers in real source material, and a style layer (fine-tuning) that captures *how* the author says things — sentence rhythm, vocabulary, rhetorical habits, tone.

### 3. Goals

- Generate responses to novel prompts that a reader familiar with the author's work would recognize as consistent with the author's voice and worldview.
- Ground factual claims and stated opinions in the actual books, with traceability back to source (so the system can say "consistent with chapter X" rather than fabricate a position).
- Avoid verbatim reproduction of substantial passages — the model should synthesize, not recite.
- Support two output modes: (a) **grounded mode** — cites and stays close to positions the author actually took, and (b) **extrapolative mode** — clearly-labeled generation of "how the author might respond" to something outside the books, styled but not attributed as an actual quote.
- Run on your own infrastructure where practical, given the GPU capacity already available.

### 4. Non-goals

- This is **not** a system for generating content to be passed off as the author's actual, verbatim writing or literal quotes.
- This is **not** a general-purpose chatbot — it should refuse or hedge on topics far outside the author's domain and known views rather than confabulate a persona-flavored answer.
- This does not aim to replace the author's own future writing — it's a study/ideation/interaction tool, not a ghostwriting-and-publish-under-their-name pipeline (see §6 for why that distinction matters legally).

### 5. Users & use cases

| User | Use case |
|---|---|
| You / the team | Interactive research and ideation — "how would he approach this new topic" |
| Students / readers | Q&A grounded in the author's actual body of work, with the author's phrasing style |
| The author (if living and involved) | A tool to draft in their own voice faster, or answer common questions at scale |
| Estate / institution (if author deceased or not directly involved) | Preserving and making accessible a body of thought, with clear provenance |

### 6. Legal, ethical & rights considerations — gating requirement

This section gates the project. Address it before building anything customer-facing.

- **Copyright in the source texts.** You need the right to use the 20 books as training data — either you hold the rights, the author has licensed them to you for this purpose, or this stays strictly internal/personal use. Training-data copyright law for LLMs is genuinely unsettled and jurisdiction-dependent; this isn't something I can give you a definitive legal read on — worth a real conversation with counsel if this ever becomes a commercial product, especially before any public release.
- **Right of publicity / identity rights.** A system that answers *as* a specific real, identifiable person is different from a system that summarizes their book. If the author is not you and hasn't given explicit written consent to be modeled this way, that's the single biggest risk in this project — bigger than the technical risk. If the author is deceased, rights typically pass to an estate; check with them.
- **Disclosure.** Any output surface (chat UI, API) should make clear the response is AI-generated in the style of the author, not an actual statement by them. This protects you and the author both.
- **Verbatim leakage.** Fine-tuning on a small corpus can cause the model to memorize and reproduce exact passages. The TRD below includes specific mitigations (dedup, decontamination checks, output filtering) — treat these as required, not optional.

### 7. Success metrics

- **Style fidelity**: blind A/B test — people who know the author's writing correctly attribute AI-generated passages to "sounds like him" at a target rate (e.g., 70%+) without being told which is which.
- **Factual/positional consistency**: sampled outputs cross-checked against the knowledge layer show no contradictions of established positions.
- **Non-reproduction**: automated overlap-detection (n-gram matching) against source texts shows no unlicensed verbatim spans over a set length (e.g., 12+ words) in generated output.
- **Groundedness**: in grounded mode, claims are traceable to a source chunk in the retrieval index at a target hit rate.

### 8. Phased scope

| Phase | Deliverable |
|---|---|
| 1 | Data pipeline: all 20 books extracted, cleaned, chunked, deduplicated |
| 2 | Knowledge layer: structured extraction of facts, opinions, recurring themes, terminology, stylometric profile |
| 3 | RAG MVP: retrieval-grounded chatbot over the knowledge layer + source chunks, using an existing strong LLM (no fine-tuning yet) — this alone may satisfy a lot of the "answer like him" ask for factual/grounded questions |
| 4 | Style fine-tune: LoRA/QLoRA fine-tune of an open-weight model on synthetically constructed style-transfer data |
| 5 | Persona orchestrator: combine retrieval + fine-tuned style model, with consistency-checking against the knowledge layer |
| 6 | Evaluation & hardening: leakage checks, blind style tests, refusal behavior for out-of-domain questions |
| 7 | Serving & integration |

Phase 3 is worth treating as a real milestone, not throwaway — you'll learn a lot about whether pure RAG is "good enough" before investing in the fine-tuning phase.

### 9. Open questions / assumptions to confirm

- Assumed: PDFs are born-digital (you mentioned text is copyable), so OCR is a **fallback** for scanned pages/images/diagrams, not the primary extraction path. TRD is written accordingly.
- Assumed: you hold or can obtain the necessary rights (§6) — this document doesn't proceed past Phase 2 without that being true.
- Not yet defined: is "grounded mode vs extrapolative mode" a hard toggle in the product, or should every response carry both a grounded core and a styled wrapper? (Recommend the latter — see TRD §5.)

---

## Part 2 — Technical Requirements Document (TRD)

### 1. Architecture overview

Seven stages, shown in the diagram above:

1. **Source** — the 20 PDFs
2. **Extraction & cleaning** — text pull + OCR fallback
3. **Knowledge layer** — structured facts, opinions, stylometric profile
4. **Retrieval index** (branch A) — vector DB for grounding
5. **Persona fine-tune** (branch B) — LoRA-tuned model for style
6. **Persona orchestrator** — merges grounded facts with styled generation
7. **Serving layer** — API / chat surface

### 2. Data ingestion pipeline

**Extraction:**
- Primary: direct text-layer extraction via `PyMuPDF` (fitz) or `pdfplumber` — since the PDFs already have a text layer, this is faster and more accurate than OCR and should be the default path for every page.
- Fallback: page-level OCR via `Tesseract` or `PaddleOCR` only for pages where text-layer extraction returns near-empty or garbled output (e.g., scanned inserts, image-based diagrams, handwritten marginalia). Flag which pages went through OCR so quality can be spot-checked separately — OCR error rates are much higher than native text extraction and you don't want typos silently entering the "author's voice."
- Layout-aware extraction matters for footnotes, sidebars, and running headers/footers — strip repeated headers/footers/page numbers (they'll otherwise pollute the corpus with junk tokens repeated hundreds of times).

**Cleaning:**
- Normalize whitespace, hyphenation across line breaks, quotation mark styles.
- De-duplicate near-identical passages (an author sometimes repeats an anecdote across books) — use MinHash/LSH for near-duplicate detection so the model doesn't over-weight repeated material.
- Tag every chunk with metadata: book title, chapter, approximate page, and a stable chunk ID — this ID is what the retrieval and knowledge layers reference later for traceability.

**Chunking:**
- Semantic chunking (not fixed-token windows) — split on paragraph/section boundaries where possible, target ~200–500 tokens per chunk, with slight overlap (~10%) to preserve context across chunk boundaries.

### 3. Knowledge layer (the structured "knowledge system")

This is the piece that keeps the system from either (a) hallucinating positions the author never held, or (b) being a pure text-retrieval bot.

Two components:

**a. Structured knowledge base**
For each book, use an LLM-assisted extraction pass (not a fine-tune — just a well-prompted strong model, e.g. Claude, run once over the corpus) to produce structured records:
- Stated opinions/positions ("author believes X because Y", with source chunk ID)
- Recurring themes and terminology (words/phrases the author uses in a specific, personal way)
- Anecdotes and examples the author returns to
- Contradictions or evolution of a view across books (useful — real authors change their minds over 20 books)

Store these as individual markdown files with YAML frontmatter (book, chapter, chunk ID, category, confidence) — one file per extracted "knowledge unit." This is deliberately similar in spirit to the Open Knowledge Format idea: durable, human-readable, diffable, and reviewable by you before it's trusted, rather than a black-box embedding.

**b. Stylometric profile**
Separately, extract a quantitative style fingerprint using NLP tooling (`spaCy`, `textstat`, custom scripts):
- Sentence length distribution, clause complexity
- Vocabulary richness / favorite words and phrases
- Rhetorical patterns (rhetorical questions, analogies, sentence openers)
- Punctuation habits, paragraph structure

This profile is used both to condition generation (as a style guide in the system prompt / fine-tune data) and as an evaluation signal (does generated text match the profile's distributions).

### 4. Retrieval layer (RAG)

- Embed all chunks with a strong open embedding model (e.g., `bge-large` or similar) into a vector DB — `Qdrant` or `Chroma`, both self-hostable on your existing infrastructure.
- Store chunk metadata (book, chapter, chunk ID) alongside vectors for citation.
- At query time: retrieve top-k relevant chunks + any matching structured knowledge units, and pass both into the generation step as grounding context.

### 5. Style/persona layer (fine-tuning)

**Base model:** an open-weight model you can run and fine-tune locally — e.g., Llama 3.3 70B or Qwen 2.5 72B (both fit comfortably across your GPU cluster at 4-bit for QLoRA). Start smaller (8B–14B class) to validate the pipeline before committing full compute to a 70B run.

**Training data construction** — this is the part that needs the most care, since you can't just fine-tune directly on the raw books (that's exactly what causes verbatim memorization):
- Generate **synthetic instruction/style pairs**: use a strong LLM to read a passage and produce a *paraphrased* continuation or a plausible question the passage answers, then pair (question → author-style paraphrase), not (question → verbatim excerpt).
- Generate **preference pairs** for DPO-style training: same question, one answer in flat/generic style, one in author style (using the stylometric profile as a target) — this teaches style contrast more directly than plain instruction tuning does.
- Explicitly exclude any pair where the "answer" is a near-verbatim match to source text (run the same n-gram overlap check used in evaluation, §7, against your own training set before you train on it).

**Method:** QLoRA (4-bit base + LoRA adapters) via `Axolotl` or Hugging Face `PEFT` + `bitsandbytes`. This is the right call given a small, high-value dataset — full fine-tuning on 20 books' worth of derived data would be both unnecessary and more prone to overfitting/leakage than a LoRA adapter.

**Hardware mapping:** your existing 17× RTX 3090 (24GB each, 408GB aggregate) is enough for QLoRA on up to 70B-class models with a multi-GPU setup (`DeepSpeed` ZeRO or `FSDP`), and comfortably enough for anything in the 8–34B range without needing multi-node complexity. This is the same class of infrastructure already running your JICA training jobs, so this can likely share the cluster's overnight training window rather than needing new hardware.

### 6. Persona orchestrator

Runtime logic that ties retrieval + fine-tune together for each incoming query:

1. Retrieve top-k chunks + knowledge units relevant to the query.
2. Classify the query: is this answerable from grounded material ("grounded mode"), or does it require extrapolation beyond the books ("extrapolative mode")?
3. Construct the prompt for the fine-tuned model: system instructions (persona + style guide from the stylometric profile) + retrieved grounding context + the query.
4. Generate with the fine-tuned model.
5. **Post-generation consistency check**: run the output back against the knowledge layer — flag if it contradicts a known stated position, and against the source corpus — flag/block if it reproduces a long verbatim span.
6. Attach mode label + source citations (in grounded mode) to the response before returning it.

### 7. Evaluation

- **Leakage/overlap check**: automated n-gram overlap scan of every generated output against the full source corpus; hard block above a threshold (e.g., any 12+ word exact match).
- **Style similarity**: compare generated text's stylometric profile (from §3b) against the author's target profile.
- **Blind human eval**: panel of people familiar with the author's work, shown mixed real/generated passages, asked to identify which is which and rate authenticity.
- **Groundedness**: sample grounded-mode responses, verify cited chunks actually support the claim made.

### 8. Serving

- Inference via `vLLM` (for throughput) or `Ollama` (for simplicity) hosting the fine-tuned model on your own GPUs.
- Thin API layer wrapping: retrieval call → orchestrator → generation → post-checks → response.
- Chat front-end (could reuse patterns from your other internal tools) with a visible "AI-generated, in the style of [author]" disclosure and mode indicator (grounded / extrapolative).

### 9. Tech stack summary

| Layer | Tooling |
|---|---|
| PDF extraction | PyMuPDF / pdfplumber |
| OCR fallback | Tesseract / PaddleOCR |
| Dedup | MinHash / LSH (`datasketch`) |
| Knowledge extraction | LLM-assisted structured extraction → markdown + YAML |
| Embeddings | bge-large or similar open embedding model |
| Vector DB | Qdrant or Chroma (self-hosted) |
| Fine-tuning | QLoRA via Axolotl or HF PEFT + bitsandbytes |
| Base model | Llama 3.3 / Qwen 2.5 (start 8–14B, scale to 70B) |
| Multi-GPU training | DeepSpeed ZeRO or FSDP across the RTX 3090 cluster |
| Serving | vLLM or Ollama |
| Style analysis | spaCy, textstat, custom stylometric scripts |

### 10. Milestone timeline (indicative)

| Milestone | Depends on |
|---|---|
| M1 — Data pipeline complete (all 20 books extracted, cleaned, chunked) | — |
| M2 — Knowledge layer v1 (structured facts + stylometric profile) | M1 |
| M3 — RAG MVP live | M2 |
| M4 — Fine-tune training data built + decontaminated | M2 |
| M5 — First LoRA fine-tune trained + evaluated (small model) | M4 |
| M6 — Persona orchestrator integrating retrieval + fine-tune | M3, M5 |
| M7 — Full evaluation pass (leakage, style, groundedness, blind panel) | M6 |
| M8 — Serving layer + disclosure UX | M7 |

### 11. Key risks

| Risk | Mitigation |
|---|---|
| Verbatim leakage / plagiarism | Synthetic paraphrase-based training data, n-gram decontamination, output-time overlap blocking |
| Consent/rights issue if author ≠ you | Gate at PRD §6 — resolve before Phase 3 |
| Small corpus → overfitting | LoRA not full fine-tune, DPO-style contrast pairs rather than raw next-token training on book text |
| Model asserts false positions as the author's | Knowledge-layer consistency check before returning any output |
| Users mistake output for real quotes | Mandatory disclosure + mode labeling in the UI |