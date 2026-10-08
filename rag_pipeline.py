
"""End-to-End RAG pipeline: PDFs -> Answers.

This is the .py conversion of RAG.ipynb. The notebook built the pipeline one
cell at a time so each stage could be inspected; here the same stages are
grouped into reusable functions so a web server (see app.py) can build the
pipeline once at startup and answer many questions:

    PDF -> text (+ OCR fallback) -> chunks -> embeddings -> FAISS -> retriever -> QA

Everything runs fully offline on locally cached models -- no HuggingFace Hub,
no API keys, no network calls.
"""

# ---------------------------------------------------------------------------
# Step 1 -- Force offline mode & quiet warnings
#
# This machine can't reach the internet, so everything runs fully offline from
# the local cache. These env vars MUST be set *before* transformers is imported,
# so they live at the very top of the module.
# ---------------------------------------------------------------------------
import os

os.environ["HF_HUB_OFFLINE"] = "1"          # never call out to the HuggingFace Hub
os.environ["TRANSFORMERS_OFFLINE"] = "1"    # load only locally cached models
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # (harmless here; kept from the notebook)
# transformers logs generation/tokenizer notices via its OWN logger, not the
# warnings module, so silence it here (set before transformers is imported).
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Step 2 -- Imports (grouped by the role each plays in the pipeline)
# ---------------------------------------------------------------------------
# Ingestion: read PDFs, wrap as Documents, split into chunks
from pypdf import PdfReader
import pymupdf          # render pages to images for the OCR fallback
import numpy as np      # image arrays for OCR
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vectorisation & search: embed text and store it in a FAISS index
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Answering: the extractive reader that finds the answer span in the passages
import transformers

transformers.logging.set_verbosity_error()  # belt-and-suspenders on the env var above
# Extractive reader. transformers 5.x removed the "question-answering" pipeline
# task, so we drive the model directly (tokenize -> logits -> best span).
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# Defaults mirror the notebook. PDF path(s) can be overridden by the caller /
# the RAG_PDF_PATHS env var (comma-separated) so the web app is configurable.
DEFAULT_PDF_PATHS = [
    p.strip()
    for p in os.environ.get("RAG_PDF_PATHS", "gk_ques_ans.pdf").split(",")
    if p.strip()
]
EMBEDDING_MODEL = "distilbert-base-uncased"  # locally cached; loaded offline
# Extractive reader (locally cached, full weights). This is a "retriever + reader"
# RAG: instead of generating text token-by-token (slow on CPU), the reader finds
# the answer span inside the retrieved passages -- ~1-2s per answer on CPU and
# very accurate for factual lookups. Override with RAG_QA_MODEL.
QA_MODEL = os.environ.get("RAG_QA_MODEL", "deepset/roberta-base-squad2")

# ---------------------------------------------------------------------------
# Backend selection -- the same pipeline runs two ways, chosen at runtime:
#
#   * "local"   (default) -- fully OFFLINE. HuggingFace embeddings + an
#                 extractive reader, no network, no API keys. This is what runs
#                 on your machine and inside the Docker image with baked models.
#   * "bedrock"           -- AWS-native. Amazon Bedrock supplies BOTH the
#                 embeddings (Titan) and a generative model for the answer. Used
#                 when the app is deployed on AWS; credentials come from the
#                 task/instance IAM role (never stored in the image).
#
# Retrieval (FAISS + the hybrid keyword arm) and all the PDF/OCR/caching code are
# shared -- only the embedding model and the "reader vs. generator" differ.
# ---------------------------------------------------------------------------
RAG_BACKEND = os.environ.get("RAG_BACKEND", "local").strip().lower()

# Bedrock model ids (only used when RAG_BACKEND=bedrock). Both are overridable so
# you can pick whatever models are enabled in your account/region. Defaults are
# broadly-available, low-cost models; enable them under Bedrock > Model access.
BEDROCK_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
BEDROCK_EMBED_MODEL = os.environ.get("RAG_BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
BEDROCK_CHAT_MODEL = os.environ.get("RAG_BEDROCK_CHAT_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")


def _effective_embedding_id():
    """Identity of the embedding model actually in use, for cache keying.

    Local and Bedrock embeddings produce different vectors (and dimensions), so
    their FAISS indexes must not collide in the on-disk cache.
    """
    return f"bedrock:{BEDROCK_EMBED_MODEL}" if RAG_BACKEND == "bedrock" else EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# Step 4 -- PDF -> text (with OCR fallback)
# ---------------------------------------------------------------------------
def ocr_pdf(path, dpi=200, log=print):
    """OCR every page of a PDF that has no extractable text (scanned / text-as-outlines)."""
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    doc = pymupdf.open(path)
    texts = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        result, _ = engine(img)
        page_text = "\n".join(line[1] for line in (result or []))
        if page_text.strip():
            texts.append(page_text)
        log(f"  OCR page {i + 1}/{doc.page_count}: {len(page_text)} chars")
    return texts


def extract_texts(pdf_paths, log=print):
    """Read every page of every PDF and collect the text.

    Two-tier strategy per file:
      1. Fast path -- pypdf.extract_text() for normal, text-based PDFs (instant).
      2. OCR fallback -- if a file yields no text, render each page to an image
         with PyMuPDF and read it with RapidOCR (self-contained, offline).
    """
    pdf_texts = []
    for path in pdf_paths:
        reader = PdfReader(path)
        extracted = [p.extract_text() for p in reader.pages]
        extracted = [t for t in extracted if t and t.strip()]

        if extracted:
            pdf_texts.extend(extracted)                    # fast path: real text found
        else:
            log(f"No embedded text in {path} -> running OCR (this can take a while)...")
            pdf_texts.extend(ocr_pdf(path, log=log))       # fallback: scanned / outlined PDF

    log(f"Extracted {len(pdf_texts)} page(s) of text.")

    if not pdf_texts:
        raise RuntimeError(
            "Still no text after OCR. The PDF may be blank, corrupt, or in a language the OCR "
            "model doesn't cover."
        )
    return pdf_texts


# ---------------------------------------------------------------------------
# Step 5 -- text -> chunks
# ---------------------------------------------------------------------------
def split_into_chunks(pdf_texts, chunk_size=1000, chunk_overlap=200, log=print):
    """Wrap each page as a Document, then split with an overlap so a sentence
    spanning a boundary still appears intact in one chunk."""
    docs = [Document(page_content=t) for t in pdf_texts]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    documents = splitter.split_documents(docs)
    log(f"Split {len(docs)} page-document(s) into {len(documents)} chunks.")
    return documents


# ---------------------------------------------------------------------------
# Steps 6-8 -- embeddings, FAISS index, and the retriever + reader QA
# ---------------------------------------------------------------------------
def load_embeddings():
    """Step 6 -- the embedding model for the active backend.

    local   -> HuggingFaceEmbeddings (offline, from the baked/cached weights).
    bedrock -> BedrockEmbeddings (Titan); imported lazily so the offline path
               never requires boto3 / langchain-aws to be installed.
    """
    if RAG_BACKEND == "bedrock":
        from langchain_aws import BedrockEmbeddings  # lazy: AWS-only dependency

        return BedrockEmbeddings(model_id=BEDROCK_EMBED_MODEL, region_name=BEDROCK_REGION)
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(documents, embedding_model, log=print):
    """Step 7 -- FAISS index over the chunk vectors."""
    vector_store = FAISS.from_documents(documents, embedding_model)
    log(f"FAISS index built over {vector_store.index.ntotal} vectors.")
    return vector_store


# Candidate chunks fed to the reader. We gather them with a HYBRID retriever:
# a few by semantic similarity (FAISS) plus a few by keyword overlap. The weak
# distilbert embeddings miss exact-term questions ("What is Acrophobia?"); the
# keyword arm catches those. Each arm is small so the reader stays fast.
SEMANTIC_K = int(os.environ.get("RAG_SEMANTIC_K", "5"))
KEYWORD_K = int(os.environ.get("RAG_KEYWORD_K", "5"))

# Very common words that shouldn't drive keyword retrieval.
_STOPWORDS = {
    "the", "a", "an", "of", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "and", "or", "by", "with", "as", "what", "which", "who", "whom",
    "whose", "where", "when", "why", "how", "that", "this", "these", "those",
    "it", "its", "from", "into", "known", "name", "many", "does", "do", "did",
    "has", "have", "had", "be", "been", "being", "there", "their", "list",
}


def _keywords(text):
    """Lower-cased content words in `text` (drops stopwords and 1-char tokens)."""
    import re

    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 1 and w not in _STOPWORDS
    }


class ExtractiveReader:
    """Runs an extractive QA model over the retrieved passages and returns the
    best answer span. Replaces the removed 'question-answering' pipeline.

    All passages are scored in ONE batched forward pass, so latency stays flat
    as the number of retrieved chunks (k) grows -- letting us keep k high for
    recall without paying per-chunk.
    """

    def __init__(self, model, tokenizer, torch_device, max_answer_len=40):
        self.model = model
        self.tok = tokenizer
        self.torch_device = torch_device  # "cpu" or "cuda"
        self.max_answer_len = max_answer_len

    def best_answer(self, question, contexts):
        """Return {"answer", "score"} for the best span across all contexts."""
        import torch

        if not contexts:
            return {"answer": "", "score": 0.0}

        enc = self.tok(
            [question] * len(contexts),
            list(contexts),
            return_tensors="pt",
            truncation="only_second",  # never truncate the question, only context
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            out = self.model(**{k: v.to(self.torch_device) for k, v in enc.items()})

        starts = out.start_logits.detach().cpu()
        ends = out.end_logits.detach().cpu()

        best = None  # (score, answer)
        for b in range(len(contexts)):
            seq_ids = enc.sequence_ids(b)  # None=special/pad, 0=question, 1=context
            s = starts[b].clone()
            e = ends[b].clone()
            for i, sid in enumerate(seq_ids):
                if sid != 1:  # only allow spans inside this passage
                    s[i] = float("-inf")
                    e[i] = float("-inf")

            start_idx = int(torch.argmax(s))
            # End within a short window after start -> a valid, bounded span.
            window = e[start_idx:start_idx + self.max_answer_len]
            end_idx = start_idx + int(torch.argmax(window))
            score = float(s[start_idx] + e[end_idx])

            if best is None or score > best[0]:
                ids = enc["input_ids"][b][start_idx:end_idx + 1]
                answer = self.tok.decode(ids, skip_special_tokens=True).strip()
                best = (score, answer)

        return {"answer": best[1], "score": best[0]}


class HybridRetriever:
    """Gathers candidate chunks two ways and unions them (de-duped by text).

    Shared by BOTH backends so retrieval behaves identically whether the answer
    is then extracted (local) or generated (bedrock):
      * semantic arm -- FAISS similarity over the embedding vectors.
      * keyword arm  -- content-word overlap; catches exact-term questions the
        embeddings miss.
    """

    def __init__(self, vector_store, all_docs):
        self.vector_store = vector_store
        self._docs = all_docs
        # Precompute each chunk's keyword set once for the keyword arm.
        self._doc_keywords = [_keywords(d.page_content) for d in all_docs]

    def _semantic(self, query):
        return self.vector_store.similarity_search(query, k=SEMANTIC_K)

    def _keyword(self, query):
        qk = _keywords(query)
        if not qk:
            return []
        scored = [
            (len(qk & kws), i) for i, kws in enumerate(self._doc_keywords)
        ]
        scored.sort(reverse=True)
        return [self._docs[i] for score, i in scored[:KEYWORD_K] if score > 0]

    def contexts(self, query):
        """Ordered, de-duplicated passage texts for `query`."""
        seen, out = set(), []
        for doc in self._semantic(query) + self._keyword(query):
            text = doc.page_content
            if text not in seen:
                seen.add(text)
                out.append(text)
        return out


class RetrieverReaderQA:
    """LOCAL backend: hybrid retriever + extractive reader.

    Kept API-compatible with the old LangChain chain: call ``.invoke({"query": q})``
    and read ``["result"]``. Returns the best answer span across the retrieved
    passages (scored in a single batched pass).
    """

    def __init__(self, retriever, reader, log=print):
        self.retriever = retriever
        self.reader = reader
        self._log = log

    def invoke(self, inputs):
        query = inputs["query"]
        contexts = self.retriever.contexts(query)
        return {"result": self.reader.best_answer(query, contexts)["answer"].strip()}


# Prompt for the Bedrock generative backend. The model must answer ONLY from the
# retrieved passages so the system stays a grounded RAG (no hallucinated facts).
_BEDROCK_SYSTEM = (
    "You are a precise question-answering assistant. Answer the user's question "
    "using ONLY the provided context passages. If the answer is not contained in "
    "the context, reply exactly: 'I could not find the answer in the document.' "
    "Keep the answer concise and factual; do not invent information."
)


class BedrockGenerativeQA:
    """BEDROCK backend: hybrid retriever + a Bedrock generative model.

    Same ``.invoke({"query": q}) -> {"result": ...}`` contract as the local path,
    so app.py / the CLI don't care which backend is active. The retrieved
    passages are stuffed into the prompt and the model generates a grounded
    answer. ChatBedrock is imported lazily (AWS-only dependency).
    """

    def __init__(self, retriever, log=print):
        from langchain_aws import ChatBedrock  # lazy: AWS-only dependency

        self.retriever = retriever
        self._log = log
        self.llm = ChatBedrock(
            model_id=BEDROCK_CHAT_MODEL,
            region_name=BEDROCK_REGION,
            model_kwargs={"temperature": 0, "max_tokens": 512},
        )

    def invoke(self, inputs):
        query = inputs["query"]
        contexts = self.retriever.contexts(query)
        if not contexts:
            return {"result": "I could not find the answer in the document."}

        context_block = "\n\n---\n\n".join(
            f"[Passage {i + 1}]\n{c}" for i, c in enumerate(contexts)
        )
        messages = [
            ("system", _BEDROCK_SYSTEM),
            ("human", f"Context passages:\n{context_block}\n\nQuestion: {query}"),
        ]
        response = self.llm.invoke(messages)
        text = getattr(response, "content", str(response))
        return {"result": (text or "").strip()}


def build_chain(vector_store, log=print):
    """Step 8 -- wire the hybrid retriever to the active backend's answerer.

    local   -> extractive reader (finds the answer span; fast, offline, CPU).
    bedrock -> Bedrock generative model (generates a grounded answer via AWS).
    """
    # All chunks, for the keyword arm of the hybrid retriever.
    all_docs = list(vector_store.docstore._dict.values())
    retriever = HybridRetriever(vector_store, all_docs)

    if RAG_BACKEND == "bedrock":
        log(f"Answerer: Amazon Bedrock (chat={BEDROCK_CHAT_MODEL}, "
            f"region={BEDROCK_REGION}, semantic_k={SEMANTIC_K}, keyword_k={KEYWORD_K})")
        qa = BedrockGenerativeQA(retriever, log=log)
        log("QA chain ready.")
        return qa

    # --- local extractive reader ---
    import torch

    # Use all logical cores (torch defaults to about half of them).
    try:
        torch.set_num_threads(os.cpu_count() or torch.get_num_threads())
    except Exception:  # noqa: BLE001 -- best-effort tuning
        pass

    # Use the GPU automatically if one is present; otherwise CPU.
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Reader device: {torch_device.upper()} "
        f"(threads={torch.get_num_threads()}, model={QA_MODEL}, "
        f"semantic_k={SEMANTIC_K}, keyword_k={KEYWORD_K})")

    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
    model.to(torch_device)
    model.eval()
    reader = ExtractiveReader(model, tokenizer, torch_device)

    log("QA chain ready.")
    return RetrieverReaderQA(retriever, reader, log=log)


def build_qa_chain(documents, log=print):
    """Embed the chunks, build the FAISS index, and wire up the retriever+reader.

    Kept as a convenience (used by the notebook parity path); the web app goes
    through build_pipeline(), which adds on-disk caching of the FAISS index.
    """
    embedding_model = load_embeddings()
    vector_store = build_vector_store(documents, embedding_model, log=log)
    return build_chain(vector_store, log=log)


# ---------------------------------------------------------------------------
# Step 9 -- ask a question
# ---------------------------------------------------------------------------
def clean(text):
    """Tidy the raw answer before returning it.

    For the local extractive reader, keep just the first paragraph (older small
    generators could ramble past the first blank line). For the Bedrock backend
    the answer is already a complete, grounded response -- return it whole so
    multi-paragraph answers aren't truncated.
    """
    text = (text or "").strip()
    if RAG_BACKEND == "bedrock":
        return text
    return text.split("\n\n")[0].strip()


def answer_query(qa_chain, query):
    """Run the full pipeline for one question and return a cleaned answer string."""
    if not query or not query.strip():
        return "Please enter a valid query."
    try:
        # .invoke() is the current API (the old .run() is deprecated); returns a dict.
        response = clean(qa_chain.invoke({"query": query})["result"])
        return response if response and response.strip() else "No answer found."
    except Exception as e:  # noqa: BLE001 -- surface any pipeline error to the caller
        return f"Error processing the query: {e}"


# ---------------------------------------------------------------------------
# On-disk caches
#
# Startup has two expensive, deterministic steps: OCR (minutes for a scanned
# PDF) and embedding every chunk into the FAISS index. We cache BOTH, keyed so
# each only re-runs when its real inputs change:
#
#   * Extracted text -- keyed on the PDF files only (path + mtime + size). The
#     scanned pages don't change unless the file does, so OCR runs ONCE and is
#     reused even if you re-chunk, change the embedding model, or clear the
#     index. This is what keeps OCR from re-running.
#   * FAISS index -- keyed on the files AND the embedding/chunk settings, since
#     changing those genuinely requires re-embedding (but never re-OCR).
# ---------------------------------------------------------------------------
CACHE_DIR = os.environ.get("RAG_CACHE_DIR", ".rag_cache")


def _files_signature(pdf_paths):
    """Stable id for the source files, based on their CONTENT; None if missing.

    Hashing the bytes (not mtime) makes the cache portable: commit the sidecar
    and it still matches after a `git checkout`, which rewrites mtimes. The PDFs
    are small so hashing is instant.
    """
    import hashlib

    h = hashlib.sha1()
    for p in pdf_paths:
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except OSError:
            return None
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _source_signature(pdf_paths):
    """Id for the files + embedding/chunk settings (the FAISS index key)."""
    import hashlib

    files = _files_signature(pdf_paths)
    if files is None:
        return None
    # Key on the EFFECTIVE embedding id so local and Bedrock indexes (different
    # vectors/dimensions) never collide in the cache.
    return hashlib.sha1(
        f"{files}|emb={_effective_embedding_id()};chunk=1000/200".encode("utf-8")
    ).hexdigest()[:16]


def get_texts(pdf_paths, log=print):
    """Extracted page text for the PDFs, running OCR only on a cache miss.

    The result is cached to a JSON sidecar keyed on the files alone, so OCR for
    a scanned PDF happens exactly once. Delete the sidecar (or the PDF changes)
    to force a re-extract.
    """
    import json

    files_sig = _files_signature(pdf_paths)
    cache_file = (
        os.path.join(CACHE_DIR, f"text_{files_sig}.json") if files_sig else None
    )

    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                texts = json.load(f)
            log(f"Loaded extracted text from {cache_file} "
                f"({len(texts)} page(s); skipping OCR).")
            return texts
        except Exception as e:  # noqa: BLE001 -- corrupt cache: fall back to extract
            log(f"Could not read text cache ({e}); re-extracting.")

    texts = extract_texts(pdf_paths, log=log)  # fast path or OCR fallback

    if cache_file:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(texts, f, ensure_ascii=False)
            log(f"Cached extracted text to {cache_file} (OCR won't re-run).")
        except Exception as e:  # noqa: BLE001 -- caching is best-effort
            log(f"Could not write text cache ({e}); continuing.")

    return texts


def build_pipeline(pdf_paths=None, log=print, use_cache=True):
    """PDF path(s) -> ready-to-query QA chain, with on-disk FAISS caching.

    First run: Step 4 (extract/OCR) -> Step 5 (chunk) -> Step 7 (embed/index),
    then the index is saved under CACHE_DIR. Later runs skip straight to loading
    that saved index -- no OCR, no re-embedding -- so startup is fast.
    """
    pdf_paths = pdf_paths or DEFAULT_PDF_PATHS
    log(f"Building RAG pipeline from: {', '.join(pdf_paths)}")

    embedding_model = load_embeddings()
    vector_store = None

    signature = _source_signature(pdf_paths) if use_cache else None
    index_dir = os.path.join(CACHE_DIR, signature) if signature else None

    # Fast path: reload a previously built index for these exact sources.
    if index_dir and os.path.isdir(index_dir):
        try:
            log(f"Loading cached index from {index_dir} (skipping OCR + embedding)...")
            vector_store = FAISS.load_local(
                index_dir,
                embedding_model,
                allow_dangerous_deserialization=True,  # our own local, trusted cache
            )
            log(f"Loaded cached index with {vector_store.index.ntotal} vectors.")
        except Exception as e:  # noqa: BLE001 -- corrupt/old cache: fall back to rebuild
            log(f"Could not load cache ({e}); rebuilding.")
            vector_store = None

    # Slow path: extract -> chunk -> embed, then persist for next time.
    # get_texts() reuses cached OCR output, so rebuilding the index here (e.g.
    # after a model/chunk change) does NOT re-run OCR.
    if vector_store is None:
        pdf_texts = get_texts(pdf_paths, log=log)
        documents = split_into_chunks(pdf_texts, log=log)
        vector_store = build_vector_store(documents, embedding_model, log=log)
        if index_dir:
            try:
                os.makedirs(index_dir, exist_ok=True)
                vector_store.save_local(index_dir)
                log(f"Cached index to {index_dir} for faster future startups.")
            except Exception as e:  # noqa: BLE001 -- caching is best-effort
                log(f"Could not write cache ({e}); continuing without it.")

    return build_chain(vector_store, log=log)


# ---------------------------------------------------------------------------
# CLI entry point -- mirrors Step 9: build the pipeline and answer questions.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # `python rag_pipeline.py extract` -- run OCR once and populate the text
    # cache (then commit .rag_cache/text_*.json so OCR never runs again).
    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        texts = get_texts(DEFAULT_PDF_PATHS)
        print(f"Extracted + cached {len(texts)} page(s). "
              f"Commit the .rag_cache/text_*.json sidecar to skip OCR everywhere.")
        sys.exit(0)

    qa = build_pipeline()

    if len(sys.argv) > 1:
        # One-shot: python rag_pipeline.py "your question"
        question = " ".join(sys.argv[1:])
        print("Q:", question)
        print("A:", answer_query(qa, question))
    else:
        # Interactive REPL
        print("\nRAG ready. Type a question (blank line or Ctrl-C to quit).\n")
        try:
            while True:
                q = input("Q: ").strip()
                if not q:
                    break
                print("A:", answer_query(qa, q), "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")