# End-to-End RAG App: PDFs → Answers

A small, self-contained **Retrieval-Augmented Generation (RAG)** app. You give it PDFs, it finds the
passages relevant to your question and answers *from those passages*.

Originally prototyped in a Jupyter notebook ([RAG.ipynb](RAG.ipynb)), it now ships as a reusable
Python module plus a web application:

```
PDF -> text (cached, +OCR fallback) -> chunks -> embeddings -> FAISS
                                                                  |
                                    hybrid retrieve (semantic + keyword)
                                                                  |
                              answerer (local reader | Bedrock LLM) -> Web UI
```

It runs two ways from **one codebase**, selected by the `RAG_BACKEND` env var:

- **`local`** (default) — **fully offline** on locally cached models. No HuggingFace Hub, no API
  keys, no network calls. This is what runs on your machine.
- **`bedrock`** — **AWS-native**. Amazon Bedrock supplies both the embeddings (Titan) and the
  generative answer. Used when the app is deployed on AWS; credentials come from an IAM role, so no
  keys are ever stored. See [Deploying to AWS](#deploying-to-aws).

Retrieval (FAISS + the keyword arm), PDF/OCR, and caching are shared — only the embedding model and
the "extractive reader vs. generative model" differ between backends.

---

## What it does

1. **Extract** text from PDFs (with an OCR fallback for scanned / text-as-outline PDFs). The
   extracted text is **cached**, so OCR runs only once per file (see [Caching](#caching--why-startup-is-fast)).
2. **Chunk** the text and wrap it as LangChain `Document` objects.
3. **Embed** each chunk into a vector with a local embedding model and **index** them in FAISS.
4. **Retrieve** candidate passages with a **hybrid retriever**: semantic similarity (FAISS) *and*
   keyword overlap. The keyword arm catches exact-term questions the weak embeddings miss.
5. **Read** the answer: an *extractive* QA model finds the best answer span inside the retrieved
   passages (scored in one batched pass). This is far faster on CPU than generating text.
6. **Serve** it through a web application with a clean, accessible interface.

---

## Performance

Measured on this repo's sample (`gk_ques_ans.pdf`, a 40-page scanned PDF), CPU-only (no GPU),
before vs. after the changes in this version:

| Metric                        | Before (generative)        | After (retriever + reader) |
|-------------------------------|----------------------------|----------------------------|
| **Per-question latency**      | ~50–75 s                   | **~2 s** (warm)            |
| **App startup (warm)**        | ~4–9 min (OCR every start) | **~8 s** (OCR cached)      |
| **OCR runs**                  | every startup / rebuild    | **once per file, ever**    |
| **Accuracy** (6-question set) | 4 / 6                      | **6 / 6**                  |

What changed, and why:

- **Generative LLM → extractive reader.** `microsoft/phi-1_5` (1.3B) generated tokens one at a time
  on CPU — the ~50–75 s wait. `deepset/roberta-base-squad2` instead *extracts* the answer span from
  the passages in a single forward pass (~2 s). (Instruction-tuned generators like
  `Qwen2.5-0.5B-Instruct` would also be fast, but their weights aren't in the local cache.)
- **Semantic-only → hybrid retrieval.** The `distilbert` embeddings are weak sentence encoders and
  missed exact-term questions ("What is Acrophobia?"). Adding a keyword arm fixed the misses (4/6 → 6/6).
- **OCR every startup → cached once.** Text extraction is now cached separately from the index, keyed
  on the PDF's content hash, so OCR never re-runs unless the file itself changes.

---

## Project structure

| File / folder                | Purpose                                                              |
|------------------------------|----------------------------------------------------------------------|
| `rag_pipeline.py`            | The RAG pipeline as a reusable module (+ a CLI). Converted from the notebook. |
| `app.py`                     | Flask web server: builds the pipeline once, serves the UI and a small JSON API. |
| `templates/index.html`       | The web page.                                                        |
| `static/css/styles.css`      | Styling — U.S. Web Design System (USWDS) color tokens, **Montserrat** font, light theme. |
| `static/js/app.js`           | Front-end: status polling + async question/answer.                   |
| `.rag_cache/`                | Cached extracted text (`text_*.json`) and FAISS index. Safe to delete. |
| `RAG.ipynb`                  | The original step-by-step notebook (kept for reference).             |
| `gk_ques_ans.pdf`            | Sample document used by default.                                     |
| `Dockerfile` / `.dockerignore` | Container image. Bakes the offline models; serves via gunicorn.    |
| `scripts/download_models.py` | Pre-downloads the local models into the image at build time.         |
| `infra/`                     | **Terraform** IaC: ECR + App Runner service + IAM roles (Bedrock).   |
| `deploy.sh`                  | One-command deploy: build → push to ECR → `terraform apply`.         |
| `.github/workflows/deploy.yml` | CI that builds and deploys to AWS (runner has internet for the build). |
| `.env.example`               | Documented runtime environment variables.                            |

---

## Models used (all local / offline)

| Role        | Model                              | Notes                                                        |
|-------------|------------------------------------|--------------------------------------------------------------|
| Embeddings  | `distilbert-base-uncased`          | Mean-pooled → 768-dim vectors for the semantic retriever     |
| Reader (QA) | `deepset/roberta-base-squad2`      | Extractive: finds the answer span in the retrieved passages  |
| OCR         | RapidOCR (`rapidocr-onnxruntime`)  | Self-contained ONNX models; used only when a PDF has no text |

> **Why these?** They're the capable models already present in the local HuggingFace cache with full
> weights. Override the reader with `RAG_QA_MODEL` if you have a better QA model cached.
>
> On the **`bedrock`** backend these are replaced by managed AWS models — Titan embeddings
> (`amazon.titan-embed-text-v2:0`) and a generative LLM (`anthropic.claude-3-haiku-20240307-v1:0` by
> default) — configurable via `RAG_BEDROCK_EMBED_MODEL` / `RAG_BEDROCK_CHAT_MODEL`.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to run

### Web application (recommended)

```bash
python app.py
```

Then open **http://127.0.0.1:5000** and ask questions.

- The one-time build (extract → embed → index → load the reader) runs in a background thread at
  startup. The page loads immediately and a status indicator shows **"Preparing index…"**; the
  **Get answer** button enables once it reads **"Index ready"**. With text + index cached, this is
  a few seconds.
- Index your own PDF(s) with an environment variable (comma-separated for multiple files):

  ```bash
  RAG_PDF_PATHS="my_document.pdf,another.pdf" python app.py            # macOS / Linux
  ```
  ```powershell
  $env:RAG_PDF_PATHS="my_document.pdf,another.pdf"; python app.py       # Windows PowerShell
  ```

#### API

| Endpoint        | Method | Body / Response                                              |
|-----------------|--------|--------------------------------------------------------------|
| `/api/status`   | GET    | `{"status": "loading｜ready｜error", "message": "..."}`       |
| `/api/ask`      | POST   | Request `{"query": "..."}` → Response `{"answer": "..."}`     |

```bash
curl -s -X POST http://127.0.0.1:5000/api/ask \
     -H "Content-Type: application/json" -d '{"query":"What is Acrophobia?"}'
# {"answer":"Fear of Heights"}
```

### Command line

```bash
python rag_pipeline.py "Who is the founder of Vaccinology?"   # one-shot
python rag_pipeline.py                                        # interactive REPL
python rag_pipeline.py extract                                # run OCR once & cache the text
```

---

## Deploying to AWS

The app is packaged to run **offline locally** and deploy **end-to-end on AWS** with no code changes —
you flip `RAG_BACKEND` to `bedrock` and the same image calls Amazon Bedrock instead of the local
models. Everything needed to deploy lives in this repo: a `Dockerfile`, Terraform in `infra/`, and a
`deploy.sh` wrapper.

### Architecture

```
          build (CI / laptop w/ internet)                 runtime (AWS)
  ┌─────────────────────────────────────┐        ┌──────────────────────────────┐
  │ docker build                         │        │ AWS App Runner (HTTPS, auto- │
  │  • pip install deps                  │  push  │ scaling, managed TLS)        │
  │  • bake HF models (offline weights)  │ ─────► │   gunicorn → Flask app       │
  │  → image in Amazon ECR               │        │   instance IAM role          │
  └─────────────────────────────────────┘        │        │  bedrock:InvokeModel │
                                                  │        ▼                     │
                                                  │   Amazon Bedrock             │
                                                  │   (Titan embed + Claude LLM) │
                                                  └──────────────────────────────┘
```

### Why these choices

| Decision | Choice | Why |
|---|---|---|
| **Compute** | **AWS App Runner** | Least infrastructure for a single containerized web service: it provisions HTTPS, load balancing, and autoscaling for you from just a container image. No VPC/ALB/cert wiring to maintain. Bedrock is a public API, so no VPC connector is needed. (ECS Fargate is the scale-up path if you later need VPC-private networking or fine-grained control.) |
| **IaC** | **Terraform** | Cloud-agnostic, readable HCL with no bootstrap step (unlike CDK, which needs a CloudFormation bootstrap stack). The whole footprint — ECR, IAM, App Runner, autoscaling — is a handful of declarative files you can `plan`/`apply`/`destroy` reproducibly. |
| **Models** | **Baked into the image** (local backend) | The runtime stays fully offline (`HF_HUB_OFFLINE=1`) — important because the corp network blocks huggingface.co. Weights are fetched **once at build time** (where the Hub is reachable) and shipped inside the image, so startup is deterministic with no runtime download. |
| **AI backend** | **Pluggable local + Bedrock** | You asked for it to work offline *and* deploy on AWS. One `RAG_BACKEND` switch satisfies both: `local` for your machine, `bedrock` for AWS. Bedrock keeps the deployed footprint light (no GPU, models are a managed service) and authenticates via an IAM role — no API keys in the image or config. |

### Prerequisites (on the machine that runs the deploy — CI or a non-offline box)

- Docker, AWS CLI v2, Terraform ≥ 1.5
- AWS credentials allowed to create ECR / IAM / App Runner
- Internet access (the image build downloads the HF models to bake in)
- **Bedrock model access enabled** in your region for the embed + chat models
  (Bedrock console → *Model access*). Defaults: `amazon.titan-embed-text-v2:0` and
  `anthropic.claude-3-haiku-20240307-v1:0`.

### Deploy

```bash
# From the repo root, on a machine with Docker + AWS CLI + Terraform:
AWS_REGION=us-east-1 IMAGE_TAG=$(git rev-parse --short HEAD) ./deploy.sh
```

`deploy.sh` runs the full flow: `terraform init` → create the ECR repo → build & push the image →
`terraform apply` the App Runner service. It prints the public **service URL** at the end. Use a
**unique `IMAGE_TAG` per deploy** (e.g. the git sha) so App Runner picks up the new image.

Prefer CI? `.github/workflows/deploy.yml` does the same on a GitHub runner (which has internet for
the build). Set the repo variable `AWS_ROLE_ARN` (OIDC, recommended) or
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` secrets, then run the **Deploy RAG app to AWS** workflow.

Tear everything down with:

```bash
terraform -chdir=infra destroy
```

### Configuration

All knobs are Terraform variables (see `infra/variables.tf` / `infra/terraform.tfvars.example`):
region, CPU/memory, autoscaling bounds, the Bedrock model ids, and `rag_backend` (set it to `local`
to run the offline backend on App Runner instead — the models are baked into the image either way).

### Testing the Bedrock backend locally

You don't need AWS to develop, but you can exercise the Bedrock path locally if you have AWS
credentials and model access:

```powershell
$env:RAG_BACKEND="bedrock"; $env:AWS_REGION="us-east-1"; python app.py   # Windows PowerShell
```
```bash
RAG_BACKEND=bedrock AWS_REGION=us-east-1 python app.py                    # macOS / Linux
```

Credentials are read from the standard AWS chain (`aws configure`, `AWS_PROFILE`, or env keys).

---

## Caching — why startup is fast

Two independent on-disk caches under `.rag_cache/` remove the repeated cost:

1. **Extracted text** (`text_<hash>.json`) — keyed on the **PDF's content hash**. OCR of a scanned
   PDF is the slowest step, and the pages never change unless the file does, so this runs **exactly
   once**. Re-chunking, changing the embedding model, or clearing the index will **not** re-run OCR.
   Because the key is content (not mtime), the sidecar is **portable**: run `python rag_pipeline.py
   extract` once and **commit** `.rag_cache/text_*.json`, and OCR won't run on any other machine or CI.
2. **FAISS index** — keyed on the files *and* the embedding/chunk settings, since changing those
   genuinely requires re-embedding (but never re-OCR).

Override the cache location with `RAG_CACHE_DIR`. Delete `.rag_cache/` to rebuild from scratch.

---

## OCR fallback

Some PDFs contain **no real text** — they're scanned images, or the text was converted to
outlines/curves. `pypdf` extracts nothing from those.

1. **Fast path** — `pypdf.extract_text()` for normal, text-based PDFs (instant).
2. **OCR fallback** — if a file yields no text, each page is rendered to an image with PyMuPDF and
   read with RapidOCR. Slow (a few minutes on CPU) but only ever runs once thanks to the text cache.

---

## Offline notes

- `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` — load only cached models, set at the top of
  `rag_pipeline.py`, *before* `transformers` is imported.
- RapidOCR ships its models inside the wheel, so OCR works without any download.
- The CSS is vendored locally; only the Montserrat webfont is fetched when online (with a fallback).

---

## Tuning & limitations

- **Retrieval knobs:** `RAG_SEMANTIC_K` (default 5) and `RAG_KEYWORD_K` (default 5) control how many
  candidate chunks each arm of the hybrid retriever contributes. Higher = better recall but more
  reader compute; too high pulls in distractors.
- **Extractive answers** are spans copied verbatim from the document — ideal for factual lookups,
  not for summaries or multi-fact synthesis.
- **Model overrides:** `RAG_QA_MODEL` swaps the reader. GPU is used automatically if available
  (`torch.cuda.is_available()`); otherwise CPU with all cores.

---

## Roadmap

See [Todo.md](Todo.md) — next up: multi-document retrieval and live source retrieval (showing which
passages an answer came from). The per-file text cache already makes adding a document OCR only that
new file.


## DashBoard

![DashBoard](/assets/dashboard.png)
