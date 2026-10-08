"""Flask web application for the RAG pipeline.

Serves a U.S. Web Design System (USWDS)-styled single page that lets you ask
questions about the indexed PDF(s). The heavy work -- OCR, embedding, building
the FAISS index and loading the LLM -- happens once in a background thread at
startup; the page reports readiness and only enables the Ask button when the
pipeline is loaded.

Run locally:
    python app.py
Then open http://127.0.0.1:5000

Run in production (e.g. inside the Docker image on AWS App Runner):
    gunicorn --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:$PORT app:app
A single worker is used so the (memory-heavy) pipeline is built once; threads
handle concurrent requests. The build is kicked off at import time (below) so it
runs under gunicorn too, not only under `python app.py`.
"""

import os
import threading

from flask import Flask, jsonify, render_template, request

import rag_pipeline

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Pipeline state -- built once in a background thread so the page can load
# immediately and poll /api/status while the (slow, one-time) build runs.
# ---------------------------------------------------------------------------
_state = {
    "qa_chain": None,
    "status": "loading",   # loading | ready | error
    "message": "Starting up...",
    "pdf_paths": rag_pipeline.DEFAULT_PDF_PATHS,
    "backend": rag_pipeline.RAG_BACKEND,
}
_lock = threading.Lock()
# Guard so the build thread starts exactly once, whether launched by gunicorn
# (import time) or by `python app.py` (__main__).
_build_started = False
_build_started_lock = threading.Lock()


def _log(msg):
    """Route pipeline progress into the shared status message + stdout."""
    print(msg, flush=True)
    with _lock:
        _state["message"] = str(msg)


def _build():
    try:
        qa = rag_pipeline.build_pipeline(log=_log)
        with _lock:
            _state["qa_chain"] = qa
            _state["status"] = "ready"
            _state["message"] = "Pipeline ready."
    except Exception as e:  # noqa: BLE001
        with _lock:
            _state["status"] = "error"
            _state["message"] = f"Failed to build pipeline: {e}"
        print(f"[error] {e}", flush=True)


def start_build_once():
    """Kick off the one-time pipeline build in a background thread (idempotent).

    Called both at import time (so gunicorn triggers it) and from __main__.
    """
    global _build_started
    with _build_started_lock:
        if _build_started:
            return
        _build_started = True
    threading.Thread(target=_build, daemon=True).start()


@app.route("/")
def index():
    return render_template("index.html", pdf_paths=", ".join(_state["pdf_paths"]))


@app.route("/healthz")
def healthz():
    """Liveness probe for the load balancer / App Runner.

    Returns 200 as soon as the web process is up -- deliberately independent of
    the (slow, one-time) pipeline build so a long index build never marks the
    container unhealthy. Readiness of the index is reported by /api/status.
    """
    return jsonify(ok=True), 200


@app.route("/api/status")
def status():
    with _lock:
        return jsonify(
            status=_state["status"],
            message=_state["message"],
            backend=_state["backend"],
        )


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    with _lock:
        qa = _state["qa_chain"]
        st = _state["status"]

    if st != "ready" or qa is None:
        return jsonify(error="The document index is still being prepared. Please wait."), 503
        
    if not query:
        return jsonify(error="Please enter a question."), 400

    answer = rag_pipeline.answer_query(qa, query)
    return jsonify(answer=answer)


# Start the build when the module is imported (this is how gunicorn/App Runner
# triggers it). Harmless under `python app.py` too -- start_build_once() is
# idempotent, and __main__ below simply ensures it has started.
start_build_once()


if __name__ == "__main__":
    start_build_once()
    # Bind to all interfaces inside a container; default to localhost otherwise.
    # PORT is honored so the same entrypoint works locally and on App Runner.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    # use_reloader=False so the reloader doesn't spawn a second (duplicate) build.
    app.run(host=host, port=port, debug=False, use_reloader=False)
