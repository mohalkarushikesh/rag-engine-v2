"""Pre-download the local (offline) models into the image at BUILD time.

The runtime container runs fully offline (HF_HUB_OFFLINE=1), so the weights for
the `local` backend must already be present on disk. This script fetches them
into HF_HOME during `docker build`; it therefore needs network access to the
HuggingFace Hub and must run where the Hub is reachable (CI / AWS CodeBuild /
a machine with internet) -- NOT on the offline runtime host.

Only the `local` backend needs these; the `bedrock` backend calls AWS APIs and
uses none of them. They are baked anyway so one image can serve either backend.
"""

import os

# This runs at build time, so force ONLINE here regardless of any inherited env.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from transformers import (  # noqa: E402  (env must be set before import)
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "distilbert-base-uncased")
QA_MODEL = os.environ.get("RAG_QA_MODEL", "deepset/roberta-base-squad2")


def _fetch(name, model_cls):
    print(f"[download_models] fetching {name} ...", flush=True)
    AutoTokenizer.from_pretrained(name)
    model_cls.from_pretrained(name)
    print(f"[download_models] done: {name}", flush=True)


if __name__ == "__main__":
    _fetch(EMBEDDING_MODEL, AutoModel)                 # embeddings backbone
    _fetch(QA_MODEL, AutoModelForQuestionAnswering)    # extractive reader
    print(f"[download_models] cache populated under HF_HOME={os.environ.get('HF_HOME')}", flush=True)