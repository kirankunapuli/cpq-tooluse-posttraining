"""One place to load the two small models, so every script shares the download.

Both are CPU-sized on purpose. The gate has to answer in milliseconds on the request
path, and the retriever has to be re-indexable inside a deploy window, so a large
encoder would be the wrong choice even if a GPU were free.
"""
from __future__ import annotations

import functools
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"          # 384 dims, 33M params
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # 6 layer cross encoder


@functools.lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


@functools.lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, device="cpu", max_length=256)


def encode(texts, batch_size: int = 32):
    """Encode to unit vectors, which makes cosine similarity a dot product."""
    import numpy as np
    vecs = embedder().encode(list(texts), batch_size=batch_size,
                             normalize_embeddings=True, show_progress_bar=False,
                             convert_to_numpy=True)
    return np.asarray(vecs, dtype="float32")


def available() -> bool:
    """True when the full stack is importable. Scripts and tests use this to skip
    gracefully instead of crashing on a machine that only has the stdlib."""
    import importlib.util
    return all(importlib.util.find_spec(m) is not None
               for m in ("numpy", "sentence_transformers", "sklearn"))
