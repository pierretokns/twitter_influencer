#!/usr/bin/env python3
"""Check whether configured Hugging Face model repos are accessible."""

from __future__ import annotations

from huggingface_hub import HfApi


REPOS = [
    "google/embeddinggemma-300m",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "nvidia/llama-nemotron-embed-1b-v2",
    "nvidia/NV-Embed-v2",
    "ibm-granite/granite-embedding-97m-multilingual-r2",
    "ibm-granite/granite-embedding-311m-multilingual-r2",
    "ibm-granite/granite-embedding-small-english-r2",
    "ibm-granite/granite-embedding-english-r2",
    "ibm-granite/granite-embedding-reranker-english-r2",
    "jinaai/jina-embeddings-v4",
    "jinaai/jina-embeddings-v4-text-code-GGUF",
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-0.6B-GGUF",
    "perplexity-ai/pplx-embed-v1-0.6b",
    "perplexity-ai/pplx-embed-context-v1-0.6b",
    "perplexity-ai/pplx-embed-v1-4b",
    "perplexity-ai/pplx-embed-context-v1-4b",
    "Qwen/Qwen3-Reranker-0.6B",
    "mixedbread-ai/mxbai-rerank-base-v2",
    "zeroentropy/zembed-1-embedding",
    "zeroentropy/zerank-2-reranker",
]


def main() -> int:
    api = HfApi()
    for repo in REPOS:
        try:
            info = api.model_info(repo, files_metadata=False)
            print(
                "OK\t{}\tgated={}\tprivate={}".format(
                    repo,
                    getattr(info, "gated", None),
                    getattr(info, "private", None),
                )
            )
        except Exception as exc:
            message = str(exc).splitlines()[0] if str(exc) else repr(exc)
            print(f"FAIL\t{repo}\t{type(exc).__name__}: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
