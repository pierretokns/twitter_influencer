#!/usr/bin/env python3
"""Model-family adapter metadata for retrieval and reranking benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AdapterKind = Literal[
    "single_vector",
    "cross_encoder",
    "late_interaction",
    "unified_embed_rerank",
    "gguf_reranker",
    "embedding_context_rerank",
    "unknown",
]


@dataclass(frozen=True)
class RetrievalModelAdapter:
    family: str
    kind: AdapterKind
    current_matrix_supported: bool
    notes: str
    query_prompt_name: str | None = None
    document_prompt_name: str | None = None
    task: str | None = None
    query_prefix: str = ""
    document_prefix: str = ""
    append_eos: str = ""

    def format_query(self, text: str) -> str:
        if self.family == "e2rank":
            task = "Given a web search query, retrieve relevant passages that answer the query"
            return f"Instruct: {task}\nQuery:{text}{self.append_eos}"
        return f"{self.query_prefix}{text}{self.append_eos}"

    def format_document(self, text: str) -> str:
        return f"{self.document_prefix}{text}{self.append_eos}"


def adapter_for_model(model_name: str) -> RetrievalModelAdapter:
    if model_name == "production_bge_m3_hybrid":
        return RetrievalModelAdapter(
            family="production_bge_m3_hybrid",
            kind="single_vector",
            current_matrix_supported=True,
            notes="Production hybrid dense+sparse sqlite-vec retrieval path.",
        )
    if model_name.startswith("Qwen/Qwen3-Embedding-"):
        return RetrievalModelAdapter(
            family="qwen3_embedding",
            kind="single_vector",
            current_matrix_supported=True,
            query_prompt_name="query",
            notes="Instruction-aware single-vector embedding; query prompt is required for fair scoring.",
        )
    if model_name.startswith("jinaai/jina-embeddings-v"):
        return RetrievalModelAdapter(
            family="jina_embeddings_v",
            kind="single_vector",
            current_matrix_supported=True,
            task="retrieval",
            query_prompt_name="query",
            document_prompt_name="passage",
            notes="Task-aware Jina embedding path; v4/v5 also have multi-vector or omni modes needing separate tests.",
        )
    if model_name.startswith("nomic-ai/nomic-embed-text-v2-moe"):
        return RetrievalModelAdapter(
            family="nomic_v2_moe",
            kind="single_vector",
            current_matrix_supported=True,
            query_prefix="search_query: ",
            document_prefix="search_document: ",
            notes="MoE embedding model with required task prefixes and Matryoshka truncation support.",
        )
    if model_name.startswith("Alibaba-NLP/E2Rank-"):
        return RetrievalModelAdapter(
            family="e2rank",
            kind="unified_embed_rerank",
            current_matrix_supported=True,
            append_eos="<|endoftext|>",
            notes="Embedding mode is supported here; listwise reranking needs a separate pseudo-query adapter.",
        )
    if "colbert" in model_name.lower() or "colqwen" in model_name.lower():
        return RetrievalModelAdapter(
            family="colbert_late_interaction",
            kind="late_interaction",
            current_matrix_supported=False,
            notes="Requires token/multi-vector embeddings and MaxSim/PLAID-style scoring, not cosine single-vector scoring.",
        )
    if model_name.endswith("-GGUF") or "GGUF" in model_name:
        return RetrievalModelAdapter(
            family="gguf",
            kind="gguf_reranker",
            current_matrix_supported=False,
            notes="Requires llama.cpp or model-specific GGUF embedding/ranking invocation, not CrossEncoder.",
        )
    return RetrievalModelAdapter(
        family="default_single_vector",
        kind="single_vector",
        current_matrix_supported=True,
        notes="Default SentenceTransformer-compatible single-vector path.",
    )


def format_texts_for_model(model_name: str, texts: list[str], is_query: bool) -> list[str]:
    adapter = adapter_for_model(model_name)
    formatter = adapter.format_query if is_query else adapter.format_document
    return [formatter(text) for text in texts]

