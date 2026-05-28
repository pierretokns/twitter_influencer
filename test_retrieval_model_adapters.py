#!/usr/bin/env python3
"""Regression checks for retrieval benchmark model-family adapters."""

from tools.retrieval_model_adapters import adapter_for_model, format_texts_for_model
from tools.retrieval_pipeline_matrix_bench import E2RANK_LISTWISE_RERANKER, build_e2rank_listwise_prompt


def test_qwen_query_prompt_metadata() -> None:
    adapter = adapter_for_model("Qwen/Qwen3-Embedding-0.6B")
    assert adapter.family == "qwen3_embedding"
    assert adapter.current_matrix_supported
    assert adapter.query_prompt_name == "query"


def test_jina_task_metadata() -> None:
    adapter = adapter_for_model("jinaai/jina-embeddings-v5-text-small-retrieval")
    assert adapter.family == "jina_embeddings_v"
    assert adapter.current_matrix_supported
    assert adapter.task == "retrieval"
    assert adapter.query_prompt_name == "query"
    assert adapter.document_prompt_name == "passage"


def test_nomic_task_prefixes() -> None:
    assert format_texts_for_model(
        "nomic-ai/nomic-embed-text-v2-moe",
        ["finance AI governance"],
        is_query=True,
    ) == ["search_query: finance AI governance"]
    assert format_texts_for_model(
        "nomic-ai/nomic-embed-text-v2-moe",
        ["J.P. Morgan model risk controls"],
        is_query=False,
    ) == ["search_document: J.P. Morgan model risk controls"]


def test_e2rank_instruction_and_eos() -> None:
    formatted = format_texts_for_model(
        "Alibaba-NLP/E2Rank-0.6B",
        ["hedge fund AI research workflow"],
        is_query=True,
    )[0]
    assert formatted.startswith("Instruct: Given a web search query")
    assert "\nQuery:hedge fund AI research workflow" in formatted
    assert formatted.endswith("<|endoftext|>")


def test_e2rank_listwise_prompt_shape() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking=False):
            assert tokenize is False
            assert add_generation_prompt is True
            assert enable_thinking is False
            return messages[0]["content"] + "\n<assistant>"

    prompt = build_e2rank_listwise_prompt(
        FakeTokenizer(),
        "finance AI workflows",
        [
            {"id": "1", "type": "web", "source": "source", "title": "J.P. Morgan AI controls", "text": "model risk", "url": "https://example.com/1"},
            {"id": "2", "type": "web", "source": "source", "title": "Visa fraud AI", "text": "payments", "url": "https://example.com/2"},
        ],
        num_input_docs=2,
    )
    assert E2RANK_LISTWISE_RERANKER == "e2rank-listwise"
    assert "Documents:\n[1]" in prompt
    assert "Search Query:finance AI workflows" in prompt
    assert prompt.count("<|endoftext|>") == 2


def test_non_single_vector_models_are_rejected_from_current_matrix() -> None:
    colbert = adapter_for_model("LiquidAI/LFM2-ColBERT-350M")
    assert colbert.kind == "late_interaction"
    assert not colbert.current_matrix_supported

    gguf = adapter_for_model("jinaai/jina-reranker-v3-GGUF")
    assert gguf.kind == "gguf_reranker"
    assert not gguf.current_matrix_supported


if __name__ == "__main__":
    test_qwen_query_prompt_metadata()
    test_jina_task_metadata()
    test_nomic_task_prefixes()
    test_e2rank_instruction_and_eos()
    test_e2rank_listwise_prompt_shape()
    test_non_single_vector_models_are_rejected_from_current_matrix()
    print("retrieval model adapter tests passed")
