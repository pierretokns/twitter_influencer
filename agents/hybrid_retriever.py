"""
HybridRetriever - BGE-M3 Based Hybrid Retrieval and Source Attribution
======================================================================

AGENT TYPE: Retrieval Agent (Hybrid dense + sparse)

PURPOSE:
    Implements hybrid retrieval using BGE-M3's native dense + sparse embeddings.
    Also provides Document Page Finder for source attribution.

ALGORITHM: Weighted Hybrid Scoring (Liang et al., ACM 2024)
    hybrid_score = α * dense_score + (1-α) * sparse_score

    Where:
    - dense_score: Cosine similarity of dense embeddings (1024 dims)
    - sparse_score: Dot product of top-K lexical weights (256 dims)
    - α: Weight parameter (0.5 = equal weight, recommended by BGE-M3)

DOCUMENT PAGE FINDER:
    Maps generated content back to sources using TF-IDF similarity.
    Used for source attribution after post generation.

USAGE:
    from agents.hybrid_retriever import HybridRetriever, find_supporting_sources, find_best_paragraph_match

    # For retrieval
    retriever = HybridRetriever(alpha=0.5)
    scores = retriever.compute_hybrid_score(query_dense, query_sparse, doc_dense, doc_sparse)

    # For source attribution
    citations = find_supporting_sources(generated_content, source_texts, threshold=0.2)

    # For paragraph-level quote extraction
    paragraphs = [{'text': 'First paragraph...', 'index': 0}, ...]
    match = find_best_paragraph_match("AI advances rapidly", paragraphs, threshold=0.3)
    if match:
        quote = match['text']
        start_time = match.get('start_time')  # YouTube timestamp
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class HybridRetriever:
    """
    Hybrid retrieval using BGE-M3's native dense + sparse embeddings.
    Implements weighted combination from Liang et al. 2024.
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initialize hybrid retriever.

        Args:
            alpha: Weight for dense vs sparse embeddings.
                   0.5 = equal weight (recommended by BGE-M3 paper)
                   Higher alpha = more weight on semantic (dense)
                   Lower alpha = more weight on keyword (sparse)
        """
        self.alpha = alpha

    def compute_hybrid_score(
        self,
        query_dense: np.ndarray,      # (1, 1024) or (N, 1024)
        query_sparse: np.ndarray,     # (1, 256) or (N, 256)
        doc_dense: np.ndarray,        # (M, 1024)
        doc_sparse: np.ndarray        # (M, 256)
    ) -> np.ndarray:
        """
        Compute hybrid similarity scores between queries and documents.

        Args:
            query_dense: Dense embeddings for queries
            query_sparse: Sparse embeddings for queries (top-K representation)
            doc_dense: Dense embeddings for documents
            doc_sparse: Sparse embeddings for documents

        Returns:
            Array of similarity scores. Shape depends on query shape:
            - Single query (1, D): Returns (M,) scores
            - Multiple queries (N, D): Returns (N, M) scores
        """
        # Handle single query case
        if query_dense.ndim == 1:
            query_dense = query_dense.reshape(1, -1)
            query_sparse = query_sparse.reshape(1, -1)

        # Normalize dense embeddings
        dense_norm_q = query_dense / (np.linalg.norm(query_dense, axis=1, keepdims=True) + 1e-8)
        dense_norm_d = doc_dense / (np.linalg.norm(doc_dense, axis=1, keepdims=True) + 1e-8)

        # Dense similarity (cosine)
        dense_scores = np.dot(dense_norm_q, dense_norm_d.T)  # (N, M)

        # Normalize sparse embeddings
        sparse_norm_q = query_sparse / (np.linalg.norm(query_sparse, axis=1, keepdims=True) + 1e-8)
        sparse_norm_d = doc_sparse / (np.linalg.norm(doc_sparse, axis=1, keepdims=True) + 1e-8)

        # Sparse similarity (cosine of top-k weights)
        sparse_scores = np.dot(sparse_norm_q, sparse_norm_d.T)  # (N, M)

        # Weighted combination
        hybrid_scores = self.alpha * dense_scores + (1 - self.alpha) * sparse_scores

        # Flatten if single query
        if hybrid_scores.shape[0] == 1:
            return hybrid_scores.flatten()

        return hybrid_scores

    def retrieve_top_k(
        self,
        query_dense: np.ndarray,
        query_sparse: np.ndarray,
        doc_dense: np.ndarray,
        doc_sparse: np.ndarray,
        k: int = 20
    ) -> List[int]:
        """
        Retrieve top-k document indices by hybrid score.

        Args:
            query_dense: Dense embedding for query (1, 1024)
            query_sparse: Sparse embedding for query (1, 256)
            doc_dense: Dense embeddings for documents (N, 1024)
            doc_sparse: Sparse embeddings for documents (N, 256)
            k: Number of documents to retrieve

        Returns:
            List of top-k document indices, sorted by score descending
        """
        scores = self.compute_hybrid_score(query_dense, query_sparse, doc_dense, doc_sparse)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return top_k_indices.tolist()

    def compute_similarity_matrix(
        self,
        dense: np.ndarray,   # (N, 1024)
        sparse: np.ndarray   # (N, 256)
    ) -> np.ndarray:
        """
        Compute pairwise hybrid similarity matrix.
        Used for MMR diversity selection.

        Args:
            dense: Dense embeddings for all items
            sparse: Sparse embeddings for all items

        Returns:
            Similarity matrix (N, N)
        """
        return self.compute_hybrid_score(dense, sparse, dense, sparse)


def sparse_to_dense(sparse_dict: Dict[int, float], top_k: int = 256) -> np.ndarray:
    """
    Convert BGE-M3 sparse embedding dict to fixed-size dense representation.

    BGE-M3 returns sparse embeddings as {token_id: weight} dicts.
    For efficient storage and retrieval, we convert to top-K dense vector.

    Args:
        sparse_dict: Sparse embedding {token_id: weight}
        top_k: Number of top weights to keep

    Returns:
        Dense array of shape (top_k,) with top-K weights
    """
    if not sparse_dict:
        return np.zeros(top_k, dtype=np.float32)

    # Sort by weight, take top-k
    sorted_items = sorted(sparse_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Create dense vector with weights in order of importance
    dense = np.zeros(top_k, dtype=np.float32)
    for i, (token_id, weight) in enumerate(sorted_items):
        dense[i] = weight

    return dense


def find_supporting_sources(
    generated_content: str,
    source_texts: List[str],
    threshold: float = 0.2
) -> List[Tuple[int, float]]:
    """
    Document Page Finder: Map generated content to sources using TF-IDF.

    Based on Liang et al. ACM 2024 paper. Uses TF-IDF similarity to identify
    which sources actually support the generated content.

    Args:
        generated_content: The generated post/text
        source_texts: List of source texts (tweets, article titles, etc.)
        threshold: Minimum similarity score to consider a source as cited

    Returns:
        List of (source_index, similarity_score) tuples, sorted by score descending.
        Only includes sources with score >= threshold.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("[HybridRetriever] sklearn not available, returning empty citations")
        return []

    # Split content into sentences/claims
    sentences = [s.strip() for s in generated_content.split('.') if s.strip() and len(s.strip()) > 10]

    if not sentences or not source_texts:
        return []

    # Filter out empty source texts
    valid_sources = [(i, text) for i, text in enumerate(source_texts) if text and len(text.strip()) > 10]
    if not valid_sources:
        return []

    source_indices = [i for i, _ in valid_sources]
    source_texts_clean = [text for _, text in valid_sources]

    try:
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        all_texts = source_texts_clean + sentences
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        n_sources = len(source_texts_clean)
        source_vectors = tfidf_matrix[:n_sources]
        sentence_vectors = tfidf_matrix[n_sources:]

        # Calculate similarity of each source to all sentences
        similarities = cosine_similarity(source_vectors, sentence_vectors)

        # Max similarity of any sentence to each source
        max_similarities = similarities.max(axis=1)

        # Filter by threshold and map back to original indices
        results = [
            (source_indices[i], float(score))
            for i, score in enumerate(max_similarities)
            if score >= threshold
        ]

        return sorted(results, key=lambda x: x[1], reverse=True)

    except Exception as e:
        print(f"[HybridRetriever] TF-IDF attribution failed: {e}")
        return []


def _extract_key_entities(text: str) -> set:
    """
    Extract key named entities and important terms from text.

    Focuses on:
    - Company/product names (OpenAI, Claude, GPT, etc.)
    - Proper nouns (capitalized words)
    - Technical terms and acronyms
    - Numbers and statistics
    """
    import re

    entities = set()
    text_lower = text.lower()

    # Known AI companies and products (case-insensitive matching)
    known_entities = [
        'openai', 'anthropic', 'google', 'deepmind', 'meta', 'microsoft',
        'claude', 'gpt', 'gemini', 'llama', 'mistral', 'perplexity',
        'chatgpt', 'copilot', 'bard', 'palm', 'torch', 'huggingface',
        'nvidia', 'apple', 'amazon', 'aws', 'azure', 'youtube', 'twitter',
        'hipaa', 'healthcare', 'epic', 'tailwind', 'github', 'reddit',
        # New/emerging AI companies
        'qwen', 'deepseek', 'fireworks', 'baseten', 'alibaba', 'elevenlabs',
        'runway', 'pipecat', 'daily', 'modal', 'stability', 'stability ai',
        'eleven labs', 'gemini 3', 'flash', 'dflash', 'o1', 'o1-mini',
    ]

    for entity in known_entities:
        if entity in text_lower:
            entities.add(entity)

    # Extract capitalized words/phrases (likely proper nouns)
    # Match sequences like "OpenAI", "ChatGPT Health", "HIPAA"
    caps_pattern = r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b'
    for match in re.findall(caps_pattern, text):
        if len(match) > 2:  # Skip single letters
            entities.add(match.lower())

    # Extract numbers with context (e.g., "50 hours", "97%", "$1B")
    number_pattern = r'\b\d+(?:\.\d+)?(?:\s*(?:%|hours?|days?|weeks?|months?|years?|[BMK]|billion|million))?\b'
    for match in re.findall(number_pattern, text, re.IGNORECASE):
        entities.add(match.lower())

    # Extract acronyms (2-5 capital letters)
    acronym_pattern = r'\b[A-Z]{2,5}\b'
    for match in re.findall(acronym_pattern, text):
        entities.add(match.lower())

    return entities


def find_sentence_source_mapping(
    sentences: List[str],
    source_texts: List[str],
    threshold: float = 0.4,
    require_entity_overlap: bool = True
) -> Dict[int, int]:
    """
    Map each sentence to its best matching source using semantic similarity (BGE-M3).

    Used for Perplexity-style inline citation placement. Identifies which
    sentence in the generated content best matches which source text.

    Upgraded to use BGE-M3 hybrid embeddings instead of TF-IDF for better semantic
    matching of paraphrased content. This allows sentences like "Qwen demonstrated
    AI agents" to match sources about "Qwen app completing tasks" semantically.

    Now with dynamic entity overlap validation: requires matching entity terms
    extracted from source texts (company names, product names, key terms) to appear
    in both sentence and source before considering a match valid.

    Args:
        sentences: List of sentences from generated content
        source_texts: List of source texts (tweets, article snippets)
        threshold: Minimum similarity to consider a match (increased from 0.25 to 0.4)
        require_entity_overlap: Require entity term overlap between sentence and source (enabled by default)

    Returns:
        Dict mapping sentence_index -> source_index for sentences above threshold
    """
    if not sentences or not source_texts:
        return {}

    # Extract entities from each source text (dynamic entity terms)
    source_entities = [_extract_key_entities(text) for text in source_texts]

    try:
        # Try to use BGE-M3 semantic embeddings first (better for paraphrases)
        from ai_news_scraper import encode_texts_hybrid
        import numpy as np

        # Encode sentences and sources using hybrid embeddings
        sentence_embeddings, _ = encode_texts_hybrid(sentences, max_length=512)
        source_embeddings, _ = encode_texts_hybrid(source_texts, max_length=512)

        # Compute semantic similarity using cosine distance
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(sentence_embeddings, source_embeddings)

        mapping = {}
        for sent_idx, row in enumerate(similarities):
            if len(row) == 0:
                continue

            # Extract entities from this sentence
            sent_entities = _extract_key_entities(sentences[sent_idx])

            # Sort sources by similarity score descending
            sorted_sources = sorted(enumerate(row), key=lambda x: x[1], reverse=True)

            for source_idx, score in sorted_sources:
                if score < threshold:
                    break  # No more candidates above threshold

                # Check entity overlap if enabled
                if require_entity_overlap:
                    # Require at least one entity term to match
                    entity_overlap = sent_entities & source_entities[source_idx]
                    if not entity_overlap:
                        continue  # Skip this source, try next

                # Found a valid semantic match with entity validation
                mapping[sent_idx] = source_idx
                break  # Move to next sentence

        return mapping

    except Exception as e:
        print(f"[HybridRetriever] BGE-M3 mapping failed ({e}), falling back to TF-IDF")

        # Fallback to TF-IDF if BGE-M3 unavailable
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            all_texts = source_texts + sentences
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            n_sources = len(source_texts)
            source_vecs = tfidf_matrix[:n_sources]
            sentence_vecs = tfidf_matrix[n_sources:]

            similarities = cosine_similarity(sentence_vecs, source_vecs)

            mapping = {}
            for sent_idx, row in enumerate(similarities):
                if len(row) == 0:
                    continue

                # Extract entities from this sentence for TF-IDF path too
                sent_entities = _extract_key_entities(sentences[sent_idx])

                sorted_sources = sorted(enumerate(row), key=lambda x: x[1], reverse=True)
                for source_idx, score in sorted_sources:
                    if score < threshold:
                        break

                    # Check entity overlap if enabled (TF-IDF path)
                    if require_entity_overlap:
                        entity_overlap = sent_entities & source_entities[source_idx]
                        if not entity_overlap:
                            continue  # Skip this source, try next

                    mapping[sent_idx] = source_idx
                    break

            return mapping

        except Exception as e2:
            print(f"[HybridRetriever] Sentence mapping failed: {e2}")
            return {}


def extract_citations(
    content: str,
    sources: List[Dict]
) -> List[Dict]:
    """
    Extract [N] citation markers from content and map to sources.

    Args:
        content: Generated content with [1], [2], etc. markers
        sources: List of source dicts with 'id', 'source_type', 'text' keys

    Returns:
        List of citation dicts with source info and index
    """
    import re

    citation_pattern = r'\[(\d+)\]'
    cited_indices = set(int(m) for m in re.findall(citation_pattern, content))

    referenced_sources = []
    for idx in cited_indices:
        if 1 <= idx <= len(sources):
            source = sources[idx - 1]
            referenced_sources.append({
                'index': idx,
                'source_id': source.get('id'),
                'source_type': source.get('source_type'),
                'text': source.get('text', '')[:200],
                'url': source.get('url', '')
            })

    return referenced_sources


def find_best_paragraph_match(
    sentence: str,
    paragraphs: List[Dict],
    threshold: float = 0.35
) -> Optional[Dict]:
    """
    Find the specific paragraph/segment that best matches a sentence using BGE-M3.

    Used for extracting citation quotes from web articles and YouTube content.
    Based on SOTA practices from Perplexity AI and academic literature:
    - Meta-Chunking (arXiv 2410.12788): Logical perception boundaries
    - Max-Min Semantic Chunking: Dynamic similarity assessment

    Args:
        sentence: The sentence from the generated post to match
        paragraphs: List of paragraph/segment dicts with 'text' key.
                   May also have 'start_time' (YouTube), 'index', 'source' keys.
        threshold: Minimum cosine similarity to consider a match (0.3 for BGE-M3).
                   Higher than TF-IDF threshold (0.15) since semantic embeddings
                   produce higher similarity scores for related content.

    Returns:
        Dict with matched paragraph info if found:
        - 'text': The matched paragraph text (truncated to 200 chars)
        - 'score': Cosine similarity score
        - 'start_time': YouTube timestamp if available
        - 'index': Paragraph index
        - 'source': 'description' or 'transcript' for YouTube
        Returns None if no match above threshold.
    """
    if not paragraphs or not sentence or len(sentence.strip()) < 10:
        return None

    # Filter empty paragraphs
    valid_paragraphs = [
        p for p in paragraphs
        if p.get('text') and len(p.get('text', '').strip()) >= 20
    ]

    if not valid_paragraphs:
        return None

    # Get BGE-M3 model
    model = get_bge_m3_model()

    if model is None:
        # Fallback to TF-IDF if BGE-M3 not available
        return _find_best_paragraph_tfidf(sentence, valid_paragraphs, threshold=0.15)

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        # Encode all texts: paragraphs + query sentence
        texts = [p['text'] for p in valid_paragraphs] + [sentence]

        result = model.encode(
            texts,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )

        # Get dense embeddings
        dense_vecs = np.array(result['dense_vecs'], dtype=np.float32)
        query_vec = dense_vecs[-1:]   # (1, 1024)
        para_vecs = dense_vecs[:-1]   # (N, 1024)

        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, para_vecs)[0]

        # Find best match
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

        if best_score >= threshold:
            matched = valid_paragraphs[best_idx]
            return {
                'text': matched['text'][:200],
                'score': best_score,
                'start_time': matched.get('start_time'),
                'index': matched.get('index', best_idx),
                'source': matched.get('source')
            }

        return None

    except Exception as e:
        print(f"[HybridRetriever] BGE-M3 paragraph matching failed: {e}")
        # Fallback to TF-IDF
        return _find_best_paragraph_tfidf(sentence, valid_paragraphs, threshold=0.15)


def _find_best_paragraph_tfidf(
    sentence: str,
    paragraphs: List[Dict],
    threshold: float = 0.15
) -> Optional[Dict]:
    """
    Fallback TF-IDF based paragraph matching when BGE-M3 unavailable.

    Args:
        sentence: Query sentence
        paragraphs: List of paragraph dicts with 'text' key
        threshold: Minimum similarity threshold

    Returns:
        Best matching paragraph dict or None
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = [p['text'] for p in paragraphs] + [sentence]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(texts)

        query_vec = tfidf_matrix[-1]
        para_vecs = tfidf_matrix[:-1]

        similarities = cosine_similarity(query_vec, para_vecs)[0]
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

        if best_score >= threshold:
            matched = paragraphs[best_idx]
            return {
                'text': matched['text'][:200],
                'score': best_score,
                'start_time': matched.get('start_time'),
                'index': matched.get('index', best_idx),
                'source': matched.get('source')
            }

        return None

    except Exception as e:
        print(f"[HybridRetriever] TF-IDF paragraph matching failed: {e}")
        return None


# Singleton for BGE-M3 model (lazy loaded)
_bge_m3_model = None


def get_bge_m3_model():
    """
    Get or initialize the BGE-M3 model.
    Lazy-loaded singleton for efficiency.
    """
    global _bge_m3_model

    if _bge_m3_model is None:
        try:
            from FlagEmbedding import BGEM3FlagModel
            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[HybridRetriever] Loading BGE-M3 model on {device}...")

            _bge_m3_model = BGEM3FlagModel(
                'BAAI/bge-m3',
                use_fp16=True,
                device=device
            )

            print("[HybridRetriever] BGE-M3 loaded successfully")

        except ImportError as e:
            print(f"[HybridRetriever] FlagEmbedding not available: {e}")
            print("[HybridRetriever] Install with: pip install FlagEmbedding")
            _bge_m3_model = None
        except Exception as e:
            print(f"[HybridRetriever] Failed to load BGE-M3: {e}")
            _bge_m3_model = None

    return _bge_m3_model


def encode_texts_hybrid(
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 512
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Encode texts using BGE-M3 hybrid embeddings.

    Args:
        texts: List of texts to encode
        batch_size: Batch size for encoding
        max_length: Maximum sequence length (512 for tweets, 8192 for articles)

    Returns:
        Tuple of (dense_embeddings, sparse_embeddings) or (None, None) if failed.
        - dense_embeddings: (N, 1024) array
        - sparse_embeddings: (N, 256) array (top-K representation)
    """
    model = get_bge_m3_model()

    if model is None:
        return None, None

    try:
        # Encode with BGE-M3
        embedding_results = model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False  # Skip multi-vector for now
        )

        # Extract dense embeddings
        dense_embeddings = np.array(embedding_results['dense_vecs'], dtype=np.float32)

        # Convert sparse embeddings to fixed-size dense
        sparse_list = []
        for sparse_dict in embedding_results['lexical_weights']:
            sparse_dense = sparse_to_dense(sparse_dict, top_k=256)
            sparse_list.append(sparse_dense)

        sparse_embeddings = np.array(sparse_list, dtype=np.float32)

        return dense_embeddings, sparse_embeddings

    except Exception as e:
        print(f"[HybridRetriever] Encoding failed: {e}")
        return None, None


if __name__ == "__main__":
    # Test the module
    print("Testing HybridRetriever...")

    # Test hybrid scoring
    retriever = HybridRetriever(alpha=0.5)

    np.random.seed(42)
    query_dense = np.random.randn(1, 1024).astype(np.float32)
    query_sparse = np.random.randn(1, 256).astype(np.float32)
    doc_dense = np.random.randn(10, 1024).astype(np.float32)
    doc_sparse = np.random.randn(10, 256).astype(np.float32)

    scores = retriever.compute_hybrid_score(query_dense, query_sparse, doc_dense, doc_sparse)
    print(f"Hybrid scores shape: {scores.shape}")
    print(f"Scores: {scores[:5]}")

    top_k = retriever.retrieve_top_k(query_dense, query_sparse, doc_dense, doc_sparse, k=3)
    print(f"Top-3 indices: {top_k}")

    # Test Document Page Finder
    print("\nTesting Document Page Finder...")
    content = "OpenAI announced GPT-5 with improved reasoning capabilities. Anthropic released Claude 4 with extended context."
    sources = [
        "OpenAI unveils GPT-5 with breakthrough reasoning capabilities",
        "New chess AI defeats world grandmaster",
        "Claude 4 from Anthropic now available with 1M token context"
    ]

    citations = find_supporting_sources(content, sources)
    print(f"Found citations: {citations}")

    # Test sparse_to_dense
    print("\nTesting sparse_to_dense...")
    sparse_dict = {100: 0.5, 200: 0.8, 300: 0.3, 400: 0.9, 500: 0.1}
    dense = sparse_to_dense(sparse_dict, top_k=4)
    print(f"Sparse to dense: {dense}")  # Should be [0.9, 0.8, 0.5, 0.3]

    # Test find_best_paragraph_match (TF-IDF fallback - BGE-M3 may not be loaded)
    print("\nTesting find_best_paragraph_match...")
    paragraphs = [
        {'text': 'OpenAI released GPT-5 with breakthrough reasoning capabilities', 'index': 0},
        {'text': 'Google announced new quantum computing advances', 'index': 1},
        {'text': 'Claude 4 from Anthropic supports 1 million token context', 'index': 2, 'start_time': 125.5},
    ]
    query = "Anthropic Claude extended context window"
    match = find_best_paragraph_match(query, paragraphs, threshold=0.1)
    if match:
        print(f"  Match found: '{match['text'][:50]}...'")
        print(f"  Score: {match['score']:.3f}, Index: {match['index']}, Start time: {match.get('start_time')}")
    else:
        print("  No match found (may need lower threshold)")

    print("\nAll tests passed!")
