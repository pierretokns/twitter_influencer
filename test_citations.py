#!/usr/bin/env python3
"""
Test script to validate hybrid citation system (3-stage pipeline).

Tests:
1. Dynamic entity extraction from sources
2. Stage 1: LLM-generated citation parsing
3. Stage 2: Citation verification (entity overlap + semantic)
4. Stage 3: Citation correction (remove/replace weak citations)
5. Full pipeline integration test

Based on research:
- CiteFix (arXiv 2504.15629): Post-processing improves accuracy 15-21%
- VeriCite (arXiv 2510.11394): 3-stage verify+refine improves F1 by 11.4%
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.variant_generator import PostVariantGenerator
from agents.news_selector import NewsSelector
from agents.hybrid_retriever import _extract_key_entities


def test_entity_extraction():
    """Test dynamic entity extraction from sources."""
    print("[Test 1] Dynamic Entity Extraction\n")

    test_texts = [
        "OpenAI releases GPT-5 with improved reasoning",
        "Google Gemini Personal Intelligence draws insights from your data",
        "Qwen App demonstrates autonomous task completion",
        "The best VR glasses on the market today",
    ]

    for i, text in enumerate(test_texts, 1):
        entities = _extract_key_entities(text)
        print(f"  Text {i}: {text[:50]}...")
        print(f"  Entities: {entities}")
        print()


def test_parse_llm_citations():
    """Test Stage 1: Parse LLM-generated citations."""
    print("[Test 2] Stage 1 - Parse LLM Citations\n")

    generator = PostVariantGenerator()

    # Simulate LLM-generated content with citations
    test_content = """OpenAI just released GPT-5 with breakthrough reasoning[1].

This is huge for the industry.

Anthropic also announced Claude improvements[3].

Meanwhile, Google is pushing Gemini updates[2].

What do you think this means for the future?

#AI #Tech #Innovation"""

    # Mock news items
    news_items = [
        {'text': 'OpenAI announces GPT-5 with improved reasoning', 'username': 'OpenAI', 'source_type': 'twitter'},
        {'text': 'Google launches Gemini 2.0 with new features', 'username': 'Google', 'source_type': 'twitter'},
        {'text': 'Anthropic releases Claude 3.5 Sonnet', 'username': 'Anthropic', 'source_type': 'twitter'},
        {'text': 'Random tech news about smartphones', 'username': 'TechNews', 'source_type': 'web'},
    ]

    content, annotated = generator.parse_llm_citations(test_content, news_items)

    print(f"  Content length: {len(content)} chars")
    print(f"  Cited sources: {sum(1 for a in annotated if a.get('is_referenced'))}/{len(annotated)}")
    print()

    for i, item in enumerate(annotated):
        if item.get('is_referenced'):
            print(f"  [{i+1}] @{item.get('username')}: CITED")
        else:
            print(f"  [{i+1}] @{item.get('username')}: not cited")
    print()

    # Verify expected citations
    expected_cited = {1, 2, 3}
    actual_cited = {i+1 for i, a in enumerate(annotated) if a.get('is_referenced')}

    if expected_cited == actual_cited:
        print("  PASS: Correctly parsed citations [1], [2], [3]")
    else:
        print(f"  FAIL: Expected {expected_cited}, got {actual_cited}")


def test_verify_citations():
    """Test Stage 2: Citation verification with entity overlap + semantic."""
    print("\n[Test 3] Stage 2 - Verify Citations\n")

    generator = PostVariantGenerator()

    # Content with mixed quality citations
    # [1] should be VERIFIED (OpenAI + GPT-5 match)
    # [2] should be WEAK (generic VR doesn't match healthcare claim)
    # [3] should be VERIFIED (Qwen entities match)
    test_content = """OpenAI released GPT-5 with multimodal capabilities[1].

ChatGPT is now helping with healthcare diagnoses[2].

Qwen demonstrated impressive autonomous task completion[3].

The future of AI is here."""

    news_items = [
        {'text': 'OpenAI announces GPT-5 with breakthrough multimodal features and reasoning capabilities', 'username': 'OpenAI', 'source_type': 'twitter'},
        {'text': 'The best VR glasses for gaming and entertainment in 2024', 'username': 'TechReview', 'source_type': 'web'},
        {'text': 'Alibaba Qwen app demonstrates autonomous task completion capabilities for everyday tasks', 'username': 'Qwen', 'source_type': 'twitter'},
        {'text': 'ChatGPT helping doctors with medical diagnosis and HIPAA compliance', 'username': 'HealthTech', 'source_type': 'twitter'},
    ]

    # Parse citations first
    content, annotated = generator.parse_llm_citations(test_content, news_items)

    # Verify citations
    results = generator.verify_citations_hybrid(
        content, annotated,
        semantic_threshold=0.4,
        require_entity_overlap=True
    )

    print(f"  Total citations found: {len(results)}")
    print()

    for r in results:
        status_icon = "PASS" if r['status'] == 'verified' else "FAIL" if r['status'] == 'weak' else "?"
        print(f"  [{r['citation']}] {status_icon} - {r['status'].upper()}")
        print(f"      Source: @{r.get('source', 'Unknown')}")
        print(f"      Reason: {r['reason'][:60]}...")
        if r.get('entity_overlap'):
            print(f"      Entities: {r['entity_overlap'][:3]}")
        print()

    # Check expected results
    verified = sum(1 for r in results if r['status'] == 'verified')
    weak = sum(1 for r in results if r['status'] == 'weak')

    print(f"  Summary: {verified} verified, {weak} weak")

    # Citation [2] should be weak (VR glasses doesn't support healthcare claim)
    citation_2 = next((r for r in results if r['citation'] == 2), None)
    if citation_2 and citation_2['status'] == 'weak':
        print("  PASS: Citation [2] correctly identified as weak (VR vs healthcare mismatch)")
    else:
        print("  WARNING: Citation [2] was not flagged as weak")


def test_correct_citations():
    """Test Stage 3: Citation correction (remove weak citations)."""
    print("\n[Test 4] Stage 3 - Correct Weak Citations\n")

    generator = PostVariantGenerator()

    # Content with weak citation [2]
    test_content = """OpenAI released GPT-5 with multimodal capabilities[1].

ChatGPT is now helping with healthcare diagnoses[2].

Qwen demonstrated impressive autonomous task completion[3]."""

    news_items = [
        {'text': 'OpenAI announces GPT-5 with breakthrough multimodal features', 'username': 'OpenAI', 'source_type': 'twitter'},
        {'text': 'The best VR glasses for gaming in 2024', 'username': 'TechReview', 'source_type': 'web'},
        {'text': 'Alibaba Qwen demonstrates autonomous task completion', 'username': 'Qwen', 'source_type': 'twitter'},
    ]

    # Parse and verify
    content, annotated = generator.parse_llm_citations(test_content, news_items)
    verification_results = generator.verify_citations_hybrid(content, annotated)

    # Mark citation [2] as weak for testing (simulate verification failure)
    for r in verification_results:
        if r['citation'] == 2:
            r['status'] = 'weak'
            r['reason'] = 'No entity overlap between sentence and source'

    # Correct with 'remove' action
    corrected, warnings = generator.correct_weak_citations(
        content, verification_results, annotated,
        action='remove',
        min_citations_after=1
    )

    print(f"  Original: {len(test_content)} chars")
    print(f"  Corrected: {len(corrected)} chars")
    print()

    print("  Warnings:")
    for w in warnings:
        print(f"    - {w}")
    print()

    # Check that [2] was removed
    if '[2]' not in corrected:
        print("  PASS: Citation [2] was removed from content")
    else:
        print("  FAIL: Citation [2] still present in content")

    # Check that [1] and [3] remain
    if '[1]' in corrected and '[3]' in corrected:
        print("  PASS: Valid citations [1] and [3] preserved")
    else:
        print("  FAIL: Valid citations were incorrectly removed")

    print()
    print("  Corrected content:")
    print(f"    {corrected[:200]}...")


def test_full_pipeline():
    """Test full 3-stage hybrid citation pipeline with real data."""
    print("\n[Test 5] Full Pipeline Integration Test\n")

    db_path = Path(__file__).parent / 'output_data' / 'ai_news.db'
    if not db_path.exists():
        print(f"  SKIP: Database not found at {db_path}")
        return

    # Select diverse news
    selector = NewsSelector(db_path)
    news_items = selector.select_diverse_news(limit=10)
    print(f"  Selected {len(news_items)} news items")

    if not news_items:
        print("  No news items available")
        return

    # Generate 1 variant
    generator = PostVariantGenerator()
    variants = generator.generate_variants(news_items, num_variants=1)
    print(f"  Generated {len(variants)} variant(s)")

    if not variants:
        print("  No variants generated")
        return

    variant = variants[0]
    original_content = variant.content
    print(f"  Original content: {len(original_content)} chars")

    # Stage 1: Parse LLM citations
    print("\n  --- Stage 1: Parse LLM Citations ---")
    content, annotated = generator.parse_llm_citations(original_content, news_items)
    cited_count = sum(1 for a in annotated if a.get('is_referenced'))
    print(f"  LLM-generated citations: {cited_count}")

    if cited_count == 0:
        print("  Falling back to post-hoc attribution...")
        annotated = generator.annotate_sources_with_attribution(content, news_items, threshold=0.25)
        content, annotated = generator.insert_citation_markers(content, annotated, max_citations=5)
        cited_count = sum(1 for a in annotated if a.get('is_referenced'))
        print(f"  Post-hoc citations: {cited_count}")

    # Stage 2: Verify citations
    print("\n  --- Stage 2: Verify Citations ---")
    verification_results = generator.verify_citations_hybrid(
        content, annotated,
        semantic_threshold=0.4,
        require_entity_overlap=True
    )

    verified = sum(1 for r in verification_results if r['status'] == 'verified')
    weak = sum(1 for r in verification_results if r['status'] == 'weak')
    print(f"  Verification: {verified} verified, {weak} weak")

    for r in verification_results:
        if r['status'] == 'weak':
            print(f"    [{r['citation']}] WEAK: {r['reason'][:50]}...")

    # Stage 3: Correct weak citations
    print("\n  --- Stage 3: Correct Citations ---")
    corrected, warnings = generator.correct_weak_citations(
        content, verification_results, annotated,
        action='remove',
        min_citations_after=1
    )

    for w in warnings:
        print(f"    {w}")

    print(f"\n  Final content: {len(corrected)} chars")

    # Re-count citations after correction
    import re
    final_citations = set(int(m) for m in re.findall(r'\[(\d+)\]', corrected))
    print(f"  Final citations: {sorted(final_citations)}")

    print()
    print("  Sample of final content (first 400 chars):")
    print(f"    {corrected[:400]}...")
    print()

    # Verify citation quality
    if verified > 0 or len(final_citations) > 0:
        print("  PASS: Pipeline produced verified citations")
    else:
        print("  WARNING: No verified citations in final output")


def test_citation_system():
    """Legacy test - citation system with new thresholds."""
    print("\n[Test 6] Legacy Citation System Test\n")

    db_path = Path(__file__).parent / 'output_data' / 'ai_news.db'
    if not db_path.exists():
        print(f"  SKIP: Database not found at {db_path}")
        return

    # Select diverse news
    selector = NewsSelector(db_path)
    news_items = selector.select_diverse_news(limit=10)
    print(f"  Selected {len(news_items)} news items")

    if not news_items:
        print("  No news items available")
        return

    # Generate variants
    generator = PostVariantGenerator()
    variants = generator.generate_variants(news_items, num_variants=1)
    print(f"  Generated {len(variants)} variant(s)")

    if not variants:
        print("  No variants generated")
        return

    variant = variants[0]
    print(f"  Variant length: {len(variant.content)} chars")
    print()

    # Test annotation
    print("  Annotating sources (threshold=0.25)...")
    annotated = generator.annotate_sources_with_attribution(variant.content, news_items, threshold=0.25)
    referenced = [s for s in annotated if s.get('is_referenced')]
    print(f"  Referenced sources: {len(referenced)}/{len(annotated)}")
    print()

    # Insert markers with entity validation
    print("  Inserting citation markers (threshold=0.4, entity overlap=True)...")
    marked_content, cited_sources = generator.insert_citation_markers(variant.content, annotated)
    citations = [s for s in cited_sources if s.get('citation_number')]

    print(f"  Total citations inserted: {len(citations)}")
    print()

    if citations:
        print("  Citation Details:")
        for c in citations[:5]:
            print(f"    [{c.get('citation_number')}] {c.get('source_author', c.get('username', 'Unknown'))}")
            print(f"        Score: {c.get('attribution_score', 0):.3f}")
            print(f"        Type: {c.get('source_type')}")
            print()

    print("  Sample of marked content (first 500 chars):")
    print(f"    {marked_content[:500]}...")
    print()


if __name__ == '__main__':
    try:
        # Unit tests (no DB required)
        test_entity_extraction()
        test_parse_llm_citations()
        test_verify_citations()
        test_correct_citations()

        # Integration tests (requires DB)
        test_full_pipeline()
        test_citation_system()

        print("\n" + "="*60)
        print("[RESULT] All hybrid citation system tests completed!")
        print("="*60)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
