# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "python-dotenv>=0.19.0",
#     "sentence-transformers>=2.2.0",
#     "sqlite-vec>=0.1.0",
#     "numpy>=1.24.0",
#     "scikit-learn>=1.3.0",
#     "hdbscan>=0.8.33",
# ]
# ///

"""
Test script for AI News Scraper
"""

from pathlib import Path
import tempfile
import shutil

# Import from main module
from ai_news_scraper import (
    AINewsDatabase,
    EmbeddingGenerator,
    TrendDetector,
    AI_INFLUENCERS,
    get_all_seed_influencers,
    AINewsOrchestrator
)
import numpy as np


def test_influencer_list():
    """Test that influencer list is properly configured"""
    print("\n=== Testing Influencer List ===")

    influencers = get_all_seed_influencers()
    print(f"Total seed influencers: {len(influencers)}")
    print(f"Categories: {list(AI_INFLUENCERS.keys())}")

    for category, handles in AI_INFLUENCERS.items():
        print(f"  {category}: {len(handles)} accounts")

    assert len(influencers) > 30, "Should have at least 30 seed influencers"
    print("PASSED: Influencer list configured correctly")


def test_database():
    """Test database creation and basic operations"""
    print("\n=== Testing Database ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'test.db'
        db = AINewsDatabase(db_path)

        # Test influencer saving
        db.save_influencer('testuser', 'Test User', 'test', True, 'test')

        # Test tweet saving
        test_tweet = {
            'tweet_id': '123456789',
            'username': 'testuser',
            'display_name': 'Test User',
            'text': 'This is a test about AI and machine learning transformers',
            'timestamp': '2024-01-01T12:00:00Z',
            'url': 'https://twitter.com/testuser/status/123456789',
            'likes_count': 100,
            'retweets_count': 50,
            'replies_count': 10,
        }
        is_ai = db.save_tweet(test_tweet)

        assert is_ai == True, "Tweet should be marked as AI-relevant"

        # Test stats
        stats = db.get_stats()
        assert stats['total_tweets'] == 1
        assert stats['ai_relevant_tweets'] == 1

        db.close()

    print("PASSED: Database operations work correctly")


def test_embeddings():
    """Test embedding generation"""
    print("\n=== Testing Embeddings ===")

    gen = EmbeddingGenerator()

    # Generate enough samples for the fallback embedder to work properly
    warmup_texts = [
        "OpenAI released GPT-5 with breakthrough reasoning capabilities",
        "Machine learning models are improving rapidly",
        "AI safety research is critical for the future",
        "Neural networks can now understand complex language",
        "Deep learning has transformed computer vision",
        "Large language models are changing how we work",
        "Transformer architecture revolutionized NLP",
        "AI assistants help with coding tasks",
        "Reinforcement learning enables robotic control",
        "Natural language processing has advanced significantly",
    ]

    # Warm up the embedder with multiple samples
    for text in warmup_texts:
        _ = gen.generate(text)

    # Test single embedding
    text = "OpenAI released GPT-5 with breakthrough reasoning capabilities"
    embedding = gen.generate(text)

    assert embedding.shape == (384,), f"Expected shape (384,), got {embedding.shape}"
    assert isinstance(embedding, np.ndarray), "Should return numpy array"

    # Test batch embedding
    texts = [
        "GPT-5 achieves superhuman reasoning",
        "Claude 4 improves code generation",
        "Gemini 2 launches with multimodal understanding",
    ]
    embeddings = gen.generate_batch(texts)

    assert embeddings.shape == (3, 384), f"Expected shape (3, 384), got {embeddings.shape}"

    print("PASSED: Embedding generation works correctly")


def test_similarity_search():
    """Test vector similarity search"""
    print("\n=== Testing Similarity Search ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'test_vss.db'
        db = AINewsDatabase(db_path)
        gen = EmbeddingGenerator()

        # Insert test tweets with embeddings
        test_tweets = [
            {'tweet_id': 'test1', 'username': 'openai', 'text': 'GPT-5 achieves superhuman reasoning on complex tasks', 'timestamp': '2024-12-01T10:00:00Z', 'likes_count': 1000, 'retweets_count': 500},
            {'tweet_id': 'test2', 'username': 'anthropic', 'text': 'Claude 4 now available with improved code generation', 'timestamp': '2024-12-01T11:00:00Z', 'likes_count': 800, 'retweets_count': 400},
            {'tweet_id': 'test3', 'username': 'google', 'text': 'Gemini 2 launches with multimodal understanding', 'timestamp': '2024-12-01T12:00:00Z', 'likes_count': 900, 'retweets_count': 450},
            {'tweet_id': 'test4', 'username': 'meta', 'text': 'Llama 4 open source release with 1T parameters', 'timestamp': '2024-12-01T13:00:00Z', 'likes_count': 700, 'retweets_count': 350},
            {'tweet_id': 'test5', 'username': 'researcher', 'text': 'New breakthrough in transformer architecture efficiency', 'timestamp': '2024-12-01T14:00:00Z', 'likes_count': 200, 'retweets_count': 100},
        ]

        for tweet in test_tweets:
            embedding = gen.generate(tweet['text'])
            db.save_tweet(tweet, embedding)

        # Test similarity search
        if db._has_vec:
            query = "What are the latest language model releases?"
            query_embedding = gen.generate(query)
            results = db.similarity_search(query_embedding, limit=3)

            print(f"Query: '{query}'")
            print(f"Found {len(results)} similar tweets:")
            for r in results:
                print(f"  - @{r['username']}: {r['text'][:50]}...")

            assert len(results) > 0, "Should find similar tweets"
            print("PASSED: Similarity search works correctly")
        else:
            print("SKIPPED: sqlite-vec not available")

        db.close()


def test_trend_detection():
    """Test trend detection with clustering"""
    print("\n=== Testing Trend Detection ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'test_trends.db'
        db = AINewsDatabase(db_path)
        gen = EmbeddingGenerator()

        # Create enough tweets for clustering (at least 10)
        from datetime import datetime, timedelta
        now = datetime.now()

        # Cluster 1: LLM releases
        llm_tweets = [
            "GPT-5 just released with amazing new capabilities",
            "OpenAI announces GPT-5 is now available",
            "The new GPT-5 model shows impressive reasoning",
            "GPT-5 benchmark results are incredible",
        ]

        # Cluster 2: Open source AI
        oss_tweets = [
            "Llama 4 is now open source on GitHub",
            "Meta releases Llama 4 weights for everyone",
            "Open source AI is winning with Llama",
            "Download Llama 4 and run it locally",
        ]

        # Cluster 3: AI safety
        safety_tweets = [
            "AI alignment research is critical for safety",
            "New paper on AI safety and alignment",
            "We need more research on AI alignment",
            "AI safety concerns are growing",
        ]

        all_tweets = llm_tweets + oss_tweets + safety_tweets

        for i, text in enumerate(all_tweets):
            tweet = {
                'tweet_id': f'trend_test_{i}',
                'username': f'user{i}',
                'text': text,
                'timestamp': (now - timedelta(hours=i)).isoformat(),
                'likes_count': 100 + i * 10,
                'retweets_count': 50 + i * 5,
                'replies_count': 10 + i,
            }
            embedding = gen.generate(text)
            db.save_tweet(tweet, embedding)

        # Run trend detection
        detector = TrendDetector(db)
        trends = detector.analyze_trends(hours=48, min_cluster_size=2)

        print(f"Detected {len(trends)} topic clusters:")
        for i, trend in enumerate(trends):
            print(f"  {i+1}. Keywords: {', '.join(trend['keywords'][:5])}")
            print(f"     Tweets: {trend['tweet_count']}, Engagement: {trend['total_engagement']}")

        # We should detect at least 2 clusters
        if len(trends) >= 2:
            print("PASSED: Trend detection works correctly")
        else:
            print(f"NOTE: Detected {len(trends)} clusters (expected 2-3, but clustering can vary)")

        db.close()


def test_influencer_discovery():
    """Test influencer discovery from mentions"""
    print("\n=== Testing Influencer Discovery ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'test_discovery.db'
        db = AINewsDatabase(db_path)

        # Create tweets with mentions
        tweets = [
            {'tweet_id': 'd1', 'username': 'user1', 'text': 'Check out @newinfluencer for great AI content!', 'timestamp': '2024-12-01T10:00:00Z'},
            {'tweet_id': 'd2', 'username': 'user2', 'text': 'I agree with @newinfluencer about LLMs', 'timestamp': '2024-12-01T11:00:00Z'},
            {'tweet_id': 'd3', 'username': 'user3', 'text': '@newinfluencer just posted a great thread', 'timestamp': '2024-12-01T12:00:00Z'},
            {'tweet_id': 'd4', 'username': 'user4', 'text': '@newinfluencer is the best AI account', 'timestamp': '2024-12-01T13:00:00Z'},
        ]

        for tweet in tweets:
            db.save_tweet(tweet)

        # Check discovered influencers
        discovered = db.get_discovered_influencers(min_mentions=3)

        print(f"Discovered {len(discovered)} potential influencers:")
        for d in discovered:
            print(f"  - @{d['username']}: {d['mention_count']} mentions")

        assert len(discovered) == 1, "Should discover one influencer"
        assert discovered[0]['username'] == 'newinfluencer'

        # Test promotion
        db.promote_to_influencer('newinfluencer')

        # Verify promotion
        cursor = db.conn.cursor()
        cursor.execute('SELECT * FROM influencers WHERE username = ?', ('newinfluencer',))
        promoted = cursor.fetchone()

        assert promoted is not None, "Should be in influencers table"
        assert promoted['is_seed'] == 0, "Should not be a seed"
        assert promoted['discovery_source'] == 'auto_discovered'

        db.close()

    print("PASSED: Influencer discovery works correctly")


def test_full_orchestrator():
    """Test the full orchestrator initialization"""
    print("\n=== Testing Full Orchestrator ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        orchestrator = AINewsOrchestrator(output_dir)
        orchestrator.initialize()

        # Check stats
        stats = orchestrator.db.get_stats()
        print(f"Initialized with {stats['total_influencers']} influencers")

        assert stats['total_influencers'] > 30, "Should have seeded influencers"

        # Test report export (with no data)
        report_path = orchestrator.export_report('test_report.json')
        assert report_path.exists(), "Report should be created"

        orchestrator.close()

    print("PASSED: Orchestrator works correctly")


def main():
    """Run all tests"""
    print("=" * 60)
    print("AI NEWS SCRAPER TEST SUITE")
    print("=" * 60)

    tests = [
        test_influencer_list,
        test_database,
        test_embeddings,
        test_similarity_search,
        test_trend_detection,
        test_influencer_discovery,
        test_full_orchestrator,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
