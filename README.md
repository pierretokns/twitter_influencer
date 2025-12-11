# Twitter Scraper Tools

A collection of Twitter/X scraping tools for bookmarks, likes, and AI news aggregation with vector similarity search.

## Env Setup

Create a `.env` file with your Twitter credentials:

```
TWITTER_USERNAME=your_username
TWITTER_PASSWORD=your_password
```

## Running

All scripts are designed to run using `uv`. If you don't have uv installed:

```bash
pip install uv
```

---

## Available Scripts

### 1. Bookmarks Scraper
Scrapes all your Twitter bookmarks with full thread support.

```bash
uv run 04_twitter_bookmarks_advanced.py
```

### 2. Likes Scraper
Scrapes all your liked tweets with complete thread extraction.

```bash
uv run 06_twitter_likes_scraper.py
```

### 3. AI News Scraper (NEW)
Scrapes AI news from curated influencers, stores in SQLite with vector similarity search, detects emerging trends, and discovers new high-signal accounts.

```bash
# Show statistics
uv run ai_news_scraper.py --stats

# Run scraping session
uv run ai_news_scraper.py --scrape

# With options
uv run ai_news_scraper.py --scrape --max-users 20 --tweets-per-user 30

# Analyze trends from recent data
uv run ai_news_scraper.py --trends --hours 48

# Semantic search across scraped content
uv run ai_news_scraper.py --search "latest LLM releases"

# Generate JSON report
uv run ai_news_scraper.py --report
```

#### AI News Scraper Features

- **45 Curated Seed Influencers** across 5 categories:
  - Researchers (Yann LeCun, Andrej Karpathy, Andrew Ng, etc.)
  - Organizations (OpenAI, Anthropic, DeepMind, Meta AI, etc.)
  - Journalists/Commentators (Sam Altman, Jim Fan, etc.)
  - Engineering/Tools (LangChain, LlamaIndex, Hugging Face, etc.)
  - AI Safety/Policy (AI Safety Institute, Dario Amodei, etc.)

- **Vector Similarity Search** using sqlite-vec for semantic search across all tweets

- **Trend Detection** using clustering (HDBSCAN/KMeans) to identify emerging topics

- **Automatic Influencer Discovery** - promotes accounts frequently mentioned by existing influencers

- **AI Relevance Filtering** - automatically tags tweets related to AI/ML topics

---

## Output

All data is stored in SQLite databases in the `output_data/` folder:

- `twitter_bookmarks.db` - Bookmarks data
- `twitter_likes.db` - Likes data
- `ai_news.db` - AI news with embeddings and trend data

The tools also export JSON files with timestamps for each scraping session.

## Running Tests

```bash
uv run test_ai_scraper.py
```
