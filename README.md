# Social Media Automation Tools

A collection of Twitter/X and LinkedIn automation tools for content scraping, AI news aggregation, and automated content publishing.

## Env Setup

Create a `.env` file with your credentials:

```
# Twitter credentials (for scraping)
TWITTER_USERNAME=your_username
TWITTER_PASSWORD=your_password

# Google OAuth (alternative login for Twitter/LinkedIn)
GOOGLE_EMAIL=your_email@gmail.com

# LinkedIn credentials (for autopilot)
LINKEDIN_EMAIL=your_email
LINKEDIN_PASSWORD=your_password

# Anthropic API (for AI-powered content generation with Claude)
ANTHROPIC_API_KEY=sk-ant-your-api-key
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

### 3. AI News Scraper
Scrapes AI news from curated influencers, stores in SQLite with vector similarity search, detects emerging trends, and discovers new high-signal accounts.

```bash
# Show statistics
uv run ai_news_scraper.py --stats

# Run scraping session with Google auth
uv run ai_news_scraper.py --scrape --google-auth=your@gmail.com

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

---

### 4. LinkedIn Autopilot
AI-powered LinkedIn content automation that generates engaging posts, creates images, posts automatically, responds to comments, and optimizes strategy based on engagement.

```bash
# Show statistics
uv run linkedin_autopilot.py --stats

# Login to LinkedIn (with Google auth)
uv run linkedin_autopilot.py --login --google-auth=your@gmail.com

# Generate content queue (7 posts)
uv run linkedin_autopilot.py --generate 7

# Post the next item from queue
uv run linkedin_autopilot.py --login --post --google-auth=your@gmail.com

# Check and respond to comments
uv run linkedin_autopilot.py --login --check-comments --google-auth=your@gmail.com

# Run a single autopilot cycle
uv run linkedin_autopilot.py --login --run --google-auth=your@gmail.com

# Run scheduled autopilot (continuous)
uv run linkedin_autopilot.py --login --schedule --google-auth=your@gmail.com
```

#### LinkedIn Autopilot Features

- **Content Generation**: Creates engaging LinkedIn posts from scraped AI news
  - Uses Claude (Anthropic) for intelligent content creation
  - Falls back to templates if no API key configured
  - Multiple post types: news breakdowns, hot takes, curated lists, questions, insights

- **Image Generation**: Creates professional images for posts
  - Quote/insight images with customizable themes
  - Statistics visualization images
  - Proper LinkedIn dimensions (1200x627)

- **Auto-Posting**: Scheduled posting at optimal times
  - Analyzes engagement data to find best posting times
  - Content queue management
  - Human-like typing and delays

- **Comment Management**: Monitors and responds to comments
  - Sentiment analysis for appropriate responses
  - AI-powered response generation
  - Tracks response history

- **Strategy Optimization**: Continuously improves based on engagement
  - Tracks likes, comments, shares, impressions
  - Identifies best-performing content types
  - Optimizes posting schedule based on data

---

## Output

All data is stored in SQLite databases in the `output_data/` folder:

- `twitter_bookmarks.db` - Bookmarks data
- `twitter_likes.db` - Likes data
- `ai_news.db` - AI news with embeddings and trend data
- `linkedin_autopilot.db` - LinkedIn posts, comments, and engagement metrics

Images are stored in `output_data/images/`.

## Typical Workflow

1. **Scrape AI News** to gather content:
   ```bash
   uv run ai_news_scraper.py --scrape --google-auth=your@gmail.com
   ```

2. **Analyze Trends** to see what's hot:
   ```bash
   uv run ai_news_scraper.py --trends
   ```

3. **Generate LinkedIn Content** from the news:
   ```bash
   uv run linkedin_autopilot.py --generate 7
   ```

4. **Review and Post**:
   ```bash
   uv run linkedin_autopilot.py --login --post --google-auth=your@gmail.com
   ```

5. **Or run on autopilot**:
   ```bash
   uv run linkedin_autopilot.py --login --schedule --google-auth=your@gmail.com
   ```

## Running Tests

```bash
uv run test_ai_scraper.py
```
