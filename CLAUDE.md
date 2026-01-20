# Project Context for Claude

## Critical Constraints

### Database Migrations
- Schema changes MUST go through `db_migrations.py` - see file header for instructions
- Never modify inline schema in database classes
- Current versions: discord_links=v2, ai_news=v7

### Scripts Are Self-Contained
- Each `.py` file runs standalone via `uv run script.py`
- Dependencies declared in `# /// script` header at top of each file
- Also mirrored in `pyproject.toml` for IDE support

### Environment Variables
- `DISCORD_TOKEN` - Discord user token for API access
- Credentials stored in `.env` (not committed)

## Architecture

```
run_scrapers.py         # Unified runner: all scrapers + dedup + alerts
ai_news_scraper.py      # Twitter/X scraping + web articles → ai_news.db
discord_link_scraper.py # Discord links → content scraping → discord_links.db
db_migrations.py        # Schema versioning (PRAGMA user_version)
linkedin_autopilot.py   # LinkedIn posting automation
query_news.py           # Vector similarity search CLI
```

## Running Scrapers

```bash
uv run python run_scrapers.py              # All scrapers + dedup + alerts
uv run python run_scrapers.py --discord    # Discord only
uv run python run_scrapers.py --twitter    # Twitter only
uv run python run_scrapers.py --dedup      # Just deduplication
uv run python run_scrapers.py --dry-run    # Preview what would run
```

Set `ALERT_WEBHOOK_URL` in `.env` for Discord/Slack notifications.

## Remote Server (Hetzner)

Server: `157.90.125.102` (port 49222) - requires VPN or whitelisted IP

### Production Database Access

**Location**: `/home/appuser/twitter_influencer/output_data/ai_news.db`

**Query Examples**:
```bash
# Tournament data
ssh -p 49222 appuser@157.90.125.102 "sqlite3 ~/twitter_influencer/output_data/ai_news.db \"SELECT run_id, status FROM tournament_runs ORDER BY run_id DESC LIMIT 10;\""

# Tournament sources and citations
ssh -p 49222 appuser@157.90.125.102 "sqlite3 ~/twitter_influencer/output_data/ai_news.db \"SELECT run_id, citation_number, source_author, citation FROM tournament_sources WHERE run_id = ? AND citation_number IS NOT NULL ORDER BY citation_number;\""

# Winner content
ssh -p 49222 appuser@157.90.125.102 "sqlite3 ~/twitter_influencer/output_data/ai_news.db \"SELECT run_id, winner_content FROM tournament_runs WHERE run_id = ?;\""
```

Cron schedule (should be in appuser crontab):
```
0 8 * * *  ~/run_scraper.sh   # 8am UTC
0 20 * * * ~/run_scraper.sh   # 8pm UTC
```

Access via Hetzner Console if SSH blocked: https://console.hetzner.cloud/

## Content Scraping Pipeline

`discord_link_scraper.py` routes URLs to bespoke scrapers:
- **YouTube**: Transcript extraction via youtube-transcript-api
- **GitHub**: README/issue content extraction
- **Social**: OpenGraph metadata fallback
- **Articles**: Full content + publication date

## Vector Search

Both databases use `sqlite-vec` for embeddings:

**ai_news.db - Hybrid Retrieval (BGE-M3)**:
- Primary model: `BAAI/bge-m3` via FlagEmbedding (1024-dim dense + 256-dim sparse)
- Fallback model: `all-MiniLM-L6-v2` (384 dimensions) if BGE-M3 unavailable
- Virtual tables: `tweet_embeddings_dense`, `tweet_embeddings_sparse`,
  `web_article_embeddings_dense`, `web_article_embeddings_sparse`,
  `youtube_video_embeddings_dense`, `youtube_video_embeddings_sparse`
- Hybrid scoring: `α * dense_score + (1-α) * sparse_score` (default α=0.5)
- Source attribution via TF-IDF Document Page Finder

**discord_links.db**:
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Virtual tables: `link_embeddings`

Extension loaded at runtime (graceful fallback if unavailable)

## Hybrid Citation System (3-Stage Pipeline)

Based on research from CiteFix (arXiv 2504.15629) and VeriCite (arXiv 2510.11394), the citation system
uses a 3-stage hybrid pipeline that combines prompt-based generation with post-processing verification:

### Stage 1: Prompt-Based Generation
- LLM generates citations directly via `GENERATION_PROMPT` in `variant_generator.py`
- Prompt includes citation rules: max 5 citations, one per source, place at sentence end
- News context formatted with `[N] @author (source_type): text` for clear numbering
- **Function**: `parse_llm_citations()` extracts [N] markers from LLM output

### Stage 2: Post-Processing Verification
- Verifies each citation using entity overlap + BGE-M3 semantic similarity
- **Threshold**: 0.4 (increased from 0.25 to reduce false positives)
- **Entity overlap**: Required - sentence must share entities with cited source
- Catches misattributions like VR glasses cited for healthcare claims
- **Function**: `verify_citations_hybrid()` returns verified/weak/invalid status

### Stage 3: Citation Correction
- Removes or replaces weak citations based on verification results
- Actions: 'remove' (default), 'replace' (find better source), 'warn' (log only)
- Preserves minimum citations to avoid empty citation sections
- **Function**: `correct_weak_citations()` returns corrected content + warnings

### Pipeline Flow (linkedin_autopilot.py)
```
1. Generate variant with prompt-based citations
2. Parse LLM citations (fallback to post-hoc if none)
3. Verify each citation (entity + semantic)
4. Correct weak citations (remove by default)
5. Re-parse to get final citation state
```

### Key Improvements Over Previous System
- **Entity overlap validation**: Prevents citing Qwen for OpenAI claims
- **Higher threshold (0.4)**: Reduces false positive semantic matches
- **LLM-generated citations**: Better contextual understanding than pure TF-IDF
- **Post-processing verification**: Catches 80% of attribution errors (per CiteFix)
- **Expected improvement**: 15-21% accuracy increase over prompt-only

### Relevant Code Files
- **3-stage pipeline**: `agents/variant_generator.py:590-910`
  - `parse_llm_citations()`: Stage 1 parsing
  - `verify_citations_hybrid()`: Stage 2 verification
  - `correct_weak_citations()`: Stage 3 correction
- **Entity extraction**: `agents/hybrid_retriever.py:255-305` (`_extract_key_entities`)
- **Pipeline integration**: `linkedin_autopilot.py:1056-1156`
- **Tests**: `test_citations.py`

### Configuration
```python
# In linkedin_autopilot.py
semantic_threshold=0.4        # Minimum BGE-M3 similarity
require_entity_overlap=True   # Require entity match
action='remove'               # How to handle weak citations
min_citations_after=1         # Don't remove if too few remain
```

### Testing
```bash
uv run python test_citations.py  # Run all citation tests
```

### Historical Context (Tournament 86 Issues - Now Fixed)
The previous post-hoc only system had issues with:
- Fireship video cited for healthcare claims (no entity overlap)
- Same source cited 4+ times for unrelated claims (now one per source)
- Generic sources matching specific claims (threshold was too low at 0.25)

These are addressed by the 3-stage hybrid pipeline.
