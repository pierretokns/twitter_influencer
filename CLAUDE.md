# Project Context for Claude

## Critical Constraints

### Database Migrations
- Schema changes MUST go through `db_migrations.py` - see file header for instructions
- Never modify inline schema in database classes
- Current versions: discord_links=v2, ai_news=v2

### Scripts Are Self-Contained
- Each `.py` file runs standalone via `uv run script.py`
- Dependencies declared in `# /// script` header at top of each file
- Also mirrored in `pyproject.toml` for IDE support

### Environment Variables
- `DISCORD_TOKEN` - Discord user token for API access
- Credentials stored in `.env` (not committed)

## Architecture

```
ai_news_scraper.py      # Twitter/X scraping + web articles → ai_news.db
discord_link_scraper.py # Discord links → content scraping → discord_links.db
db_migrations.py        # Schema versioning (PRAGMA user_version)
linkedin_autopilot.py   # LinkedIn posting automation
query_news.py           # Vector similarity search CLI
```

## Content Scraping Pipeline

`discord_link_scraper.py` routes URLs to bespoke scrapers:
- **YouTube**: Transcript extraction via youtube-transcript-api
- **GitHub**: README/issue content extraction
- **Social**: OpenGraph metadata fallback
- **Articles**: Full content + publication date

## Vector Search

Both databases use `sqlite-vec` for embeddings:
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Virtual tables: `link_embeddings`, `tweet_embeddings`
- Extension loaded at runtime (graceful fallback if unavailable)
