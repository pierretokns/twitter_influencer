# Project Context for Claude

## Database Schema Management

This project uses a **simple version-based SQLite migration system** stored in `db_migrations.py`.

### How It Works

- Schema version tracked via SQLite's built-in `PRAGMA user_version`
- Version number stored in the database file itself (survives copies/deployments)
- Migrations run automatically when database classes initialize
- Safe to run multiple times (idempotent)

### Current Schema Versions

| Database | File | Current Version | Class |
|----------|------|-----------------|-------|
| discord_links | `output_data/discord_links.db` | v2 | `DiscordLinksDatabase` |
| ai_news | `output_data/ai_news.db` | v2 | `AINewsDatabase` |

### Adding New Migrations

**CRITICAL**: When modifying database schema, you MUST:

1. **Bump the version number** in `db_migrations.py`:
   ```python
   DISCORD_DB_VERSION = 3  # was 2
   ```

2. **Add migration SQL** to the migrations dict:
   ```python
   DISCORD_MIGRATIONS[3] = [
       "ALTER TABLE discord_links ADD COLUMN new_field TEXT",
       "CREATE INDEX IF NOT EXISTS idx_new_field ON discord_links(new_field)",
   ]
   ```

3. **Never modify existing migrations** - only add new ones

4. **Test migrations** before committing:
   ```bash
   python db_migrations.py --check  # see pending
   python db_migrations.py          # run migrations
   ```

### SQLite ALTER TABLE Limitations

SQLite has limited ALTER TABLE support:
- **CAN**: Add columns, rename columns/tables
- **CANNOT**: Drop columns, change column types, add constraints

For complex schema changes, create a new table and migrate data:
```python
MIGRATIONS[N] = [
    "CREATE TABLE new_table AS SELECT ... FROM old_table",
    "DROP TABLE old_table",
    "ALTER TABLE new_table RENAME TO old_table",
]
```

### Migration Files Location

- `db_migrations.py` - All migrations and version tracking
- Database classes auto-import and run migrations on init

### CLI Commands

```bash
# Check migration status
python db_migrations.py --check

# Run pending migrations
python db_migrations.py

# Specify custom paths
python db_migrations.py --discord path/to/discord.db --ai-news path/to/ai.db
```

## Project Structure

```
twitter_influencer/
├── ai_news_scraper.py      # Twitter/X + web article scraping
├── discord_link_scraper.py # Discord link extraction + content scraping
├── db_migrations.py        # Schema migrations (CRITICAL)
├── linkedin_autopilot.py   # LinkedIn automation
├── output_data/            # Database files
│   ├── ai_news.db
│   └── discord_links.db
└── pyproject.toml          # Dependencies
```

## Key Dependencies

- `sqlite-vec` - Vector similarity search extension
- `sentence-transformers` - Embedding generation (all-MiniLM-L6-v2)
- `youtube-transcript-api` - YouTube transcript extraction
- `beautifulsoup4` - HTML parsing
- `dateparser` - Publication date extraction
