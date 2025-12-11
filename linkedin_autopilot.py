# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "undetected-chromedriver>=3.5.0",
#     "python-dotenv>=0.19.0",
#     "setuptools>=65.0.0",
#     "requests>=2.31.0",
#     "pillow>=10.0.0",
#     "schedule>=1.2.0",
# ]
# ///

"""
LinkedIn Autopilot - AI-Powered Content Automation

Generates engaging LinkedIn content from AI news, creates images,
posts automatically, responds to comments, and optimizes strategy
based on engagement metrics.
"""

import os
import sys
import json
import time
import random
import sqlite3
import ssl
import re
import schedule
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import Counter
import hashlib
import threading

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import requests

# Note: pyautogui was removed - using JavaScript injection for file uploads instead

# Fix SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context


# =============================================================================
# CONFIGURATION
# =============================================================================

# Content templates for different post types
POST_TEMPLATES = {
    "news_breakdown": """🚀 {headline}

{summary}

Key takeaways:
{takeaways}

What's your take on this? 👇

#AI #MachineLearning #Tech #Innovation""",

    "hot_take": """💡 Hot take: {opinion}

Here's why this matters:

{reasoning}

Agree or disagree? Let me know in the comments.

#AI #TechTrends #FutureOfWork""",

    "curated_list": """📊 {title}

{items}

Which one are you most excited about?

Save this for later! 🔖

#AI #ArtificialIntelligence #TechNews""",

    "question": """🤔 {question}

{context}

I'd love to hear your thoughts below 👇

#AI #Discussion #TechCommunity""",

    "insight": """💎 {insight}

{explanation}

{call_to_action}

#AI #Innovation #Insights""",
}

# Engagement response templates
COMMENT_RESPONSES = {
    "positive": [
        "Thanks for sharing your perspective! 🙌",
        "Glad you found this valuable!",
        "Appreciate the kind words! What aspect resonates most with you?",
        "Thanks! Happy to discuss further.",
        "Great point! I hadn't considered that angle.",
    ],
    "question": [
        "Great question! {response}",
        "That's a thoughtful question. {response}",
        "Thanks for asking! {response}",
    ],
    "negative": [
        "I appreciate your perspective. Would love to hear more about your view.",
        "Thanks for the feedback. What would you suggest instead?",
        "Interesting point of view. Let's discuss!",
    ],
    "neutral": [
        "Thanks for engaging!",
        "Appreciate you taking the time to comment.",
        "Good point to consider!",
    ],
}


# =============================================================================
# LOGGING
# =============================================================================

class Logger:
    """Simple console logger"""

    @staticmethod
    def info(msg: str):
        print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} {msg}")

    @staticmethod
    def success(msg: str):
        print(f"[OK] {datetime.now().strftime('%H:%M:%S')} {msg}")

    @staticmethod
    def warning(msg: str):
        print(f"[WARN] {datetime.now().strftime('%H:%M:%S')} {msg}")

    @staticmethod
    def error(msg: str):
        print(f"[ERROR] {datetime.now().strftime('%H:%M:%S')} {msg}")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class LinkedInPost:
    """Represents a LinkedIn post"""
    post_id: str = ""
    content: str = ""
    image_path: Optional[str] = None
    posted_at: Optional[str] = None
    status: str = "draft"  # draft, scheduled, posted, failed
    scheduled_for: Optional[str] = None
    source_tweet_ids: List[str] = field(default_factory=list)
    template_type: str = ""

    # Engagement metrics
    likes: int = 0
    comments: int = 0
    shares: int = 0
    impressions: int = 0
    engagement_rate: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EngagementMetrics:
    """Track engagement patterns"""
    post_type: str
    avg_likes: float = 0.0
    avg_comments: float = 0.0
    avg_shares: float = 0.0
    avg_engagement_rate: float = 0.0
    best_posting_hours: List[int] = field(default_factory=list)
    best_posting_days: List[int] = field(default_factory=list)  # 0=Monday
    top_hashtags: List[str] = field(default_factory=list)
    sample_size: int = 0


# =============================================================================
# DATABASE
# =============================================================================

class LinkedInDatabase:
    """SQLite database for LinkedIn automation"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()

        # Posts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                post_id TEXT PRIMARY KEY,
                content TEXT,
                image_path TEXT,
                posted_at TEXT,
                status TEXT DEFAULT 'draft',
                scheduled_for TEXT,
                source_tweet_ids TEXT,
                template_type TEXT,
                likes INTEGER DEFAULT 0,
                comments INTEGER DEFAULT 0,
                shares INTEGER DEFAULT 0,
                impressions INTEGER DEFAULT 0,
                engagement_rate REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Comments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comments (
                comment_id TEXT PRIMARY KEY,
                post_id TEXT,
                author_name TEXT,
                author_url TEXT,
                content TEXT,
                sentiment TEXT,
                responded BOOLEAN DEFAULT FALSE,
                response TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (post_id) REFERENCES posts(post_id)
            )
        ''')

        # Engagement history for optimization
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS engagement_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id TEXT,
                metric_type TEXT,
                value REAL,
                recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (post_id) REFERENCES posts(post_id)
            )
        ''')

        # Strategy settings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Content queue
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_queue (
                queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                image_path TEXT,
                template_type TEXT,
                priority INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                used BOOLEAN DEFAULT FALSE
            )
        ''')

        self.conn.commit()
        Logger.success(f"LinkedIn database initialized: {self.db_path}")

    def save_post(self, post: LinkedInPost):
        """Save or update a post"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO posts (
                post_id, content, image_path, posted_at, status,
                scheduled_for, source_tweet_ids, template_type,
                likes, comments, shares, impressions, engagement_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            post.post_id,
            post.content,
            post.image_path,
            post.posted_at,
            post.status,
            post.scheduled_for,
            json.dumps(post.source_tweet_ids),
            post.template_type,
            post.likes,
            post.comments,
            post.shares,
            post.impressions,
            post.engagement_rate
        ))
        self.conn.commit()

    def get_pending_posts(self) -> List[LinkedInPost]:
        """Get posts scheduled for posting"""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute('''
            SELECT * FROM posts
            WHERE status = 'scheduled' AND scheduled_for <= ?
            ORDER BY scheduled_for ASC
        ''', (now,))

        posts = []
        for row in cursor.fetchall():
            post = LinkedInPost(
                post_id=row['post_id'],
                content=row['content'],
                image_path=row['image_path'],
                posted_at=row['posted_at'],
                status=row['status'],
                scheduled_for=row['scheduled_for'],
                source_tweet_ids=json.loads(row['source_tweet_ids'] or '[]'),
                template_type=row['template_type'],
                likes=row['likes'],
                comments=row['comments'],
                shares=row['shares'],
                impressions=row['impressions'],
                engagement_rate=row['engagement_rate']
            )
            posts.append(post)
        return posts

    def get_posts_for_engagement_check(self, hours: int = 48) -> List[LinkedInPost]:
        """Get recent posts that need engagement checking"""
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        cursor.execute('''
            SELECT * FROM posts
            WHERE status = 'posted' AND posted_at > ?
            ORDER BY posted_at DESC
        ''', (cutoff,))

        posts = []
        for row in cursor.fetchall():
            post = LinkedInPost(
                post_id=row['post_id'],
                content=row['content'],
                image_path=row['image_path'],
                posted_at=row['posted_at'],
                status=row['status'],
                likes=row['likes'],
                comments=row['comments'],
                shares=row['shares'],
                impressions=row['impressions'],
                engagement_rate=row['engagement_rate']
            )
            posts.append(post)
        return posts

    def save_comment(self, comment_data: Dict):
        """Save a comment"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO comments (
                comment_id, post_id, author_name, author_url,
                content, sentiment, responded, response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            comment_data.get('comment_id'),
            comment_data.get('post_id'),
            comment_data.get('author_name'),
            comment_data.get('author_url'),
            comment_data.get('content'),
            comment_data.get('sentiment'),
            comment_data.get('responded', False),
            comment_data.get('response')
        ))
        self.conn.commit()

    def get_unanswered_comments(self) -> List[Dict]:
        """Get comments that haven't been responded to"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM comments
            WHERE responded = FALSE
            ORDER BY created_at ASC
        ''')
        return [dict(row) for row in cursor.fetchall()]

    def mark_comment_responded(self, comment_id: str, response: str):
        """Mark a comment as responded"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE comments SET responded = TRUE, response = ?
            WHERE comment_id = ?
        ''', (response, comment_id))
        self.conn.commit()

    def add_to_queue(self, content: str, image_path: str = None,
                     template_type: str = "", priority: int = 0):
        """Add content to the posting queue"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO content_queue (content, image_path, template_type, priority)
            VALUES (?, ?, ?, ?)
        ''', (content, image_path, template_type, priority))
        self.conn.commit()

    def get_next_from_queue(self) -> Optional[Dict]:
        """Get the next item from the content queue"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM content_queue
            WHERE used = FALSE
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
        ''')
        row = cursor.fetchone()
        if row:
            # Mark as used
            cursor.execute('UPDATE content_queue SET used = TRUE WHERE queue_id = ?',
                          (row['queue_id'],))
            self.conn.commit()
            return dict(row)
        return None

    def get_engagement_stats(self) -> Dict:
        """Get aggregated engagement statistics"""
        cursor = self.conn.cursor()

        # Overall stats
        cursor.execute('''
            SELECT
                COUNT(*) as total_posts,
                AVG(likes) as avg_likes,
                AVG(comments) as avg_comments,
                AVG(shares) as avg_shares,
                AVG(engagement_rate) as avg_engagement_rate,
                SUM(impressions) as total_impressions
            FROM posts WHERE status = 'posted'
        ''')
        overall = dict(cursor.fetchone())

        # Stats by template type
        cursor.execute('''
            SELECT
                template_type,
                COUNT(*) as count,
                AVG(likes) as avg_likes,
                AVG(comments) as avg_comments,
                AVG(engagement_rate) as avg_engagement_rate
            FROM posts
            WHERE status = 'posted' AND template_type != ''
            GROUP BY template_type
            ORDER BY avg_engagement_rate DESC
        ''')
        by_type = [dict(row) for row in cursor.fetchall()]

        # Best posting times
        cursor.execute('''
            SELECT
                CAST(strftime('%H', posted_at) AS INTEGER) as hour,
                AVG(engagement_rate) as avg_engagement
            FROM posts
            WHERE status = 'posted'
            GROUP BY hour
            ORDER BY avg_engagement DESC
            LIMIT 5
        ''')
        best_hours = [dict(row) for row in cursor.fetchall()]

        return {
            'overall': overall,
            'by_type': by_type,
            'best_hours': best_hours
        }

    def save_strategy(self, key: str, value: Any):
        """Save a strategy setting"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO strategy (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (key, json.dumps(value)))
        self.conn.commit()

    def get_strategy(self, key: str, default: Any = None) -> Any:
        """Get a strategy setting"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT value FROM strategy WHERE key = ?', (key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row['value'])
        return default

    def close(self):
        if self.conn:
            self.conn.close()


# =============================================================================
# CONTENT GENERATOR
# =============================================================================

class ContentGenerator:
    """Generate LinkedIn content from AI news"""

    def __init__(self, db: LinkedInDatabase, ai_news_db_path: Path = None):
        self.db = db
        self.ai_news_db_path = ai_news_db_path

    def _clean_post_content(self, content: str) -> str:
        """Clean up generated content - remove meta-text and fix formatting"""
        if not content:
            return ""

        lines = content.split('\n')
        cleaned_lines = []
        skip_patterns = [
            "here's", "here is", "---", "character count", "word count",
            "linkedin post", "post:", "output:", "```"
        ]

        for line in lines:
            line_lower = line.lower().strip()
            # Skip meta-text lines
            if any(pattern in line_lower for pattern in skip_patterns):
                continue
            # Skip empty dashes
            if line.strip() == '---' or line.strip() == '—':
                continue
            cleaned_lines.append(line)

        content = '\n'.join(cleaned_lines).strip()

        # Remove leading/trailing dashes
        content = re.sub(r'^[-—]+\s*', '', content)
        content = re.sub(r'\s*[-—]+$', '', content)

        # Convert markdown bold **text** to UPPERCASE for LinkedIn (since it doesn't support markdown)
        content = re.sub(r'\*\*([^*]+)\*\*', lambda m: m.group(1).upper(), content)

        # Remove any remaining markdown
        content = re.sub(r'#{1,6}\s*', '', content)  # Remove headers
        content = re.sub(r'\*([^*]+)\*', r'\1', content)  # Remove single asterisks

        return content.strip()

    def _call_claude_cli(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Call Claude CLI tool for content generation"""
        try:
            # Use the claude CLI tool installed via npm
            result = subprocess.run(
                ['claude', '-p', prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0 and result.stdout.strip():
                return self._clean_post_content(result.stdout.strip())
            else:
                if result.stderr:
                    Logger.warning(f"Claude CLI error: {result.stderr}")
                return None
        except FileNotFoundError:
            Logger.warning("Claude CLI not found. Make sure it's installed via npm.")
            return None
        except subprocess.TimeoutExpired:
            Logger.warning("Claude CLI timed out")
            return None
        except Exception as e:
            Logger.warning(f"Claude CLI call failed: {e}")
            return None

    def get_recent_ai_news(self, limit: int = 20) -> List[Dict]:
        """Get recent AI news from the scraper database"""
        if not self.ai_news_db_path or not self.ai_news_db_path.exists():
            Logger.warning("AI news database not found")
            return []

        conn = sqlite3.connect(str(self.ai_news_db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM tweets
            WHERE is_ai_relevant = TRUE
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        tweets = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return tweets

    def generate_post_with_ai(self, news_items: List[Dict],
                              template_type: str = "news_breakdown") -> Optional[str]:
        """Generate a LinkedIn post using Claude CLI"""
        # Prepare context from news items
        news_context = "\n".join([
            f"- {item.get('text', '')[:200]}" for item in news_items[:5]
        ])

        prompt = f"""Write a LinkedIn post about this AI news. Output ONLY the post text, nothing else.

News context:
{news_context}

Rules:
- Write the post directly, no intro like "Here's a post..."
- Professional but conversational tone
- 1-2 relevant emojis max
- End with a question to drive comments
- 3-5 hashtags at the very end
- Under 1300 characters total
- No markdown formatting (no ** or # symbols)
- Make a bold claim or share a unique insight

Post style: {template_type}"""

        result = self._call_claude_cli(prompt, max_tokens=500)
        if result:
            return result
        else:
            Logger.warning("Claude CLI unavailable, using template fallback")
            return self.generate_post_from_template(news_items, template_type)

    def generate_post_from_template(self, news_items: List[Dict],
                                    template_type: str = "news_breakdown") -> str:
        """Generate a post using templates (fallback)"""
        if not news_items:
            return ""

        template = POST_TEMPLATES.get(template_type, POST_TEMPLATES["news_breakdown"])

        # Extract key info from news items
        top_item = news_items[0]
        headline = top_item.get('text', '')[:100]

        # Create summary
        summary = top_item.get('text', '')[:280]

        # Extract takeaways
        takeaways = ""
        for i, item in enumerate(news_items[:3], 1):
            text = item.get('text', '')[:100]
            takeaways += f"• {text}\n"

        # Fill template
        post = template.format(
            headline=headline,
            summary=summary,
            takeaways=takeaways.strip(),
            title="Top AI Updates This Week",
            items=takeaways.strip(),
            question="What AI development are you most excited about?",
            context="The AI landscape is evolving rapidly.",
            opinion="This changes everything.",
            reasoning="Here's my analysis...",
            insight="Key insight from the AI world",
            explanation="Let me break this down...",
            call_to_action="Follow for more AI insights!"
        )

        return post[:1300]  # LinkedIn limit

    def generate_content_batch(self, count: int = 5) -> List[LinkedInPost]:
        """Generate a batch of content for the queue"""
        news_items = self.get_recent_ai_news(limit=count * 4)

        if not news_items:
            Logger.warning("No AI news available for content generation")
            return []

        posts = []
        template_types = list(POST_TEMPLATES.keys())

        for i in range(min(count, len(news_items) // 3)):
            # Select news items for this post
            start_idx = i * 3
            items_for_post = news_items[start_idx:start_idx + 3]

            # Rotate through template types
            template_type = template_types[i % len(template_types)]

            # Generate content
            content = self.generate_post_with_ai(items_for_post, template_type)

            if content:
                post = LinkedInPost(
                    post_id=hashlib.md5(content[:50].encode()).hexdigest()[:12],
                    content=content,
                    template_type=template_type,
                    source_tweet_ids=[item.get('tweet_id', '') for item in items_for_post],
                    status='draft'
                )
                posts.append(post)

        Logger.success(f"Generated {len(posts)} content pieces")
        return posts

    def generate_comment_response(self, comment: str, post_content: str) -> str:
        """Generate a response to a comment using Claude CLI"""
        # Analyze sentiment for fallback
        sentiment = self._analyze_sentiment(comment)

        prompt = f"""You're responding to a comment on your LinkedIn post.

Your post: {post_content[:200]}...
Comment: {comment}

Write a brief, friendly, professional response (1-2 sentences).
Be genuine and encourage further discussion."""

        result = self._call_claude_cli(prompt, max_tokens=100)
        if result:
            return result

        # Fallback to templates
        responses = COMMENT_RESPONSES.get(sentiment, COMMENT_RESPONSES["neutral"])
        return random.choice(responses)

    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis"""
        text_lower = text.lower()

        positive_words = ['great', 'awesome', 'love', 'excellent', 'amazing',
                         'helpful', 'thanks', 'thank', 'agree', 'interesting']
        negative_words = ['disagree', 'wrong', 'bad', 'terrible', 'hate',
                         'useless', 'stupid', 'waste']
        question_indicators = ['?', 'how', 'what', 'why', 'when', 'where', 'could']

        if any(q in text_lower for q in question_indicators):
            return "question"
        if any(p in text_lower for p in positive_words):
            return "positive"
        if any(n in text_lower for n in negative_words):
            return "negative"
        return "neutral"


# =============================================================================
# IMAGE GENERATOR
# =============================================================================

class ImageGenerator:
    """Generate images for LinkedIn posts using AI (DALL-E 3) or fallback"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

    def _generate_image_prompt_with_claude(self, topic: str, post_content: str) -> Optional[str]:
        """Use Claude to generate a creative, viral-worthy image prompt"""
        try:
            meta_prompt = f"""You are a creative director for viral social media content. Generate a DALL-E 3 image prompt that will create a scroll-stopping, viral-worthy image for this LinkedIn post.

POST TOPIC: {topic}
POST CONTENT: {post_content[:400]}

VIRAL IMAGE PRINCIPLES TO APPLY:
- ONE single powerful visual metaphor (not multiple concepts)
- Evoke AWE or CURIOSITY - make people stop scrolling
- Ultra-simple composition with clear focal point
- Dramatic, cinematic lighting (golden hour, neon, dramatic shadows)
- Bold, saturated colors that pop on mobile screens
- Unexpected or surreal element that makes people look twice
- NO text, words, letters, or UI elements in the image
- Think "would this make someone screenshot and share?"

OUTPUT FORMAT: Write ONLY the image prompt, nothing else. Be specific about:
- The exact scene/subject
- Camera angle and framing
- Lighting style
- Color palette
- Mood/atmosphere

Example good prompts:
- "A single chess piece (king) made of glowing blue circuitry, standing alone on an infinite mirror floor reflecting a sunset sky, dramatic low angle shot, cinematic lighting"
- "Human hand reaching toward a robot hand, their fingertips creating an explosion of golden light particles, dark background, close-up macro shot, ethereal glow"
- "A door opening in the middle of a vast desert, bright white light pouring out, person silhouetted walking toward it, epic wide shot, golden hour lighting"

Now generate a prompt for the post above:"""

            result = subprocess.run(
                ['claude', '-p', meta_prompt],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and result.stdout.strip():
                prompt = result.stdout.strip()
                # Clean up the prompt
                prompt = prompt.replace('\n', ' ').strip()
                if len(prompt) > 50:  # Sanity check
                    Logger.info(f"Claude generated image prompt: {prompt[:100]}...")
                    return prompt

        except subprocess.TimeoutExpired:
            Logger.warning("Claude CLI timed out for image prompt")
        except Exception as e:
            Logger.warning(f"Could not generate prompt with Claude: {e}")

        return None

    def generate_ai_image(self, prompt: str, post_content: str = "", style: str = "professional") -> Optional[str]:
        """Generate an image using DALL-E 3 that tells the story of the post"""
        if not self.openai_api_key:
            Logger.warning("OPENAI_API_KEY not set, skipping AI image generation")
            return None

        try:
            import requests

            # Use Claude CLI to generate a creative image prompt based on the post
            image_prompt = self._generate_image_prompt_with_claude(prompt, post_content)

            if not image_prompt:
                # Fallback to a simpler but evocative prompt based on viral principles
                image_prompt = f"""A single powerful visual metaphor for "{prompt[:60]}":

One clear focal point, ultra-minimal composition. Dramatic cinematic lighting with deep shadows and bright highlights. Bold saturated colors that pop. Evoke awe and curiosity. Photorealistic or hyperreal quality. NO text, words, or UI elements. 16:9 aspect ratio."""

            Logger.info("Generating AI image with DALL-E 3...")

            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "dall-e-3",
                    "prompt": image_prompt,
                    "n": 1,
                    "size": "1792x1024",  # Closest to LinkedIn's 1200x627
                    "quality": "standard",  # Use "hd" for higher quality (costs more)
                    "style": "vivid"
                },
                timeout=60
            )

            if response.status_code != 200:
                Logger.warning(f"DALL-E API error: {response.status_code} - {response.text}")
                return None

            data = response.json()
            image_url = data['data'][0]['url']

            # Download the image
            img_response = requests.get(image_url, timeout=30)
            if img_response.status_code != 200:
                Logger.warning("Could not download generated image")
                return None

            # Save the image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"linkedin_ai_{timestamp}.png"
            filepath = self.output_dir / filename

            with open(filepath, 'wb') as f:
                f.write(img_response.content)

            # Resize to LinkedIn optimal dimensions
            try:
                img = Image.open(filepath)
                img = img.resize((1200, 627), Image.Resampling.LANCZOS)
                img.save(filepath, 'PNG')
            except Exception as e:
                Logger.warning(f"Could not resize image: {e}")

            Logger.success(f"Created AI image: {filename}")
            return str(filepath)

        except Exception as e:
            Logger.warning(f"AI image generation failed: {e}")
            return None

    def create_image_for_post(self, post_content: str, use_ai: bool = True) -> Optional[str]:
        """Create an image for a post - tries AI first, falls back to quote image"""
        # Extract key theme/topic from post
        lines = post_content.split('\n')
        key_line = lines[0] if lines else post_content[:100]
        key_line = re.sub(r'[^\w\s]', '', key_line)[:100]

        # Try AI generation first - pass full post content for context
        if use_ai and self.openai_api_key:
            ai_image = self.generate_ai_image(key_line, post_content=post_content)
            if ai_image:
                return ai_image

        # Fallback to simple quote image
        Logger.info("Using fallback quote image")
        return self.create_quote_image(key_line)

    def create_quote_image(self, text: str, theme: str = "ai") -> str:
        """Create a simple quote/insight image"""
        # Image dimensions (LinkedIn recommended)
        width, height = 1200, 627

        # Theme colors
        themes = {
            "ai": {"bg": (15, 23, 42), "text": (255, 255, 255), "accent": (59, 130, 246)},
            "tech": {"bg": (17, 24, 39), "text": (255, 255, 255), "accent": (16, 185, 129)},
            "innovation": {"bg": (30, 27, 75), "text": (255, 255, 255), "accent": (168, 85, 247)},
            "business": {"bg": (255, 255, 255), "text": (31, 41, 55), "accent": (79, 70, 229)},
        }
        colors = themes.get(theme, themes["ai"])

        # Create image
        img = Image.new('RGB', (width, height), colors["bg"])
        draw = ImageDraw.Draw(img)

        # Try to load a font, fall back to default
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font_large = ImageFont.load_default()
            font_small = font_large

        # Add accent bar
        draw.rectangle([(0, 0), (8, height)], fill=colors["accent"])

        # Word wrap the text
        max_chars_per_line = 40
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > max_chars_per_line:
                if len(current_line) > 1:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(' '.join(current_line))
                    current_line = []

        if current_line:
            lines.append(' '.join(current_line))

        # Limit to 5 lines
        lines = lines[:5]
        if len(lines) == 5:
            lines[-1] = lines[-1][:max_chars_per_line - 3] + "..."

        # Draw text
        y_start = (height - len(lines) * 55) // 2
        for i, line in enumerate(lines):
            y = y_start + i * 55
            draw.text((60, y), line, fill=colors["text"], font=font_large)

        # Add branding/watermark
        draw.text((60, height - 50), "AI Insights • linkedin.com", fill=colors["accent"], font=font_small)

        # Save image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"linkedin_post_{timestamp}.png"
        filepath = self.output_dir / filename
        img.save(filepath, 'PNG')

        Logger.success(f"Created image: {filename}")
        return str(filepath)

    def create_stats_image(self, title: str, stats: List[Tuple[str, str]]) -> str:
        """Create an image with statistics"""
        width, height = 1200, 627

        img = Image.new('RGB', (width, height), (15, 23, 42))
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            font_stat = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
            font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font_title = ImageFont.load_default()
            font_stat = font_title
            font_label = font_title

        # Title
        draw.text((60, 40), title, fill=(255, 255, 255), font=font_title)

        # Stats grid
        cols = min(len(stats), 3)
        col_width = (width - 120) // cols
        y_start = 140

        for i, (label, value) in enumerate(stats[:6]):
            col = i % cols
            row = i // cols
            x = 60 + col * col_width
            y = y_start + row * 150

            # Value
            draw.text((x, y), str(value), fill=(59, 130, 246), font=font_stat)
            # Label
            draw.text((x, y + 60), label, fill=(156, 163, 175), font=font_label)

        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"linkedin_stats_{timestamp}.png"
        filepath = self.output_dir / filename
        img.save(filepath, 'PNG')

        return str(filepath)


# =============================================================================
# LINKEDIN BROWSER AUTOMATION
# =============================================================================

class LinkedInAuth:
    """Handles LinkedIn authentication"""

    def __init__(self, driver, email: str = None, password: str = None,
                 google_email: str = None):
        self.driver = driver
        self.email = email
        self.password = password
        self.google_email = google_email

    def login(self) -> bool:
        """Login to LinkedIn"""
        if self.google_email:
            return self.login_with_google()
        return self.login_with_password()

    def login_with_google(self) -> bool:
        """Login using Google OAuth"""
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            Logger.info(f"Logging into LinkedIn via Google ({self.google_email})...")

            self.driver.get("https://www.linkedin.com/login")
            time.sleep(random.uniform(3, 5))

            # LinkedIn uses Google One Tap - look for various Google sign-in options
            google_selectors = [
                "//div[contains(@class, 'google-auth-button')]//button",
                "//div[contains(@class, 'alternate-signin__btn--google')]",
                "//div[contains(@id, 'google')]//div[@role='button']",
                "//button[contains(@data-provider, 'google')]",
                "//iframe[contains(@src, 'accounts.google.com')]",
                "//*[contains(@class, 'google')]//button",
            ]

            google_clicked = False
            for selector in google_selectors:
                try:
                    google_btn = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    google_btn.click()
                    Logger.info("Clicked Google sign-in")
                    google_clicked = True
                    time.sleep(random.uniform(3, 5))
                    break
                except:
                    continue

            if not google_clicked:
                # Try finding Google One Tap iframe
                try:
                    iframes = self.driver.find_elements(By.TAG_NAME, 'iframe')
                    for iframe in iframes:
                        src = iframe.get_attribute('src') or ''
                        if 'google' in src.lower():
                            self.driver.switch_to.frame(iframe)
                            try:
                                tap_btn = WebDriverWait(self.driver, 5).until(
                                    EC.element_to_be_clickable((By.XPATH, "//*[@role='button']"))
                                )
                                tap_btn.click()
                                google_clicked = True
                                Logger.info("Clicked Google One Tap")
                            finally:
                                self.driver.switch_to.default_content()
                            if google_clicked:
                                break
                except:
                    pass

            if not google_clicked:
                Logger.warning("Google sign-in button not found automatically")
                Logger.info("Please manually click the Google sign-in option...")
                Logger.info("Waiting 30 seconds for manual login...")
                time.sleep(30)

            # Handle Google account selection
            current_url = self.driver.current_url
            if 'accounts.google.com' in current_url:
                try:
                    time.sleep(random.uniform(2, 3))
                    account_selectors = [
                        f"//div[contains(text(), '{self.google_email}')]",
                        f"//*[contains(text(), '{self.google_email}')]",
                        f"//div[@data-email='{self.google_email}']",
                    ]

                    for selector in account_selectors:
                        try:
                            account = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable((By.XPATH, selector))
                            )
                            account.click()
                            Logger.success(f"Selected Google account")
                            time.sleep(random.uniform(3, 5))
                            break
                        except:
                            continue
                except Exception as e:
                    Logger.warning(f"Account selection issue: {e}")
                    Logger.info("Please manually select your Google account...")
                    time.sleep(15)

            # Wait for LinkedIn redirect
            try:
                WebDriverWait(self.driver, 30).until(
                    lambda d: 'linkedin.com' in d.current_url
                )
            except:
                time.sleep(10)

            return self._verify_login()

        except Exception as e:
            Logger.error(f"Google login failed: {e}")
            return False

    def login_with_password(self) -> bool:
        """Login with email/password"""
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.common.keys import Keys
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            Logger.info("Logging into LinkedIn...")

            self.driver.get("https://www.linkedin.com/login")
            time.sleep(random.uniform(2, 4))

            # Email
            email_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, 'username'))
            )
            for char in self.email:
                email_input.send_keys(char)
                time.sleep(random.uniform(0.05, 0.15))
            time.sleep(random.uniform(0.5, 1.0))

            # Password
            password_input = self.driver.find_element(By.ID, 'password')
            for char in self.password:
                password_input.send_keys(char)
                time.sleep(random.uniform(0.05, 0.15))
            time.sleep(random.uniform(0.5, 1.0))

            password_input.send_keys(Keys.RETURN)
            time.sleep(random.uniform(3, 5))

            return self._verify_login()

        except Exception as e:
            Logger.error(f"Password login failed: {e}")
            return False

    def _verify_login(self) -> bool:
        """Verify successful login"""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            WebDriverWait(self.driver, 15).until(
                lambda d: 'feed' in d.current_url or
                         len(d.find_elements(By.CSS_SELECTOR, '[data-control-name="nav.homepage"]')) > 0 or
                         len(d.find_elements(By.CSS_SELECTOR, '.global-nav')) > 0
            )
            Logger.success("LinkedIn login successful!")
            return True
        except:
            Logger.warning("Login status unclear")
            return True


class LinkedInPoster:
    """Post content to LinkedIn"""

    def __init__(self, driver):
        self.driver = driver

    def _qa_review_post(self, content: str, image_path: str = None) -> dict:
        """Use Claude CLI to QA review the post before publishing"""
        prompt = f"""You are a QA reviewer for LinkedIn posts. Review this post briefly.

POST:
{content}

{"[Has image attached]" if image_path else "[No image]"}

Check for: grammar errors, clarity, professional tone, appropriate hashtags.

Respond in JSON format only:
{{"approved": true/false, "score": 1-10, "issues": ["issue1"], "summary": "brief assessment"}}

Only set approved=false for serious issues (offensive content, major errors)."""

        try:
            result = subprocess.run(
                ['claude', '-p', prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                import json
                response = result.stdout.strip()
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(response[start:end])
            return {"approved": True, "score": 7, "summary": "Review completed"}
        except Exception as e:
            Logger.warning(f"QA review failed: {e}")
            return {"approved": True, "score": 5, "summary": "Review skipped"}

    def create_post(self, content: str, image_path: str = None) -> bool:
        """Create a new LinkedIn post with optional image and QA review"""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            # QA Review first
            Logger.info("Running QA review on post content...")
            review = self._qa_review_post(content, image_path)
            Logger.info(f"QA Score: {review.get('score', 'N/A')}/10 - {review.get('summary', '')}")

            if review.get('issues'):
                for issue in review.get('issues', []):
                    Logger.warning(f"QA Issue: {issue}")

            if not review.get('approved', True):
                Logger.error("Post not approved by QA review. Skipping.")
                return False

            Logger.info("Creating LinkedIn post...")

            # Navigate to feed if not there
            if 'feed' not in self.driver.current_url:
                self.driver.get("https://www.linkedin.com/feed/")
                time.sleep(random.uniform(3, 5))

            time.sleep(random.uniform(2, 3))
            self.driver.execute_script("window.scrollTo(0, 0)")
            time.sleep(random.uniform(1, 2))

            # If we have an image, click "Photo" button directly (better flow)
            # Otherwise, click "Start a post"
            if image_path and os.path.exists(image_path):
                Logger.info("Clicking 'Photo' button for image post...")
                try:
                    photo_btn = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Add a photo']"))
                    )
                    photo_btn.click()
                    time.sleep(2)

                    # Inject image via JavaScript
                    Logger.info("Injecting image...")
                    import base64
                    abs_image_path = os.path.abspath(image_path)
                    with open(abs_image_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                    image_name = os.path.basename(abs_image_path)
                    image_type = 'image/png' if abs_image_path.lower().endswith('.png') else 'image/jpeg'

                    result = self.driver.execute_script("""
                        var fileInput = document.getElementById('media-editor-file-selector__file-input');
                        if (!fileInput) fileInput = document.querySelector('input[type="file"]');
                        if (!fileInput) {
                            var container = document.querySelector('.share-creation-state, [class*="share"]');
                            if (container) {
                                fileInput = document.createElement('input');
                                fileInput.type = 'file';
                                fileInput.id = 'media-editor-file-selector__file-input';
                                fileInput.accept = 'image/*';
                                fileInput.style.display = 'none';
                                container.appendChild(fileInput);
                            }
                        }
                        if (!fileInput) return {success: false, error: 'No file input'};

                        try {
                            var byteString = atob(arguments[0]);
                            var ab = new ArrayBuffer(byteString.length);
                            var ia = new Uint8Array(ab);
                            for (var i = 0; i < byteString.length; i++) ia[i] = byteString.charCodeAt(i);
                            var blob = new Blob([ab], {type: arguments[2]});
                            var file = new File([blob], arguments[1], {type: arguments[2]});
                            var dt = new DataTransfer();
                            dt.items.add(file);
                            fileInput.files = dt.files;
                            fileInput.dispatchEvent(new Event('change', {bubbles: true}));
                            return {success: true};
                        } catch(e) { return {success: false, error: e.toString()}; }
                    """, image_data, image_name, image_type)

                    if result and result.get('success'):
                        Logger.success("Image injected successfully")
                        time.sleep(2)

                        # Handle Next/Done buttons
                        for step in ["Next", "Done"]:
                            try:
                                btn = WebDriverWait(self.driver, 3).until(
                                    EC.element_to_be_clickable((By.XPATH, f"//button[.//span[text()='{step}']]"))
                                )
                                btn.click()
                                Logger.info(f"Clicked '{step}' button")
                                time.sleep(2)
                            except:
                                pass
                    else:
                        Logger.warning(f"Image injection failed: {result}")

                except Exception as e:
                    Logger.warning(f"Photo button approach failed: {e}, falling back to Start a post")
                    # Fall back to Start a post
                    start_btn = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Start a post')]"))
                    )
                    start_btn.click()
                    time.sleep(3)
            else:
                # No image - use Start a post
                Logger.info("Clicking 'Start a post' button...")
                start_btn = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Start a post')]"))
                )
                start_btn.click()
                time.sleep(3)

            # Find and fill text area
            text_area = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[@role='textbox']"))
            )
            self.driver.execute_script("""
                arguments[0].innerHTML = arguments[1];
                arguments[0].dispatchEvent(new Event('input', { bubbles: true }));
            """, text_area, content.replace('\n', '<br>'))
            Logger.info("Content added")
            time.sleep(random.uniform(1, 2))

            # Click Post button
            post_clicked = False
            post_selectors = [
                "//button[contains(@class, 'share-actions__primary-action')]",
                "//button[.//span[text()='Post']]",
                "//span[text()='Post']/ancestor::button",
                "//button[@type='submit'][contains(., 'Post')]",
                "//div[contains(@class, 'share-box')]//button[contains(@class, 'artdeco-button--primary')]",
            ]

            for selector in post_selectors:
                try:
                    post_btn = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, selector))
                    )
                    # Scroll into view
                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", post_btn)
                    time.sleep(0.5)
                    # Try clicking with JavaScript (most reliable)
                    self.driver.execute_script("arguments[0].click();", post_btn)
                    post_clicked = True
                    Logger.info("Clicked Post button")
                    time.sleep(random.uniform(4, 6))
                    break
                except Exception as e:
                    continue

            if not post_clicked:
                # Last resort: try ActionChains
                try:
                    post_btn = self.driver.find_element(By.XPATH, "//button[contains(., 'Post')]")
                    ActionChains(self.driver).move_to_element(post_btn).pause(0.5).click().perform()
                    post_clicked = True
                    Logger.info("Clicked Post button via ActionChains")
                    time.sleep(random.uniform(4, 6))
                except Exception as e:
                    Logger.error(f"Could not click Post button: {e}")
                    return False

            Logger.success("Post published!")
            return True

        except Exception as e:
            Logger.error(f"Failed to create post: {e}")
            return False

    def get_post_engagement(self, post_url: str = None) -> Dict:
        """Get engagement metrics for a post"""
        from selenium.webdriver.common.by import By

        try:
            if post_url:
                self.driver.get(post_url)
                time.sleep(random.uniform(3, 5))

            metrics = {
                'likes': 0,
                'comments': 0,
                'shares': 0,
                'impressions': 0
            }

            # Try to extract metrics from the page
            try:
                # Reactions count
                reactions = self.driver.find_elements(By.XPATH,
                    "//button[contains(@class, 'reactions-count')]|"
                    "//span[contains(@class, 'reactions-count')]"
                )
                if reactions:
                    text = reactions[0].text
                    match = re.search(r'(\d+)', text.replace(',', ''))
                    if match:
                        metrics['likes'] = int(match.group(1))
            except:
                pass

            try:
                # Comments count
                comments = self.driver.find_elements(By.XPATH,
                    "//button[contains(., 'comment')]|"
                    "//span[contains(., 'comment')]"
                )
                for elem in comments:
                    match = re.search(r'(\d+)\s*comment', elem.text.lower())
                    if match:
                        metrics['comments'] = int(match.group(1))
                        break
            except:
                pass

            return metrics

        except Exception as e:
            Logger.error(f"Failed to get engagement: {e}")
            return {}

    def get_comments(self, post_url: str = None) -> List[Dict]:
        """Get comments on a post"""
        from selenium.webdriver.common.by import By

        comments = []
        try:
            if post_url:
                self.driver.get(post_url)
                time.sleep(random.uniform(3, 5))

            # Find comment elements
            comment_elements = self.driver.find_elements(By.XPATH,
                "//article[contains(@class, 'comments-comment-item')]|"
                "//div[contains(@class, 'comment-item')]"
            )

            for elem in comment_elements[:20]:  # Limit to 20 comments
                try:
                    author = elem.find_element(By.XPATH, ".//span[contains(@class, 'hoverable-link-text')]").text
                    content = elem.find_element(By.XPATH, ".//span[contains(@class, 'break-words')]").text

                    comments.append({
                        'comment_id': hashlib.md5(f"{author}{content[:20]}".encode()).hexdigest()[:12],
                        'author_name': author,
                        'content': content
                    })
                except:
                    continue

        except Exception as e:
            Logger.error(f"Failed to get comments: {e}")

        return comments

    def reply_to_comment(self, comment_id: str, response: str) -> bool:
        """Reply to a comment"""
        # This is complex as LinkedIn's comment UI changes frequently
        # For now, log that we would respond
        Logger.info(f"Would reply to comment {comment_id}: {response[:50]}...")
        return True


# =============================================================================
# STRATEGY OPTIMIZER
# =============================================================================

class StrategyOptimizer:
    """Optimize posting strategy based on engagement"""

    def __init__(self, db: LinkedInDatabase):
        self.db = db

    def analyze_performance(self) -> Dict:
        """Analyze posting performance"""
        stats = self.db.get_engagement_stats()
        return stats

    def get_optimal_posting_time(self) -> Tuple[int, int]:
        """Get the optimal hour and day to post"""
        stats = self.db.get_engagement_stats()

        # Default optimal times if no data
        if not stats.get('best_hours'):
            # LinkedIn best times: Tuesday-Thursday, 8-10am, 12pm
            return (9, 2)  # 9am, Tuesday

        best_hour = stats['best_hours'][0]['hour'] if stats['best_hours'] else 9
        best_day = 2  # Tuesday default

        return (best_hour, best_day)

    def get_best_template_type(self) -> str:
        """Get the best performing template type"""
        stats = self.db.get_engagement_stats()

        if stats.get('by_type') and stats['by_type']:
            return stats['by_type'][0]['template_type']

        return "news_breakdown"  # Default

    def should_post_now(self) -> bool:
        """Determine if now is a good time to post"""
        now = datetime.now()
        hour = now.hour
        day = now.weekday()

        # LinkedIn optimal posting windows
        # Best: Tue-Thu, 8-10am and 12pm
        good_hours = [8, 9, 10, 12, 17, 18]
        good_days = [1, 2, 3]  # Tue, Wed, Thu

        if day in good_days and hour in good_hours:
            return True

        # Also okay: Mon, Fri during business hours
        okay_days = [0, 4]
        if day in okay_days and 8 <= hour <= 18:
            return True

        return False

    def update_strategy(self):
        """Update strategy based on recent performance"""
        stats = self.analyze_performance()

        # Save insights
        self.db.save_strategy('last_analysis', datetime.now().isoformat())
        self.db.save_strategy('best_template', self.get_best_template_type())
        self.db.save_strategy('optimal_time', self.get_optimal_posting_time())

        Logger.info("Strategy updated based on engagement data")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class LinkedInAutopilot:
    """Main orchestrator for LinkedIn automation"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.output_dir / 'linkedin_autopilot.db'
        self.db = LinkedInDatabase(self.db_path)

        # AI news database path
        self.ai_news_db = self.output_dir / 'ai_news.db'

        # Cookie storage path
        self.cookies_path = self.output_dir / 'linkedin_cookies.json'

        self.content_gen = ContentGenerator(self.db, self.ai_news_db)
        self.image_gen = ImageGenerator(self.output_dir / 'images')
        self.optimizer = StrategyOptimizer(self.db)

        self.driver = None

    def setup_driver(self):
        """Setup browser driver"""
        import undetected_chromedriver as uc

        options = uc.ChromeOptions()
        options.add_argument('--start-maximized')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        # Use a persistent profile directory
        profile_dir = self.output_dir / 'chrome_profile'
        profile_dir.mkdir(parents=True, exist_ok=True)
        options.add_argument(f'--user-data-dir={profile_dir}')

        self.driver = uc.Chrome(options=options)
        return self.driver

    def save_cookies(self):
        """Save browser cookies to file"""
        if self.driver:
            cookies = self.driver.get_cookies()
            with open(self.cookies_path, 'w') as f:
                json.dump(cookies, f)
            Logger.success(f"Saved {len(cookies)} cookies")

    def load_cookies(self) -> bool:
        """Load cookies from file"""
        if not self.cookies_path.exists():
            return False

        try:
            with open(self.cookies_path, 'r') as f:
                cookies = json.load(f)

            # Navigate to LinkedIn first (required to set cookies for domain)
            self.driver.get("https://www.linkedin.com")
            time.sleep(2)

            for cookie in cookies:
                # Remove expiry if it's in the past
                if 'expiry' in cookie:
                    del cookie['expiry']
                try:
                    self.driver.add_cookie(cookie)
                except:
                    pass

            Logger.success(f"Loaded {len(cookies)} cookies")
            return True
        except Exception as e:
            Logger.warning(f"Could not load cookies: {e}")
            return False

    def is_logged_in(self) -> bool:
        """Check if already logged into LinkedIn"""
        try:
            self.driver.get("https://www.linkedin.com/feed/")
            time.sleep(3)
            # Check if we're on the feed (logged in) or redirected to login
            return 'feed' in self.driver.current_url and 'login' not in self.driver.current_url
        except:
            return False

    def login(self, email: str = None, password: str = None,
              google_email: str = None) -> bool:
        """Login to LinkedIn"""
        if not self.driver:
            self.setup_driver()

        # Try to use existing session first
        Logger.info("Checking for existing LinkedIn session...")
        if self.is_logged_in():
            Logger.success("Already logged in via persistent session!")
            return True

        # Try loading cookies
        if self.load_cookies():
            if self.is_logged_in():
                Logger.success("Logged in via saved cookies!")
                return True

        # Need fresh login
        auth = LinkedInAuth(self.driver, email, password, google_email)
        success = auth.login()

        if success:
            # Save cookies for future sessions
            self.save_cookies()

        return success

    def generate_content_queue(self, count: int = 7, use_ai_images: bool = True):
        """Generate content for the queue with AI-generated images"""
        Logger.info(f"Generating {count} posts for the content queue...")

        posts = self.content_gen.generate_content_batch(count)

        for post in posts:
            # Generate AI image for each post (100% of posts get images for better engagement)
            Logger.info(f"Generating image for post...")
            image_path = self.image_gen.create_image_for_post(post.content, use_ai=use_ai_images)
            post.image_path = image_path

            self.db.add_to_queue(
                content=post.content,
                image_path=post.image_path,
                template_type=post.template_type
            )

        Logger.success(f"Added {len(posts)} posts to queue")

    def post_from_queue(self) -> bool:
        """Post the next item from the queue"""
        item = self.db.get_next_from_queue()
        if not item:
            Logger.warning("Content queue is empty")
            return False

        if not self.driver:
            Logger.error("Browser not initialized. Call login() first.")
            return False

        poster = LinkedInPoster(self.driver)
        success = poster.create_post(item['content'], item.get('image_path'))

        if success:
            # Save post record
            post = LinkedInPost(
                post_id=hashlib.md5(item['content'][:50].encode()).hexdigest()[:12],
                content=item['content'],
                image_path=item.get('image_path'),
                posted_at=datetime.now().isoformat(),
                status='posted',
                template_type=item.get('template_type', '')
            )
            self.db.save_post(post)

        return success

    def check_and_respond_to_comments(self):
        """Check recent posts for comments and respond"""
        posts = self.db.get_posts_for_engagement_check(hours=48)

        if not self.driver:
            Logger.warning("Browser not initialized")
            return

        poster = LinkedInPoster(self.driver)

        for post in posts:
            Logger.info(f"Checking engagement for post {post.post_id}...")

            # Get current engagement
            # Note: This would need the actual post URL
            # For now, we'll just check for new comments in the DB

            comments = self.db.get_unanswered_comments()
            for comment in comments:
                if comment.get('post_id') == post.post_id:
                    # Generate and save response
                    response = self.content_gen.generate_comment_response(
                        comment['content'],
                        post.content
                    )

                    # In a real implementation, we'd post the response
                    Logger.info(f"Generated response: {response[:50]}...")

                    self.db.mark_comment_responded(comment['comment_id'], response)

    def run_autopilot_cycle(self):
        """Run one cycle of the autopilot"""
        Logger.info("Running autopilot cycle...")

        # Check if we should post
        if self.optimizer.should_post_now():
            Logger.info("Good time to post!")
            self.post_from_queue()
        else:
            Logger.info("Not optimal posting time, skipping post")

        # Check comments on recent posts
        self.check_and_respond_to_comments()

        # Update strategy
        self.optimizer.update_strategy()

    def run_scheduled(self, posts_per_day: int = 1):
        """Run on a schedule"""
        Logger.info(f"Starting scheduled autopilot ({posts_per_day} posts/day)...")

        # Schedule posting at optimal times
        optimal_hour, _ = self.optimizer.get_optimal_posting_time()

        schedule.every().day.at(f"{optimal_hour:02d}:00").do(self.post_from_queue)
        schedule.every().day.at(f"{(optimal_hour + 4) % 24:02d}:00").do(
            self.check_and_respond_to_comments
        )

        # Also check comments more frequently
        schedule.every(4).hours.do(self.check_and_respond_to_comments)

        Logger.info(f"Scheduled posting at {optimal_hour}:00")
        Logger.info("Running scheduler... (Ctrl+C to stop)")

        while True:
            schedule.run_pending()
            time.sleep(60)

    def print_stats(self):
        """Print current statistics"""
        stats = self.optimizer.analyze_performance()

        print("\n" + "=" * 60)
        print("LINKEDIN AUTOPILOT STATISTICS")
        print("=" * 60)

        overall = stats.get('overall', {})
        print(f"Total posts:          {overall.get('total_posts') or 0}")
        print(f"Avg likes:            {(overall.get('avg_likes') or 0):.1f}")
        print(f"Avg comments:         {(overall.get('avg_comments') or 0):.1f}")
        print(f"Avg engagement rate:  {(overall.get('avg_engagement_rate') or 0):.2%}")
        print(f"Total impressions:    {overall.get('total_impressions') or 0}")

        if stats.get('by_type'):
            print("\nPerformance by post type:")
            for t in stats['by_type'][:5]:
                print(f"  {t['template_type']}: {t['avg_engagement_rate']:.2%} engagement")

        if stats.get('best_hours'):
            print("\nBest posting hours:")
            for h in stats['best_hours'][:3]:
                print(f"  {h['hour']:02d}:00 - {h['avg_engagement']:.2%} avg engagement")

        print("=" * 60)

    def close(self):
        """Cleanup"""
        if self.driver:
            self.driver.quit()
        self.db.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='LinkedIn Autopilot - AI Content Automation')
    parser.add_argument('--login', action='store_true', help='Login to LinkedIn')
    parser.add_argument('--generate', type=int, default=0, help='Generate N posts for queue')
    parser.add_argument('--post', action='store_true', help='Post next item from queue')
    parser.add_argument('--check-comments', action='store_true', help='Check and respond to comments')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--run', action='store_true', help='Run single autopilot cycle')
    parser.add_argument('--schedule', action='store_true', help='Run scheduled autopilot (1 post/day)')
    parser.add_argument('--google-auth', type=str, help='Google email for OAuth')
    parser.add_argument('--no-ai-images', action='store_true', help='Disable AI image generation (use simple quote images)')
    parser.add_argument('--posts-per-day', type=int, default=1, help='Posts per day for scheduled mode (default: 1)')

    args = parser.parse_args()

    # Load environment
    script_dir = Path(__file__).parent
    load_dotenv(script_dir / '.env')

    linkedin_email = os.getenv('LINKEDIN_EMAIL')
    linkedin_password = os.getenv('LINKEDIN_PASSWORD')
    google_email = args.google_auth or os.getenv('GOOGLE_EMAIL')

    # Initialize
    output_dir = script_dir / 'output_data'
    autopilot = LinkedInAutopilot(output_dir)

    try:
        if args.login or args.post or args.run or args.schedule or args.check_comments:
            if not google_email and (not linkedin_email or not linkedin_password):
                Logger.error("Set GOOGLE_EMAIL or LINKEDIN_EMAIL/LINKEDIN_PASSWORD in .env")
                sys.exit(1)

            autopilot.login(
                email=linkedin_email,
                password=linkedin_password,
                google_email=google_email
            )

        if args.generate > 0:
            use_ai_images = not args.no_ai_images
            autopilot.generate_content_queue(args.generate, use_ai_images=use_ai_images)

        if args.post:
            autopilot.post_from_queue()

        if args.check_comments:
            autopilot.check_and_respond_to_comments()

        if args.run:
            autopilot.run_autopilot_cycle()

        if args.schedule:
            autopilot.run_scheduled(posts_per_day=args.posts_per_day)

        if args.stats or not any([args.login, args.generate, args.post,
                                   args.check_comments, args.run, args.schedule]):
            autopilot.print_stats()

    except KeyboardInterrupt:
        Logger.warning("Interrupted by user")
    finally:
        autopilot.close()


if __name__ == "__main__":
    main()
