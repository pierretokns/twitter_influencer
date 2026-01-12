# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "flask>=3.0.0",
#     "python-dotenv>=0.19.0",
# ]
# ///
"""
LinkedIn Feed Clone - Daily Winning Posts Viewer
================================================

A realistic LinkedIn-style feed showing tournament winning posts.
Features like/reaction functionality with backend persistence.

Usage:
    uv run python linkedin_feed.py
    Then open: http://localhost:5002

Features:
    - Realistic LinkedIn post UI
    - Like/react to posts (persisted to database)
    - Daily timeline of winning posts
    - Filter by date range
    - One-click publish to LinkedIn
"""

import os
import sys
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

OUTPUT_DIR = Path(__file__).parent / "output_data"
AI_NEWS_DB = OUTPUT_DIR / "ai_news.db"


def get_db():
    from db_migrations import migrate_ai_news_db
    conn = sqlite3.connect(str(AI_NEWS_DB))
    conn.row_factory = sqlite3.Row
    migrate_ai_news_db(conn)
    return conn


def get_user_id():
    """Generate a simple anonymous user ID based on IP"""
    ip = request.remote_addr or 'unknown'
    return hashlib.md5(ip.encode()).hexdigest()[:12]


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>AI Content Feed</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #f4f2ee;
            --bg-card: #ffffff;
            --border: #e0dfdc;
            --text-primary: #000000e6;
            --text-secondary: #00000099;
            --text-tertiary: #00000066;
            --linkedin-blue: #0a66c2;
            --linkedin-blue-hover: #004182;
            --reaction-bg: #f0f0f0;
            --like-red: #df3a3a;
            --celebrate-green: #44712e;
            --support-purple: #715eae;
            --love-red: #c37d16;
            --insightful-yellow: #e7a33e;
            --funny-teal: #2e8b8b;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-main);
            color: var(--text-primary);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
        }

        /* Header - LinkedIn style */
        .header {
            background: var(--bg-card);
            border-bottom: 1px solid var(--border);
            padding: 0 24px;
            height: 52px;
            display: flex;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .header-inner {
            max-width: 1128px;
            width: 100%;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 20px;
            font-weight: 700;
            color: var(--linkedin-blue);
            text-decoration: none;
        }
        .logo svg { width: 34px; height: 34px; }

        .nav-links {
            display: flex;
            gap: 24px;
        }
        .nav-link {
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 12px;
            font-weight: 500;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
            padding: 4px 12px;
            border-radius: 4px;
            transition: all 0.15s;
        }
        .nav-link:hover { background: var(--reaction-bg); color: var(--text-primary); }
        .nav-link.active { color: var(--text-primary); border-bottom: 2px solid var(--text-primary); }
        .nav-link svg { width: 24px; height: 24px; }

        /* Main layout */
        .main-container {
            max-width: 1128px;
            margin: 0 auto;
            padding: 24px;
            display: grid;
            grid-template-columns: 225px 1fr 300px;
            gap: 24px;
        }

        @media (max-width: 1024px) {
            .main-container { grid-template-columns: 1fr; padding: 16px; }
            .sidebar-left, .sidebar-right { display: none; }
        }

        /* Sidebar */
        .sidebar-left, .sidebar-right {
            position: sticky;
            top: 76px;
            height: fit-content;
        }

        .profile-card {
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border);
            overflow: hidden;
        }
        .profile-banner {
            height: 56px;
            background: linear-gradient(135deg, var(--linkedin-blue) 0%, #004182 100%);
        }
        .profile-info {
            padding: 12px 16px;
            text-align: center;
            margin-top: -32px;
        }
        .profile-avatar {
            width: 64px;
            height: 64px;
            border-radius: 50%;
            border: 2px solid var(--bg-card);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 24px;
            margin: 0 auto 8px;
        }
        .profile-name { font-weight: 600; font-size: 16px; }
        .profile-title { font-size: 12px; color: var(--text-secondary); }

        .stats-card {
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border);
            padding: 12px 16px;
            margin-top: 8px;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            font-size: 12px;
            border-bottom: 1px solid var(--border);
        }
        .stat-row:last-child { border-bottom: none; }
        .stat-label { color: var(--text-secondary); }
        .stat-value { font-weight: 600; color: var(--linkedin-blue); }

        /* Feed */
        .feed { display: flex; flex-direction: column; gap: 8px; }

        /* Post Card - LinkedIn style */
        .post-card {
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border);
            overflow: hidden;
        }

        .post-header {
            padding: 12px 16px;
            display: flex;
            gap: 8px;
        }
        .post-avatar {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 18px;
            flex-shrink: 0;
        }
        .post-avatar.winner {
            background: linear-gradient(135deg, #ffd700 0%, #ffb700 100%);
            color: #333;
        }
        .post-avatar.default {
            background: linear-gradient(135deg, var(--linkedin-blue) 0%, #004182 100%);
            color: white;
        }

        .post-meta { flex: 1; }
        .post-author {
            font-weight: 600;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .post-author a { color: inherit; text-decoration: none; }
        .post-author a:hover { color: var(--linkedin-blue); text-decoration: underline; }

        .badge {
            font-size: 10px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 4px;
            text-transform: uppercase;
        }
        .badge-winner { background: #ffd700; color: #333; }
        .badge-published { background: #057642; color: white; }

        .post-subtitle {
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 2px;
        }
        .post-time {
            font-size: 12px;
            color: var(--text-tertiary);
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .post-menu {
            color: var(--text-secondary);
            background: none;
            border: none;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .post-menu:hover { background: var(--reaction-bg); }

        /* Post content */
        .post-content {
            padding: 0 16px 12px;
            font-size: 14px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .post-content .hook {
            font-weight: 600;
        }
        .post-content.collapsed {
            max-height: 120px;
            overflow: hidden;
            position: relative;
        }
        .post-content.collapsed::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 40px;
            background: linear-gradient(transparent, var(--bg-card));
        }
        .see-more {
            color: var(--text-secondary);
            font-weight: 600;
            cursor: pointer;
            padding: 4px 16px 12px;
            display: block;
        }
        .see-more:hover { color: var(--linkedin-blue); }

        /* Metrics bar */
        .post-metrics {
            padding: 8px 16px;
            display: flex;
            gap: 16px;
            font-size: 11px;
            color: var(--text-tertiary);
        }
        .metric {
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 4px 8px;
            background: var(--reaction-bg);
            border-radius: 4px;
        }
        .metric.high { background: #e8f5e9; color: #2e7d32; }

        /* Reactions summary */
        .reactions-summary {
            padding: 8px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border);
        }
        .reaction-icons {
            display: flex;
            align-items: center;
        }
        .reaction-icon {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            margin-left: -4px;
            border: 1px solid var(--bg-card);
        }
        .reaction-icon:first-child { margin-left: 0; }
        .reaction-icon.like { background: var(--linkedin-blue); color: white; }
        .reaction-icon.celebrate { background: var(--celebrate-green); color: white; }
        .reaction-icon.love { background: var(--like-red); color: white; }

        /* Action buttons */
        .post-actions {
            padding: 4px 8px;
            display: flex;
            justify-content: space-around;
        }
        .action-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 12px 8px;
            border: none;
            background: transparent;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            color: var(--text-secondary);
            flex: 1;
            transition: all 0.15s;
        }
        .action-btn:hover {
            background: var(--reaction-bg);
            color: var(--text-primary);
        }
        .action-btn.liked {
            color: var(--linkedin-blue);
        }
        .action-btn svg { width: 24px; height: 24px; }

        /* Sources panel */
        .sources-panel {
            border-top: 1px solid var(--border);
            padding: 16px;
            background: var(--reaction-bg);
        }
        .sources-loading {
            color: var(--text-secondary);
            text-align: center;
            padding: 12px;
        }
        .source-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .source-item {
            display: block;
            padding: 10px 12px;
            margin-bottom: 8px;
            background: white;
            border-radius: 8px;
            border: 1px solid var(--border);
            text-decoration: none;
            color: inherit;
            transition: all 0.15s ease;
            cursor: pointer;
        }
        .source-item:hover {
            border-color: var(--linkedin-blue);
            box-shadow: 0 2px 8px rgba(0, 102, 194, 0.15);
            transform: translateY(-1px);
        }
        .source-item:last-child { margin-bottom: 0; }
        .source-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
        }
        .source-type {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            padding: 2px 6px;
            border-radius: 4px;
            background: var(--linkedin-blue);
            color: white;
        }
        .source-type.twitter { background: #1DA1F2; }
        .source-type.youtube { background: #FF0000; }
        .source-type.web { background: #34A853; }
        .source-type.discord { background: #5865F2; }
        .source-author {
            font-weight: 600;
            font-size: 13px;
            color: var(--text-primary);
        }
        .source-url {
            font-size: 11px;
            color: var(--text-tertiary);
            margin-left: auto;
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .source-text {
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.4;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .source-link {
            font-size: 12px;
            color: var(--linkedin-blue);
            text-decoration: none;
        }
        .source-link:hover { text-decoration: underline; }
        .no-sources {
            text-align: center;
            color: var(--text-secondary);
            font-style: italic;
        }

        /* Date separator */
        .date-separator {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 16px 0;
            color: var(--text-secondary);
            font-size: 12px;
            font-weight: 600;
        }
        .date-separator::before,
        .date-separator::after {
            content: '';
            flex: 1;
            height: 1px;
            background: var(--border);
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 48px 24px;
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        .empty-state svg { width: 48px; height: 48px; color: var(--text-tertiary); margin-bottom: 16px; }
        .empty-state h3 { font-size: 16px; margin-bottom: 8px; }
        .empty-state p { font-size: 14px; color: var(--text-secondary); }

        /* Filters */
        .filters-card {
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border);
            padding: 16px;
            margin-bottom: 16px;
        }
        .filters-card h3 {
            font-size: 14px;
            margin-bottom: 12px;
            color: var(--text-secondary);
        }
        .filter-row {
            display: flex;
            gap: 8px;
            margin-bottom: 8px;
        }
        .filter-input {
            flex: 1;
            padding: 8px 12px;
            border: 1px solid var(--border);
            border-radius: 4px;
            font-size: 14px;
        }
        .filter-btn {
            padding: 8px 16px;
            background: var(--linkedin-blue);
            color: white;
            border: none;
            border-radius: 20px;
            font-weight: 600;
            cursor: pointer;
            font-size: 14px;
        }
        .filter-btn:hover { background: var(--linkedin-blue-hover); }

        /* Right sidebar */
        .trending-card {
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border);
            padding: 16px;
        }
        .trending-card h3 {
            font-size: 16px;
            margin-bottom: 16px;
        }
        .trending-item {
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
        }
        .trending-item:last-child { border-bottom: none; }
        .trending-topic { font-size: 14px; font-weight: 600; }
        .trending-count { font-size: 12px; color: var(--text-secondary); }

        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.75);
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        .modal-overlay.active { display: flex; }
        .modal {
            background: var(--bg-card);
            border-radius: 8px;
            max-width: 552px;
            width: 90%;
            max-height: 90vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .modal-header {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .modal-header h2 { font-size: 18px; }
        .modal-close {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: var(--text-secondary);
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .modal-close:hover { background: var(--reaction-bg); }
        .modal-body { padding: 16px 24px; overflow-y: auto; flex: 1; }
        .modal-content {
            background: var(--bg-main);
            border-radius: 8px;
            padding: 16px;
            white-space: pre-wrap;
            line-height: 1.6;
            font-size: 14px;
        }
        .modal-footer {
            padding: 12px 24px;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 8px;
            justify-content: flex-end;
        }
        .btn {
            padding: 8px 24px;
            border-radius: 20px;
            font-weight: 600;
            cursor: pointer;
            font-size: 14px;
            border: none;
            transition: all 0.15s;
        }
        .btn-secondary {
            background: transparent;
            border: 1px solid var(--linkedin-blue);
            color: var(--linkedin-blue);
        }
        .btn-secondary:hover { background: rgba(10,102,194,0.1); }
        .btn-primary {
            background: var(--linkedin-blue);
            color: white;
        }
        .btn-primary:hover { background: var(--linkedin-blue-hover); }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-inner">
            <a href="/" class="logo">
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M20.5 2h-17A1.5 1.5 0 002 3.5v17A1.5 1.5 0 003.5 22h17a1.5 1.5 0 001.5-1.5v-17A1.5 1.5 0 0020.5 2zM8 19H5v-9h3zM6.5 8.25A1.75 1.75 0 118.3 6.5a1.78 1.78 0 01-1.8 1.75zM19 19h-3v-4.74c0-1.42-.6-1.93-1.38-1.93A1.74 1.74 0 0013 14.19a.66.66 0 000 .14V19h-3v-9h2.9v1.3a3.11 3.11 0 012.7-1.4c1.55 0 3.36.86 3.36 3.66z"/>
                </svg>
                AI Feed
            </a>
            <nav class="nav-links">
                <a href="/" class="nav-link active">
                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M23 9v2h-2v7a3 3 0 01-3 3h-4v-6h-4v6H6a3 3 0 01-3-3v-7H1V9l11-7 5 3.18V2h3v5.09z"/></svg>
                    Feed
                </a>
                <a href="https://agents.brandonsneider.com" class="nav-link">
                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M17 6V5a3 3 0 00-3-3h-4a3 3 0 00-3 3v1H2v4a3 3 0 003 3h14a3 3 0 003-3V6zM9 5a1 1 0 011-1h4a1 1 0 011 1v1H9zm10 9a4 4 0 003-1.38V17a3 3 0 01-3 3H5a3 3 0 01-3-3v-4.38A4 4 0 005 14z"/></svg>
                    Tournament
                </a>
            </nav>
        </div>
    </header>

    <div class="main-container">
        <aside class="sidebar-left">
            <div class="profile-card">
                <div class="profile-banner"></div>
                <div class="profile-info">
                    <div class="profile-avatar">AI</div>
                    <div class="profile-name">AI Content Generator</div>
                    <div class="profile-title">Multi-Agent Tournament System</div>
                </div>
            </div>
            <div class="stats-card" id="statsCard">
                <div class="stat-row">
                    <span class="stat-label">Total Winners</span>
                    <span class="stat-value" id="statTotal">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Published</span>
                    <span class="stat-value" id="statPublished">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Total Likes</span>
                    <span class="stat-value" id="statLikes">-</span>
                </div>
            </div>
        </aside>

        <main class="feed" id="feedContainer">
            <div class="filters-card">
                <h3>Filter Posts</h3>
                <div class="filter-row">
                    <input type="date" id="dateFrom" class="filter-input">
                    <input type="date" id="dateTo" class="filter-input">
                    <button class="filter-btn" onclick="loadFeed()">Filter</button>
                </div>
            </div>
            <div id="postsContainer">
                <div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
                    <h3>Loading posts...</h3>
                </div>
            </div>
        </main>

        <aside class="sidebar-right">
            <div class="trending-card">
                <h3>Sources</h3>
                <div class="trending-item">
                    <div class="trending-topic">Twitter/X</div>
                    <div class="trending-count">AI influencer tweets</div>
                </div>
                <div class="trending-item">
                    <div class="trending-topic">YouTube</div>
                    <div class="trending-count">AI channel videos</div>
                </div>
                <div class="trending-item">
                    <div class="trending-topic">Web Articles</div>
                    <div class="trending-count">AI news sites</div>
                </div>
            </div>
        </aside>
    </div>

    <div class="modal-overlay" id="publishModal">
        <div class="modal">
            <div class="modal-header">
                <h2>Share to LinkedIn</h2>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div class="modal-content" id="modalContent"></div>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="copyContent()">Copy</button>
                <button class="btn btn-primary" onclick="publishPost()">Post to LinkedIn</button>
            </div>
        </div>
    </div>

    <script>
        let postsData = {};
        let selectedPost = null;
        let userLikes = new Set();

        // Set default dates
        const today = new Date();
        const weekAgo = new Date(Date.now() - 14*24*60*60*1000);
        document.getElementById('dateTo').value = today.toISOString().split('T')[0];
        document.getElementById('dateFrom').value = weekAgo.toISOString().split('T')[0];

        function loadFeed() {
            const from = document.getElementById('dateFrom').value;
            const to = document.getElementById('dateTo').value;

            fetch(`/api/feed?from=${from}&to=${to}`)
                .then(r => r.json())
                .then(data => {
                    userLikes = new Set(data.user_likes || []);
                    renderStats(data.stats);
                    renderPosts(data.posts);
                })
                .catch(err => {
                    document.getElementById('postsContainer').innerHTML =
                        '<div class="empty-state"><h3>Error loading feed</h3><p>' + err + '</p></div>';
                });
        }

        function renderStats(stats) {
            document.getElementById('statTotal').textContent = stats.total || 0;
            document.getElementById('statPublished').textContent = stats.published || 0;
            document.getElementById('statLikes').textContent = stats.total_likes || 0;
        }

        function renderPosts(posts) {
            if (!posts || posts.length === 0) {
                document.getElementById('postsContainer').innerHTML = `
                    <div class="empty-state">
                        <svg viewBox="0 0 24 24" fill="currentColor"><path d="M19 3H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V5a2 2 0 00-2-2zm-7 14l-5-5 1.41-1.41L12 14.17l4.59-4.58L18 11l-6 6z"/></svg>
                        <h3>No winning posts yet</h3>
                        <p>Run a tournament to generate AI content</p>
                    </div>`;
                return;
            }

            // Group by day
            const byDay = {};
            posts.forEach(post => {
                const day = (post.completed_at || '').split('T')[0];
                if (!byDay[day]) byDay[day] = [];
                byDay[day].push(post);
            });

            let html = '';
            Object.keys(byDay).sort().reverse().forEach(day => {
                const dateStr = new Date(day + 'T00:00:00').toLocaleDateString('en-US', {
                    weekday: 'long', month: 'long', day: 'numeric'
                });
                html += `<div class="date-separator">${dateStr}</div>`;

                byDay[day].forEach(post => {
                    postsData[post.run_id] = post;
                    html += renderPostCard(post);
                });
            });

            document.getElementById('postsContainer').innerHTML = html;
        }

        function renderPostCard(post) {
            const content = post.winner_content || '';
            const lines = content.split('\\n');
            const hook = lines[0] || '';
            const rest = lines.slice(1).join('\\n');
            const isLong = content.length > 300;
            const initial = (post.hook_style || 'W')[0].toUpperCase();
            const isLiked = userLikes.has(post.run_id);
            const likeCount = post.like_count || 0;
            const timeAgo = getTimeAgo(post.completed_at);

            return `
            <article class="post-card" data-run-id="${post.run_id}">
                <div class="post-header">
                    <div class="post-avatar ${post.was_published ? 'winner' : 'default'}">${initial}</div>
                    <div class="post-meta">
                        <div class="post-author">
                            <a href="#">${post.hook_style || 'Tournament Winner'}</a>
                            <span class="badge badge-winner">WINNER</span>
                            ${post.was_published ? '<span class="badge badge-published">PUBLISHED</span>' : ''}
                        </div>
                        <div class="post-subtitle">Tournament #${post.run_id} &bull; ${post.num_variants} variants &bull; ${post.num_rounds} rounds</div>
                        <div class="post-time">${timeAgo}</div>
                    </div>
                    <button class="post-menu" onclick="showPostMenu(${post.run_id})">
                        <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M14 12a2 2 0 11-4 0 2 2 0 014 0zm6 0a2 2 0 11-4 0 2 2 0 014 0zM6 12a2 2 0 11-4 0 2 2 0 014 0z"/></svg>
                    </button>
                </div>

                <div class="post-content ${isLong ? 'collapsed' : ''}" id="content-${post.run_id}">
                    <span class="hook">${escapeHtml(hook)}</span>${rest ? '\\n' + escapeHtml(rest) : ''}
                </div>
                ${isLong ? `<span class="see-more" onclick="expandPost(${post.run_id})">...see more</span>` : ''}

                <div class="post-metrics">
                    <span class="metric ${post.winner_qe_score >= 75 ? 'high' : ''}">QE: ${post.winner_qe_score || 0}/100</span>
                    <span class="metric">ELO: ${Math.round(post.winner_elo || 1000)}</span>
                    <span class="metric">${post.total_debates || 0} debates</span>
                </div>

                <div class="reactions-summary">
                    <div style="display: flex; align-items: center; gap: 4px;">
                        ${likeCount > 0 ? `
                            <span class="reaction-icons">
                                <span class="reaction-icon like">+</span>
                            </span>
                            <span>${likeCount}</span>
                        ` : ''}
                    </div>
                </div>

                <div class="post-actions">
                    <button class="action-btn ${isLiked ? 'liked' : ''}" onclick="toggleLike(${post.run_id})">
                        <svg viewBox="0 0 24 24" fill="${isLiked ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2">
                            <path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3zM7 22H4a2 2 0 01-2-2v-7a2 2 0 012-2h3"/>
                        </svg>
                        Like
                    </button>
                    <button class="action-btn" onclick="openPublish(${post.run_id})">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                        </svg>
                        Share
                    </button>
                    <button class="action-btn" onclick="copyToClipboard(${post.run_id})">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
                        </svg>
                        Copy
                    </button>
                    <button class="action-btn" onclick="toggleSources(${post.run_id})">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"/>
                            <path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"/>
                        </svg>
                        Sources ${post.source_count ? '(' + post.source_count + ')' : ''}
                    </button>
                </div>

                <!-- Sources panel (hidden by default) -->
                <div class="sources-panel" id="sources-${post.run_id}" style="display: none;">
                    <div class="sources-loading">Loading sources...</div>
                </div>
            </article>`;
        }

        function escapeHtml(text) {
            if (!text) return '';
            return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        function getTimeAgo(dateStr) {
            if (!dateStr) return '';
            const date = new Date(dateStr);
            const now = new Date();
            const diff = Math.floor((now - date) / 1000);

            if (diff < 60) return 'just now';
            if (diff < 3600) return Math.floor(diff / 60) + 'm';
            if (diff < 86400) return Math.floor(diff / 3600) + 'h';
            if (diff < 604800) return Math.floor(diff / 86400) + 'd';
            return Math.floor(diff / 604800) + 'w';
        }

        function expandPost(runId) {
            const content = document.getElementById('content-' + runId);
            content.classList.remove('collapsed');
            content.nextElementSibling.style.display = 'none';
        }

        function toggleLike(runId) {
            fetch('/api/like', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ run_id: runId })
            })
            .then(r => r.json())
            .then(data => {
                if (data.liked) {
                    userLikes.add(runId);
                } else {
                    userLikes.delete(runId);
                }
                // Update UI
                const card = document.querySelector(`[data-run-id="${runId}"]`);
                const likeBtn = card.querySelector('.action-btn');
                likeBtn.classList.toggle('liked', data.liked);

                // Update like count in stats
                loadFeed();
            });
        }

        function showPostMenu(runId) {
            // Could add dropdown menu here
        }

        function copyToClipboard(runId) {
            const post = postsData[runId];
            if (post) {
                navigator.clipboard.writeText(post.winner_content);
                alert('Copied to clipboard!');
            }
        }

        async function toggleSources(runId) {
            const panel = document.getElementById(`sources-${runId}`);
            if (!panel) return;

            // Toggle visibility
            if (panel.style.display === 'none') {
                panel.style.display = 'block';

                // Load sources if not already loaded
                if (panel.querySelector('.sources-loading')) {
                    try {
                        const response = await fetch(`/api/sources/${runId}`);
                        const data = await response.json();

                        if (data.sources && data.sources.length > 0) {
                            panel.innerHTML = `
                                <ul class="source-list">
                                    ${data.sources.map(s => `
                                        <a href="${s.source_url || '#'}" target="_blank" class="source-item" ${!s.source_url ? 'style="pointer-events:none"' : ''}>
                                            <div class="source-header">
                                                <span class="source-type ${s.source_type}">${s.source_type}</span>
                                                <span class="source-author">@${s.source_author || 'unknown'}</span>
                                                <span class="source-url">${s.source_url ? new URL(s.source_url).hostname : ''}</span>
                                            </div>
                                            <div class="source-text">${escapeHtml(s.source_text || '')}</div>
                                        </a>
                                    `).join('')}
                                </ul>
                            `;
                        } else {
                            panel.innerHTML = '<div class="no-sources">No sources tracked for this post</div>';
                        }
                    } catch (e) {
                        panel.innerHTML = '<div class="no-sources">Failed to load sources</div>';
                    }
                }
            } else {
                panel.style.display = 'none';
            }
        }

        function openPublish(runId) {
            selectedPost = postsData[runId];
            if (selectedPost) {
                document.getElementById('modalContent').textContent = selectedPost.winner_content;
                document.getElementById('publishModal').classList.add('active');
            }
        }

        function closeModal() {
            document.getElementById('publishModal').classList.remove('active');
            selectedPost = null;
        }

        function copyContent() {
            if (selectedPost) {
                navigator.clipboard.writeText(selectedPost.winner_content);
            }
        }

        function publishPost() {
            if (!selectedPost) return;
            closeModal();

            fetch('/api/publish', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    run_id: selectedPost.run_id,
                    content: selectedPost.winner_content
                })
            })
            .then(r => r.json())
            .then(data => {
                alert(data.message);
                if (data.success) loadFeed();
            });
        }

        // Initial load
        loadFeed();
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/feed')
def get_feed():
    """Get winning posts for date range with like info"""
    from_date = request.args.get('from', '')
    to_date = request.args.get('to', '')
    user_id = get_user_id()

    conn = get_db()
    cursor = conn.cursor()

    # Build query
    query = """
        SELECT r.*, v.hook_style,
               COALESCE(r.like_count, 0) as like_count
        FROM tournament_runs r
        LEFT JOIN tournament_variants v ON r.run_id = v.run_id AND v.final_rank = 1
        WHERE r.status = 'complete'
    """
    params = []

    if from_date:
        query += " AND date(r.completed_at) >= ?"
        params.append(from_date)
    if to_date:
        query += " AND date(r.completed_at) <= ?"
        params.append(to_date)

    query += " ORDER BY r.completed_at DESC"

    cursor.execute(query, params)
    posts = [dict(row) for row in cursor.fetchall()]

    # Get source counts for each post
    for post in posts:
        try:
            cursor.execute(
                "SELECT COUNT(*) as count FROM tournament_sources WHERE run_id = ?",
                (post['run_id'],)
            )
            result = cursor.fetchone()
            post['source_count'] = result['count'] if result else 0
        except:
            post['source_count'] = 0

    # Get user's likes
    try:
        cursor.execute(
            "SELECT run_id FROM post_likes WHERE user_id = ?",
            (user_id,)
        )
        user_likes = [row['run_id'] for row in cursor.fetchall()]
    except:
        user_likes = []

    # Get stats
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN was_published THEN 1 ELSE 0 END) as published,
            SUM(COALESCE(like_count, 0)) as total_likes
        FROM tournament_runs
        WHERE status = 'complete'
    """)
    stats = dict(cursor.fetchone())

    conn.close()

    return jsonify({
        "posts": posts,
        "stats": stats,
        "user_likes": user_likes
    })


@app.route('/api/sources/<int:run_id>')
def get_sources(run_id):
    """Get source news items used to generate a post"""
    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT source_type, source_text, source_url, source_author, source_timestamp
            FROM tournament_sources
            WHERE run_id = ?
            ORDER BY source_type, source_timestamp DESC
        """, (run_id,))
        sources = [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        sources = []
        print(f"Error fetching sources: {e}")

    conn.close()

    return jsonify({
        "run_id": run_id,
        "sources": sources,
        "count": len(sources)
    })


@app.route('/api/like', methods=['POST'])
def toggle_like():
    """Toggle like on a post"""
    data = request.json
    run_id = data.get('run_id')
    user_id = get_user_id()

    if not run_id:
        return jsonify({"error": "No run_id provided"}), 400

    conn = get_db()
    cursor = conn.cursor()

    # Check if already liked
    cursor.execute(
        "SELECT like_id FROM post_likes WHERE run_id = ? AND user_id = ?",
        (run_id, user_id)
    )
    existing = cursor.fetchone()

    if existing:
        # Unlike
        cursor.execute(
            "DELETE FROM post_likes WHERE run_id = ? AND user_id = ?",
            (run_id, user_id)
        )
        cursor.execute(
            "UPDATE tournament_runs SET like_count = COALESCE(like_count, 0) - 1 WHERE run_id = ?",
            (run_id,)
        )
        liked = False
    else:
        # Like
        cursor.execute(
            "INSERT INTO post_likes (run_id, user_id) VALUES (?, ?)",
            (run_id, user_id)
        )
        cursor.execute(
            "UPDATE tournament_runs SET like_count = COALESCE(like_count, 0) + 1 WHERE run_id = ?",
            (run_id,)
        )
        liked = True

    conn.commit()

    # Get new like count
    cursor.execute("SELECT like_count FROM tournament_runs WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()
    like_count = row['like_count'] if row else 0

    conn.close()

    return jsonify({"liked": liked, "like_count": like_count})


@app.route('/api/publish', methods=['POST'])
def publish():
    """Publish winning post to LinkedIn"""
    data = request.json
    run_id = data.get('run_id')
    content = data.get('content')

    if not content:
        return jsonify({"success": False, "message": "No content provided"})

    try:
        from linkedin_autopilot import LinkedInAutopilot

        script_dir = Path(__file__).parent
        load_dotenv(script_dir / '.env')

        output_dir = script_dir / 'output_data'
        autopilot = LinkedInAutopilot(output_dir)

        google_email = os.getenv('GOOGLE_EMAIL')
        linkedin_email = os.getenv('LINKEDIN_EMAIL')
        linkedin_password = os.getenv('LINKEDIN_PASSWORD')

        if not google_email and (not linkedin_email or not linkedin_password):
            return jsonify({
                "success": False,
                "message": "LinkedIn credentials not configured"
            })

        autopilot.login(
            email=linkedin_email,
            password=linkedin_password,
            google_email=google_email
        )

        success = autopilot.poster.create_post(content)
        autopilot.close()

        if success and run_id:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE tournament_runs
                SET was_published = TRUE, published_at = ?
                WHERE run_id = ?
            """, (datetime.now().isoformat(), run_id))
            conn.commit()
            conn.close()

        return jsonify({
            "success": success,
            "message": "Posted successfully!" if success else "Failed to publish"
        })

    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("LinkedIn Feed Clone")
    print("="*60)
    print("\nOpen http://localhost:5002 in your browser")
    print("Press Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
