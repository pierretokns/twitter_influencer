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

A feed-style UI showing all winning posts from tournaments, organized by day.
Each post shows the tournament winner with QE score, ELO, and option to publish.

Usage:
    uv run python linkedin_feed.py
    Then open: http://localhost:5002

Features:
    - Daily timeline of winning posts
    - Filter by date range
    - Show all tournament history
    - One-click publish to LinkedIn
    - View full debate history for each winner
"""

import os
import sys
import sqlite3
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
    conn = sqlite3.connect(str(AI_NEWS_DB))
    conn.row_factory = sqlite3.Row
    return conn


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>AI Content Feed - Daily Winners</title>
    <meta charset="utf-8">
    <style>
        :root {
            --bg: #f3f2ef;
            --card: #ffffff;
            --border: #e0e0e0;
            --accent: #0077b5;
            --accent2: #00a0dc;
            --green: #10b981;
            --text: #191919;
            --muted: #666666;
            --gold: #ffd700;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }

        /* LinkedIn-style header */
        .header {
            background: var(--card);
            border-bottom: 1px solid var(--border);
            padding: 0.75rem 1rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .header-inner {
            max-width: 1128px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--accent);
        }
        .logo svg { width: 34px; height: 34px; }

        .controls {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .control-group label {
            font-size: 0.875rem;
            color: var(--muted);
        }
        select, input[type="date"] {
            padding: 0.5rem;
            border: 1px solid var(--border);
            border-radius: 4px;
            font-size: 0.875rem;
        }

        /* Main layout */
        .main {
            max-width: 680px;
            margin: 1.5rem auto;
            padding: 0 1rem;
        }

        /* Stats bar */
        .stats-bar {
            background: var(--card);
            border-radius: 8px;
            border: 1px solid var(--border);
            padding: 1rem;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-around;
            text-align: center;
        }
        .stat { flex: 1; }
        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--accent);
        }
        .stat-label {
            font-size: 0.75rem;
            color: var(--muted);
            text-transform: uppercase;
        }

        /* Day section */
        .day-section {
            margin-bottom: 1.5rem;
        }
        .day-header {
            font-size: 0.875rem;
            color: var(--muted);
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .day-count {
            background: var(--accent);
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.75rem;
        }

        /* Post card - LinkedIn style */
        .post-card {
            background: var(--card);
            border-radius: 8px;
            border: 1px solid var(--border);
            margin-bottom: 0.75rem;
            overflow: hidden;
            transition: box-shadow 0.2s;
        }
        .post-card:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .post-card.published {
            border-left: 4px solid var(--green);
        }

        /* Post header */
        .post-header {
            padding: 1rem;
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
        }
        .avatar {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 1.25rem;
            flex-shrink: 0;
        }
        .avatar.gold {
            background: linear-gradient(135deg, #ffd700 0%, #ffb700 100%);
            color: #333;
        }
        .post-meta {
            flex: 1;
        }
        .post-author {
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .winner-badge {
            background: var(--gold);
            color: #333;
            padding: 0.125rem 0.375rem;
            border-radius: 4px;
            font-size: 0.625rem;
            font-weight: 700;
        }
        .post-info {
            font-size: 0.75rem;
            color: var(--muted);
            margin-top: 0.25rem;
        }
        .post-scores {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.375rem;
        }
        .score-badge {
            padding: 0.125rem 0.375rem;
            border-radius: 4px;
            font-size: 0.625rem;
            font-weight: 600;
        }
        .score-elo { background: #e3f2fd; color: #1565c0; }
        .score-qe { background: #e8f5e9; color: #2e7d32; }
        .score-record { background: #fce4ec; color: #c2185b; }

        /* Post content */
        .post-content {
            padding: 0 1rem 1rem;
            white-space: pre-wrap;
            line-height: 1.5;
            font-size: 0.9375rem;
        }
        .post-content .hook {
            font-weight: 600;
            color: var(--text);
        }
        .post-content .more {
            color: var(--accent);
            cursor: pointer;
            font-weight: 500;
        }

        /* Post actions */
        .post-actions {
            border-top: 1px solid var(--border);
            padding: 0.5rem 1rem;
            display: flex;
            justify-content: space-around;
        }
        .action-btn {
            display: flex;
            align-items: center;
            gap: 0.375rem;
            padding: 0.5rem 1rem;
            border: none;
            background: transparent;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.875rem;
            color: var(--muted);
            transition: all 0.2s;
        }
        .action-btn:hover {
            background: rgba(0,0,0,0.05);
            color: var(--text);
        }
        .action-btn.publish:hover {
            background: rgba(0,119,181,0.1);
            color: var(--accent);
        }
        .action-btn svg {
            width: 20px;
            height: 20px;
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            background: var(--card);
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        .empty-state svg {
            width: 64px;
            height: 64px;
            color: var(--muted);
            opacity: 0.5;
            margin-bottom: 1rem;
        }
        .empty-state h3 { color: var(--muted); margin-bottom: 0.5rem; }
        .empty-state p { color: var(--muted); font-size: 0.875rem; }

        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.6);
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        .modal-overlay.active { display: flex; }
        .modal {
            background: var(--card);
            border-radius: 12px;
            max-width: 550px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }
        .modal-header {
            padding: 1rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .modal-header h2 { font-size: 1.125rem; }
        .modal-close {
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: var(--muted);
        }
        .modal-body { padding: 1rem; }
        .modal-content {
            background: var(--bg);
            border-radius: 8px;
            padding: 1rem;
            white-space: pre-wrap;
            line-height: 1.6;
            max-height: 300px;
            overflow-y: auto;
        }
        .modal-actions {
            padding: 1rem;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 0.75rem;
            justify-content: flex-end;
        }
        .btn {
            padding: 0.625rem 1.25rem;
            border-radius: 20px;
            font-weight: 600;
            cursor: pointer;
            font-size: 0.875rem;
            border: none;
            transition: all 0.2s;
        }
        .btn-secondary {
            background: transparent;
            border: 1px solid var(--accent);
            color: var(--accent);
        }
        .btn-primary {
            background: var(--accent);
            color: white;
        }
        .btn-primary:hover { background: #006097; }

        /* Loading */
        .loading {
            text-align: center;
            padding: 2rem;
            color: var(--muted);
        }

        /* Responsive */
        @media (max-width: 768px) {
            .controls { flex-direction: column; gap: 0.5rem; }
            .stats-bar { flex-wrap: wrap; }
            .stat { min-width: 33%; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-inner">
            <div class="logo">
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M20.5 2h-17A1.5 1.5 0 002 3.5v17A1.5 1.5 0 003.5 22h17a1.5 1.5 0 001.5-1.5v-17A1.5 1.5 0 0020.5 2zM8 19H5v-9h3zM6.5 8.25A1.75 1.75 0 118.3 6.5a1.78 1.78 0 01-1.8 1.75zM19 19h-3v-4.74c0-1.42-.6-1.93-1.38-1.93A1.74 1.74 0 0013 14.19a.66.66 0 000 .14V19h-3v-9h2.9v1.3a3.11 3.11 0 012.7-1.4c1.55 0 3.36.86 3.36 3.66z"/>
                </svg>
                AI Content Feed
            </div>
            <div class="controls">
                <div class="control-group">
                    <label>From:</label>
                    <input type="date" id="dateFrom" value="">
                </div>
                <div class="control-group">
                    <label>To:</label>
                    <input type="date" id="dateTo" value="">
                </div>
                <button class="btn btn-primary" onclick="loadFeed()">Filter</button>
            </div>
        </div>
    </div>

    <main class="main">
        <div class="stats-bar" id="statsBar">
            <div class="stat">
                <div class="stat-value" id="totalPosts">-</div>
                <div class="stat-label">Total Winners</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="publishedCount">-</div>
                <div class="stat-label">Published</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="avgQE">-</div>
                <div class="stat-label">Avg QE Score</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="avgELO">-</div>
                <div class="stat-label">Avg ELO</div>
            </div>
        </div>

        <div id="feedContainer">
            <div class="loading">Loading feed...</div>
        </div>
    </main>

    <div class="modal-overlay" id="publishModal">
        <div class="modal">
            <div class="modal-header">
                <h2>Publish to LinkedIn</h2>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-body">
                <p style="color: var(--muted); margin-bottom: 1rem; font-size: 0.875rem;">
                    Review the content below before publishing:
                </p>
                <div class="modal-content" id="modalContent"></div>
            </div>
            <div class="modal-actions">
                <button class="btn btn-secondary" onclick="copyContent()">Copy to Clipboard</button>
                <button class="btn btn-primary" onclick="publishPost()">Publish Now</button>
            </div>
        </div>
    </div>

    <script>
        let selectedPost = null;

        // Set default dates (last 7 days)
        document.getElementById('dateTo').value = new Date().toISOString().split('T')[0];
        document.getElementById('dateFrom').value = new Date(Date.now() - 7*24*60*60*1000).toISOString().split('T')[0];

        function loadFeed() {
            const from = document.getElementById('dateFrom').value;
            const to = document.getElementById('dateTo').value;

            fetch(`/api/feed?from=${from}&to=${to}`)
                .then(r => r.json())
                .then(data => {
                    renderStats(data.stats);
                    renderFeed(data.posts);
                })
                .catch(err => {
                    document.getElementById('feedContainer').innerHTML =
                        '<div class="empty-state"><h3>Error loading feed</h3><p>' + err + '</p></div>';
                });
        }

        function renderStats(stats) {
            document.getElementById('totalPosts').textContent = stats.total || 0;
            document.getElementById('publishedCount').textContent = stats.published || 0;
            document.getElementById('avgQE').textContent = stats.avg_qe ? Math.round(stats.avg_qe) : '-';
            document.getElementById('avgELO').textContent = stats.avg_elo ? Math.round(stats.avg_elo) : '-';
        }

        function renderFeed(posts) {
            if (!posts || posts.length === 0) {
                document.getElementById('feedContainer').innerHTML = `
                    <div class="empty-state">
                        <svg viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                        </svg>
                        <h3>No winning posts yet</h3>
                        <p>Run a tournament to generate content</p>
                    </div>`;
                return;
            }

            // Group by day
            const byDay = {};
            posts.forEach(post => {
                const day = post.completed_at.split('T')[0];
                if (!byDay[day]) byDay[day] = [];
                byDay[day].push(post);
            });

            let html = '';
            Object.keys(byDay).sort().reverse().forEach(day => {
                const dayPosts = byDay[day];
                const dateStr = new Date(day).toLocaleDateString('en-US', {
                    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
                });

                html += `<div class="day-section">
                    <div class="day-header">
                        <span>${dateStr}</span>
                        <span class="day-count">${dayPosts.length} winner${dayPosts.length > 1 ? 's' : ''}</span>
                    </div>`;

                dayPosts.forEach(post => {
                    const content = post.winner_content || '';
                    const hook = content.split('\\n')[0];
                    const rest = content.split('\\n').slice(1).join('\\n');
                    const truncated = rest.length > 200;
                    const displayRest = truncated ? rest.substring(0, 200) + '...' : rest;
                    const initial = (post.hook_style || 'W')[0].toUpperCase();

                    html += `
                    <div class="post-card ${post.was_published ? 'published' : ''}" data-run-id="${post.run_id}">
                        <div class="post-header">
                            <div class="avatar ${post.was_published ? 'gold' : ''}">${initial}</div>
                            <div class="post-meta">
                                <div class="post-author">
                                    ${post.hook_style || 'Winner'}
                                    <span class="winner-badge">WINNER</span>
                                    ${post.was_published ? '<span class="winner-badge" style="background:#10b981;color:white;">PUBLISHED</span>' : ''}
                                </div>
                                <div class="post-info">
                                    Tournament #${post.run_id} &bull; ${post.num_variants} variants &bull; ${post.num_rounds} rounds
                                </div>
                                <div class="post-scores">
                                    <span class="score-badge score-elo">ELO ${Math.round(post.winner_elo || 1000)}</span>
                                    <span class="score-badge score-qe">QE ${post.winner_qe_score || 0}/100</span>
                                    <span class="score-badge score-record">${post.total_debates || 0} debates</span>
                                </div>
                            </div>
                        </div>
                        <div class="post-content">
                            <span class="hook">${escapeHtml(hook)}</span>
${displayRest ? '\\n' + escapeHtml(displayRest) : ''}
                            ${truncated ? '<span class="more" onclick="showFull(' + post.run_id + ')">...see more</span>' : ''}
                        </div>
                        <div class="post-actions">
                            <button class="action-btn" onclick="copyToClipboard(${post.run_id})">
                                <svg viewBox="0 0 24 24" fill="currentColor"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>
                                Copy
                            </button>
                            <button class="action-btn publish" onclick="openPublish(${post.run_id})">
                                <svg viewBox="0 0 24 24" fill="currentColor"><path d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92s2.92-1.31 2.92-2.92-1.31-2.92-2.92-2.92z"/></svg>
                                ${post.was_published ? 'Re-share' : 'Publish'}
                            </button>
                            <button class="action-btn" onclick="viewDetails(${post.run_id})">
                                <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/></svg>
                                Details
                            </button>
                        </div>
                    </div>`;
                });

                html += '</div>';
            });

            document.getElementById('feedContainer').innerHTML = html;

            // Store posts for actions
            window.postsData = {};
            posts.forEach(p => window.postsData[p.run_id] = p);
        }

        function escapeHtml(text) {
            if (!text) return '';
            return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        function copyToClipboard(runId) {
            const post = window.postsData[runId];
            if (post) {
                navigator.clipboard.writeText(post.winner_content);
                alert('Copied to clipboard!');
            }
        }

        function showFull(runId) {
            openPublish(runId);
        }

        function openPublish(runId) {
            selectedPost = window.postsData[runId];
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
                alert('Copied!');
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

        function viewDetails(runId) {
            window.location.href = `/details/${runId}`;
        }

        // Initial load
        loadFeed();
    </script>
</body>
</html>
'''


DETAILS_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Tournament Details - Run #{{ run_id }}</title>
    <meta charset="utf-8">
    <style>
        :root {
            --bg: #f3f2ef;
            --card: #ffffff;
            --border: #e0e0e0;
            --accent: #0077b5;
            --green: #10b981;
            --text: #191919;
            --muted: #666666;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: var(--bg); color: var(--text); }
        .header { background: var(--card); border-bottom: 1px solid var(--border); padding: 1rem; }
        .header-inner { max-width: 900px; margin: 0 auto; display: flex; align-items: center; gap: 1rem; }
        .back-btn { color: var(--accent); text-decoration: none; display: flex; align-items: center; gap: 0.25rem; }
        .main { max-width: 900px; margin: 1.5rem auto; padding: 0 1rem; }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }
        .card h2 { margin-bottom: 1rem; font-size: 1.125rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
        .stat { text-align: center; padding: 1rem; background: var(--bg); border-radius: 8px; }
        .stat-value { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
        .stat-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; }
        .variant { padding: 1rem; border: 1px solid var(--border); border-radius: 8px; margin-bottom: 0.75rem; }
        .variant.winner { border-color: var(--green); background: rgba(16,185,129,0.05); }
        .variant-header { display: flex; justify-content: space-between; margin-bottom: 0.5rem; }
        .variant-rank { font-weight: 700; }
        .variant-scores { display: flex; gap: 0.5rem; }
        .score { padding: 0.125rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
        .score-elo { background: #e3f2fd; color: #1565c0; }
        .score-qe { background: #e8f5e9; color: #2e7d32; }
        .variant-content { font-size: 0.875rem; line-height: 1.5; color: var(--muted); white-space: pre-wrap; max-height: 100px; overflow: hidden; }
        .debates-list { max-height: 400px; overflow-y: auto; }
        .debate { padding: 0.75rem; border-bottom: 1px solid var(--border); font-size: 0.875rem; }
        .debate:last-child { border-bottom: none; }
        .debate-result { font-weight: 600; }
        .debate-result.win { color: var(--green); }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-inner">
            <a href="/" class="back-btn">&larr; Back to Feed</a>
            <h1>Tournament #{{ run_id }}</h1>
        </div>
    </div>
    <main class="main">
        <div class="card">
            <h2>Overview</h2>
            <div class="grid">
                <div class="stat">
                    <div class="stat-value">{{ run.num_variants }}</div>
                    <div class="stat-label">Variants</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{{ run.num_rounds }}</div>
                    <div class="stat-label">Rounds</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{{ run.total_debates }}</div>
                    <div class="stat-label">Total Debates</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{{ run.winner_qe_score or '-' }}</div>
                    <div class="stat-label">Winner QE</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>All Variants (Ranked)</h2>
            {% for v in variants %}
            <div class="variant {{ 'winner' if v.final_rank == 1 else '' }}">
                <div class="variant-header">
                    <span class="variant-rank">#{{ v.final_rank }} - {{ v.hook_style }}</span>
                    <div class="variant-scores">
                        <span class="score score-elo">ELO {{ v.elo_rating|round|int }}</span>
                        <span class="score score-qe">QE {{ v.qe_score }}/100</span>
                    </div>
                </div>
                <div class="variant-content">{{ v.content[:200] }}{% if v.content|length > 200 %}...{% endif %}</div>
            </div>
            {% endfor %}
        </div>

        <div class="card">
            <h2>Debate History</h2>
            <div class="debates-list">
                {% for d in debates %}
                <div class="debate">
                    <span class="debate-result {{ 'win' if d.winner_id else '' }}">
                        {{ d.variant_a_id }} vs {{ d.variant_b_id }}
                        {% if d.winner_id %} &rarr; {{ d.winner_id }} wins{% endif %}
                    </span>
                    {% if d.reasoning %}
                    <div style="color: var(--muted); margin-top: 0.25rem; font-size: 0.8rem;">{{ d.reasoning[:150] }}...</div>
                    {% endif %}
                </div>
                {% else %}
                <div class="debate" style="color: var(--muted);">No debate records found</div>
                {% endfor %}
            </div>
        </div>
    </main>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/feed')
def get_feed():
    """Get winning posts for date range"""
    from_date = request.args.get('from', '')
    to_date = request.args.get('to', '')

    conn = get_db()
    cursor = conn.cursor()

    # Build query
    query = """
        SELECT r.*, v.hook_style
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

    # Get stats
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN was_published THEN 1 ELSE 0 END) as published,
            AVG(winner_qe_score) as avg_qe,
            AVG(winner_elo) as avg_elo
        FROM tournament_runs
        WHERE status = 'complete'
    """)
    stats = dict(cursor.fetchone())

    conn.close()

    return jsonify({"posts": posts, "stats": stats})


@app.route('/details/<int:run_id>')
def details(run_id):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM tournament_runs WHERE run_id = ?", (run_id,))
    run = cursor.fetchone()
    if not run:
        return "Tournament not found", 404

    cursor.execute("""
        SELECT * FROM tournament_variants
        WHERE run_id = ?
        ORDER BY final_rank
    """, (run_id,))
    variants = cursor.fetchall()

    cursor.execute("""
        SELECT * FROM tournament_debates
        WHERE run_id = ?
        ORDER BY debate_id
    """, (run_id,))
    debates = cursor.fetchall()

    conn.close()

    return render_template_string(
        DETAILS_TEMPLATE,
        run_id=run_id,
        run=dict(run),
        variants=[dict(v) for v in variants],
        debates=[dict(d) for d in debates]
    )


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
            # Mark as published
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
    print("LinkedIn Feed Clone - Daily Winners")
    print("="*60)
    print("\nOpen http://localhost:5002 in your browser")
    print("Press Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
