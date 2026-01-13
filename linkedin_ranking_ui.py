# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "flask>=3.0.0",
#     "python-dotenv>=0.19.0",
#     "schedule>=1.2.0",
#     "pillow>=10.0.0",
#     "undetected-chromedriver>=3.5.0",
# ]
# ///

"""
LinkedIn Post Ranking UI - Visual Tournament with Human-in-the-loop approval

Run with: uv run linkedin_ranking_ui.py
Then open: http://localhost:5001
"""

import os
import sys
import json
import sqlite3
import subprocess
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

app = Flask(__name__)


# =============================================================================
# TOURNAMENT PERSISTENCE
# =============================================================================

def save_tournament_to_db(
    db_path: Path,
    num_variants: int,
    num_rounds: int,
    ranked_variants: list,
    debates: list = None,
    news_sources: list = None
) -> int:
    """
    Save tournament results to database.

    Saves:
    - Tournament run metadata
    - All variants with rankings
    - Debate history
    - Source news items used (for traceability)

    Returns the run_id of the saved tournament.
    """
    from db_migrations import migrate_ai_news_db

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Ensure tables exist (run migrations)
    migrate_ai_news_db(conn)

    cursor = conn.cursor()

    try:
        # Get winner info
        winner = ranked_variants[0] if ranked_variants else None

        # Insert tournament run
        cursor.execute("""
            INSERT INTO tournament_runs (
                started_at, completed_at, num_variants, num_rounds,
                total_debates, winner_variant_id, winner_content,
                winner_elo, winner_qe_score, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            num_variants,
            num_rounds,
            len(debates) if debates else 0,
            winner.variant_id if winner else None,
            winner.content if winner else None,
            winner.elo_rating if winner else None,
            winner.qe_score if winner else None,
            'complete'
        ))

        run_id = cursor.lastrowid

        # Insert all variants with their final rankings
        for rank, v in enumerate(ranked_variants, 1):
            cursor.execute("""
                INSERT INTO tournament_variants (
                    run_id, variant_id, hook_style, content, elo_rating,
                    qe_score, qe_feedback, wins, losses, generation,
                    evolved_from, evolution_feedback, is_duplicate, final_rank
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                v.variant_id,
                v.hook_style,
                v.content,
                v.elo_rating,
                v.qe_score,
                v.qe_feedback,
                v.wins,
                v.losses,
                v.generation,
                v.evolved_from,
                v.evolution_feedback,
                v.is_duplicate,
                rank
            ))

        # Insert debates
        if debates:
            for debate in debates:
                cursor.execute("""
                    INSERT INTO tournament_debates (
                        run_id, round_num, variant_a_id, variant_b_id,
                        winner_id, reasoning
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    debate.get('round', 1),
                    debate.get('a'),
                    debate.get('b'),
                    debate.get('winner'),
                    debate.get('reasoning', '')
                ))

        # Insert source news items for traceability
        # Save the top sources used for generation (limit to avoid bloat)
        # TODO: Implement better source tracking in generator to identify exact sources used
        MAX_SOURCES = 20  # Save top 20 most relevant sources
        if news_sources and winner:
            saved_count = 0
            for idx, item in enumerate(news_sources[:MAX_SOURCES], 1):
                source_type = item.get('source_type', 'unknown')
                source_id = item.get('id', '')

                # Build URL based on source type
                source_url = item.get('url', '')
                if not source_url:
                    if source_type == 'twitter':
                        source_url = f"https://x.com/{item.get('username', '')}/status/{source_id}"
                    elif source_type == 'youtube':
                        source_url = f"https://youtube.com/watch?v={source_id}"

                cursor.execute("""
                    INSERT INTO tournament_sources (
                        run_id, source_type, source_id, source_text,
                        source_url, source_author, source_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    source_type,
                    str(source_id),
                    item.get('text', '')[:500],  # Truncate long text
                    source_url,
                    item.get('username', ''),
                    item.get('timestamp', '')
                ))
                saved_count += 1

        conn.commit()
        sources_count = saved_count if news_sources and winner else 0
        print(f"[DB] Saved tournament run #{run_id} with {len(ranked_variants)} variants and {sources_count} sources")
        return run_id

    except Exception as e:
        conn.rollback()
        print(f"[DB ERROR] Failed to save tournament: {e}")
        raise
    finally:
        conn.close()


def mark_tournament_published(db_path: Path, run_id: int):
    """Mark a tournament run as published"""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE tournament_runs
        SET was_published = TRUE, published_at = ?
        WHERE run_id = ?
    """, (datetime.now().isoformat(), run_id))
    conn.commit()
    conn.close()

# Global state - Enhanced for Google Co-Scientist inspired pipeline
ranking_state = {
    "status": "idle",
    "phase": "",
    "current_step": 0,
    "total_steps": 6,
    # Pipeline data
    "news_items": [],           # Source news
    "variants": [],             # All variants with metadata
    "generated_posts": [],      # Full post previews
    "qe_results": [],           # QE evaluation details
    "proximity_results": [],    # Similarity scores & duplicates
    "evolution_results": [],    # Which posts were evolved
    "debates": [],              # All debate results
    "matches": [],              # Tournament matches
    "rankings": [],             # Final rankings
    "logs": [],
    "error": None,
}

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>LinkedIn Post Tournament</title>
    <meta charset="utf-8">
    <style>
        :root {
            --bg: #0a0a0f;
            --card: #12121a;
            --border: #1e1e2e;
            --accent: #0077b5;
            --accent2: #00a0dc;
            --green: #10b981;
            --red: #ef4444;
            --yellow: #f59e0b;
            --text: #e5e7eb;
            --muted: #6b7280;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }

        .header {
            background: linear-gradient(135deg, #0077b5 0%, #00a0dc 100%);
            padding: 2rem;
            text-align: center;
        }
        .header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
        .header p { opacity: 0.9; }

        .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }

        .control-bar {
            display: flex;
            gap: 1rem;
            align-items: center;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }

        .btn {
            padding: 0.875rem 2rem;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .btn-primary {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
            color: white;
        }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,119,181,0.4); }
        .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .btn-success { background: var(--green); color: white; }
        .btn-success:hover { background: #059669; }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--card);
            border-radius: 9999px;
            font-size: 0.875rem;
        }
        .status-dot {
            width: 10px; height: 10px;
            border-radius: 50%;
            background: var(--muted);
        }
        .status-dot.active { background: var(--green); animation: pulse 1.5s infinite; }
        .status-dot.complete { background: var(--accent); }
        .status-dot.error { background: var(--red); }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

        .phase-indicator {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 2rem;
        }
        .phase {
            flex: 1;
            padding: 1rem;
            background: var(--card);
            border-radius: 8px;
            text-align: center;
            border: 2px solid transparent;
            transition: all 0.3s;
        }
        .phase.active { border-color: var(--accent); background: rgba(0,119,181,0.1); }
        .phase.complete { border-color: var(--green); }
        .phase-num {
            width: 28px; height: 28px;
            border-radius: 50%;
            background: var(--border);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .phase.active .phase-num { background: var(--accent); }
        .phase.complete .phase-num { background: var(--green); }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 2rem;
        }

        @media (max-width: 1200px) {
            .main-grid { grid-template-columns: 1fr; }
        }

        /* Tournament Bracket */
        .tournament {
            background: var(--card);
            border-radius: 12px;
            padding: 1.5rem;
            min-height: 500px;
        }
        .tournament h2 { margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }

        .bracket {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 2rem;
            padding: 2rem 0;
            overflow-x: auto;
        }

        .round {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            min-width: 200px;
        }
        .round-title {
            text-align: center;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--muted);
            margin-bottom: 0.5rem;
        }

        .match {
            background: var(--bg);
            border-radius: 8px;
            padding: 0.75rem;
            border: 1px solid var(--border);
        }
        .match.active { border-color: var(--yellow); box-shadow: 0 0 20px rgba(245,158,11,0.2); }
        .match.complete { border-color: var(--green); }

        .fighter {
            padding: 0.5rem;
            border-radius: 4px;
            font-size: 0.875rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 0.25rem 0;
        }
        .fighter.winner { background: rgba(16,185,129,0.2); color: var(--green); }
        .fighter.loser { opacity: 0.5; }
        .fighter-name { font-weight: 600; }
        .fighter-score { font-size: 0.75rem; color: var(--muted); }

        .vs {
            text-align: center;
            font-size: 0.75rem;
            color: var(--muted);
            padding: 0.25rem;
        }

        /* Live Log */
        .live-log {
            background: var(--card);
            border-radius: 12px;
            padding: 1.5rem;
            max-height: 400px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .live-log h2 { margin-bottom: 1rem; }
        .log-entries {
            flex: 1;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            line-height: 1.8;
        }
        .log-entry { padding: 0.25rem 0; border-bottom: 1px solid var(--border); }
        .log-entry.step { color: var(--yellow); font-weight: bold; }
        .log-entry.success { color: var(--green); }
        .log-entry.error { color: var(--red); }
        .log-entry .time { color: var(--muted); margin-right: 0.5rem; }

        /* Results */
        .results {
            margin-top: 2rem;
        }
        .results h2 { margin-bottom: 1.5rem; text-align: center; }

        .podium {
            display: flex;
            justify-content: center;
            align-items: flex-end;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .podium-place {
            text-align: center;
            transition: all 0.3s;
        }
        .podium-place:hover { transform: translateY(-5px); }
        .podium-bar {
            width: 120px;
            background: var(--card);
            border-radius: 8px 8px 0 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-end;
            padding: 1rem;
            border: 2px solid var(--border);
        }
        .podium-place.first .podium-bar { height: 200px; border-color: gold; background: linear-gradient(180deg, rgba(255,215,0,0.1) 0%, var(--card) 100%); }
        .podium-place.second .podium-bar { height: 160px; border-color: silver; }
        .podium-place.third .podium-bar { height: 120px; border-color: #cd7f32; }
        .podium-rank {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .podium-place.first .podium-rank { color: gold; }
        .podium-place.second .podium-rank { color: silver; }
        .podium-place.third .podium-rank { color: #cd7f32; }
        .podium-name { font-size: 0.875rem; color: var(--muted); }
        .podium-elo { font-size: 1.25rem; font-weight: bold; margin-top: 0.5rem; }

        .post-cards {
            display: grid;
            gap: 1.5rem;
        }
        .post-card {
            background: var(--card);
            border-radius: 12px;
            padding: 1.5rem;
            border: 2px solid var(--border);
            transition: all 0.3s;
        }
        .post-card:hover { border-color: var(--accent); }
        .post-card.gold { border-color: gold; background: linear-gradient(135deg, rgba(255,215,0,0.05) 0%, var(--card) 100%); }

        .post-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1rem;
        }
        .post-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .post-rank-badge {
            width: 36px; height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.25rem;
        }
        .post-card.gold .post-rank-badge { background: gold; color: black; }

        .post-meta {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }
        .meta-tag {
            padding: 0.25rem 0.5rem;
            background: var(--bg);
            border-radius: 4px;
            font-size: 0.75rem;
        }
        .meta-tag.high { background: rgba(16,185,129,0.2); color: var(--green); }

        .post-content {
            background: var(--bg);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            max-height: 200px;
            overflow-y: auto;
            white-space: pre-wrap;
            line-height: 1.6;
            font-size: 0.9rem;
        }
        .hook-line { color: var(--yellow); font-weight: 600; }

        .post-actions {
            display: flex;
            gap: 0.75rem;
            margin-top: 1rem;
        }

        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.8);
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        .modal-overlay.active { display: flex; }
        .modal {
            background: var(--card);
            border-radius: 16px;
            padding: 2rem;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }
        .modal h2 { margin-bottom: 1rem; }
        .modal-actions { display: flex; gap: 1rem; justify-content: flex-end; margin-top: 1.5rem; }

        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--muted);
        }
        .empty-state svg { width: 80px; height: 80px; margin-bottom: 1rem; opacity: 0.3; }
    </style>
</head>
<body>
    <div class="header">
        <h1>LinkedIn Post Tournament</h1>
        <p>Multi-agent ELO ranking with human-in-the-loop approval</p>
    </div>

    <div class="container">
        <div class="control-bar">
            <button class="btn btn-primary" id="startBtn" onclick="startTournament()">
                <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                Start Tournament
            </button>
            <div style="display:flex;gap:0.5rem;align-items:center;">
                <label style="font-size:0.875rem;color:var(--muted);">Variants:</label>
                <select id="variantCount" style="background:var(--card);border:1px solid var(--border);color:var(--text);padding:0.5rem;border-radius:4px;">
                    <option value="3">3</option>
                    <option value="5" selected>5</option>
                    <option value="7">7</option>
                    <option value="10">10</option>
                    <option value="15">15</option>
                </select>
            </div>
            <div style="display:flex;gap:0.5rem;align-items:center;">
                <label style="font-size:0.875rem;color:var(--muted);">Rounds:</label>
                <select id="roundCount" style="background:var(--card);border:1px solid var(--border);color:var(--text);padding:0.5rem;border-radius:4px;">
                    <option value="2">2</option>
                    <option value="3" selected>3</option>
                    <option value="5">5</option>
                    <option value="7">7</option>
                    <option value="10">10</option>
                </select>
            </div>
            <div class="status-badge">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Ready</span>
            </div>
        </div>

        <div class="phase-indicator">
            <div class="phase" id="phase1">
                <div class="phase-num">1</div>
                <div>Generate</div>
            </div>
            <div class="phase" id="phase2">
                <div class="phase-num">2</div>
                <div>QE Score</div>
            </div>
            <div class="phase" id="phase3">
                <div class="phase-num">3</div>
                <div>Proximity</div>
            </div>
            <div class="phase" id="phase4">
                <div class="phase-num">4</div>
                <div>Evolution</div>
            </div>
            <div class="phase" id="phase5">
                <div class="phase-num">5</div>
                <div>Tournament</div>
            </div>
            <div class="phase" id="phase6">
                <div class="phase-num">6</div>
                <div>Results</div>
            </div>
        </div>

        <div class="main-grid">
            <div class="tournament">
                <h2>
                    <svg width="24" height="24" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"/></svg>
                    Tournament Bracket
                </h2>
                <div id="bracketArea">
                    <div class="empty-state">
                        <svg fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
                        <h3>No tournament yet</h3>
                        <p>Click "Start Tournament" to generate and rank posts</p>
                    </div>
                </div>
            </div>

            <div class="live-log">
                <h2>Live Log</h2>
                <div class="log-entries" id="logEntries">
                    <div class="log-entry"><span class="time">--:--:--</span> Waiting to start...</div>
                </div>
            </div>
        </div>

        <!-- Generated Posts Preview (shown after generation, before tournament completes) -->
        <div class="results" id="previewArea" style="display: none;">
            <h2>Generated Post Variants</h2>
            <p style="color: var(--muted); text-align: center; margin-bottom: 1.5rem;">Posts generated from today's AI news - now running QE evaluation and ELO tournament...</p>
            <div class="post-cards" id="previewCards"></div>
        </div>

        <div class="results" id="resultsArea" style="display: none;">
            <h2>Final Rankings</h2>
            <div class="podium" id="podium"></div>
            <div class="post-cards" id="postCards"></div>
        </div>
    </div>

    <div class="modal-overlay" id="modal">
        <div class="modal">
            <h2>Publish to LinkedIn?</h2>
            <p style="color: var(--muted); margin-bottom: 1rem;">This will post the following content to your LinkedIn account:</p>
            <div class="post-content" id="modalContent"></div>
            <div class="modal-actions">
                <button class="btn" onclick="closeModal()" style="background: var(--border);">Cancel</button>
                <button class="btn btn-success" onclick="publishPost()">Publish Now</button>
            </div>
        </div>
    </div>

    <script>
        let pollInterval = null;
        let selectedPost = null;

        function startTournament() {
            document.getElementById('startBtn').disabled = true;
            document.getElementById('statusDot').className = 'status-dot active';
            document.getElementById('statusText').textContent = 'Starting...';
            document.getElementById('logEntries').innerHTML = '';
            document.getElementById('resultsArea').style.display = 'none';
            document.getElementById('bracketArea').innerHTML = '<div class="empty-state"><div class="status-dot active" style="width:40px;height:40px;margin:0 auto 1rem;"></div><h3>Generating variants...</h3></div>';

            // Reset phases
            document.querySelectorAll('.phase').forEach(p => p.classList.remove('active', 'complete'));

            const variants = document.getElementById('variantCount').value;
            const rounds = document.getElementById('roundCount').value;

            fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ variants: parseInt(variants), rounds: parseInt(rounds) })
            })
                .then(r => r.json())
                .then(data => {
                    if (data.started) {
                        pollInterval = setInterval(pollStatus, 1000);
                    }
                })
                .catch(err => {
                    addLog('Error starting: ' + err, 'error');
                    document.getElementById('startBtn').disabled = false;
                });
        }

        function pollStatus() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    updateUI(data);

                    if (data.status === 'complete' || data.status === 'error') {
                        clearInterval(pollInterval);
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('statusDot').className = data.status === 'error' ? 'status-dot error' : 'status-dot complete';
                    }
                });
        }

        function updateUI(data) {
            // Update status
            document.getElementById('statusText').textContent = data.phase || data.status;

            // Update phases (now 6 phases for Google Co-Scientist pipeline)
            const phaseMap = {
                'generating': 1,
                'evaluating': 2,
                'proximity': 3,
                'evolving': 4,
                'tournament': 5,
                'complete': 6
            };
            const currentPhase = phaseMap[data.status] || 0;
            for (let i = 1; i <= 6; i++) {
                const el = document.getElementById('phase' + i);
                if (!el) continue;
                el.classList.remove('active', 'complete');
                if (i < currentPhase) el.classList.add('complete');
                if (i === currentPhase) el.classList.add('active');
            }

            // Update logs
            if (data.logs && data.logs.length > 0) {
                const logEl = document.getElementById('logEntries');
                logEl.innerHTML = data.logs.map(log =>
                    `<div class="log-entry ${log.type || ''}"><span class="time">${log.time}</span> ${log.msg}</div>`
                ).join('');
                logEl.scrollTop = logEl.scrollHeight;
            }

            // Update bracket
            if (data.matches && data.matches.length > 0) {
                renderBracket(data.matches, data.variants || []);
            }

            // Update generated posts preview (show during tournament)
            if (data.generated_posts && data.generated_posts.length > 0 && data.status !== 'complete') {
                renderPreview(data.generated_posts, data.variants || []);
            }

            // Hide preview when complete, show final results
            if (data.status === 'complete') {
                document.getElementById('previewArea').style.display = 'none';
            }

            // Update results
            if (data.rankings && data.rankings.length > 0) {
                renderResults(data.rankings);
            }
        }

        function renderPreview(posts, variants) {
            document.getElementById('previewArea').style.display = 'block';

            // Create a map of variant data for QE scores
            const variantMap = {};
            variants.forEach(v => variantMap[v.id] = v);

            const cardsHtml = posts.map((p, i) => {
                const varData = variantMap[p.id] || {};
                const qeScore = varData.qe !== undefined ? varData.qe : '...';
                const qeClass = varData.qe >= 75 ? 'high' : '';

                return `
                <div class="post-card">
                    <div class="post-header">
                        <div class="post-badge">
                            <div class="post-rank-badge" style="background: var(--border); color: var(--text);">${i + 1}</div>
                            <div>
                                <div style="font-weight:600">${p.style.replace('_', ' ')}</div>
                                <div style="font-size:0.75rem;color:var(--muted)">${p.id}</div>
                            </div>
                        </div>
                        <div class="post-meta">
                            <span class="meta-tag ${qeClass}">QE: ${qeScore}/100</span>
                            <span class="meta-tag">${p.content.length} chars</span>
                        </div>
                    </div>
                    <div class="post-content"><span class="hook-line">${escapeHtml(p.content.split('\\n')[0].trim())}</span>\\n${escapeHtml(p.content.split('\\n').slice(1).join('\\n'))}</div>
                </div>`;
            }).join('');

            document.getElementById('previewCards').innerHTML = cardsHtml;
        }

        function renderBracket(matches, variants) {
            const variantMap = {};
            variants.forEach(v => variantMap[v.id] = v);

            // Group matches by round
            const rounds = {};
            matches.forEach(m => {
                if (!rounds[m.round]) rounds[m.round] = [];
                rounds[m.round].push(m);
            });

            let html = '<div class="bracket">';
            Object.keys(rounds).sort().forEach(roundNum => {
                html += `<div class="round"><div class="round-title">Round ${roundNum}</div>`;
                rounds[roundNum].forEach(match => {
                    const a = variantMap[match.a] || { name: match.a, elo: '?' };
                    const b = variantMap[match.b] || { name: match.b, elo: '?' };
                    const isComplete = match.winner !== null;
                    const isActive = match.active;

                    html += `<div class="match ${isActive ? 'active' : ''} ${isComplete ? 'complete' : ''}">`;
                    html += `<div class="fighter ${match.winner === match.a ? 'winner' : (isComplete ? 'loser' : '')}">
                        <span class="fighter-name">${a.name || match.a}</span>
                        <span class="fighter-score">ELO ${Math.round(a.elo || 1000)}</span>
                    </div>`;
                    html += `<div class="vs">${isActive ? '⚔️ FIGHTING' : 'vs'}</div>`;
                    html += `<div class="fighter ${match.winner === match.b ? 'winner' : (isComplete ? 'loser' : '')}">
                        <span class="fighter-name">${b.name || match.b}</span>
                        <span class="fighter-score">ELO ${Math.round(b.elo || 1000)}</span>
                    </div>`;
                    html += '</div>';
                });
                html += '</div>';
            });
            html += '</div>';

            document.getElementById('bracketArea').innerHTML = html;
        }

        function renderResults(rankings) {
            document.getElementById('resultsArea').style.display = 'block';

            // Podium
            const podiumHtml = rankings.slice(0, 3).map((r, i) => {
                const places = ['first', 'second', 'third'];
                const medals = ['🥇', '🥈', '🥉'];
                return `<div class="podium-place ${places[i]}">
                    <div class="podium-bar">
                        <div class="podium-rank">${medals[i]}</div>
                        <div class="podium-name">${r.style}</div>
                        <div class="podium-elo">${Math.round(r.elo)}</div>
                    </div>
                </div>`;
            }).join('');

            // Reorder for visual (2nd, 1st, 3rd)
            const podiumEl = document.getElementById('podium');
            if (rankings.length >= 3) {
                podiumEl.innerHTML = `
                    <div class="podium-place second">
                        <div class="podium-bar">
                            <div class="podium-rank">🥈</div>
                            <div class="podium-name">${rankings[1].style}</div>
                            <div class="podium-elo">${Math.round(rankings[1].elo)}</div>
                        </div>
                    </div>
                    <div class="podium-place first">
                        <div class="podium-bar">
                            <div class="podium-rank">🥇</div>
                            <div class="podium-name">${rankings[0].style}</div>
                            <div class="podium-elo">${Math.round(rankings[0].elo)}</div>
                        </div>
                    </div>
                    <div class="podium-place third">
                        <div class="podium-bar">
                            <div class="podium-rank">🥉</div>
                            <div class="podium-name">${rankings[2].style}</div>
                            <div class="podium-elo">${Math.round(rankings[2].elo)}</div>
                        </div>
                    </div>
                `;
            }

            // Post cards with evolution and debate info
            const cardsHtml = rankings.map((r, i) => {
                const isEvolved = r.generation && r.generation > 1;
                const evolutionBadge = isEvolved ?
                    '<span class="meta-tag" style="background:rgba(245,158,11,0.2);color:var(--yellow);">EVOLVED Gen ' + r.generation + '</span>' : '';
                const debateCount = r.debate_history ? r.debate_history.length : 0;

                return '<div class="post-card ' + (i === 0 ? 'gold' : '') + '">' +
                    '<div class="post-header">' +
                        '<div class="post-badge">' +
                            '<div class="post-rank-badge">' + (i + 1) + '</div>' +
                            '<div>' +
                                '<div style="font-weight:600">' + r.style.replace('_', ' ') + '</div>' +
                                '<div style="font-size:0.75rem;color:var(--muted)">' + r.id + '</div>' +
                            '</div>' +
                        '</div>' +
                        '<div class="post-meta">' +
                            '<span class="meta-tag ' + (r.qe_score >= 75 ? 'high' : '') + '">QE: ' + (r.qe_score || r.qe) + '/100</span>' +
                            '<span class="meta-tag">ELO: ' + Math.round(r.elo) + '</span>' +
                            '<span class="meta-tag">W' + r.wins + '-L' + r.losses + '</span>' +
                            '<span class="meta-tag">Debates: ' + debateCount + '</span>' +
                            evolutionBadge +
                        '</div>' +
                    '</div>' +
                    (isEvolved && r.evolved_from ?
                        '<div style="padding:0.5rem;background:rgba(245,158,11,0.1);border-radius:4px;margin-bottom:0.5rem;font-size:0.8rem;">' +
                            'Evolved from ' + r.evolved_from + ': ' + (r.evolution_feedback || 'Improved based on QE feedback') +
                        '</div>' : '') +
                    '<div class="post-content"><span class="hook-line">' + escapeHtml(r.content.split('\\n')[0].trim()) + '</span>\\n' + escapeHtml(r.content.split('\\n').slice(1).join('\\n')) + '</div>' +
                    (r.debate_history && r.debate_history.length > 0 ?
                        '<details style="margin:0.5rem 0;padding:0.5rem;background:var(--bg);border-radius:4px;">' +
                            '<summary style="cursor:pointer;font-size:0.85rem;color:var(--accent);">View Debate History (' + r.debate_history.length + ' debates)</summary>' +
                            '<div style="margin-top:0.5rem;font-size:0.8rem;">' +
                                r.debate_history.map(d =>
                                    '<div style="padding:0.25rem 0;border-bottom:1px solid var(--border);">' +
                                        (d.won ? 'Won' : 'Lost') + ' vs ' + d.opponent +
                                        (d.reasoning ? ': ' + d.reasoning.substring(0, 100) + '...' : '') +
                                    '</div>'
                                ).join('') +
                            '</div>' +
                        '</details>' : '') +
                    '<div class="post-actions">' +
                        '<button class="btn btn-success" onclick="selectForPublish(' + i + ')">' +
                            '<svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>' +
                            ' Select & Publish' +
                        '</button>' +
                        '<button class="btn" style="background:var(--border)" onclick="copyContent(' + i + ')">' +
                            '<svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>' +
                            ' Copy' +
                        '</button>' +
                    '</div>' +
                '</div>';
            }).join('');

            document.getElementById('postCards').innerHTML = cardsHtml;
            window.rankingsData = rankings;
        }

        function escapeHtml(text) {
            if (!text) return '';
            return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        function addLog(msg, type) {
            const time = new Date().toLocaleTimeString();
            const logEl = document.getElementById('logEntries');
            logEl.innerHTML += `<div class="log-entry ${type || ''}"><span class="time">${time}</span> ${msg}</div>`;
            logEl.scrollTop = logEl.scrollHeight;
        }

        function selectForPublish(index) {
            selectedPost = window.rankingsData[index];
            document.getElementById('modalContent').textContent = selectedPost.content;
            document.getElementById('modal').classList.add('active');
        }

        function closeModal() {
            document.getElementById('modal').classList.remove('active');
            selectedPost = null;
        }

        function publishPost() {
            if (!selectedPost) return;
            closeModal();

            document.getElementById('statusDot').className = 'status-dot active';
            document.getElementById('statusText').textContent = 'Publishing to LinkedIn...';

            fetch('/api/publish', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: selectedPost.content })
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('statusDot').className = data.success ? 'status-dot complete' : 'status-dot error';
                document.getElementById('statusText').textContent = data.message;
                addLog(data.message, data.success ? 'success' : 'error');
            });
        }

        function copyContent(index) {
            const content = window.rankingsData[index].content;
            navigator.clipboard.writeText(content);
            addLog('Copied post #' + (index + 1) + ' to clipboard', 'success');
        }
    </script>
</body>
</html>
'''


def log(msg, msg_type=""):
    """Add to ranking state logs"""
    ranking_state["logs"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "msg": msg,
        "type": msg_type
    })
    print(f"[{msg_type.upper() or 'INFO'}] {msg}")


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/status')
def get_status():
    return jsonify(ranking_state)


@app.route('/api/start', methods=['POST'])
def start_tournament():
    """Start tournament in background thread"""
    data = request.json or {}
    num_variants = data.get('variants', 5)
    num_rounds = data.get('rounds', 3)

    def run_tournament(num_variants, num_rounds):
        """Run the full Google Co-Scientist inspired pipeline"""
        global ranking_state
        ranking_state = {
            "status": "generating",
            "phase": "Initializing pipeline...",
            "current_step": 0,
            "total_steps": 6,
            "news_items": [],
            "variants": [],
            "generated_posts": [],
            "qe_results": [],
            "proximity_results": [],
            "evolution_results": [],
            "debates": [],
            "matches": [],
            "rankings": [],
            "logs": [],
            "error": None,
        }

        try:
            script_dir = Path(__file__).parent
            ai_news_db = script_dir / 'output_data' / 'ai_news.db'

            log(f"Starting Google Co-Scientist Pipeline: {num_variants} variants, {num_rounds} rounds", "step")

            # Import here to avoid circular imports
            from linkedin_autopilot import PostRankingSystem

            ranking_system = PostRankingSystem(ai_news_db_path=ai_news_db)

            # Progress callback to update UI state
            def on_progress(step, phase, data=None):
                ranking_state["current_step"] = step
                if phase == "news_loaded":
                    ranking_state["phase"] = f"Loaded {data['count']} news items"
                    log(f"Loaded {data['count']} news items from Twitter/X and web sources", "success")
                elif phase == "generating":
                    ranking_state["status"] = "generating"
                    ranking_state["phase"] = f"Generating {data['num_variants']} variants..."
                    log(f"[STEP 1] Generating {data['num_variants']} post variants...", "step")
                elif phase == "generated":
                    log(f"Generated {data['count']} variants", "success")
                elif phase == "evaluating":
                    ranking_state["status"] = "evaluating"
                    ranking_state["phase"] = f"QE evaluating {data['count']} variants..."
                    log(f"[STEP 2] Running QE Agent evaluation...", "step")
                elif phase == "evaluated":
                    ranking_state["qe_results"] = data["variants"]
                    # Update generated_posts with QE scores
                    for v in data["variants"]:
                        ranking_state["generated_posts"].append({
                            "id": v["id"],
                            "style": v["hook_style"],
                            "content": v["content"],
                            "qe_score": v["qe_score"],
                            "qe_feedback": v.get("qe_feedback", ""),
                        })
                        log(f"  {v['id']}: QE {v['qe_score']}/100", "")
                elif phase == "proximity_check":
                    ranking_state["status"] = "proximity"
                    ranking_state["phase"] = "Checking for duplicates..."
                    log(f"[STEP 3] Running Proximity Agent...", "step")
                elif phase == "proximity_done":
                    log(f"Found {data['duplicates']} duplicates, {data['unique']} unique variants", "success")
                elif phase == "proximity_skipped":
                    log("[STEP 3] Proximity check skipped", "")
                elif phase == "evolving":
                    ranking_state["status"] = "evolving"
                    ranking_state["phase"] = f"Evolving {data['count']} low-scoring posts..."
                    log(f"[STEP 4] Evolution Agent improving {data['count']} posts below QE {data['threshold']}...", "step")
                elif phase == "evolved":
                    log(f"Evolved {data['evolved_count']} posts", "success")
                elif phase == "evolution_skipped":
                    log("[STEP 4] All posts above threshold, skipping evolution", "")
                elif phase == "evolution_disabled":
                    log("[STEP 4] Evolution disabled", "")
                elif phase == "tournament_start":
                    ranking_state["status"] = "tournament"
                    ranking_state["phase"] = f"Tournament: {data['variants']} variants, {data['rounds']} rounds"
                    log(f"[STEP 5] ELO Tournament with debate-based ranking...", "step")
                elif phase == "tournament_done":
                    ranking_state["debates"] = data["debates"]
                    # Log debate count
                    log(f"Completed {len(data['debates'])} debates", "success")

            # Run the full pipeline
            ranked_variants = ranking_system.generate_and_rank_posts(
                num_variants=num_variants,
                elo_rounds=num_rounds,
                evolution_threshold=70,
                enable_evolution=True,
                enable_proximity=True,
                progress_callback=on_progress
            )

            # Update variants list for UI
            ranking_state["variants"] = [
                {
                    "id": v.variant_id,
                    "name": v.hook_style[:8],
                    "elo": v.elo_rating,
                    "style": v.hook_style,
                    "qe": v.qe_score,
                    "generation": v.generation,
                }
                for v in ranked_variants
            ]

            # Final rankings with full data
            ranking_state["rankings"] = [
                {
                    "id": v.variant_id,
                    "style": v.hook_style,
                    "elo": v.elo_rating,
                    "qe_score": v.qe_score,
                    "qe": v.qe_score,  # For backwards compatibility
                    "qe_feedback": v.qe_feedback,
                    "wins": v.wins,
                    "losses": v.losses,
                    "content": v.content,
                    "generation": v.generation,
                    "evolved_from": v.evolved_from,
                    "evolution_feedback": v.evolution_feedback,
                    "debate_history": v.debate_history,
                    "is_duplicate": v.is_duplicate,
                }
                for v in ranked_variants
            ]

            ranking_state["status"] = "complete"
            ranking_state["phase"] = "Pipeline complete!"

            # Save tournament results to database
            try:
                # Get news items used from ranking system
                news_items_used = ranking_system.pipeline_state.get("news_items", [])

                run_id = save_tournament_to_db(
                    db_path=ai_news_db,
                    num_variants=num_variants,
                    num_rounds=num_rounds,
                    ranked_variants=ranked_variants,
                    debates=ranking_state.get("debates", []),
                    news_sources=news_items_used
                )
                ranking_state["run_id"] = run_id
                ranking_state["news_items"] = news_items_used  # Store for UI
                log(f"Tournament saved to database (run #{run_id}) with {len(news_items_used)} source links", "success")
            except Exception as db_err:
                log(f"Warning: Could not save to database: {db_err}", "error")

            log("="*50, "step")
            log("FINAL RANKINGS", "step")
            for i, v in enumerate(ranked_variants[:3]):
                gen_tag = f" [EVOLVED Gen {v.generation}]" if v.generation > 1 else ""
                log(f"#{i+1}: {v.variant_id}{gen_tag} - ELO {v.elo_rating:.0f}, QE {v.qe_score}/100", "success")
            log("="*50, "step")
            log("Select a winner to publish!", "success")

        except Exception as e:
            ranking_state["status"] = "error"
            ranking_state["phase"] = f"Error: {str(e)}"
            ranking_state["error"] = str(e)
            log(f"Error: {str(e)}", "error")
            traceback.print_exc()

    thread = threading.Thread(target=run_tournament, args=(num_variants, num_rounds))
    thread.start()

    return jsonify({"started": True})


@app.route('/api/publish', methods=['POST'])
def publish():
    """Publish post to LinkedIn"""
    data = request.json
    content = data.get('content')

    if not content:
        return jsonify({"success": False, "message": "No content provided"})

    try:
        script_dir = Path(__file__).parent
        load_dotenv(script_dir / '.env')

        from linkedin_autopilot import LinkedInAutopilot

        output_dir = script_dir / 'output_data'
        autopilot = LinkedInAutopilot(output_dir)

        google_email = os.getenv('GOOGLE_EMAIL')
        linkedin_email = os.getenv('LINKEDIN_EMAIL')
        linkedin_password = os.getenv('LINKEDIN_PASSWORD')

        if not google_email and (not linkedin_email or not linkedin_password):
            return jsonify({
                "success": False,
                "message": "LinkedIn credentials not configured. Set GOOGLE_EMAIL or LINKEDIN_EMAIL/PASSWORD in .env"
            })

        log("Logging into LinkedIn...", "step")
        autopilot.login(
            email=linkedin_email,
            password=linkedin_password,
            google_email=google_email
        )

        log("Publishing post...", "step")
        success = autopilot.poster.create_post(content)
        autopilot.close()

        if success:
            # Mark tournament as published in database
            run_id = ranking_state.get("run_id")
            if run_id:
                try:
                    ai_news_db = script_dir / 'output_data' / 'ai_news.db'
                    mark_tournament_published(ai_news_db, run_id)
                    log(f"Tournament #{run_id} marked as published", "success")
                except Exception as e:
                    log(f"Warning: Could not update publish status: {e}", "error")
            return jsonify({"success": True, "message": "Posted successfully to LinkedIn!"})
        else:
            return jsonify({"success": False, "message": "Failed to publish"})

    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("LinkedIn Post Tournament")
    print("="*60)
    print("\nOpen http://localhost:5001 in your browser")
    print("Press Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
