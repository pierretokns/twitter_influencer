#!/usr/bin/env python3
"""
Fetch YouTube channel IDs by hitting RSS feed URLs.
RSS feeds reveal channel_id in the URL parameters.

Usage:
    python fetch_channel_ids.py
"""

import requests
import re
from typing import Optional, Dict, List

# YouTube handles to lookup
HANDLES_TO_LOOKUP = [
    "@DailyHQ",  # Pipecat
    "@elevenlabsio",  # ElevenLabs
    "@MistralAIofficial",  # Mistral AI
    "@liquid-ai-inc",  # Liquid AI
    "@TheBrowserCompany",  # Arc
    "@perplexity_ai",  # Perplexity AI
    "@modal_labs",  # Modal
    "@stabilityai",  # Stability AI
    "@togethercompute",  # Together AI
]

def get_channel_id_from_rss(handle: str) -> Optional[str]:
    """
    Try to extract channel_id from RSS feed.
    YouTube redirects @handle URLs to traditional URLs with channel_id.
    """
    # First, try direct RSS URL lookup via YouTube's API redirect
    # When you visit youtube.com/@handle, YouTube serves a page that contains the channel_id

    try:
        # Visit the handle URL and look for channelId in the page source
        url = f"https://www.youtube.com/{handle}"
        response = requests.get(url, timeout=10, allow_redirects=True)

        # Look for channelId in the page HTML
        match = re.search(r'"channelId":"([^"]+)"', response.text)
        if match:
            return match.group(1)

        # Alternative pattern
        match = re.search(r'channelId["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{24})', response.text)
        if match:
            return match.group(1)

    except Exception as e:
        print(f"[!] Error fetching {handle}: {e}")

    return None

def verify_channel_id(channel_id: str) -> bool:
    """Verify a channel ID works by checking its RSS feed."""
    try:
        rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        response = requests.head(rss_url, timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("=" * 70)
    print("YouTube Channel ID Extractor")
    print("=" * 70)
    print()

    results: Dict[str, Optional[str]] = {}

    for handle in HANDLES_TO_LOOKUP:
        print(f"[*] Looking up {handle}...", end=" ", flush=True)
        channel_id = get_channel_id_from_rss(handle)

        if channel_id:
            if verify_channel_id(channel_id):
                print(f"✓ {channel_id}")
                results[handle] = channel_id
            else:
                print(f"[?] Found {channel_id} but RSS verify failed")
                results[handle] = channel_id  # Still return it
        else:
            print("[!] Not found")
            results[handle] = None

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    found = 0
    for handle, channel_id in results.items():
        status = "✓" if channel_id else "✗"
        cid = channel_id or "(not found)"
        print(f"  {status} {handle:<25} {cid}")
        if channel_id:
            found += 1

    print()
    print(f"Found: {found}/{len(HANDLES_TO_LOOKUP)}")
    print()
    print("SEED_CHANNELS format to add:")
    print("-" * 70)
    for handle, channel_id in results.items():
        if channel_id:
            # Extract company name from handle
            name = handle.lstrip("@").replace("ai", "AI").title()
            print(f'    ("{channel_id}", "{name}", "official"),')

if __name__ == "__main__":
    main()
