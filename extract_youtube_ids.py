#!/usr/bin/env python3
"""
Extract YouTube channel IDs by analyzing page HTML for embedded JSON data.
YouTube embeds the channelId in the initial HTML as JSON.
"""

import re
import requests
from typing import Optional

# Map of handle to expected channel info
CHANNELS = {
    "DailyHQ": "Daily (Pipecat)",
    "elevenlabsio": "ElevenLabs",
    "MistralAIofficial": "Mistral AI",
    "liquid-ai-inc": "Liquid AI",
    "TheBrowserCompany": "Arc (Browser Company)",
    "perplexity_ai": "Perplexity AI",
    "modal_labs": "Modal",
    "stabilityai": "Stability AI",
    "togethercompute": "Together AI",
}

def extract_channel_id_from_handle(handle: str) -> Optional[str]:
    """Extract channel ID from a YouTube handle by fetching and parsing the page."""
    url = f"https://www.youtube.com/@{handle}"

    try:
        # Fetch the page with a proper user agent
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        response.raise_for_status()

        html = response.text

        # Pattern 1: Look for "channelId" in JSON-like structures
        match = re.search(r'"channelId":"([^"]+)"', html)
        if match:
            return match.group(1)

        # Pattern 2: Look for channelId with different quote styles
        match = re.search(r'channelId["\']?\s*:\s*["\']([^"\']+)["\']', html)
        if match:
            return match.group(1)

        # Pattern 3: Look in meta tags
        match = re.search(r'<meta\s+property="og:url"\s+content="[^"]*(?:channel/|@)([^/"]+)', html)
        if match:
            channel_id = match.group(1)
            # If it starts with UC, it's a valid channel ID
            if channel_id.startswith('UC'):
                return channel_id

        # Pattern 4: Browse feature data - YouTube embeds this for the initial page load
        match = re.search(r'"browseId":"(UC[A-Za-z0-9_-]{22})"', html)
        if match:
            return match.group(1)

    except Exception as e:
        print(f"  [!] Error fetching {handle}: {e}")

    return None

def verify_channel_id(channel_id: str) -> bool:
    """Test if channel ID is valid by checking RSS feed."""
    try:
        rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        response = requests.head(rss_url, timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("=" * 80)
    print("YouTube Channel ID Extractor")
    print("=" * 80)
    print()

    results = {}

    for handle, display_name in CHANNELS.items():
        print(f"[*] {display_name:<30} (@{handle})...", end=" ", flush=True)

        channel_id = extract_channel_id_from_handle(handle)

        if channel_id:
            # Verify it works
            if verify_channel_id(channel_id):
                print(f"✓ {channel_id}")
                results[display_name] = (handle, channel_id)
            else:
                print(f"? {channel_id} (RSS unverified)")
                results[display_name] = (handle, channel_id)
        else:
            print("[!] Not found")
            results[display_name] = (handle, None)

    print()
    print("=" * 80)
    print("RESULTS - Ready for SEED_CHANNELS")
    print("=" * 80)
    print()

    found_count = sum(1 for _, (_, cid) in results.items() if cid)

    for display_name, (handle, channel_id) in results.items():
        if channel_id:
            # Format for Python tuple
            print(f'    ("{channel_id}", "{display_name}", "official"),  # @{handle}')

    print()
    print(f"Successfully extracted: {found_count}/{len(CHANNELS)}")
    print()
    print("=" * 80)
    print("NOT FOUND (may need to add manually or via API):")
    print("=" * 80)
    for display_name, (handle, channel_id) in results.items():
        if not channel_id:
            print(f"  - {display_name} (@{handle})")

if __name__ == "__main__":
    main()
