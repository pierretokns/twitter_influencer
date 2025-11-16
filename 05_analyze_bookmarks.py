# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "python-dotenv>=0.19.0",
# ]
# ///

"""
Twitter Bookmarks Analyzer
Analyzes saved bookmark data and provides filtering/search capabilities
"""

import os
import json
import csv
from datetime import datetime
from collections import Counter

def load_bookmarks(filepath):
    """Load bookmarks from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return []

def analyze_bookmarks(bookmarks):
    """Provide comprehensive analysis of bookmarks"""
    print("\n" + "="*60)
    print("BOOKMARK ANALYSIS")
    print("="*60)
    
    print(f"\n📊 Total Bookmarks: {len(bookmarks)}")
    
    if not bookmarks:
        print("No bookmarks to analyze.")
        return
    
    # Media type distribution
    media_types = [b.get('media_type', 'none') for b in bookmarks]
    media_counter = Counter(media_types)
    print(f"\n📷 Media Distribution:")
    for media_type, count in media_counter.most_common():
        percentage = (count / len(bookmarks)) * 100
        print(f"   {media_type}: {count} ({percentage:.1f}%)")
    
    # Top users
    usernames = [b.get('username', 'N/A') for b in bookmarks]
    user_counter = Counter(usernames)
    print(f"\n👤 Top 10 Users:")
    for username, count in user_counter.most_common(10):
        print(f"   {username}: {count} bookmarks")
    
    # Engagement statistics
    total_likes = sum(int(b.get('likes', '0').replace(',', '')) for b in bookmarks if b.get('likes', '0') != 'N/A')
    total_retweets = sum(int(b.get('retweets', '0').replace(',', '')) for b in bookmarks if b.get('retweets', '0') != 'N/A')
    total_replies = sum(int(b.get('replies', '0').replace(',', '')) for b in bookmarks if b.get('replies', '0') != 'N/A')
    
    print(f"\n💬 Engagement Totals:")
    print(f"   Likes: {total_likes:,}")
    print(f"   Retweets: {total_retweets:,}")
    print(f"   Replies: {total_replies:,}")
    
    # Retweets vs original
    retweets = [b for b in bookmarks if b.get('is_retweet') == 'Yes']
    print(f"\n🔄 Retweets: {len(retweets)} ({len(retweets)/len(bookmarks)*100:.1f}%)")
    print(f"   Original tweets: {len(bookmarks) - len(retweets)}")
    
    # Date range (if timestamps available)
    timestamps = [b.get('timestamp') for b in bookmarks if b.get('timestamp') != 'N/A']
    if timestamps:
        dates = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
        oldest = min(dates)
        newest = max(dates)
        print(f"\n📅 Date Range:")
        print(f"   Oldest: {oldest.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Newest: {newest.strftime('%Y-%m-%d %H:%M')}")

def search_bookmarks(bookmarks, query):
    """Search bookmarks by text content"""
    query_lower = query.lower()
    results = []
    
    for bookmark in bookmarks:
        text = bookmark.get('text', '').lower()
        username = bookmark.get('username', '').lower()
        display_name = bookmark.get('display_name', '').lower()
        
        if query_lower in text or query_lower in username or query_lower in display_name:
            results.append(bookmark)
    
    return results

def filter_by_user(bookmarks, username):
    """Filter bookmarks by username"""
    username_lower = username.lower()
    return [b for b in bookmarks if username_lower in b.get('username', '').lower()]

def filter_by_media(bookmarks, media_type):
    """Filter bookmarks by media type"""
    return [b for b in bookmarks if b.get('media_type', 'none') == media_type]

def filter_by_date(bookmarks, start_date=None, end_date=None):
    """Filter bookmarks by date range"""
    results = []
    
    for bookmark in bookmarks:
        timestamp = bookmark.get('timestamp')
        if timestamp and timestamp != 'N/A':
            try:
                tweet_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                if start_date and tweet_date < start_date:
                    continue
                if end_date and tweet_date > end_date:
                    continue
                
                results.append(bookmark)
            except:
                continue
    
    return results

def export_filtered(bookmarks, output_path, format='json'):
    """Export filtered bookmarks"""
    if format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(bookmarks, f, indent=2, ensure_ascii=False)
    elif format == 'csv':
        if bookmarks:
            keys = bookmarks[0].keys()
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(bookmarks)
    
    print(f"✓ Exported {len(bookmarks)} bookmarks to: {output_path}")

def interactive_menu(bookmarks):
    """Interactive menu for analyzing and filtering bookmarks"""
    while True:
        print("\n" + "="*60)
        print("BOOKMARK ANALYZER MENU")
        print("="*60)
        print(f"Currently working with: {len(bookmarks)} bookmarks")
        print("\nOptions:")
        print("1. Show analysis")
        print("2. Search by text")
        print("3. Filter by user")
        print("4. Filter by media type")
        print("5. Filter by date range")
        print("6. Export current results")
        print("7. Reset to all bookmarks")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            analyze_bookmarks(bookmarks)
        
        elif choice == '2':
            query = input("Enter search query: ").strip()
            results = search_bookmarks(bookmarks, query)
            print(f"\nFound {len(results)} bookmarks matching '{query}'")
            if results and input("View results? (y/n): ").lower() == 'y':
                for i, b in enumerate(results[:10], 1):
                    print(f"\n{i}. @{b.get('username')}")
                    print(f"   {b.get('text', 'N/A')[:100]}...")
                if len(results) > 10:
                    print(f"\n... and {len(results) - 10} more")
            bookmarks = results
        
        elif choice == '3':
            username = input("Enter username (without @): ").strip()
            results = filter_by_user(bookmarks, username)
            print(f"\nFound {len(results)} bookmarks from @{username}")
            bookmarks = results
        
        elif choice == '4':
            print("\nMedia types: photo, video, gif, none")
            media_type = input("Enter media type: ").strip().lower()
            results = filter_by_media(bookmarks, media_type)
            print(f"\nFound {len(results)} bookmarks with {media_type}")
            bookmarks = results
        
        elif choice == '5':
            print("\nEnter dates in YYYY-MM-DD format (leave blank to skip)")
            start = input("Start date: ").strip()
            end = input("End date: ").strip()
            
            start_date = datetime.fromisoformat(start) if start else None
            end_date = datetime.fromisoformat(end) if end else None
            
            results = filter_by_date(bookmarks, start_date, end_date)
            print(f"\nFound {len(results)} bookmarks in date range")
            bookmarks = results
        
        elif choice == '6':
            filename = input("Enter output filename (without extension): ").strip()
            format_choice = input("Format (json/csv): ").strip().lower()
            
            output_dir = os.path.join(os.path.dirname(__file__), 'output_data')
            os.makedirs(output_dir, exist_ok=True)
            
            ext = 'json' if format_choice == 'json' else 'csv'
            output_path = os.path.join(output_dir, f"{filename}.{ext}")
            
            export_filtered(bookmarks, output_path, format_choice)
        
        elif choice == '7':
            print("\nTo reset, please reload the original file")
            return
        
        elif choice == '8':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output_data')
    
    # Check if output directory exists
    if not os.path.exists(output_dir):
        print(f"❌ No output_data directory found at: {output_dir}")
        print("   Please run the scraper first to generate bookmark data.")
        return
    
    # List available JSON files
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    
    if not json_files:
        print("❌ No bookmark files found in output_data/")
        print("   Please run the scraper first to generate bookmark data.")
        return
    
    print("\n" + "="*60)
    print("Available bookmark files:")
    print("="*60)
    for i, filename in enumerate(json_files, 1):
        filepath = os.path.join(output_dir, filename)
        size = os.path.getsize(filepath)
        modified = datetime.fromtimestamp(os.path.getmtime(filepath))
        print(f"{i}. {filename}")
        print(f"   Size: {size:,} bytes | Modified: {modified.strftime('%Y-%m-%d %H:%M')}")
    
    # Let user choose file
    try:
        choice = int(input("\nSelect file number: ").strip())
        if 1 <= choice <= len(json_files):
            selected_file = json_files[choice - 1]
            filepath = os.path.join(output_dir, selected_file)
            
            print(f"\nLoading: {selected_file}")
            bookmarks = load_bookmarks(filepath)
            
            if bookmarks:
                print(f"✓ Loaded {len(bookmarks)} bookmarks")
                
                # Show initial analysis
                analyze_bookmarks(bookmarks)
                
                # Start interactive menu
                if input("\nEnter interactive mode? (y/n): ").lower() == 'y':
                    interactive_menu(bookmarks)
            else:
                print("❌ Failed to load bookmarks or file is empty")
        else:
            print("❌ Invalid selection")
    
    except ValueError:
        print("❌ Invalid input")
    except KeyboardInterrupt:
        print("\n\nExiting...")

if __name__ == "__main__":
    main()
