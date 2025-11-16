# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "selenium>=4.0.0",
#     "python-dotenv>=0.19.0",
# ]
# ///

"""
Twitter Bookmarks Scraper - Step 2: Extract Bookmark Data
This script logs in, loads bookmarks, and extracts all bookmark data with scrolling.
"""

import os
import time
import json
import csv
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

def setup_driver():
    """Configure and return a Chrome WebDriver instance"""
    chrome_options = Options()
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def login_to_twitter(driver, username, password):
    """Log into Twitter"""
    try:
        print("Logging into Twitter...")
        driver.get("https://twitter.com/i/flow/login")
        time.sleep(3)
        
        # Enter username
        username_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[autocomplete="username"]'))
        )
        username_input.send_keys(username)
        username_input.send_keys(Keys.RETURN)
        time.sleep(3)
        
        # Enter password
        password_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="password"]'))
        )
        password_input.send_keys(password)
        password_input.send_keys(Keys.RETURN)
        time.sleep(5)
        
        print("✓ Login successful!")
        return True
        
    except Exception as e:
        print(f"❌ Login failed: {str(e)}")
        return False

def extract_tweet_data(tweet_element):
    """Extract data from a single tweet element"""
    try:
        tweet_data = {}
        
        # Try to extract username
        try:
            username_elem = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"]')
            tweet_data['username'] = username_elem.text
        except:
            tweet_data['username'] = 'N/A'
        
        # Try to extract tweet text
        try:
            text_elem = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]')
            tweet_data['text'] = text_elem.text
        except:
            tweet_data['text'] = 'N/A'
        
        # Try to extract timestamp
        try:
            time_elem = tweet_element.find_element(By.CSS_SELECTOR, 'time')
            tweet_data['timestamp'] = time_elem.get_attribute('datetime')
        except:
            tweet_data['timestamp'] = 'N/A'
        
        # Try to extract tweet link
        try:
            link_elem = tweet_element.find_element(By.CSS_SELECTOR, 'a[href*="/status/"]')
            tweet_data['link'] = link_elem.get_attribute('href')
        except:
            tweet_data['link'] = 'N/A'
        
        # Try to extract engagement metrics
        try:
            # Replies
            reply_elem = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="reply"]')
            tweet_data['replies'] = reply_elem.text
        except:
            tweet_data['replies'] = '0'
        
        try:
            # Retweets
            retweet_elem = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="retweet"]')
            tweet_data['retweets'] = retweet_elem.text
        except:
            tweet_data['retweets'] = '0'
        
        try:
            # Likes
            like_elem = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="like"]')
            tweet_data['likes'] = like_elem.text
        except:
            tweet_data['likes'] = '0'
        
        # Check if tweet has media
        try:
            media_elem = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="tweetPhoto"], [data-testid="videoPlayer"]')
            tweet_data['has_media'] = 'Yes'
        except:
            tweet_data['has_media'] = 'No'
        
        return tweet_data
        
    except Exception as e:
        print(f"⚠️  Error extracting tweet data: {str(e)}")
        return None

def scrape_bookmarks(driver, script_dir, max_scrolls=50):
    """Scrape all bookmarks with scrolling"""
    try:
        print("\n" + "="*60)
        print("Starting bookmark extraction...")
        print("="*60)
        
        # Navigate to bookmarks
        driver.get("https://twitter.com/i/bookmarks")
        time.sleep(5)
        
        all_tweets = []
        seen_tweet_ids = set()
        scroll_count = 0
        no_new_tweets_count = 0
        
        while scroll_count < max_scrolls:
            scroll_count += 1
            print(f"\nScroll {scroll_count}/{max_scrolls}")
            
            # Find all tweet articles on the current page
            tweet_elements = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')
            print(f"Found {len(tweet_elements)} tweet elements on page")
            
            new_tweets_this_scroll = 0
            
            for tweet_elem in tweet_elements:
                # Try to get a unique identifier for the tweet
                try:
                    tweet_link = tweet_elem.find_element(By.CSS_SELECTOR, 'a[href*="/status/"]')
                    tweet_id = tweet_link.get_attribute('href')
                    
                    if tweet_id not in seen_tweet_ids:
                        seen_tweet_ids.add(tweet_id)
                        tweet_data = extract_tweet_data(tweet_elem)
                        
                        if tweet_data:
                            all_tweets.append(tweet_data)
                            new_tweets_this_scroll += 1
                            print(f"  ✓ Extracted tweet {len(all_tweets)}: {tweet_data.get('text', 'N/A')[:50]}...")
                
                except Exception as e:
                    continue
            
            print(f"  New tweets extracted this scroll: {new_tweets_this_scroll}")
            
            # Check if we're still finding new tweets
            if new_tweets_this_scroll == 0:
                no_new_tweets_count += 1
                if no_new_tweets_count >= 3:
                    print("\n✓ No new tweets found after 3 scrolls. Reached end of bookmarks.")
                    break
            else:
                no_new_tweets_count = 0
            
            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Save progress periodically
            if scroll_count % 10 == 0:
                save_bookmarks(all_tweets, script_dir, f"bookmarks_progress_{scroll_count}.json")
        
        print(f"\n" + "="*60)
        print(f"Extraction complete! Total bookmarks: {len(all_tweets)}")
        print("="*60)
        
        return all_tweets
        
    except Exception as e:
        print(f"\n❌ Error during scraping: {str(e)}")
        return all_tweets if 'all_tweets' in locals() else []

def save_bookmarks(bookmarks, script_dir, filename="twitter_bookmarks.json"):
    """Save bookmarks to JSON and CSV files"""
    try:
        output_dir = os.path.join(script_dir, 'output_data')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        json_path = os.path.join(output_dir, filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(bookmarks, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved JSON: {json_path}")
        
        # Save as CSV
        if bookmarks:
            csv_filename = filename.replace('.json', '.csv')
            csv_path = os.path.join(output_dir, csv_filename)
            
            keys = bookmarks[0].keys()
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(bookmarks)
            print(f"✓ Saved CSV: {csv_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving bookmarks: {str(e)}")
        return False

def main():
    # Load environment variables
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    load_dotenv(env_path)
    
    username = os.getenv('TWITTER_USERNAME')
    password = os.getenv('TWITTER_PASSWORD')
    
    if not username or not password or username == 'your_username_here':
        print("❌ Error: Please set TWITTER_USERNAME and TWITTER_PASSWORD in .env file")
        print(f"   Location: {env_path}")
        return
    
    driver = None
    
    try:
        print("Setting up Chrome driver...")
        driver = setup_driver()
        
        # Login
        if not login_to_twitter(driver, username, password):
            return
        
        # Scrape bookmarks
        bookmarks = scrape_bookmarks(driver, script_dir, max_scrolls=50)
        
        if bookmarks:
            # Save final results
            save_bookmarks(bookmarks, script_dir, "twitter_bookmarks_final.json")
            
            print("\n" + "="*60)
            print("SUCCESS!")
            print(f"Total bookmarks extracted: {len(bookmarks)}")
            print("="*60)
        else:
            print("\n⚠️  No bookmarks were extracted.")
        
        # Keep browser open to review
        input("\nPress Enter to close the browser...")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        if driver:
            screenshot_path = os.path.join(script_dir, 'output_images', 'error_extraction.png')
            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
            driver.save_screenshot(screenshot_path)
            print(f"Error screenshot saved to: {screenshot_path}")
        raise
    
    finally:
        if driver:
            print("\nClosing browser...")
            driver.quit()

if __name__ == "__main__":
    main()
