# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "undetected-chromedriver>=3.5.0",
#     "python-dotenv>=0.19.0",
#     "setuptools>=65.0.0",
#     "certifi>=2023.0.0",
# ]
# ///

"""
Twitter Bookmarks Scraper - Step 1: Login and Navigate to Bookmarks
This script logs into Twitter, navigates to bookmarks, and saves the page for analysis.
"""

import os
import time
import json
import random
import ssl
import certifi
from dotenv import load_dotenv
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

# Fix SSL certificate verification for macOS/Python 3.13
ssl._create_default_https_context = ssl._create_unverified_context

def human_delay(min_sec=1, max_sec=3):
    """Add random delay to mimic human behavior"""
    time.sleep(random.uniform(min_sec, max_sec))

def setup_driver():
    """Configure and return an undetected Chrome WebDriver instance"""
    options = uc.ChromeOptions()
    options.add_argument('--start-maximized')
    options.add_argument('--disable-blink-features=AutomationControlled')

    # Use undetected chromedriver to bypass bot detection
    driver = uc.Chrome(options=options, version_main=None)

    return driver

def save_page_state(driver, step_name, script_dir):
    """Save page source and screenshot for analysis"""
    output_pages_dir = os.path.join(script_dir, 'output_pages')
    output_images_dir = os.path.join(script_dir, 'output_images')
    
    os.makedirs(output_pages_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    
    # Save HTML
    html_path = os.path.join(output_pages_dir, f'{step_name}.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(driver.page_source)
    print(f"✓ Saved HTML: {html_path}")
    
    # Save screenshot
    screenshot_path = os.path.join(output_images_dir, f'{step_name}.png')
    driver.save_screenshot(screenshot_path)
    print(f"✓ Saved screenshot: {screenshot_path}")
    
    return html_path, screenshot_path

def login_to_twitter(driver, username, password, script_dir):
    """Log into Twitter with provided credentials"""
    try:
        print("\n" + "="*60)
        print("STEP 1: Loading Twitter login page...")
        print("="*60)

        # Visit homepage first (more human-like)
        print("Visiting Twitter homepage first...")
        driver.get("https://twitter.com")
        human_delay(2, 4)

        driver.get("https://twitter.com/i/flow/login")
        human_delay(3, 5)

        # Save initial login page
        save_page_state(driver, "01_login_page_initial", script_dir)

        print("\nSTEP 2: Entering username...")
        # Wait for username input and enter username
        username_input = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[autocomplete="username"]'))
        )

        # Type slowly like a human
        for char in username:
            username_input.send_keys(char)
            time.sleep(random.uniform(0.05, 0.15))

        human_delay(0.5, 1.5)
        username_input.send_keys(Keys.RETURN)
        human_delay(2, 4)

        save_page_state(driver, "02_after_username", script_dir)
        
        # Check if there's an unusual activity check or verification
        try:
            page_text = driver.find_element(By.TAG_NAME, 'body').text.lower()
            if 'unusual' in page_text or 'verify' in page_text or 'suspicious' in page_text:
                print("\n⚠️  VERIFICATION REQUIRED!")
                print("Twitter is asking for verification.")
                save_page_state(driver, "03_verification_required", script_dir)
                print("\nPlease complete the verification manually in the browser window")
                print("Press Enter when done...")
                input()
        except:
            pass  # No unusual activity check

        print("\nSTEP 3: Entering password...")
        # Wait for password input and enter password
        password_input = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="password"]'))
        )

        # Type password slowly like a human
        for char in password:
            password_input.send_keys(char)
            time.sleep(random.uniform(0.05, 0.15))

        human_delay(0.5, 1.5)
        password_input.send_keys(Keys.RETURN)
        human_delay(3, 6)

        save_page_state(driver, "04_after_password", script_dir)

        # Check for login errors
        try:
            page_text = driver.find_element(By.TAG_NAME, 'body').text.lower()
            if 'could not' in page_text or 'wrong' in page_text or 'please try again' in page_text:
                print("\n❌ Login error detected!")
                save_page_state(driver, "error_login_blocked", script_dir)
                print("Twitter blocked the login attempt.")
                print("\nTroubleshooting:")
                print("1. Try logging in manually through a regular browser first")
                print("2. Wait 15-30 minutes before trying again")
                print("3. Use a different network/IP if possible")
                return False
        except:
            pass

        # Check if login was successful
        try:
            WebDriverWait(driver, 15).until(
                lambda d: "home" in d.current_url.lower() or
                         len(d.find_elements(By.CSS_SELECTOR, '[data-testid="AppTabBar_Home_Link"]')) > 0 or
                         len(d.find_elements(By.CSS_SELECTOR, '[data-testid="SideNav_AccountSwitcher_Button"]')) > 0
            )
            print("✓ Login successful!")
            return True
        except:
            print("⚠️  Login status unclear. Please check the browser.")
            save_page_state(driver, "login_unclear", script_dir)
            print("\nIf you see your Twitter home feed, press Enter to continue...")
            print("Otherwise, press Ctrl+C to exit.")
            input()
            return True

    except Exception as e:
        print(f"\n❌ Login failed: {str(e)}")
        save_page_state(driver, "error_login", script_dir)
        return False

def navigate_to_bookmarks(driver, script_dir):
    """Navigate to the bookmarks page"""
    try:
        print("\n" + "="*60)
        print("STEP 4: Navigating to bookmarks...")
        print("="*60)
        
        driver.get("https://twitter.com/i/bookmarks")
        time.sleep(5)
        
        save_page_state(driver, "05_bookmarks_page", script_dir)
        
        print("✓ Reached bookmarks page!")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to reach bookmarks: {str(e)}")
        save_page_state(driver, "error_bookmarks_nav", script_dir)
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
        
        # Step 1: Login
        if not login_to_twitter(driver, username, password, script_dir):
            return
        
        # Step 2: Navigate to bookmarks
        if not navigate_to_bookmarks(driver, script_dir):
            return
        
        print("\n" + "="*60)
        print("SUCCESS! Initial setup complete.")
        print("="*60)
        print("\nNext steps:")
        print("1. Review the saved HTML files in output_pages/")
        print("2. Review the screenshots in output_images/")
        print("3. Run the next script to extract bookmark data")
        print("="*60)
        
        # Keep browser open to see the result
        input("\nPress Enter to close the browser...")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        if driver:
            error_screenshot = os.path.join(script_dir, 'output_images', 'error_final.png')
            driver.save_screenshot(error_screenshot)
            print(f"Error screenshot saved to: {error_screenshot}")
        raise
    
    finally:
        if driver:
            print("\nClosing browser...")
            driver.quit()

if __name__ == "__main__":
    main()
