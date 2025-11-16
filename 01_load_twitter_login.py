# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "selenium>=4.0.0",
#     "python-dotenv>=0.19.0",
# ]
# ///

"""
Twitter Login Page Loader
This script loads the Twitter login page, saves the HTML, and takes a screenshot
for analysis to determine the correct selectors for logging in.
"""

import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def setup_driver():
    """Configure and return a Chrome WebDriver instance"""
    chrome_options = Options()
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def main():
    # Get script directory for saving files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_pages_dir = os.path.join(script_dir, 'output_pages')
    output_images_dir = os.path.join(script_dir, 'output_images')
    
    # Create directories if they don't exist
    os.makedirs(output_pages_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    
    driver = None
    
    try:
        print("Setting up Chrome driver...")
        driver = setup_driver()
        
        print("Navigating to Twitter login page...")
        driver.get("https://twitter.com/i/flow/login")
        
        # Wait for page to load
        time.sleep(5)
        
        # Save page source
        page_source_path = os.path.join(output_pages_dir, '01_twitter_login_page.html')
        with open(page_source_path, 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        print(f"✓ Page source saved to: {page_source_path}")
        
        # Take screenshot
        screenshot_path = os.path.join(output_images_dir, '01_twitter_login_page.png')
        driver.save_screenshot(screenshot_path)
        print(f"✓ Screenshot saved to: {screenshot_path}")
        
        print("\n" + "="*60)
        print("Page loaded successfully!")
        print("Next steps:")
        print("1. Analyze the saved HTML to find login form selectors")
        print("2. Create login script with proper credentials")
        print("="*60)
        
        # Keep browser open for a moment
        time.sleep(2)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        if driver:
            error_screenshot = os.path.join(output_images_dir, '01_error_screenshot.png')
            driver.save_screenshot(error_screenshot)
            print(f"Error screenshot saved to: {error_screenshot}")
        raise
    
    finally:
        if driver:
            print("\nClosing browser...")
            driver.quit()

if __name__ == "__main__":
    main()
