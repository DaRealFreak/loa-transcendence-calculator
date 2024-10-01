import os
import shutil
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.firefox import GeckoDriverManager

# Define the local bin directory path
BIN_DIR = os.path.join(os.getcwd(), 'bin')
GECKODRIVER_PATH = os.path.join(BIN_DIR, 'geckodriver')


def download_geckodriver_to_bin():
    """
    Download geckodriver using webdriver_manager and move it to the bin directory.
    """
    print("Geckodriver not found locally. Downloading...")
    # Use webdriver_manager to download geckodriver
    gecko_path = GeckoDriverManager().install()

    # Ensure the bin directory exists
    os.makedirs(BIN_DIR, exist_ok=True)

    # Move the downloaded geckodriver to the bin folder
    shutil.copy(gecko_path, GECKODRIVER_PATH)
    print(f"Geckodriver downloaded and saved to {GECKODRIVER_PATH}")


def setup_firefox():
    """
    Setup Firefox WebDriver using geckodriver, either from the local bin directory or downloading it on first run.
    """
    # Check if geckodriver is already in the bin folder
    if not os.path.exists(GECKODRIVER_PATH):
        # If geckodriver is not found, download it and place it in the bin folder
        download_geckodriver_to_bin()

    # Setup Firefox options
    firefox_options = Options()
    firefox_options.add_argument("--headless")  # Run in headless mode

    # Initialize the WebDriver with the local geckodriver binary
    service = Service(GECKODRIVER_PATH)

    # Start Firefox with the specified driver and options
    driver = webdriver.Firefox(service=service, options=firefox_options)

    return driver


def toggle_checkbox_and_confirm(driver):
    try:
        # Wait for the parent <div> to be present
        parent_div = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.bg-white.dark\\:bg-zinc-700"))
        )

        # Find the checkbox within the parent <div> and toggle it
        checkbox = parent_div.find_element(By.CSS_SELECTOR, "input[type='checkbox'][role='switch']")
        driver.execute_script("arguments[0].click();", checkbox)
        print("Checkbox toggled.")

        # Wait for the Confirm button inside the parent <div> and click it
        confirm_button = parent_div.find_element(By.XPATH,
                                                 ".//button[contains(text(), '확인') or contains(text(), 'Confirm')]")
        driver.execute_script("arguments[0].click();", confirm_button)
        print("Confirm button clicked.")

    except Exception as e:
        print(f"Error occurred: {e}")


def open_url_and_interact(driver, url):
    # Open the provided URL
    driver.get(url)
    print(f"Opened URL: {url}")

    # Toggle the checkbox and click the confirm button
    toggle_checkbox_and_confirm(driver)

    # Take a screenshot after interaction
    driver.save_screenshot('page_screenshot.png')
    print("Screenshot saved as 'page_screenshot.png'")


if __name__ == "__main__":
    start = time.time()
    url = "https://cho.elphago.work/en"

    # Setup the Firefox driver using the locally installed geckodriver
    driver = setup_firefox()

    # Perform interactions and take a screenshot
    open_url_and_interact(driver, url)

    # Calculate the time taken
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")

    # Quit the driver
    driver.quit()
