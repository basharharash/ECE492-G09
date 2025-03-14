import os
import time
import json
import shutil
import urllib.parse
from selenium import webdriver
from dotenv import load_dotenv
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

load_dotenv()  # loads .env if present

VALID_FOLDER = "valid_roasts"
INVALID_FOLDER = "invalid_roasts"

os.makedirs(VALID_FOLDER, exist_ok=True)
os.makedirs(INVALID_FOLDER, exist_ok=True)

# Create a separate temporary download directory for in-flight exports.
TEMP_DOWNLOADS_DIR = os.path.join(os.getcwd(), "temp_downloads")
os.makedirs(TEMP_DOWNLOADS_DIR, exist_ok=True)

def construct_filtered_url(
    name_filter: str, start_date: str, end_date: str, include_consumed: bool = True
) -> str:
    """
    Constructs a filtered URL for the roast overview page, including pagination.
    """
    base_url = "https://c-sar.cropster.com/apps/roast/overview?filters="
    filters = {
        "name": name_filter,
        "creationDate": [start_date, end_date],
        "includeConsumed": include_consumed,
        "pageSize": 200,
    }
    filters_json = json.dumps(filters)
    filters_encoded = urllib.parse.quote(filters_json)
    full_url = base_url + filters_encoded
    return full_url

def login_to_cropster(username: str, password: str) -> webdriver.Chrome:
    """
    Launches Chrome, logs into Cropster, and returns the driver.
    """
    options = webdriver.ChromeOptions()
    # Set the default download directory to our temp_downloads folder
    prefs = {
        "download.default_directory": TEMP_DOWNLOADS_DIR,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True
    }
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.maximize_window()
    driver.get("https://c-sar.cropster.com")
    
    # Wait for login fields
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "username"))
    )
    
    driver.find_element(By.ID, "username").clear()
    driver.find_element(By.ID, "username").send_keys(username)
    driver.find_element(By.ID, "password").clear()
    driver.find_element(By.ID, "password").send_keys(password)
    driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]').click()
    
    # Wait for login to complete
    time.sleep(5)
    return driver

def get_overview_rows(driver):
    """
    Extracts data from each row in the roast overview table.
    Returns a list of dictionaries with:
      'id'    : Roast ID (from the ID-Tag column)
      'url'   : Detail page URL (from the Name column)
      'valid' : True if a green check mark is found in the Goals cell, else False.
    """
    rows_data = []
    try:
        table = driver.find_elements(By.CLASS_NAME, "table--striped")
        tbody = table[0].find_element(By.TAG_NAME, "tbody")
        rows = tbody.find_elements(By.TAG_NAME, "tr")
        for r in rows:
            row = r.find_elements(By.TAG_NAME, "td")
            div = row[1].find_element(By.CLASS_NAME,"ember-view")
            try:
                roast_id = div.find_element(By.TAG_NAME, "a").text
            except Exception:
                roast_id = "unknown_id"
            try:
                link = div.find_element(By.TAG_NAME, "a").get_attribute("href")
            except Exception as e:
                print("Skipping row; detail link not found:", e)
                continue
            try:
                goals_cell = row[9]
                try:
                    div2 = goals_cell.find_element(By.CLASS_NAME, "ember-view")
                    # check for the green check mark
                    _ = div2.find_element(By.CLASS_NAME, "icon--green")
                    valid = True
                except Exception:
                    valid = False
            except Exception as e:
                print("Skipping row; validity not found:", e)
                continue
            rows_data.append({"id": roast_id, "url": link, "valid": valid})
            break
    except Exception as e:
        print("Error extracting table rows:", e)
    return rows_data

def wait_for_newest_file(download_dir: str, before_download_timestamp: float, timeout: int = 30) -> str:
    """
    Polls the download directory for up to `timeout` seconds.
    Returns the *newest* file that appears after `before_download_timestamp`.
    Raises an exception if no new file appears in time.
    """
    end_time = time.time() + timeout
    
    newest_file_path = None
    newest_mtime = before_download_timestamp

    while time.time() < end_time:
        # Look for the file with mtime > before_download_timestamp
        files = [os.path.join(download_dir, f) for f in os.listdir(download_dir)]
        if not files:
            time.sleep(1)
            continue

        # Identify the newest file
        candidate_path = max(files, key=os.path.getmtime)
        candidate_mtime = os.path.getmtime(candidate_path)

        # If it was created/updated after we started waiting, consider it
        if candidate_mtime > newest_mtime:
            newest_file_path = candidate_path
            newest_mtime = candidate_mtime

        # Also ensure it's not a *.crdownload (i.e. partial/in-progress)
        # Chrome downloads partial files with .crdownload extension
        if newest_file_path and not newest_file_path.endswith(".crdownload"):
            # We assume the file is fully downloaded now.
            return newest_file_path
        
        time.sleep(1)

    raise TimeoutError(
        f"No completed file appeared in {download_dir} after {timeout} seconds."
    )

def process_detail(driver, detail_url: str, roast_id: str, valid: bool):
    """
    Opens the roast detail page in a new tab.
    Then clicks on the 'Export' dropdown and chooses 'Excel (.xls)'.
    Then waits for the file to finish downloading to our TEMP_DOWNLOADS_DIR,
    and moves it to either the valid or invalid folder.
    """
    # Determine which folder we want to move the file into
    target_folder = VALID_FOLDER if valid else INVALID_FOLDER

    # Record current time, so we can detect the *new* file that appears
    before_download_timestamp = time.time()
    
    # Open detail page in a new tab
    driver.execute_script("window.open(arguments[0]);", detail_url)
    driver.switch_to.window(driver.window_handles[-1])
    
    # Wait for detail page to load
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "section#section-roast-stats"))
    )

    # Click the 'Export' button
    export_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Export')]"))
    )
    export_button.click()

    # Click the 'Excel (.xls)' option
    excel_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[span[text()='Excel (.xls)']]"))
    )
    excel_button.click()

    # Click the final 'Export to Excel' button
    final_export_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[span[text()='Export to Excel']]"))
    )
    time.sleep(0.5)
    driver.execute_script("arguments[0].click();", final_export_button)

    # Wait for the new file to show up in TEMP_DOWNLOADS_DIR
    print(f"Waiting for file download for roast {roast_id}...")
    downloaded_file_path = wait_for_newest_file(TEMP_DOWNLOADS_DIR, before_download_timestamp, timeout=60)

    # Construct a new file path in either 'valid_roasts' or 'invalid_roasts'
    # Use roast_id as the filename (with .xls extension) to avoid collisions.
    new_filename = f"{roast_id}.xls"
    destination_path = os.path.join(target_folder, new_filename)

    # Move the file to the appropriate folder
    shutil.move(downloaded_file_path, destination_path)
    print(f"Moved downloaded file to {destination_path}")

    # Close this tab and return to the main overview tab
    driver.close()
    driver.switch_to.window(driver.window_handles[0])

def process_all_pages(driver):
    """
    Paginate purely by updating the 'page' parameter in the URL
    until we see the 'no matching records' text.
    """
    page = 1
    while True:
        print(f"Processing page {page}...")
        
        # Construct the filtered URL with the current page
        filtered_url = construct_filtered_url(
            name_filter="Guatemala",
            start_date="2024-09-01",
            end_date="2024-12-31",
        )
        driver.get(filtered_url + "&page=" + str(page))
        time.sleep(5)  # wait for page to load; adjust as needed

        # Check if the "no data" element exists
        try:
            no_data_element = driver.find_element(By.CLASS_NAME, "c-table__no-data")
            # If found, it means there's no data on this page
            print("No matching records found. Stopping.")
            break
        except NoSuchElementException:
            # If that element isn't found, we presumably have data
            pass

        # Otherwise, parse table rows & process them
        rows = get_overview_rows(driver)
        if not rows:
            print("No rows found (empty table). Stopping.")
            break

        for row in rows:
            process_detail(driver, row["url"], row["id"], row["valid"])
        
        page += 1


def main():
    # Read credentials from environment
    username = os.getenv("CROPSTER_USERNAME")
    password = os.getenv("CROPSTER_PASSWORD")
    if not username or not password:
        raise ValueError("Missing environment variables CROPSTER_USERNAME or CROPSTER_PASSWORD")
    
    driver = login_to_cropster(username, password)
    try:
        # Construct the filtered URL
        name_filter = "Guatemala"
        start_date = "2024-09-01"
        end_date = "2024-12-31"
        filtered_url = construct_filtered_url(name_filter, start_date, end_date)
        driver.get(filtered_url)
        time.sleep(10)
        process_all_pages(driver)
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
