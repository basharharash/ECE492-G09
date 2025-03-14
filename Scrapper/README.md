# Cropster Scraper

A Python script that logs into Cropster, filters for certain roasts, exports them to Excel, and saves them locally.

---

## 1. Requirements

- Python 3.9+ recommended
- Google Chrome installed (latest version)
- [ChromeDriver](https://chromedriver.chromium.org/) matching your Chrome version *(auto-managed by `webdriver_manager.chrome`)*

---

## 2. Installation

1. **Clone** or **download** this repository.
2. In a terminal/command prompt, **create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   # or on Windows:
   # venv\Scripts\activate

## 3. Credentials
Make sure to enter your Cropster username/email and password in the ```.env``` file before running the script
