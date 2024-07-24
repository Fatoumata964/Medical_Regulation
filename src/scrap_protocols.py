import sys
import time
import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import chromedriver_autoinstaller

def main(medicine_csv, download_dir):
    # setup chrome options
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless') # ensure GUI is off
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,  # Désactive l'invite de téléchargement
        "directory_upgrade": True               # S'assure que le répertoire est créé ou mis à jour
    }

    # Ajout des préférences expérimentales à chrome_options
    chrome_options.add_experimental_option('prefs', prefs)

    # set path to chromedriver as per your configuration
    chromedriver_autoinstaller.install()

    # set up the webdriver
    driver = webdriver.Chrome(options=chrome_options)
    driver.get('https://www.clinicaltrialsregister.eu/ctr-search/search')

    # Read the medicine data
    medecines = pd.read_csv(medicine_csv)

    protocols = []
    for name in medecines['ProductName']:
        driver.find_element("id", "query").clear()
        inputElement = driver.find_element("id", "query")
        inputElement.send_keys(name)
        time.sleep(1)
        inputElement.submit()
        time.sleep(2)
        usa = driver.find_element(By.XPATH, '//*[@id="tabs-1"]')
        driver.execute_script('arguments[0].scrollIntoView(true)', usa)

        liste = driver.find_elements(By.CSS_SELECTOR, "#tabs-1 > div.results.grid_8plus")
        for e in liste:
            trials = e.find_elements(By.CSS_SELECTOR, '#tabs-1 > div.results.grid_8plus > table:nth-child(1) > tbody > tr:nth-child(7) > td > a')
            for trial in trials:
                protocols.append(trial.get_attribute('href'))
        driver.execute_script("window.scrollBy(0,0)", "")

    for protocol in protocols:
        driver.get(protocol)
        download = driver.find_element(By.XPATH, '//*[@id="downloadCTA"]')
        download.click()
        time.sleep(5)

        # Renommer le fichier téléchargé
        original_filename = "trial.txt"  # Remplacez par le nom exact du fichier
        original_filepath = os.path.join(download_dir, original_filename)
        match = re.search(r'trial/(\d{4}-\d{6}-\d{2})/(\w{2})', protocol)
        if match:
            new_filename = os.path.join(download_dir, f"{match.group(1)}_{match.group(2)}.txt")
            os.rename(original_filepath, new_filename)
            print(f"File saved as: {new_filename}")

        time.sleep(5)

    driver.quit()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <medicine_csv> <download_dir>")
    else:
        medicine_csv = sys.argv[1]
        download_dir = sys.argv[2]
        main(medicine_csv, download_dir)
