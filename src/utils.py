import requests
import os
from zipfile import ZipFile


# Function to download the file
def download_file(url, download_path):
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(download_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded to {download_path}.")
    else:
        print(f"Failed to download file. HTTP Status Code: {response.status_code}")
        response.raise_for_status()


# Function to extract the zip file
def extract_zip(file_path, extract_path):
    print(f"Extracting {file_path} to {extract_path}...")
    os.makedirs(extract_path, exist_ok=True)
    with ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extracted to {extract_path}.")