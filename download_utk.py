# src/data/download_utk.py
"""
Download UTKFace dataset from Kaggle into data/UTKFace.
Make sure you have kaggle.json in your user .kaggle folder or set KAGGLE_USERNAME/KAGGLE_KEY env vars.
"""
import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_utk(destination: str = "data/UTKFace", dataset_slug: str = "jangedoo/utkface-new"):
    os.makedirs(destination, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print("Downloading UTKFace from Kaggle (this may take a while)...")
    api.dataset_download_files(dataset_slug, path=destination, unzip=True)
    print("Download complete. Files are in:", os.path.abspath(destination))

if __name__ == "__main__":
    download_utk()
