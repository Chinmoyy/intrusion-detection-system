"""
Download the NSL-KDD dataset from the University of New Brunswick.
Run this script once before training.
"""

import os
import urllib.request
import zipfile

DATA_DIR = "data"
FILES = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    "KDDTest+.txt":  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
}


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    for filename, url in FILES.items():
        dest = os.path.join(DATA_DIR, filename)
        if os.path.exists(dest):
            print(f"[✓] {filename} already exists — skipping.")
            continue
        print(f"[+] Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, dest)
            size = os.path.getsize(dest) / 1024
            print(f"    Saved {dest} ({size:.1f} KB)")
        except Exception as e:
            print(f"[!] Failed to download {filename}: {e}")
            print(f"    Manual download: {url}")

    print("\n[✓] Dataset ready. Run: python src/train.py")


if __name__ == "__main__":
    download()
