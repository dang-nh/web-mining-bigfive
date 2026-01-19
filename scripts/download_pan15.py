#!/usr/bin/env python3
"""
Download PAN15 Author Profiling Dataset.
Windows-compatible alternative to download_pan15.sh
"""
import os
import sys
import zipfile
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_URL = "https://zenodo.org/records/3745945/files/pan15-author-profiling-training-dataset-2015-04-23.zip?download=1"
TEST_URL = "https://zenodo.org/records/3745945/files/pan15-author-profiling-test-dataset-2015-04-23.zip?download=1"


def download_file(url: str, output_path: Path) -> None:
    """Download a file from URL."""
    print(f"Downloading {output_path.name}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded: {output_path}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip file to directory."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to: {extract_to}")


def find_english_zip(train_dir: Path) -> Path:
    """Find English subset zip file in training directory."""
    for pattern in ["*english*.zip", "*English*.zip"]:
        matches = list(train_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def find_english_dir(train_dir: Path) -> Path:
    """Find English subset directory in training directory."""
    for item in train_dir.rglob("*"):
        if item.is_dir() and "english" in item.name.lower():
            return item
    return None


def main():
    print("=== Downloading PAN15 Author Profiling Dataset ===")
    
    # Create raw directory
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(RAW_DIR)
    
    # Download training set
    train_zip = RAW_DIR / "pan15_train.zip"
    if not train_zip.exists():
        download_file(TRAIN_URL, train_zip)
    else:
        print("Training set already downloaded.")
    
    # Download test set
    test_zip = RAW_DIR / "pan15_test.zip"
    if not test_zip.exists():
        download_file(TEST_URL, test_zip)
    else:
        print("Test set already downloaded.")
    
    print("\n=== Extracting datasets ===")
    
    # Extract training set
    train_dir = RAW_DIR / "pan15_train"
    if not train_dir.exists():
        extract_zip(train_zip, RAW_DIR)
    else:
        print("Training set already extracted.")
    
    # Extract test set
    test_dir = RAW_DIR / "pan15_test"
    if not test_dir.exists():
        extract_zip(test_zip, RAW_DIR)
    else:
        print("Test set already extracted.")
    
    print("\n=== Extracting English subset ===")
    
    # Extract English subset
    english_dir = RAW_DIR / "pan15_train_en"
    if not english_dir.exists():
        english_zip = find_english_zip(train_dir)
        if english_zip:
            print(f"Found English subset: {english_zip}")
            extract_zip(english_zip, RAW_DIR)
        else:
            # Try finding English directory
            english_source = find_english_dir(train_dir)
            if english_source:
                print(f"Found English directory: {english_source}")
                import shutil
                shutil.copytree(english_source, english_dir)
            else:
                print("Warning: English subset not found in training set.")
    else:
        print("English subset already extracted.")
    
    print("\n=== Download complete ===")
    print(f"Data location: {RAW_DIR}")
    print("\nFiles and directories:")
    for item in sorted(RAW_DIR.iterdir()):
        if item.is_dir():
            print(f"  [DIR]  {item.name}/")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  [FILE] {item.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
