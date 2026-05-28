#!/usr/bin/env python3
"""
Script to download the Kaggle News Category Dataset
"""

import os
import subprocess
import sys
import zipfile
import pandas as pd

def check_kaggle_installed():
    """Check if kaggle is installed"""
    try:
        import kaggle
        return True
    except ImportError:
        return False

def install_kaggle():
    """Install kaggle package"""
    print("Installing kaggle package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])

def download_dataset():
    """Download the dataset from Kaggle"""
    print("Downloading News Category Dataset from Kaggle...")

    # The dataset is: rmisra/news-category-dataset
    dataset_name = "rmisra/news-category-dataset"

    try:
        # Download the dataset
        subprocess.run([
            "kaggle", "datasets", "download", "-d", dataset_name
        ], check=True)

        print("Dataset downloaded successfully!")

        # Extract the zip file
        zip_file = "news-category-dataset.zip"
        if os.path.exists(zip_file):
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(".")
            print("Dataset extracted successfully!")

            # Remove zip file
            os.remove(zip_file)
            print("Cleanup completed!")

            return True
        else:
            print("Error: Dataset file not found after download")
            return False

    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have a Kaggle account")
        print("2. Download your API key from: https://www.kaggle.com/account")
        print("3. Place kaggle.json in ~/.kaggle/ directory")
        print("4. Set proper permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False

def check_dataset_files():
    """Check what dataset files are available"""
    print("\nChecking for dataset files...")

    # Look for common dataset file names
    possible_files = [
        "News_Category_Dataset_v3.csv",
        "News_Category_Dataset_v2.csv",
        "News_Category_Dataset_v1.csv",
        "News_Category_Dataset.csv",
        "news_category_dataset.csv"
    ]

    found_files = []
    for file in possible_files:
        if os.path.exists(file):
            found_files.append(file)
            print(f"✓ Found: {file}")

    if found_files:
        # Check the content of the first file
        df = pd.read_csv(found_files[0])
        print(f"\nDataset info:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample categories: {df['category'].unique()[:10].tolist()}")
        return found_files[0]
    else:
        print("No dataset files found")
        return None

def main():
    print("🚀 KAGGLE DATASET DOWNLOADER")
    print("=" * 50)

    # Check if kaggle is installed
    if not check_kaggle_installed():
        print("Kaggle package not found. Installing...")
        install_kaggle()

    # Check if dataset already exists
    dataset_file = check_dataset_files()
    if dataset_file:
        print(f"\n✅ Dataset already available: {dataset_file}")
        return dataset_file

    # Download dataset
    print("\nDownloading dataset...")
    if download_dataset():
        dataset_file = check_dataset_files()
        if dataset_file:
            print(f"\n✅ Dataset ready: {dataset_file}")
            return dataset_file
        else:
            print("\n❌ Dataset download failed")
            return None
    else:
        print("\n❌ Could not download dataset")
        return None

if __name__ == "__main__":
    dataset_file = main()

    if dataset_file:
        print(f"\n🎉 SUCCESS! Dataset ready: {dataset_file}")
        print("\nNext steps:")
        print("1. Run: python main_pipeline.py")
        print("2. Or run: python run_demo.py (for quick demo)")
    else:
        print("\n❌ FAILED! Could not obtain dataset")
        print("\nManual download option:")
        print("1. Go to: https://www.kaggle.com/datasets/rmisra/news-category-dataset")
        print("2. Download the dataset manually")
        print("3. Place the CSV file in this directory")
        print("4. Rename it to 'News_Category_Dataset_v3.csv'")

