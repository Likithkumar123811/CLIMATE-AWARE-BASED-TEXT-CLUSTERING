"""
Data Preparation Module for Climate Dataset
Loads dataset, performs cleaning, and splits into train/test
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import os


class DataPreparator:
    def __init__(self, data_path=None):
        """
        Initialize DataPreparator
        """
        # ✅ MODIFIED
        self.data_path = data_path or 'climate_dataset_10_categories_5000.csv'
        self.df = None
        self.train_df = None
        self.test_df = None

    def load_data(self):
        """Load dataset"""
        try:
            print(f"Loading data from {self.data_path}...")
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
            print(f"Columns: {list(self.df.columns)}")
            return True
        except FileNotFoundError:
            print(f"Error: Could not find {self.data_path}")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def clean_data(self):
        """
        Clean dataset:
        - Keep Text & Category
        - Remove nulls
        - Clean text
        """
        if self.df is None:
            print("Error: No data loaded.")
            return False

        print("Cleaning data...")

        # ✅ MODIFIED
        required_columns = ['Text', 'Category']
        missing_columns = [c for c in required_columns if c not in self.df.columns]

        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return False

        self.df = self.df[required_columns].copy()

        # Remove nulls
        initial_count = len(self.df)
        self.df.dropna(inplace=True)
        print(f"Removed {initial_count - len(self.df)} null rows")

        # Clean text
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
            return ' '.join(text.split())

        # ✅ MODIFIED
        self.df['Text'] = self.df['Text'].apply(clean_text)

        self.df = self.df[self.df['Text'].str.len() > 0]

        print(f"After cleaning: {len(self.df)} rows")
        print(self.df['Category'].value_counts().head(10))

        return True

    def split_data(self, test_size=0.2, random_state=42):
        """Split dataset"""
        if self.df is None:
            print("Error: Clean data first.")
            return False

        self.train_df, self.test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df['Category']  # ✅ MODIFIED
        )

        print(f"Train rows: {len(self.train_df)}")
        print(f"Test rows: {len(self.test_df)}")

        return True

    def get_data_summary(self):
        """Summary"""
        if self.df is None:
            return None

        return {
            'total_rows': len(self.df),
            'unique_categories': self.df['Category'].nunique(),  # ✅ MODIFIED
            'avg_text_length': self.df['Text'].str.len().mean(), # ✅ MODIFIED
            'min_text_length': self.df['Text'].str.len().min(),
            'max_text_length': self.df['Text'].str.len().max()
        }

    def save_processed_data(self, output_dir='processed_data'):
        """Save files"""
        if self.train_df is None or self.test_df is None:
            print("Run split_data() first.")
            return False

        os.makedirs(output_dir, exist_ok=True)

        self.train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
        self.test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)

        print("Processed data saved successfully.")
        return True


def main():
    preparator = DataPreparator()

    if not preparator.load_data():
        return
    if not preparator.clean_data():
        return
    if not preparator.split_data():
        return

    summary = preparator.get_data_summary()
    print("\nData Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    preparator.save_processed_data()


if __name__ == "__main__":
    main()