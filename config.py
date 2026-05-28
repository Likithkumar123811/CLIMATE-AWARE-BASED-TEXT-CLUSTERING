"""
Configuration file for News Category Classification System
"""

import os
from typing import Optional

class Config:
    """Configuration class for the news classification system"""

    # Data configuration
    DATA_PATH = "News_Category_Dataset_v3.csv"
    SAMPLE_SIZE = None  # None for full dataset, int for sample size
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # ChromaDB configuration
    CHROMA_COLLECTION_NAME = "news_embeddings"
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # DeepSeek configuration
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL = "deepseek-reasoner"
    DEEPSEEK_MAX_TOKENS = 1000
    DEEPSEEK_TEMPERATURE = 0.7

    # Clustering configuration
    CLUSTERING_METHODS = ['kmeans', 'agglomerative']
    N_CLUSTERS_RANGE = (5, 20)
    CLUSTERING_RANDOM_STATE = 42

    # Visualization configuration
    FIGURE_SIZE = (12, 8)
    DPI = 300
    PLOT_STYLE = 'seaborn-v0_8'

    # Output configuration
    OUTPUT_DIR = "output"
    RESULTS_DIR = "results"
    VISUALIZATIONS_DIR = "visualizations"

    # Processing configuration
    BATCH_SIZE = 100
    MAX_RETRIES = 3
    REQUEST_DELAY = 1.0  # seconds between API requests

    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def get_output_path(cls, subdir: str = "") -> str:
        """Get output path with optional subdirectory"""
        if subdir:
            return os.path.join(cls.OUTPUT_DIR, subdir)
        return cls.OUTPUT_DIR

    @classmethod
    def get_results_path(cls) -> str:
        """Get results directory path"""
        return os.path.join(cls.OUTPUT_DIR, cls.RESULTS_DIR)

    @classmethod
    def get_visualizations_path(cls) -> str:
        """Get visualizations directory path"""
        return os.path.join(cls.OUTPUT_DIR, cls.VISUALIZATIONS_DIR)

    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration"""
        # Check if data file exists
        if not os.path.exists(cls.DATA_PATH):
            print(f"Warning: Data file not found at {cls.DATA_PATH}")
            print("Please download the Kaggle News Category Dataset and place it in the project directory")
            return False

        # Check if API key is set (optional)
        if not cls.DEEPSEEK_API_KEY:
            print("Warning: DeepSeek API key not set")
            print("Set DEEPSEEK_API_KEY environment variable for full functionality")

        return True

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("Current Configuration:")
        print(f"  Data Path: {cls.DATA_PATH}")
        print(f"  Sample Size: {cls.SAMPLE_SIZE}")
        print(f"  Output Directory: {cls.OUTPUT_DIR}")
        print(f"  DeepSeek API Key: {'Set' if cls.DEEPSEEK_API_KEY else 'Not set'}")
        print(f"  ChromaDB Directory: {cls.CHROMA_PERSIST_DIRECTORY}")
        print(f"  Clustering Methods: {cls.CLUSTERING_METHODS}")
        print(f"  Figure Size: {cls.FIGURE_SIZE}")

# Example usage and validation
if __name__ == "__main__":
    Config.print_config()
    Config.validate_config()

