"""
Setup script for News Category Classification System
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported.")
        print("Please use Python 3.8 or higher.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor} is compatible.")
        return True

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")

    directories = [
        "output",
        "output/results",
        "output/visualizations",
        "chroma_db",
        "processed_data"
    ]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False

    return True

def check_dataset():
    """Check if dataset is available"""
    print("\nChecking dataset...")

    dataset_path = "News_Category_Dataset_v3.csv"
    if os.path.exists(dataset_path):
        print(f"✅ Dataset found at {dataset_path}")
        return True
    else:
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please download the Kaggle News Category Dataset and place it in the project directory.")
        print("Download from: https://www.kaggle.com/datasets/rmisra/news-category-dataset")
        return False

def run_tests():
    """Run system tests"""
    print("\nRunning system tests...")

    try:
        result = subprocess.run([sys.executable, "test_system.py"],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED!")
    print("="*60)

    print("\nNext steps:")
    print("1. Download the Kaggle News Category Dataset:")
    print("   https://www.kaggle.com/datasets/rmisra/news-category-dataset")
    print("   Place 'News_Category_Dataset_v3.csv' in the project directory")

    print("\n2. (Optional) Set up DeepSeek API key:")
    print("   export DEEPSEEK_API_KEY='your_api_key_here'")
    print("   Or set it in your environment variables")

    print("\n3. Run the system:")
    print("   python demo.py              # Quick demonstration")
    print("   python main_pipeline.py     # Full pipeline")
    print("   python test_system.py       # Run tests again")

    print("\n4. Check the output:")
    print("   - output/results/           # Analysis results")
    print("   - output/visualizations/    # Generated plots")
    print("   - output/example_cases_results.json  # Example cases")

    print("\n5. Explore individual modules:")
    print("   - data_preparation.py       # Data loading and cleaning")
    print("   - embedding_generator.py    # Vector embeddings")
    print("   - vector_storage_retrieval.py  # Similarity search")
    print("   - deepseek_integration.py   # AI predictions")
    print("   - clustering_pipeline.py    # Clustering analysis")
    print("   - results_analysis.py       # Results processing")
    print("   - visualization.py          # Plot generation")
    print("   - example_cases.py          # Demo cases")

def main():
    """Main setup function"""
    print("🚀 NEWS CATEGORY CLASSIFICATION SYSTEM SETUP")
    print("="*60)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install requirements
    if not install_requirements():
        print("Please install the requirements manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)

    # Create directories
    if not create_directories():
        print("Please create the directories manually.")
        sys.exit(1)

    # Check dataset
    dataset_available = check_dataset()

    # Run tests
    if not run_tests():
        print("Some tests failed, but the system might still work.")
        print("Check the error messages above for details.")

    # Print next steps
    print_next_steps()

    if not dataset_available:
        print("\n⚠️  Remember to download the dataset before running the full pipeline!")

if __name__ == "__main__":
    main()