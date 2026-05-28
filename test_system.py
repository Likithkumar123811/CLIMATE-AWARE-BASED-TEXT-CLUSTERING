"""
Test System Script
Tests all components of the news classification system
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import traceback

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")

    try:
        from data_preparation import DataPreparator
        print("✓ data_preparation imported successfully")
    except Exception as e:
        print(f"✗ data_preparation import failed: {e}")
        return False

    try:
        from embedding_generator import EmbeddingGenerator
        print("✓ embedding_generator imported successfully")
    except Exception as e:
        print(f"✗ embedding_generator import failed: {e}")
        return False

    try:
        from vector_storage_retrieval import VectorStorageRetrieval
        print("✓ vector_storage_retrieval imported successfully")
    except Exception as e:
        print(f"✗ vector_storage_retrieval import failed: {e}")
        return False

    try:
        from deepseek_integration import DeepSeekReasoning
        print("✓ deepseek_integration imported successfully")
    except Exception as e:
        print(f"✗ deepseek_integration import failed: {e}")
        return False

    try:
        from clustering_pipeline import NewsClusteringPipeline
        print("✓ clustering_pipeline imported successfully")
    except Exception as e:
        print(f"✗ clustering_pipeline import failed: {e}")
        return False

    try:
        from results_analysis import ResultsAnalyzer
        print("✓ results_analysis imported successfully")
    except Exception as e:
        print(f"✗ results_analysis import failed: {e}")
        return False

    try:
        from visualization import NewsClassificationVisualizer
        print("✓ visualization imported successfully")
    except Exception as e:
        print(f"✗ visualization import failed: {e}")
        return False

    try:
        from example_cases import ExampleCasesDemo
        print("✓ example_cases imported successfully")
    except Exception as e:
        print(f"✗ example_cases import failed: {e}")
        return False

    try:
        from config import Config
        print("✓ config imported successfully")
    except Exception as e:
        print(f"✗ config import failed: {e}")
        return False

    return True

def test_data_preparation():
    """Test data preparation module"""
    print("\nTesting data preparation...")

    try:
        from data_preparation import DataPreparator

        # Create sample data
        sample_data = pd.DataFrame({
            'short_description': [
                'This is a test news article about politics.',
                'Sports news about basketball game.',
                'Technology update on new software.'
            ],
            'category': ['POLITICS', 'SPORTS', 'TECH']
        })

        preparator = DataPreparator()
        preparator.df = sample_data

        # Test cleaning
        result = preparator.clean_data()
        if result:
            print("✓ Data cleaning test passed")
        else:
            print("✗ Data cleaning test failed")
            return False

        # Test splitting
        result = preparator.split_data(test_size=0.3, random_state=42)
        if result:
            print("✓ Data splitting test passed")
        else:
            print("✗ Data splitting test failed")
            return False

        return True

    except Exception as e:
        print(f"✗ Data preparation test failed: {e}")
        traceback.print_exc()
        return False

def test_embedding_generator():
    """Test embedding generator module"""
    print("\nTesting embedding generator...")

    try:
        from embedding_generator import EmbeddingGenerator

        generator = EmbeddingGenerator()
        print("✓ EmbeddingGenerator initialized successfully")

        # Test collection stats
        stats = generator.get_collection_stats()
        print("✓ Collection stats retrieved successfully")

        return True

    except Exception as e:
        print(f"✗ Embedding generator test failed: {e}")
        traceback.print_exc()
        return False

def test_vector_storage():
    """Test vector storage module"""
    print("\nTesting vector storage...")

    try:
        from vector_storage_retrieval import VectorStorageRetrieval

        storage = VectorStorageRetrieval()
        print("✓ VectorStorageRetrieval initialized successfully")

        # Test collection stats
        stats = storage.get_collection_stats()
        print("✓ Collection stats retrieved successfully")

        return True

    except Exception as e:
        print(f"✗ Vector storage test failed: {e}")
        traceback.print_exc()
        return False

def test_deepseek_integration():
    """Test DeepSeek integration module"""
    print("\nTesting DeepSeek integration...")

    try:
        from deepseek_integration import DeepSeekReasoning

        deepseek = DeepSeekReasoning()
        print("✓ DeepSeekReasoning initialized successfully")

        # Test without API key (should work but with limited functionality)
        categories = ['POLITICS', 'SPORTS', 'TECH']
        result = deepseek.predict_category("Test news article", categories)
        print("✓ Prediction method test passed")

        return True

    except Exception as e:
        print(f"✗ DeepSeek integration test failed: {e}")
        traceback.print_exc()
        return False

def test_clustering_pipeline():
    """Test clustering pipeline module"""
    print("\nTesting clustering pipeline...")

    try:
        from clustering_pipeline import NewsClusteringPipeline

        pipeline = NewsClusteringPipeline()
        print("✓ NewsClusteringPipeline initialized successfully")

        # Test with sample data
        sample_embeddings = np.random.rand(10, 384)  # 10 samples, 384-dim embeddings
        sample_categories = ['POLITICS', 'SPORTS', 'TECH'] * 3 + ['POLITICS']
        sample_texts = [f"Sample text {i}" for i in range(10)]

        result = pipeline.prepare_embeddings(sample_embeddings, sample_categories, sample_texts)
        if result:
            print("✓ Embedding preparation test passed")
        else:
            print("✗ Embedding preparation test failed")
            return False

        return True

    except Exception as e:
        print(f"✗ Clustering pipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_results_analysis():
    """Test results analysis module"""
    print("\nTesting results analysis...")

    try:
        from results_analysis import ResultsAnalyzer

        analyzer = ResultsAnalyzer()
        print("✓ ResultsAnalyzer initialized successfully")

        # Test with sample data
        sample_results = analyzer.collect_prediction_results(
            input_texts=["Test article 1", "Test article 2"],
            ground_truth_categories=["POLITICS", "SPORTS"],
            predicted_categories=["POLITICS", "TECH"],
            deepseek_explanations=[{"confidence": 0.8}, {"confidence": 0.6}],
            similarity_scores=[0.7, 0.5],
            similar_articles=[[], []]
        )

        if sample_results is not None:
            print("✓ Results collection test passed")
        else:
            print("✗ Results collection test failed")
            return False

        # Test analysis
        analysis = analyzer.analyze_results()
        if analysis:
            print("✓ Results analysis test passed")
        else:
            print("✗ Results analysis test failed")
            return False

        return True

    except Exception as e:
        print(f"✗ Results analysis test failed: {e}")
        traceback.print_exc()
        return False

def test_visualization():
    """Test visualization module"""
    print("\nTesting visualization...")

    try:
        from visualization import NewsClassificationVisualizer

        visualizer = NewsClassificationVisualizer()
        print("✓ NewsClassificationVisualizer initialized successfully")

        # Test with sample data
        sample_df = pd.DataFrame({
            'category': ['POLITICS', 'SPORTS', 'TECH', 'POLITICS', 'SPORTS'],
            'short_description': ['Text 1', 'Text 2', 'Text 3', 'Text 4', 'Text 5']
        })

        # Test category distribution plot (without saving)
        fig = visualizer.plot_category_distribution(sample_df, save=False)
        if fig:
            print("✓ Category distribution plot test passed")
        else:
            print("✗ Category distribution plot test failed")
            return False

        return True

    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        traceback.print_exc()
        return False

def test_example_cases():
    """Test example cases module"""
    print("\nTesting example cases...")

    try:
        from example_cases import ExampleCasesDemo

        demo = ExampleCasesDemo()
        print("✓ ExampleCasesDemo initialized successfully")

        # Test creating demo cases
        cases = demo.create_demo_cases()
        if cases and len(cases) == 3:
            print("✓ Demo cases creation test passed")
        else:
            print("✗ Demo cases creation test failed")
            return False

        return True

    except Exception as e:
        print(f"✗ Example cases test failed: {e}")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration module"""
    print("\nTesting configuration...")

    try:
        from config import Config

        print("✓ Config class imported successfully")

        # Test configuration methods
        output_path = Config.get_output_path()
        results_path = Config.get_results_path()
        viz_path = Config.get_visualizations_path()

        print("✓ Configuration methods test passed")

        # Test validation
        Config.validate_config()
        print("✓ Configuration validation test passed")

        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("🧪 RUNNING SYSTEM TESTS")
    print("="*50)

    tests = [
        ("Import Test", test_imports),
        ("Data Preparation", test_data_preparation),
        ("Embedding Generator", test_embedding_generator),
        ("Vector Storage", test_vector_storage),
        ("DeepSeek Integration", test_deepseek_integration),
        ("Clustering Pipeline", test_clustering_pipeline),
        ("Results Analysis", test_results_analysis),
        ("Visualization", test_visualization),
        ("Example Cases", test_example_cases),
        ("Configuration", test_config)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")

    print("\n" + "="*50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

def main():
    """Main function"""
    success = run_all_tests()

    if success:
        print("\n🚀 System is ready! You can now:")
        print("1. Run 'python demo.py' for a quick demonstration")
        print("2. Run 'python main_pipeline.py' for the full pipeline")
        print("3. Use individual modules in your own code")
    else:
        print("\n🔧 Please fix the failing tests before using the system.")
        sys.exit(1)

if __name__ == "__main__":
    main()