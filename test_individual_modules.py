"""
Test individual modules by importing them directly
"""

import sys
import os
import importlib.util

def test_module_import(module_name, file_path):
    """Test importing a module from a file"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✅ {module_name} imported successfully")
        return True
    except Exception as e:
        print(f"❌ {module_name} import failed: {e}")
        return False

def main():
    """Test all modules"""
    print("🧪 TESTING INDIVIDUAL MODULES")
    print("="*40)

    modules_to_test = [
        ("data_preparation", "1_data_preparation.py"),
        ("embedding_generator", "2_embedding_generator.py"),
        ("vector_storage_retrieval", "3_vector_storage_retrieval.py"),
        ("deepseek_integration", "4_deepseek_integration.py"),
        ("clustering_pipeline", "5_clustering_pipeline.py"),
        ("results_analysis", "6_results_analysis.py"),
        ("visualization", "7_visualization.py"),
        ("example_cases", "8_example_cases.py"),
        ("config", "config.py")
    ]

    passed = 0
    total = len(modules_to_test)

    for module_name, file_path in modules_to_test:
        if os.path.exists(file_path):
            if test_module_import(module_name, file_path):
                passed += 1
        else:
            print(f"❌ {file_path} not found")

    print(f"\n📊 RESULTS: {passed}/{total} modules imported successfully")

    if passed == total:
        print("🎉 All modules are working correctly!")
        print("\nYou can now:")
        print("1. Use individual modules in your code")
        print("2. Run the full pipeline with real data")
        print("3. Customize components as needed")
    else:
        print("⚠️  Some modules have issues. Check the errors above.")

if __name__ == "__main__":
    main()
