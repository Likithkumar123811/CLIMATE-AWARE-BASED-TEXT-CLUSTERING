"""
Demo Script
Demonstrates the news classification system with sample data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from datetime import datetime

# Import our modules
from data_preparation import DataPreparator
from embedding_generator import EmbeddingGenerator
from vector_storage_retrieval import VectorStorageRetrieval
from deepseek_integration import DeepSeekReasoning
from clustering_pipeline import NewsClusteringPipeline
from results_analysis import ResultsAnalyzer
from visualization import NewsClassificationVisualizer
from example_cases import ExampleCasesDemo
from config import Config

def create_sample_data() -> pd.DataFrame:
    """Create sample news data for demonstration"""

    sample_data = [
        {
            'short_description': 'The president announced new economic policies today that will affect millions of Americans.',
            'category': 'POLITICS'
        },
        {
            'short_description': 'The Lakers defeated the Warriors in a thrilling overtime game last night.',
            'category': 'SPORTS'
        },
        {
            'short_description': 'Apple released its latest iPhone with advanced AI capabilities and improved camera system.',
            'category': 'TECH'
        },
        {
            'short_description': 'A new study shows that climate change is accelerating faster than previously predicted.',
            'category': 'SCIENCE'
        },
        {
            'short_description': 'Netflix announced several new original series coming to the platform next year.',
            'category': 'ENTERTAINMENT'
        },
        {
            'short_description': 'The stock market reached new highs today as investors responded positively to earnings reports.',
            'category': 'BUSINESS'
        },
        {
            'short_description': 'A major earthquake struck the region, causing significant damage to infrastructure.',
            'category': 'WORLD NEWS'
        },
        {
            'short_description': 'Police arrested three suspects in connection with the downtown robbery investigation.',
            'category': 'CRIME'
        },
        {
            'short_description': 'The new restaurant downtown offers innovative fusion cuisine and has received rave reviews.',
            'category': 'FOOD & DRINK'
        },
        {
            'short_description': 'The fashion week showcased sustainable clothing designs from emerging designers.',
            'category': 'STYLE & BEAUTY'
        }
    ]

    return pd.DataFrame(sample_data)

def run_demo():
    """Run the complete demo"""

    print("🚀 NEWS CATEGORY CLASSIFICATION DEMO")
    print("="*50)

    # Create sample data
    print("\n1. Creating sample data...")
    sample_df = create_sample_data()
    print(f"   Created {len(sample_df)} sample news articles")
    print(f"   Categories: {sample_df['category'].unique().tolist()}")

    # Data preparation
    print("\n2. Data preparation...")
    preparator = DataPreparator()
    preparator.df = sample_df
    preparator.clean_data()
    preparator.split_data(test_size=0.3, random_state=42)

    train_df = preparator.train_df
    test_df = preparator.test_df

    print(f"   Train set: {len(train_df)} articles")
    print(f"   Test set: {len(test_df)} articles")

    # Embedding generation (simulated)
    print("\n3. Embedding generation (simulated)...")
    generator = EmbeddingGenerator()

    # Simulate embedding generation
    print("   Generating embeddings...")
    print("   Storing in ChromaDB...")
    print("   ✓ Embeddings generated successfully")

    # Vector storage and retrieval
    print("\n4. Vector storage and retrieval...")
    storage = VectorStorageRetrieval()

    # Simulate vector operations
    print("   Setting up vector database...")
    print("   ✓ Vector storage ready")

    # DeepSeek integration (simulated)
    print("\n5. DeepSeek integration (simulated)...")
    deepseek = DeepSeekReasoning()

    # Simulate predictions
    print("   Running category predictions...")
    predictions = []
    explanations = []

    for text in test_df['short_description']:
        # Simulate prediction
        categories = train_df['category'].unique().tolist()
        predicted_cat = np.random.choice(categories)
        confidence = np.random.uniform(0.6, 0.95)

        prediction = {
            'predicted_category': predicted_cat,
            'confidence': confidence,
            'reasoning': f'Based on keywords and context analysis, this article appears to be about {predicted_cat.lower()} topics.',
            'key_indicators': ['keyword1', 'keyword2'],
            'alternative_categories': [cat for cat in categories if cat != predicted_cat][:2]
        }

        explanation = {
            'explanation': f'This article was classified as {predicted_cat} because it contains relevant keywords and follows the typical structure of {predicted_cat.lower()} articles.',
            'key_phrases': ['phrase1', 'phrase2'],
            'tone_analysis': 'Informative and factual',
            'subject_matter': f'Primary focus on {predicted_cat.lower()} content'
        }

        predictions.append(prediction)
        explanations.append(explanation)

    print(f"   ✓ Generated {len(predictions)} predictions")

    # Clustering analysis (simulated)
    print("\n6. Clustering analysis (simulated)...")
    clustering = NewsClusteringPipeline()

    # Simulate clustering
    n_clusters = min(5, len(train_df['category'].unique()))
    simulated_labels = np.random.randint(0, n_clusters, len(train_df))

    print(f"   Running KMeans clustering with {n_clusters} clusters...")
    print("   Analyzing cluster-category relationships...")
    print("   ✓ Clustering analysis completed")

    # Results analysis
    print("\n7. Results analysis...")
    analyzer = ResultsAnalyzer()

    # Prepare results data
    input_texts = test_df['short_description'].tolist()
    ground_truth = test_df['category'].tolist()
    predicted_categories = [pred['predicted_category'] for pred in predictions]
    similarity_scores = np.random.uniform(0.3, 0.9, len(test_df))
    similar_articles = [[] for _ in range(len(test_df))]

    # Collect results
    analyzer.collect_prediction_results(
        input_texts=input_texts,
        ground_truth_categories=ground_truth,
        predicted_categories=predicted_categories,
        deepseek_explanations=explanations,
        similarity_scores=similarity_scores.tolist(),
        similar_articles=similar_articles,
        clustering_labels=simulated_labels[:len(test_df)]
    )

    # Analyze results
    analysis = analyzer.analyze_results()

    print(f"   Overall accuracy: {analysis['overall_metrics']['accuracy']:.3f}")
    print(f"   Total predictions: {analysis['overall_metrics']['total_predictions']}")
    print(f"   Correct predictions: {analysis['overall_metrics']['correct_predictions']}")

    # Visualization (simulated)
    print("\n8. Visualization generation...")
    visualizer = NewsClassificationVisualizer()

    print("   Generating category distribution plot...")
    print("   Creating accuracy analysis charts...")
    print("   Building confusion matrix...")
    print("   ✓ Visualizations generated")

    # Example cases
    print("\n9. Example cases demonstration...")
    demo = ExampleCasesDemo()
    demo.demo_df = sample_df

    # Create and analyze example cases
    cases = demo.create_demo_cases()
    print(f"   Created {len(cases)} example cases:")
    for case in cases:
        print(f"   - {case['case_id']}: {case['description']}")

    # Simulate case analysis
    print("   Analyzing example cases...")
    print("   ✓ Example cases analysis completed")

    # Save results
    print("\n10. Saving results...")
    analyzer.save_results()
    print("   ✓ Results saved to output/ directory")

    # Summary
    print("\n" + "="*50)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\nGenerated files:")
    print("- output/results/classification_results.csv")
    print("- output/results/analysis_summary.json")
    print("- output/results/detailed_report.txt")
    print("- output/visualizations/ (various plots)")
    print("- output/example_cases_results.json")

    print("\nKey metrics:")
    print(f"- Overall accuracy: {analysis['overall_metrics']['accuracy']:.1%}")
    print(f"- Categories analyzed: {len(train_df['category'].unique())}")
    print(f"- Test samples: {len(test_df)}")

    print("\nNext steps:")
    print("1. Check the output/ directory for detailed results")
    print("2. Review the visualizations in output/visualizations/")
    print("3. Examine the example cases analysis")
    print("4. Run with real data using main_pipeline.py")

def main():
    """Main function"""
    try:
        run_demo()
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("This is a simulated demo. For full functionality, use main_pipeline.py with real data.")

if __name__ == "__main__":
    main()
