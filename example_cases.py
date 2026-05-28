"""
Example Cases Demonstration
Demonstrates three specific cases: Mismatch, Reasonable but Incorrect, and Correct Match
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime

# Import our custom modules
from data_preparation import DataPreparator
from embedding_generator import EmbeddingGenerator
from vector_storage_retrieval import VectorStorageRetrieval
from deepseek_integration import DeepSeekReasoning
from clustering_pipeline import NewsClusteringPipeline
from results_analysis import ResultsAnalyzer
from visualization import NewsClassificationVisualizer

class ExampleCasesDemo:
    def __init__(self,
                 deepseek_api_key: Optional[str] = None,
                 data_path: str = "News_Category_Dataset_v3.csv"):
        """
        Initialize Example Cases Demo

        Args:
            deepseek_api_key (str): DeepSeek API key
            data_path (str): Path to the dataset
        """
        self.deepseek_api_key = deepseek_api_key
        self.data_path = data_path

        # Initialize components
        self.data_preparator = DataPreparator(data_path)
        self.embedding_generator = EmbeddingGenerator()
        self.vector_storage = VectorStorageRetrieval()
        self.deepseek = DeepSeekReasoning(api_key=deepseek_api_key)
        self.clustering_pipeline = NewsClusteringPipeline()
        self.results_analyzer = ResultsAnalyzer()
        self.visualizer = NewsClassificationVisualizer()

        # Demo data
        self.demo_cases = []
        self.results_data = None

    def prepare_demo_data(self, sample_size: int = 100) -> bool:
        """
        Prepare a small sample of data for demonstration

        Args:
            sample_size (int): Number of samples to use for demo

        Returns:
            bool: Success status
        """
        print("Preparing demo data...")

        # Load and clean data
        if not self.data_preparator.load_data():
            return False

        if not self.data_preparator.clean_data():
            return False

        # Take a sample for demo
        self.demo_df = self.data_preparator.df.sample(n=min(sample_size, len(self.data_preparator.df)),
                                                     random_state=42)

        print(f"Demo dataset prepared with {len(self.demo_df)} samples")
        print(f"Categories in demo: {self.demo_df['category'].value_counts().to_dict()}")

        return True

    def create_demo_cases(self) -> List[Dict[str, Any]]:
        """
        Create three specific demo cases

        Returns:
            List[Dict]: Three demo cases
        """
        print("Creating specific demo cases...")

        # Case 1: Mismatch (TRAVEL vs ENTERTAINMENT)
        case1 = {
            'case_id': 'mismatch_travel_entertainment',
            'case_type': 'Mismatch',
            'description': 'TRAVEL vs ENTERTAINMENT mismatch',
            'input_text': 'The new theme park in Orlando features the world\'s largest roller coaster and promises to be the ultimate family entertainment destination with hotels, restaurants, and shopping centers.',
            'ground_truth_category': 'TRAVEL',
            'expected_behavior': 'Should be classified as TRAVEL but might be misclassified as ENTERTAINMENT due to theme park content'
        }

        # Case 2: Reasonable but Incorrect (CRIME vs WORLD NEWS)
        case2 = {
            'case_id': 'reasonable_incorrect_crime_world',
            'case_type': 'Reasonable but Incorrect',
            'description': 'CRIME vs WORLD NEWS - reasonable confusion',
            'input_text': 'International authorities have arrested a major drug trafficking ring that operated across three continents, with suspects from multiple countries now facing extradition proceedings.',
            'ground_truth_category': 'CRIME',
            'expected_behavior': 'Could reasonably be classified as WORLD NEWS due to international scope, but ground truth is CRIME'
        }

        # Case 3: Correct Match (MEDIA)
        case3 = {
            'case_id': 'correct_match_media',
            'case_type': 'Correct Match',
            'description': 'Clear MEDIA category classification',
            'input_text': 'The streaming service announced record-breaking viewership numbers for its original series, with the latest season premiere attracting over 50 million viewers worldwide.',
            'ground_truth_category': 'MEDIA',
            'expected_behavior': 'Should be correctly classified as MEDIA due to clear streaming service and entertainment industry content'
        }

        self.demo_cases = [case1, case2, case3]

        print(f"Created {len(self.demo_cases)} demo cases")
        return self.demo_cases

    def run_vector_analysis(self, input_text: str, k: int = 5) -> Dict[str, Any]:
        """
        Run vector similarity analysis for input text

        Args:
            input_text (str): Input text to analyze
            k (int): Number of similar articles to retrieve

        Returns:
            Dict: Vector analysis results
        """
        try:
            # Get similar articles
            similar_articles = self.vector_storage.retrieve_top_k_similar(input_text, k=k)

            # Find most similar category
            similar_categories = self.vector_storage.find_most_similar_category(input_text, top_k=3)

            return {
                'similar_articles': similar_articles,
                'similar_categories': similar_categories,
                'similarity_scores': [art['similarity'] for art in similar_articles],
                'most_similar_category': similar_categories[0]['category'] if similar_categories else 'Unknown'
            }

        except Exception as e:
            print(f"Error in vector analysis: {e}")
            return {
                'similar_articles': [],
                'similar_categories': [],
                'similarity_scores': [],
                'most_similar_category': 'Unknown',
                'error': str(e)
            }

    def run_deepseek_analysis(self, input_text: str, available_categories: List[str]) -> Dict[str, Any]:
        """
        Run DeepSeek analysis for input text

        Args:
            input_text (str): Input text to analyze
            available_categories (List[str]): Available categories

        Returns:
            Dict: DeepSeek analysis results
        """
        try:
            # Predict category
            prediction = self.deepseek.predict_category(input_text, available_categories)

            # Generate explanation
            explanation = self.deepseek.generate_explanation(
                input_text,
                prediction['predicted_category']
            )

            return {
                'prediction': prediction,
                'explanation': explanation,
                'predicted_category': prediction.get('predicted_category', 'Unknown'),
                'confidence': prediction.get('confidence', 0.0),
                'reasoning': prediction.get('reasoning', '')
            }

        except Exception as e:
            print(f"Error in DeepSeek analysis: {e}")
            return {
                'prediction': {'predicted_category': 'Unknown', 'confidence': 0.0, 'reasoning': 'API Error'},
                'explanation': {'explanation': 'Failed to generate explanation due to API error'},
                'predicted_category': 'Unknown',
                'confidence': 0.0,
                'reasoning': 'API Error',
                'error': str(e)
            }

    def analyze_case(self, case: Dict[str, Any], available_categories: List[str]) -> Dict[str, Any]:
        """
        Analyze a single case comprehensively

        Args:
            case (Dict): Case to analyze
            available_categories (List[str]): Available categories

        Returns:
            Dict: Complete analysis results
        """
        print(f"Analyzing case: {case['case_id']} - {case['case_type']}")

        input_text = case['input_text']
        ground_truth = case['ground_truth_category']

        # Run vector analysis
        vector_results = self.run_vector_analysis(input_text)

        # Run DeepSeek analysis
        deepseek_results = self.run_deepseek_analysis(input_text, available_categories)

        # Determine if prediction is correct
        predicted_category = deepseek_results['predicted_category']
        is_correct = predicted_category == ground_truth

        # Compile results
        analysis_results = {
            'case_info': case,
            'input_text': input_text,
            'ground_truth_category': ground_truth,
            'predicted_category': predicted_category,
            'is_correct': is_correct,
            'match_status': 'Correct' if is_correct else 'Incorrect',
            'vector_analysis': vector_results,
            'deepseek_analysis': deepseek_results,
            'analysis_timestamp': datetime.now().isoformat()
        }

        return analysis_results

    def run_all_example_cases(self) -> List[Dict[str, Any]]:
        """
        Run analysis on all three example cases

        Returns:
            List[Dict]: Analysis results for all cases
        """
        print("Running all example cases...")

        # Get available categories
        available_categories = self.demo_df['category'].unique().tolist()
        print(f"Available categories: {available_categories}")

        # Create demo cases
        cases = self.create_demo_cases()

        # Analyze each case
        analysis_results = []
        for case in cases:
            result = self.analyze_case(case, available_categories)
            analysis_results.append(result)

            # Print results for this case
            self.print_case_results(result)

        self.results_data = analysis_results
        return analysis_results

    def print_case_results(self, analysis_result: Dict[str, Any]):
        """
        Print detailed results for a single case

        Args:
            analysis_result (Dict): Analysis results to print
        """
        case_info = analysis_result['case_info']
        print("\n" + "="*80)
        print(f"CASE: {case_info['case_id']} - {case_info['case_type']}")
        print("="*80)

        print(f"\nInput Text:")
        print(f'"{analysis_result["input_text"]}"')

        print(f"\nGround Truth Category: {analysis_result['ground_truth_category']}")
        print(f"Predicted Category: {analysis_result['predicted_category']}")
        print(f"Match Status: {analysis_result['match_status']}")

        # Vector analysis results
        vector_analysis = analysis_result['vector_analysis']
        print(f"\nVector Analysis:")
        print(f"Most Similar Category: {vector_analysis['most_similar_category']}")
        print(f"Similarity Scores: {[f'{s:.3f}' for s in vector_analysis['similarity_scores']]}")

        # DeepSeek analysis results
        deepseek_analysis = analysis_result['deepseek_analysis']
        prediction = deepseek_analysis['prediction']
        explanation = deepseek_analysis['explanation']

        print(f"\nDeepSeek Prediction:")
        print(f"Category: {prediction.get('predicted_category', 'Unknown')}")
        print(f"Confidence: {prediction.get('confidence', 0.0):.3f}")
        print(f"Reasoning: {prediction.get('reasoning', 'N/A')}")

        if 'key_indicators' in prediction:
            print(f"Key Indicators: {', '.join(prediction['key_indicators'])}")

        if 'alternative_categories' in prediction:
            print(f"Alternative Categories: {', '.join(prediction['alternative_categories'])}")

        print(f"\nDetailed Explanation:")
        print(f"{explanation.get('explanation', 'N/A')}")

        if 'key_phrases' in explanation:
            print(f"Key Phrases: {', '.join(explanation['key_phrases'])}")

        if 'tone_analysis' in explanation:
            print(f"Tone Analysis: {explanation['tone_analysis']}")

        if 'subject_matter' in explanation:
            print(f"Subject Matter: {explanation['subject_matter']}")

        # Analysis of correctness
        if not analysis_result['is_correct']:
            print(f"\nMismatch Analysis:")
            print(f"Expected: {case_info['expected_behavior']}")
            print(f"Actual: Predicted as {analysis_result['predicted_category']} instead of {analysis_result['ground_truth_category']}")

        print("\n" + "="*80)

    def generate_summary_report(self) -> str:
        """
        Generate a summary report of all cases

        Returns:
            str: Summary report
        """
        if not self.results_data:
            return "No results data available"

        report = []
        report.append("EXAMPLE CASES DEMONSTRATION SUMMARY")
        report.append("="*50)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overall statistics
        total_cases = len(self.results_data)
        correct_cases = sum(1 for result in self.results_data if result['is_correct'])
        accuracy = correct_cases / total_cases if total_cases > 0 else 0

        report.append("OVERALL RESULTS:")
        report.append(f"Total Cases: {total_cases}")
        report.append(f"Correct Predictions: {correct_cases}")
        report.append(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        report.append("")

        # Case-by-case summary
        report.append("CASE-BY-CASE SUMMARY:")
        for i, result in enumerate(self.results_data, 1):
            case_info = result['case_info']
            report.append(f"\n{i}. {case_info['case_id']} - {case_info['case_type']}")
            report.append(f"   Ground Truth: {result['ground_truth_category']}")
            report.append(f"   Predicted: {result['predicted_category']}")
            report.append(f"   Status: {result['match_status']}")
            report.append(f"   Confidence: {result['deepseek_analysis']['confidence']:.3f}")

        # Vector vs DeepSeek comparison
        report.append("\nVECTOR vs DEEPSEEK COMPARISON:")
        for i, result in enumerate(self.results_data, 1):
            vector_category = result['vector_analysis']['most_similar_category']
            deepseek_category = result['predicted_category']
            ground_truth = result['ground_truth_category']

            vector_correct = vector_category == ground_truth
            deepseek_correct = deepseek_category == ground_truth

            report.append(f"\nCase {i}:")
            report.append(f"  Vector Prediction: {vector_category} ({'✓' if vector_correct else '✗'})")
            report.append(f"  DeepSeek Prediction: {deepseek_category} ({'✓' if deepseek_correct else '✗'})")
            report.append(f"  Ground Truth: {ground_truth}")

        return "\n".join(report)

    def save_demo_results(self, filename: str = "demo_results.json") -> bool:
        """
        Save demo results to file

        Args:
            filename (str): Output filename

        Returns:
            bool: Success status
        """
        try:
            if not self.results_data:
                print("No results data to save")
                return False

            # Prepare data for JSON serialization
            save_data = {
                'demo_cases': self.demo_cases,
                'analysis_results': self.results_data,
                'summary_report': self.generate_summary_report(),
                'timestamp': datetime.now().isoformat()
            }

            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)

            print(f"Demo results saved to {filename}")
            return True

        except Exception as e:
            print(f"Error saving demo results: {e}")
            return False

def main():
    """Main function to run the example cases demonstration"""
    print("Example Cases Demonstration")
    print("="*50)

    # Initialize demo
    demo = ExampleCasesDemo()

    # Note: This is a demonstration script
    # In practice, you would need to:
    # 1. Have the actual dataset file
    # 2. Set up ChromaDB with embeddings
    # 3. Have a valid DeepSeek API key

    print("\nThis script demonstrates the three example cases:")
    print("1. Mismatch (TRAVEL vs ENTERTAINMENT)")
    print("2. Reasonable but Incorrect (CRIME vs WORLD NEWS)")
    print("3. Correct Match (MEDIA)")

    print("\nTo run the full demonstration:")
    print("1. Ensure you have the Kaggle News Category Dataset")
    print("2. Set up ChromaDB with embeddings")
    print("3. Provide a valid DeepSeek API key")
    print("4. Run: demo.prepare_demo_data()")
    print("5. Run: demo.run_all_example_cases()")

    # Create demo cases for reference
    demo_cases = demo.create_demo_cases()

    print(f"\nDemo cases created:")
    for case in demo_cases:
        print(f"- {case['case_id']}: {case['description']}")

if __name__ == "__main__":
    main()
