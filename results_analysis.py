"""
Results Collection and Analysis Module
Collects and analyzes results from the news classification pipeline
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import Counter


class ResultsAnalyzer:
    def __init__(
        self,
        output_dir: str = "results",
        results_filename: str = "classification_results.csv",
    ):
        self.output_dir = output_dir
        self.results_filename = results_filename
        self.results_df = None
        self.analysis_summary = {}

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------

    def collect_prediction_results(
        self,
        input_texts: List[str],
        ground_truth_categories: List[str],
        predicted_categories: List[str],
        deepseek_explanations: List[Dict],
        similarity_scores: List[float],
        similar_articles: List[List[Dict]],
        clustering_labels: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Collect all prediction results into a structured DataFrame
        """

        print("Collecting prediction results...")

        lengths = [
            len(input_texts),
            len(ground_truth_categories),
            len(predicted_categories),
            len(deepseek_explanations),
            len(similarity_scores),
            len(similar_articles),
        ]

        min_length = min(lengths)

        input_texts = input_texts[:min_length]
        ground_truth_categories = ground_truth_categories[:min_length]
        predicted_categories = predicted_categories[:min_length]
        deepseek_explanations = deepseek_explanations[:min_length]
        similarity_scores = similarity_scores[:min_length]
        similar_articles = similar_articles[:min_length]

        if clustering_labels is not None:
            clustering_labels = clustering_labels[:min_length]

        results_data = []

        for i in range(min_length):
            explanation = deepseek_explanations[i] or {}

            is_match = ground_truth_categories[i] == predicted_categories[i]

            similar_arts = similar_articles[i]
            similar_categories = [art.get("category", "Unknown") for art in similar_arts]
            category_counts = Counter(similar_categories)
            most_common_similar_category = (
                category_counts.most_common(1)[0][0]
                if category_counts
                else "None"
            )

            result_record = {
                "index": i,
                "input_text": input_texts[i],
                "ground_truth_category": ground_truth_categories[i],
                "predicted_category": predicted_categories[i],
                "is_match": is_match,
                "match_status": "Correct" if is_match else "Incorrect",
                "similarity_score": similarity_scores[i],
                "deepseek_confidence": explanation.get("confidence", 0.0),
                "deepseek_reasoning": explanation.get("reasoning", ""),
                "deepseek_key_indicators": ", ".join(
                    explanation.get("key_indicators", [])
                ),
                "deepseek_alternative_categories": ", ".join(
                    explanation.get("alternative_categories", [])
                ),
                "similar_articles_count": len(similar_arts),
                "most_common_similar_category": most_common_similar_category,
                "similar_category_distribution": dict(category_counts),
                "text_length": len(input_texts[i]),
                "word_count": len(input_texts[i].split()),
                "clustering_label": (
                    int(clustering_labels[i])
                    if clustering_labels is not None
                    else None
                ),
                "timestamp": datetime.now().isoformat(),
            }

            results_data.append(result_record)

        self.results_df = pd.DataFrame(results_data)
        print(f"Collected {len(self.results_df)} prediction results")

        return self.results_df

    # ------------------------------------------------------------------

    def analyze_results(self) -> Dict[str, Any]:
        if self.results_df is None:
            print("No results to analyze")
            return {}

        total = len(self.results_df)
        correct = self.results_df["is_match"].sum()
        accuracy = correct / total if total else 0.0

        analysis = {
            "overall_metrics": {
                "total_predictions": total,
                "correct_predictions": int(correct),
                "accuracy": accuracy,
                "error_rate": 1 - accuracy,
            },
            "analysis_timestamp": datetime.now().isoformat(),
        }

        self.analysis_summary = analysis
        return analysis

    # ------------------------------------------------------------------

    def save_results(self) -> bool:
        try:
            if self.results_df is not None:
                path = os.path.join(self.output_dir, self.results_filename)
                self.results_df.to_csv(path, index=False)
                print(f"Results saved to {path}")

            if self.analysis_summary:
                analysis_path = os.path.join(
                    self.output_dir, "analysis_summary.json"
                )
                with open(analysis_path, "w") as f:
                    json.dump(self.analysis_summary, f, indent=2)
                print(f"Analysis summary saved to {analysis_path}")

            return True

        except Exception as e:
            print(f"Error saving results: {e}")
            return False


# ----------------------------------------------------------------------

def main():
    print("ResultsAnalyzer loaded successfully.")


if __name__ == "__main__":
    main()
