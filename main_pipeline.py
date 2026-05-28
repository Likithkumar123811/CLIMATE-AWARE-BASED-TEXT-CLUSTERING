"""
Main Pipeline Script
Modified for Climate Dataset Clustering + Evaluation + Reasoning
"""

import os
import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"    

# Import modules
from data_preparation import DataPreparator
from embedding_generator import EmbeddingGenerator
from vector_storage_retrieval import VectorStorageRetrieval
from deepseek_integration import DeepSeekReasoning
from clustering_pipeline import NewsClusteringPipeline
from results_analysis import ResultsAnalyzer
from visualization import NewsClassificationVisualizer
from example_cases import ExampleCasesDemo
from config import Config

# ✅ NEW IMPORT
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


class NewsClassificationPipeline:
    def __init__(self,
                 data_path: str = "climate_dataset_10_categories_5000.csv",  # ✅ MODIFIED
                 deepseek_api_key: Optional[str] = None,
                 output_dir: str = "output",
                 sample_size: Optional[int] = None):

        self.data_path = data_path
        self.deepseek_api_key = deepseek_api_key
        self.output_dir = output_dir
        self.sample_size = sample_size

        os.makedirs(output_dir, exist_ok=True)

        self.data_preparator = DataPreparator(data_path)
        self.embedding_generator = EmbeddingGenerator()
        self.vector_storage = VectorStorageRetrieval()
        self.deepseek = DeepSeekReasoning(api_key=deepseek_api_key)
        self.clustering_pipeline = NewsClusteringPipeline()
        self.results_analyzer = ResultsAnalyzer(output_dir=output_dir)
        self.visualizer = NewsClassificationVisualizer(
            output_dir=os.path.join(output_dir, "visualizations")
        )

        self.train_data = None
        self.embeddings_generated = False
        self.clustering_done = False

        print("Pipeline initialized successfully!")

    # -------------------------------------------------------------

    def run_data_preparation(self) -> bool:
        print("\nSTEP 1: DATA PREPARATION")

        # ✅ MODIFIED: Direct load (skip old preprocessing)
        self.train_data = pd.read_csv(self.data_path)

        print(f"Records: {len(self.train_data)}")
        print(f"Categories: {len(self.train_data['Category'].unique())}")

        return True

    # -------------------------------------------------------------

    def run_embedding_generation(self) -> bool:
        print("\nSTEP 2: EMBEDDING GENERATION")

        try:
            # ✅ MODIFIED: Use Text column
            texts = self.train_data["Text"].tolist()
            categories = self.train_data["Category"].tolist()

            embeddings = self.embedding_generator.model.encode(texts)

            self.embeddings = embeddings
            self.embeddings_generated = True

            print("Embeddings generated:", len(embeddings))
            return True

        except Exception as e:
            print(f"Error: {e}")
            return False

    # -------------------------------------------------------------

    def run_clustering_analysis(self) -> bool:
        print("\nSTEP 3: CLUSTERING")

        if not self.embeddings_generated:
            return False

        from sklearn.cluster import KMeans

        # ✅ MODIFIED: real clustering (not random)
        k = len(self.train_data["Category"].unique())

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(self.embeddings)

        self.train_data["cluster"] = labels

        self.clustering_done = True

        print(f"Clustering completed with {k} clusters")
        return True

    # -------------------------------------------------------------

    def run_prediction_and_analysis(self) -> bool:
        print("\nSTEP 4: EVALUATION + REASONING")

        if not self.clustering_done:
            return False

        # ✅ NEW: ARI & NMI
        ari = adjusted_rand_score(
            self.train_data["Category"],
            self.train_data["cluster"]
        )

        nmi = normalized_mutual_info_score(
            self.train_data["Category"],
            self.train_data["cluster"]
        )

        print("ARI:", ari)
        print("NMI:", nmi)

        # ✅ MODIFIED: DeepSeek reasoning (sample only)
        sample_df = self.train_data.sample(50)

        reasons = []
        for _, row in sample_df.iterrows():

            prompt = f"""
            Text: {row['Text']}
            Predicted Cluster: {row['cluster']}
            Actual Category: {row['Category']}

            Explain why this classification makes sense.
            """

            if self.deepseek_api_key:
                result = self.deepseek.generate_response(prompt)
            else:
                result = "Semantic similarity based reasoning"

            reasons.append(result)

        sample_df["reason"] = reasons
        sample_df.to_csv("output/sample_reasoning.csv", index=False)

        print("Reasoning generated!")

        return True

    # -------------------------------------------------------------

    def run_complete_pipeline(self) -> bool:

        steps = [
            self.run_data_preparation,
            self.run_embedding_generation,
            self.run_clustering_analysis,
            self.run_prediction_and_analysis,
        ]

        for step in steps:
            if not step():
                print("Pipeline failed.")
                return False

        # ✅ SAVE FINAL OUTPUT
        self.train_data.to_csv("output/final_results.csv", index=False)

        # ✅ VISUALIZATION
        self.visualizer.plot_category_distribution(self.train_data)
        self.visualizer.plot_cluster_distribution(self.train_data)

        print("\n✅ PIPELINE COMPLETED SUCCESSFULLY")
        return True


# -------------------------------------------------------------

def main():
    pipeline = NewsClassificationPipeline(
        data_path="climate_dataset_10_categories_5000.csv",
        deepseek_api_key=None,
        sample_size=None
    )

    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()