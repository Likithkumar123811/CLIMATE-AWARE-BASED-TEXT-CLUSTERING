#!/usr/bin/env python3

import sys
import importlib.util
import pandas as pd
import os
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_OFFLINE"] = "1"


def dynamic_import(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


print("Loading modules...")

embedding_generator = dynamic_import("embedding_generator", "embedding_generator.py")
vector_storage_retrieval = dynamic_import("vector_storage_retrieval", "vector_storage_retrieval.py")
visualization = dynamic_import("visualization", "visualization.py")

EmbeddingGenerator = embedding_generator.EmbeddingGenerator
VectorStorageRetrieval = vector_storage_retrieval.VectorStorageRetrieval
NewsClassificationVisualizer = visualization.NewsClassificationVisualizer


class NewsClassificationPipeline:
    def __init__(self,
                 data_path: str = "climate_dataset_10_categories_5000.csv"):

        self.data_path = data_path

        print("Loading embedding model once...")
        self.model = EmbeddingGenerator().model

        self.vector_db = VectorStorageRetrieval()

        print("Pipeline Initialized")

    def generate_reason(self, text):
        text = text.lower()

        if "rain" in text or "flood" in text:
            return "Indicates rainfall or water overflow event"
        elif "hail" in text:
            return "Mentions hailstorm with ice particles"
        elif "thunder" in text or "lightning" in text:
            return "Indicates thunderstorm activity"
        elif "heat" in text:
            return "Indicates extreme heat conditions"
        elif "cold" in text:
            return "Indicates cold wave conditions"
        elif "drought" in text:
            return "Indicates water scarcity"
        elif "fire" in text:
            return "Indicates wildfire spreading"
        elif "landslide" in text:
            return "Indicates land movement"
        elif "pollution" in text or "air quality" in text:
            return "Indicates air pollution"
        else:
            return "General climate event"

    def explain_misclassification(self, text, actual, predicted):
        text = text.lower()

        if actual != predicted:
            if ("rain" in text or "flood" in text):
                return "Rain-related event grouped with other weather events due to semantic similarity"
            elif "thunder" in text:
                return "Thunderstorm confused with cyclone/storm due to similar features"
            elif "heat" in text:
                return "Heatwave grouped with drought due to temperature relation"
            elif "cold" in text:
                return "Cold wave merged into general extreme weather cluster"
            elif "fire" in text:
                return "Wildfire grouped with heat-related events"
            else:
                return "Semantic overlap between categories caused misclassification"
        else:
            return "Correctly clustered"

    def run_full_pipeline(self):
        print("\nSTEP 1: LOAD DATA")
        df = pd.read_csv(self.data_path)

        print(f"Total rows: {len(df)}")
        print(df.head(3))

        print("\nSTEP 2: EMBEDDINGS")
        embeddings = self.model.encode(df["Text"].tolist())
        print("Embedding shape:", embeddings.shape)

        print("\nSTEP 3: PRE-CLUSTER REASONING")
        df['pre_reason'] = df['Text'].apply(self.generate_reason)
        print(df[['Text', 'pre_reason']].head(5))

        print("\nSTEP 4: STORE IN CHROMADB")
        self.vector_db.collection.add(
            documents=df["Text"].tolist(),
            embeddings=embeddings.tolist(),
            metadatas=[{"Category": c} for c in df["Category"]],
            ids=[str(i) for i in range(len(df))]
        )
        print("Stored successfully")

        print("\nSTEP 5: CLUSTERING (10 CLUSTERS)")
        from sklearn.cluster import KMeans

        k = 10
        kmeans = KMeans(n_clusters=k, random_state=42)
        df["cluster"] = kmeans.fit_predict(embeddings)

        print(df['cluster'].value_counts())

        cluster_map = df.groupby('cluster')['Category'].agg(
            lambda x: x.value_counts().index[0]
        )
        df['final_category'] = df['cluster'].map(cluster_map)

        print("\nSTEP 6: EVALUATION")
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        print("ARI:", adjusted_rand_score(df["Category"], df["cluster"]))
        print("NMI:", normalized_mutual_info_score(df["Category"], df["cluster"]))

        print("\nSTEP 7: WRONG CLUSTER ANALYSIS")
        df['is_wrong'] = df['Category'] != df['final_category']
        df['error_reason'] = df.apply(
            lambda row: self.explain_misclassification(
                row['Text'],
                row['Category'],
                row['final_category']
            ),
            axis=1
        )

        wrong_df = df[df['is_wrong']].head(5)
        for _, row in wrong_df.iterrows():
            print("\nText:", row['Text'])
            print("Actual:", row['Category'])
            print("Predicted:", row['final_category'])
            print("Reason:", row['error_reason'])

        print("\nSTEP 8: VISUALIZATION")
        visualizer = NewsClassificationVisualizer()
        visualizer.generate_all_visualizations(df, embeddings)

        print("\nSTEP 9: SAVE RESULTS")
        os.makedirs("output", exist_ok=True)
        df.to_csv("output/final_results.csv", index=False)

        print("\nSTEP 10: GENERATE RESEARCH TABLES")
        from classification_report_extra import generate_report
        generate_report("output/final_results.csv")

        print("\n✅ PIPELINE COMPLETE")


def main():
    pipeline = NewsClassificationPipeline()
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()