"""
Clustering Pipeline Module
Implements clustering using KMeans and Agglomerative Clustering
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import pickle
import os
from datetime import datetime


class NewsClusteringPipeline:
    def __init__(self,
                 n_clusters_range: Tuple[int, int] = (5, 20),
                 random_state: int = 42):

        self.n_clusters_range = n_clusters_range
        self.random_state = random_state

        self.clustering_results = {}
        self.embeddings = None
        self.labels = None
        self.categories = None

        self.evaluation_metrics = {}
        self.visualization_data = {}

    # ------------------------------------------------------------------

    def prepare_embeddings(self,
                          embeddings_data: List[np.ndarray],
                          categories: List[str],
                          texts: List[str]) -> bool:
        try:
            # ✅ MODIFIED (safe casting)
            self.embeddings = np.array(embeddings_data)
            self.categories = np.array(categories).astype(str)
            self.texts = np.array(texts).astype(str)

            scaler = StandardScaler()
            self.embeddings_scaled = scaler.fit_transform(self.embeddings)

            print(f"Prepared {len(self.embeddings)} embeddings")
            print(f"Embedding dimension: {self.embeddings.shape[1]}")
            print(f"Unique categories: {len(np.unique(self.categories))}")

            return True

        except Exception as e:
            print(f"Error preparing embeddings: {e}")
            return False

    # ------------------------------------------------------------------

    def perform_clustering(self,
                          n_clusters: int,
                          method: str = 'kmeans') -> Dict[str, Any]:

        if self.embeddings is None:
            print("Error: No embeddings prepared.")
            return {}

        print(f"Performing {method} clustering with {n_clusters} clusters...")

        if method == 'kmeans':
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            labels = clusterer.fit_predict(self.embeddings_scaled)

            result = {
                'method': 'kmeans',
                'n_clusters': n_clusters,
                'labels': labels,
                'cluster_centers': clusterer.cluster_centers_,
                'inertia': clusterer.inertia_,
                'silhouette_score': silhouette_score(self.embeddings_scaled, labels),
                'ari_score': adjusted_rand_score(self.categories, labels),
                'nmi_score': normalized_mutual_info_score(self.categories, labels)
            }

        elif method == 'agglomerative':
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            labels = clusterer.fit_predict(self.embeddings_scaled)

            result = {
                'method': 'agglomerative',
                'n_clusters': n_clusters,
                'labels': labels,
                'silhouette_score': silhouette_score(self.embeddings_scaled, labels),
                'ari_score': adjusted_rand_score(self.categories, labels),
                'nmi_score': normalized_mutual_info_score(self.categories, labels)
            }

        else:
            print(f"Unknown method: {method}")
            return {}

        self.labels = labels
        self.current_clustering = result

        print(f"Silhouette: {result['silhouette_score']:.3f}")
        print(f"ARI: {result['ari_score']:.3f}, NMI: {result['nmi_score']:.3f}")

        return result

    # ------------------------------------------------------------------

    def analyze_clusters(self,
                        clustering_result: Optional[Dict] = None) -> Dict[str, Any]:

        if clustering_result is None:
            clustering_result = self.current_clustering

        if clustering_result is None:
            print("No clustering result available")
            return {}

        labels = clustering_result['labels']

        analysis_df = pd.DataFrame({
            'text': self.texts,
            'true_category': self.categories,
            'cluster': labels
        })

        cluster_analysis = {}

        for cluster_id in np.unique(labels):
            cluster_data = analysis_df[analysis_df['cluster'] == cluster_id]

            most_common_category = cluster_data['true_category'].mode().iloc[0]
            category_counts = cluster_data['true_category'].value_counts()

            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'dominant_category': most_common_category,
                'purity': category_counts.iloc[0] / len(cluster_data),
                'sample_texts': cluster_data['text'].head(3).tolist()
            }

        return {
            'cluster_analysis': cluster_analysis,
            'overall_purity': np.mean([c['purity'] for c in cluster_analysis.values()])
        }

    # ------------------------------------------------------------------

    def prepare_visualization_data(self,
                                  method: str = 'pca',
                                  n_components: int = 2) -> Dict[str, Any]:

        if self.embeddings is None:
            print("No embeddings")
            return {}

        print(f"Preparing visualization using {method}")

        if method == 'pca':
            reducer = PCA(n_components=n_components)
            reduced = reducer.fit_transform(self.embeddings_scaled)
        else:
            reducer = TSNE(n_components=n_components)
            reduced = reducer.fit_transform(self.embeddings_scaled)

        viz_data = {
            'reduced_embeddings': reduced,
            'categories': self.categories,
            'texts': self.texts
        }

        if self.labels is not None:
            viz_data['cluster_labels'] = self.labels

        self.visualization_data = viz_data
        return viz_data

    # ------------------------------------------------------------------

    def save_clustering_results(self,
                               filename: str = "clustering_results.pkl") -> bool:

        try:
            data = {
                'labels': self.labels,
                'categories': self.categories,
                'texts': self.texts,
                'embeddings': self.embeddings
            }

            with open(filename, 'wb') as f:
                pickle.dump(data, f)

            print(f"Saved to {filename}")
            return True

        except Exception as e:
            print(f"Error saving: {e}")
            return False


# ----------------------------------------------------------------------

def main():
    print("Clustering pipeline ready!")


if __name__ == "__main__":
    main()