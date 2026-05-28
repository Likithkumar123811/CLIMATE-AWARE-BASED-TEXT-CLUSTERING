"""
Vector Storage & Retrieval Module
Enhanced ChromaDB integration with advanced retrieval and storage functions
"""

import chromadb
from chromadb.config import Settings
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class VectorStorageRetrieval:
    def __init__(self,
                 collection_name: str = "news_embeddings",
                 persist_directory: str = "./chroma_db"):

        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = None
        self._initialize_collection()

        self.cache = {
            'category_centroids': {},
            'category_stats': {},
            'last_updated': None
        }

    # ---------------------------------------------------------

    def _initialize_collection(self):
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "News category embeddings with metadata"}
            )

    # ---------------------------------------------------------

    def store_embeddings_with_metadata(self,
                                     texts: List[str],
                                     categories: List[str],
                                     additional_metadata: Optional[List[Dict]] = None,
                                     batch_size: int = 100) -> bool:

        if len(texts) != len(categories):
            return False

        if additional_metadata is None:
            additional_metadata = [{}] * len(texts)

        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(texts))

            batch_texts = texts[start_idx:end_idx]
            batch_categories = categories[start_idx:end_idx]
            batch_metadata = additional_metadata[start_idx:end_idx]

            batch_ids = [
                f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                for i in range(start_idx, end_idx)
            ]

            enhanced_metadata = []
            for text, category, meta in zip(batch_texts, batch_categories, batch_metadata):
                enhanced_meta = {
                    "category": category,
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "timestamp": datetime.now().isoformat(),
                    **meta
                }
                enhanced_metadata.append(enhanced_meta)

            try:
                self.collection.add(
                    documents=batch_texts,
                    metadatas=enhanced_metadata,
                    ids=batch_ids
                )
            except Exception:
                continue

        return True

    # ---------------------------------------------------------

    def retrieve_top_k_similar(self,
                              query_text: str,
                              k: int = 5,
                              category_filter: Optional[str] = None):

        try:
            where_clause = {"category": category_filter} if category_filter else None

            results = self.collection.query(
                query_texts=[query_text],
                n_results=k,
                where=where_clause
            )

            similar_articles = []
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                similarity = 1 - distance

                similar_articles.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'similarity': similarity,
                    'category': results['metadatas'][0][i].get('category', 'Unknown')
                })

            return similar_articles

        except Exception:
            return []

    # ---------------------------------------------------------

    def get_category_statistics(self) -> Dict[str, Any]:
        try:
            all_docs = self.collection.get()

            if not all_docs['documents']:
                return {}

            categories = [doc['category'] for doc in all_docs['metadatas']]
            category_counts = pd.Series(categories).value_counts()

            return {
                'total_documents': len(all_docs['documents']),
                'unique_categories': len(category_counts),
                'category_distribution': category_counts.to_dict()
            }

        except Exception:
            return {}

    # ---------------------------------------------------------

    def export_embeddings_data(self,
                              output_file: str = "embeddings_export.pkl") -> bool:

        try:
            all_docs = self.collection.get()

            export_data = {
                'documents': all_docs['documents'],
                'metadatas': all_docs['metadatas'],
                'ids': all_docs['ids'],
                'timestamp': datetime.now().isoformat()
            }

            with open(output_file, 'wb') as f:
                pickle.dump(export_data, f)

            return True

        except Exception:
            return False

    # ---------------------------------------------------------

    def import_embeddings_data(self,
                              input_file: str = "embeddings_export.pkl") -> bool:

        try:
            with open(input_file, 'rb') as f:
                data = pickle.load(f)

            self.client.delete_collection(name=self.collection_name)

            self.collection = self.client.create_collection(
                name=self.collection_name
            )

            self.collection.add(
                documents=data['documents'],
                metadatas=data['metadatas'],
                ids=data['ids']
            )

            return True

        except Exception:
            return False


# ---------------------------------------------------------

def main():
    print("VectorStorageRetrieval ready (clean mode)")


if __name__ == "__main__":
    main()