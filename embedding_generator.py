"""
Embedding Generation Module using ChromaDB
Creates vector embeddings for climate text and stores them in ChromaDB
"""

import chromadb
from chromadb.config import Settings
import pandas as pd
from typing import List, Dict, Any, Optional
import os
import json
from tqdm import tqdm
import uuid

#  ADDED (IMPORTANT)
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    def __init__(
        self,
        collection_name: str = "climate_embeddings",  # ✅ MODIFIED
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

        #  ADDED MODEL INITIALIZATION
        self.model = SentenceTransformer(self.embedding_model)

        self.metadata = {
            "total_documents": 0,
            "categories": set(),
            "embedding_dimension": None,
            "model_name": embedding_model
        }

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Climate embeddings with metadata"}  # ✅ MODIFIED
        )

        self.metadata["total_documents"] = self.collection.count()


    # ------------------------------------------------------------------

    def generate_embeddings_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "Text",          # ✅ MODIFIED
        category_column: str = "Category",  # ✅ MODIFIED
        batch_size: int = 100
    ) -> bool:

        if df is None or df.empty:
            print("Error: DataFrame is empty or None")
            return False

        print(f"Generating embeddings for {len(df)} documents...")
        print(f"Embedding model: {self.embedding_model}")

        #  MODIFIED
        texts = df[text_column].astype(str).tolist()
        categories = df[category_column].astype(str).tolist()

        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))

            batch_texts = texts[start:end]
            batch_categories = categories[start:end]

            batch_ids = [str(uuid.uuid4()) for _ in batch_texts]

            batch_metadata = [
                {
                    "category": cat,
                    "text_length": len(text),
                    "batch_id": batch_idx
                }
                for cat, text in zip(batch_categories, batch_texts)
            ]

            try:
                self.collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                self.metadata["categories"].update(batch_categories)

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")

        self.metadata["total_documents"] = self.collection.count()
        self.metadata["categories"] = list(self.metadata["categories"])

        print(f"Embedding generation completed")
        print(f"Total documents: {self.metadata['total_documents']}")
        print(f"Unique categories: {len(self.metadata['categories'])}")

        return True

    # ------------------------------------------------------------------

    def get_similar_documents(
        self,
        query_text: str,
        n_results: int = 5,
        where_clause: Optional[Dict] = None
    ) -> List[Dict]:

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_clause
            )

            return [
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
                for i in range(len(results["documents"][0]))
            ]

        except Exception as e:
            print(f"Error retrieving similar documents: {e}")
            return []

    # ------------------------------------------------------------------

    def get_embeddings_by_category(self, category: str) -> List[Dict]:

        try:
            results = self.collection.get(where={"category": category})

            return [
                {
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
                for i in range(len(results["documents"]))
            ]

        except Exception as e:
            print(f"Error retrieving category {category}: {e}")
            return []

    # ------------------------------------------------------------------

    def get_collection_stats(self) -> Dict[str, Any]:

        try:
            all_docs = self.collection.get()
            categories = [meta["category"] for meta in all_docs["metadatas"]]

            return {
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "total_documents": self.collection.count(),
                "unique_categories": len(set(categories)),
                "category_distribution": pd.Series(categories).value_counts().to_dict()
            }

        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}

    # ------------------------------------------------------------------

    def save_metadata(self, filepath: str = "embedding_metadata.json") -> bool:

        try:
            metadata_copy = self.metadata.copy()
            metadata_copy["categories"] = list(metadata_copy["categories"])

            with open(filepath, "w") as f:
                json.dump(metadata_copy, f, indent=2)

            print(f"Metadata saved to {filepath}")
            return True

        except Exception as e:
            print(f"Error saving metadata: {e}")
            return False

    # ------------------------------------------------------------------

    def load_metadata(self, filepath: str = "embedding_metadata.json") -> bool:

        try:
            with open(filepath, "r") as f:
                metadata = json.load(f)

            metadata["categories"] = set(metadata.get("categories", []))
            self.metadata.update(metadata)

            print(f"Metadata loaded from {filepath}")
            return True

        except Exception as e:
            print(f"Error loading metadata: {e}")
            return False

    # ------------------------------------------------------------------

    def clear_collection(self) -> bool:

        try:
            self.client.delete_collection(name=self.collection_name)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Climate embeddings with metadata"}  #  MODIFIED
            )

            self.metadata = {
                "total_documents": 0,
                "categories": set(),
                "embedding_dimension": None,
                "model_name": self.embedding_model
            }

            print("Collection cleared successfully")
            return True

        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False


# ----------------------------------------------------------------------

def main():
    print("EmbeddingGenerator module loaded successfully.")


if __name__ == "__main__":
    main()