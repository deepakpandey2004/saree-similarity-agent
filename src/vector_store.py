"""
ChromaDB wrapper - store and search embeddings + metadata.
Ye persistent hai - data disk pe save rehta hai.
"""
import chromadb
from chromadb.config import Settings
import numpy as np
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
from typing import List, Dict, Optional
from src import config


class VectorStore:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.client = chromadb.PersistentClient(
                path=str(config.CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name=config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}  # cosine similarity
            )
            self._initialized = True
            print(f"✅ VectorStore ready. Current count: {self.collection.count()}")
    
    def add(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict],
    ):
        """Add embeddings + metadata to store"""
        # Convert numpy arrays to lists for ChromaDB
        embeddings_list = [e.tolist() for e in embeddings]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 25
    ) -> List[Dict]:
        """
        Search by embedding vector.
        Returns list of dicts with: id, score, metadata
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Parse results
        parsed = []
        if results and results.get('ids') and results['ids'][0]:
            ids = results['ids'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]
            
            for i in range(len(ids)):
                # Cosine distance -> similarity
                similarity = 1.0 - distances[i]
                parsed.append({
                    "id": ids[i],
                    "score": float(similarity),
                    "metadata": metadatas[i]
                })
        
        return parsed
    
    def count(self) -> int:
        return self.collection.count()
    
    def reset(self):
        """Delete all data - use carefully"""
        self.client.delete_collection(config.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )


def get_store() -> VectorStore:
    return VectorStore()