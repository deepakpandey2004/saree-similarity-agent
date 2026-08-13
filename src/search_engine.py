"""
Hybrid Search Engine - THE quality differentiator.

Strategy:
1. Fast: Fashion-CLIP embedding search → get top-K candidates from ChromaDB
2. Slow but accurate: Re-rank candidates using:
   - CLIP similarity (semantic/overall look)  → 55%
   - Color histogram similarity (color palette match) → 30%
   - Texture similarity (fabric/weave pattern) → 15%

Why this beats basic embedding search:
- Sarees are visually similar overall, so pure CLIP returns loose matches
- Color histograms catch color combinations (e.g., pink+gold, red+black)
- Texture (LBP) catches fabric type (silk vs cotton vs organza)
- Weighted combination gives fine-grained ranking
"""
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Optional

from src import config
from src.embeddings import get_embedder
from src.feature_extractor import extract_all_features, histogram_intersection
from src.vector_store import get_store


class HybridSearchEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.embedder = get_embedder()
            self.store = get_store()
            self._initialized = True
    
    def search(
        self,
        query_image: Image.Image,
        top_k: int = None,
        initial_k: int = None,
    ) -> List[Dict]:
        """
        Full hybrid search pipeline.
        
        Args:
            query_image: PIL Image of the query saree
            top_k: final results to return (default from config)
            initial_k: candidates to fetch before re-ranking
        
        Returns:
            List of dicts with: sku, name, price, image_path, website_link,
                                score, clip_score, color_score, texture_score
        """
        top_k = top_k or config.FINAL_RESULTS_K
        initial_k = initial_k or config.INITIAL_SEARCH_K
        
        # ==== STAGE 1: Generate query features ====
        query_embedding = self.embedder.embed_image(query_image)
        query_features = extract_all_features(query_image)
        query_color = query_features["color"]
        query_texture = query_features["texture"]
        
        # ==== STAGE 2: Get top-K candidates from ChromaDB (fast) ====
        candidates = self.store.search(query_embedding, top_k=initial_k)
        
        if not candidates:
            return []
        
        # ==== STAGE 3: Re-rank using hybrid scoring ====
        reranked = []
        for cand in candidates:
            metadata = cand["metadata"]
            clip_score = cand["score"]  # cosine similarity from ChromaDB
            
            # Deserialize stored features
            try:
                cand_color = np.array(json.loads(metadata["color_hist"]), dtype=np.float32)
                cand_texture = np.array(json.loads(metadata["texture_hist"]), dtype=np.float32)
            except Exception:
                # Fallback: use only CLIP score
                cand_color = query_color
                cand_texture = query_texture
            
            # Color similarity (histogram intersection - good for histograms)
            color_score = histogram_intersection(query_color, cand_color)
            
            # Texture similarity
            texture_score = histogram_intersection(query_texture, cand_texture)
            
            # Weighted hybrid score
            final_score = (
                config.WEIGHT_CLIP * clip_score +
                config.WEIGHT_COLOR * color_score +
                config.WEIGHT_TEXTURE * texture_score
            )
            
            reranked.append({
                "sku": metadata["sku"],
                "name": metadata["name"],
                "price": metadata["price"],
                "discounted_price": metadata["discounted_price"],
                "website_link": metadata["website_link"],
                "image_path": metadata["image_path"],
                "score": round(float(final_score), 4),
                "clip_score": round(float(clip_score), 4),
                "color_score": round(float(color_score), 4),
                "texture_score": round(float(texture_score), 4),
            })
        
        # Sort by final score, take top_k
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]
    
    def search_from_path(self, image_path: str, top_k: int = None) -> List[Dict]:
        """Convenience: load image from path and search"""
        image = Image.open(image_path).convert("RGB")
        return self.search(image, top_k=top_k)
    
    def search_from_url(self, url: str, top_k: int = None) -> List[Dict]:
        """Convenience: download image from URL and search"""
        import requests
        from io import BytesIO
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return self.search(image, top_k=top_k)


# Singleton
_engine = None

def get_search_engine() -> HybridSearchEngine:
    global _engine
    if _engine is None:
        _engine = HybridSearchEngine()
    return _engine