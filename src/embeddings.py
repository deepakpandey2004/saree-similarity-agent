"""
Fashion-CLIP based image embeddings.
Fashion-CLIP is specifically fine-tuned on fashion data,
giving MUCH better results than vanilla CLIP for sarees.
"""
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from src import config


class FashionCLIPEmbedder:
    _instance = None
    _model = None
    _processor = None
    
    def __new__(cls):
        # Singleton pattern - model ek hi baar load ho
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            print(f"🔄 Loading Fashion-CLIP model: {config.CLIP_MODEL_NAME}")
            print("   (First time takes 1-2 min to download ~600MB)")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   Device: {self.device}")
            
            self._model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
            self._processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
            self._model.to(self.device)
            self._model.eval()
            
            print("✅ Fashion-CLIP loaded!")
    
    @property
    def model(self):
        return self._model
    
    @property
    def processor(self):
        return self._processor
    
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Given a PIL image, return normalized embedding vector (512-dim).
        Normalization allows cosine similarity via dot product.
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
            # L2 normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()
    
    def embed_image_path(self, image_path: str) -> np.ndarray:
        """Convenience: load image from path and embed"""
        image = Image.open(image_path)
        return self.embed_image(image)


# Global instance
_embedder = None

def get_embedder() -> FashionCLIPEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = FashionCLIPEmbedder()
    return _embedder