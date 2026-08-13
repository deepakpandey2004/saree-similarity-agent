"""
Configuration file - saari constants aur paths yahan
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ==================== PATHS ====================
BASE_DIR = Path(__file__).parent.parent  # project root
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
CHROMA_DIR = DATA_DIR / "chroma_db"
EXCEL_FILE = BASE_DIR / "byrappa_tejas_31july.csv"

# Create directories if missing
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ==================== API KEYS ====================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==================== MODEL CONFIG ====================
# Fashion-CLIP: specifically trained on fashion images
# Fallback: use OpenCLIP ViT-B-32 if Fashion-CLIP fails
CLIP_MODEL_NAME = "patrickjohncyh/fashion-clip"
CLIP_EMBEDDING_DIM = 512

# LLM Config (Groq - FREE)
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.3

# ==================== VECTOR DB ====================
COLLECTION_NAME = "saree_collection"

# ==================== SEARCH CONFIG ====================
# How many candidates to fetch from vector DB before re-ranking
INITIAL_SEARCH_K = 25
# Final results to show user
FINAL_RESULTS_K = 5

# Hybrid scoring weights (must sum to 1.0)
# CLIP = semantic/visual overall
# Color = color palette match (sarees mein color CRUCIAL hai)
# Texture = fabric/pattern match
WEIGHT_CLIP = 0.55
WEIGHT_COLOR = 0.30
WEIGHT_TEXTURE = 0.15

# ==================== IMAGE PROCESSING ====================
IMAGE_SIZE = (224, 224)  # CLIP input size
COLOR_BINS = 32  # HSV histogram bins per channel
LBP_RADIUS = 3
LBP_POINTS = 24

# ==================== EXCEL COLUMNS ====================
COL_NAME = "Name"
COL_SKU = "SKU"
COL_PRICE = "Retail Price"
COL_DISCOUNT = "Discounted Price"
COL_IMAGE_URL = "image_url"
COL_WEBSITE = "Website Link"

# ==================== DOWNLOAD CONFIG ====================
DOWNLOAD_TIMEOUT = 15  # seconds
DOWNLOAD_RETRIES = 2