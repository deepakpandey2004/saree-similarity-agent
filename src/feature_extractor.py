"""
Extra features for fine-grained saree similarity:
1. Color Histogram (HSV) - captures color palette
2. LBP (Local Binary Patterns) - captures fabric/weave texture

Ye features CLIP ke saath combine hoke fine-grained match dete hain.
"""
import numpy as np
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern
from src import config


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR format"""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    rgb = np.array(pil_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def extract_color_histogram(pil_image: Image.Image) -> np.ndarray:
    """
    Extract HSV color histogram.
    HSV is better than RGB for color similarity (hue captures color meaningfully).
    Returns normalized histogram (sums to 1).
    """
    bgr = pil_to_cv2(pil_image)
    # Resize for consistency
    bgr = cv2.resize(bgr, config.IMAGE_SIZE)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # Compute 3D histogram (H, S, V) - captures color combinations
    hist = cv2.calcHist(
        [hsv], 
        [0, 1, 2], 
        None, 
        [config.COLOR_BINS, config.COLOR_BINS, config.COLOR_BINS],
        [0, 180, 0, 256, 0, 256]
    )
    
    # Normalize and flatten
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def extract_texture_features(pil_image: Image.Image) -> np.ndarray:
    """
    Extract LBP (Local Binary Pattern) texture features.
    Captures fabric weave patterns - crucial for saree matching.
    """
    bgr = pil_to_cv2(pil_image)
    bgr = cv2.resize(bgr, config.IMAGE_SIZE)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # LBP
    lbp = local_binary_pattern(
        gray, 
        P=config.LBP_POINTS, 
        R=config.LBP_RADIUS, 
        method='uniform'
    )
    
    # Histogram of LBP values
    n_bins = config.LBP_POINTS + 2  # uniform method
    hist, _ = np.histogram(
        lbp.ravel(), 
        bins=n_bins, 
        range=(0, n_bins),
        density=True
    )
    
    return hist.astype(np.float32)


def extract_all_features(pil_image: Image.Image) -> dict:
    """
    Extract all features from a single image.
    Returns dict with 'color' and 'texture' arrays.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    return {
        "color": extract_color_histogram(pil_image),
        "texture": extract_texture_features(pil_image)
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between 2 vectors"""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


def histogram_intersection(a: np.ndarray, b: np.ndarray) -> float:
    """
    Histogram intersection - better for color/texture histograms.
    Returns value in [0, 1].
    """
    return float(np.sum(np.minimum(a, b)) / (np.sum(a) + 1e-8))