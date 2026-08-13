"""
Full pipeline:
1. Read CSV
2. Download all images (skip if already exist)
3. Generate Fashion-CLIP embeddings for each image
4. Extract color + texture features (stored as metadata)
5. Index into ChromaDB

Run: python scripts/build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests
import time
import json
from tqdm import tqdm
from PIL import Image
from io import BytesIO

from src import config
from src.embeddings import get_embedder
from src.feature_extractor import extract_all_features
from src.vector_store import get_store


def download_images():
    """Download all images from CSV to local folder"""
    print("\n" + "="*60)
    print("STEP 1: DOWNLOADING IMAGES")
    print("="*60)
    
    df = pd.read_csv(config.EXCEL_FILE)
    print(f"✅ CSV loaded: {len(df)} rows")
    
    df = df.dropna(subset=[config.COL_IMAGE_URL, config.COL_SKU])
    df = df[df[config.COL_IMAGE_URL].str.startswith('http', na=False)]
    print(f"✅ Valid rows: {len(df)}")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        sku = str(row[config.COL_SKU]).strip()
        url = str(row[config.COL_IMAGE_URL]).strip()
        
        ext = url.split('.')[-1].lower().split('?')[0]
        if ext not in ['jpg', 'jpeg', 'png', 'webp']:
            ext = 'jpg'
        
        save_path = config.IMAGES_DIR / f"{sku}.{ext}"
        
        if save_path.exists():
            skipped += 1
            continue
        
        success = False
        for attempt in range(config.DOWNLOAD_RETRIES):
            try:
                response = requests.get(
                    url, 
                    timeout=config.DOWNLOAD_TIMEOUT,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                response.raise_for_status()
                
                img = Image.open(BytesIO(response.content))
                img.verify()
                
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded += 1
                success = True
                break
            except Exception:
                if attempt < config.DOWNLOAD_RETRIES - 1:
                    time.sleep(1)
        
        if not success:
            failed += 1
    
    print(f"\n✅ Downloaded: {downloaded} | Skipped: {skipped} | Failed: {failed}")
    return df


def build_vector_index(df):
    """Generate embeddings + features and index into ChromaDB"""
    print("\n" + "="*60)
    print("STEP 2: BUILDING VECTOR INDEX")
    print("="*60)
    
    embedder = get_embedder()
    store = get_store()
    
    # Reset if already indexed (fresh build)
    existing_count = store.count()
    if existing_count > 0:
        print(f"⚠️  Existing {existing_count} items found in index.")
        response = input("Delete and rebuild? (y/n): ").strip().lower()
        if response == 'y':
            store.reset()
            print("🗑️  Old index cleared.")
        else:
            print("✅ Keeping existing index. Exiting.")
            return
    
    # ✨ FIX: Remove duplicate SKUs (keep first occurrence)
    df = df.dropna(subset=[config.COL_IMAGE_URL, config.COL_SKU])
    initial_count = len(df)
    df = df.drop_duplicates(subset=[config.COL_SKU], keep='first')
    duplicates_removed = initial_count - len(df)
    if duplicates_removed > 0:
        print(f"🧹 Removed {duplicates_removed} duplicate SKUs from CSV")
    
    ids_to_add = []
    embeddings_to_add = []
    metadatas_to_add = []
    seen_ids = set()  # ✨ Extra safety: track already-added IDs
    
    batch_size = 50
    processed = 0
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Indexing"):
        sku = str(row[config.COL_SKU]).strip()
        
        # Skip if already seen (extra guard)
        if sku in seen_ids:
            skipped += 1
            continue
        
        # Find local image file
        image_path = None
        for ext in ['webp', 'jpg', 'jpeg', 'png']:
            candidate = config.IMAGES_DIR / f"{sku}.{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            skipped += 1
            continue
        
        try:
            image = Image.open(image_path).convert("RGB")
            embedding = embedder.embed_image(image)
            features = extract_all_features(image)
            
            metadata = {
                "sku": sku,
                "name": str(row.get(config.COL_NAME, "")),
                "price": float(row.get(config.COL_PRICE, 0) or 0),
                "discounted_price": float(row.get(config.COL_DISCOUNT, 0) or 0),
                "website_link": str(row.get(config.COL_WEBSITE, "")),
                "image_path": str(image_path),
                "color_hist": json.dumps(features["color"].tolist()),
                "texture_hist": json.dumps(features["texture"].tolist()),
            }
            
            ids_to_add.append(sku)
            embeddings_to_add.append(embedding)
            metadatas_to_add.append(metadata)
            seen_ids.add(sku)
            
            # Batch insert
            if len(ids_to_add) >= batch_size:
                store.add(ids_to_add, embeddings_to_add, metadatas_to_add)
                processed += len(ids_to_add)
                ids_to_add, embeddings_to_add, metadatas_to_add = [], [], []
        
        except Exception as e:
            skipped += 1
            continue
    
    # Insert leftovers
    if ids_to_add:
        store.add(ids_to_add, embeddings_to_add, metadatas_to_add)
        processed += len(ids_to_add)
    
    print(f"\n✅ Indexed: {processed}")
    print(f"⚠️  Skipped: {skipped}")
    print(f"📊 Final store count: {store.count()}")


if __name__ == "__main__":
    df = download_images()
    build_vector_index(df)
    
    print("\n" + "="*60)
    print("✅ ALL DONE! Vector index ready.")
    print("="*60)
    print(f"\n📁 Images: {config.IMAGES_DIR}")
    print(f"💾 Vector DB: {config.CHROMA_DIR}")
    print("\n👉 Next: Run the Streamlit app: streamlit run app.py")