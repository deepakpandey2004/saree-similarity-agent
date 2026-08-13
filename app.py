"""
Streamlit Chat UI for Saree Similarity Agent.

Features:
- Natural chat interface
- Image upload (file) or URL input
- Beautiful results display with images side-by-side
- Score breakdown for transparency
"""
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import os
import re

from src.agent import get_agent
from src.search_engine import get_search_engine
from src.vector_store import get_store


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Saree Similarity Search AI",
    page_icon="🥻",
    layout="wide",
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #B8336A;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .score-badge {
        background: #B8336A;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ==================== HEADER ====================
st.markdown('<h1 class="main-title">🥻 Saree Similarity Search AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a saree image and find visually similar ones from our catalog</p>', unsafe_allow_html=True)


# ==================== HELPERS ====================
def load_image_from_url(url: str) -> Image.Image:
    """
    Load image from URL. Handles both direct image URLs and webpage URLs
    (tries to extract og:image or first <img> from HTML).
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, timeout=15, headers=headers)
    response.raise_for_status()

    content_type = response.headers.get('Content-Type', '').lower()

    # Case 1: Direct image URL
    if 'image' in content_type:
        return Image.open(BytesIO(response.content)).convert("RGB")

    # Case 2: HTML page - try to extract og:image
    html = response.text
    og_match = re.search(
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        html, re.IGNORECASE
    )
    if og_match:
        img_url = og_match.group(1)
        img_response = requests.get(img_url, timeout=15, headers=headers)
        img_response.raise_for_status()
        return Image.open(BytesIO(img_response.content)).convert("RGB")

    # Case 3: Fallback - first product image
    img_match = re.search(r'<img[^>]+src=["\']([^"\']+\.(?:jpg|jpeg|png|webp))["\']', html, re.IGNORECASE)
    if img_match:
        img_url = img_match.group(1)
        if not img_url.startswith('http'):
            from urllib.parse import urljoin
            img_url = urljoin(url, img_url)
        img_response = requests.get(img_url, timeout=15, headers=headers)
        img_response.raise_for_status()
        return Image.open(BytesIO(img_response.content)).convert("RGB")

    raise ValueError("Could not find an image at the given URL. Please provide a direct image URL.")


# ==================== SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Namaste! 🙏 I'm your Byrappa Silks assistant. Upload a saree image (or paste an image URL) in the sidebar, and I'll find visually similar sarees from our catalog. How can I help you today?"}
    ]

if "current_image" not in st.session_state:
    st.session_state.current_image = None

if "last_results" not in st.session_state:
    st.session_state.last_results = None


# ==================== SIDEBAR: Image Input ====================
with st.sidebar:
    st.header("📸 Upload Saree Image")

    input_method = st.radio("Choose input method:", ["Upload File", "Image URL"], horizontal=True)

    query_image = None

    if input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload a saree image to find similar ones"
        )
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert("RGB")
    else:
        url = st.text_input("Paste image URL:", placeholder="https://...")
        if url:
            try:
                query_image = load_image_from_url(url)
            except Exception as e:
                st.error(f"Could not load image: {e}")

    # Display uploaded image
    if query_image is not None:
        st.session_state.current_image = query_image
        st.image(query_image, caption="Your Query Image")

    st.divider()

    # Stats
    try:
        store = get_store()
        st.metric("📊 Catalog Size", f"{store.count()} sarees")
    except Exception:
        pass

    if st.button("🔄 Reset Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat reset! Upload a new image to start."}
        ]
        st.session_state.current_image = None
        st.session_state.last_results = None
        get_agent().reset()
        st.rerun()


# ==================== MAIN: Chat + Results ====================
col_chat, col_results = st.columns([1, 1])

with col_chat:
    st.subheader("💬 Chat")

    # Display chat messages
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask me anything or say 'find similar sarees'...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            try:
                agent = get_agent()
                response = agent.chat(user_input, uploaded_image=st.session_state.current_image)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Cache results for visual display if user asked for similarity
                if st.session_state.current_image is not None:
                    search_keywords = ["similar", "find", "match", "like this", "search", "show me"]
                    if any(kw in user_input.lower() for kw in search_keywords):
                        engine = get_search_engine()
                        results = engine.search(st.session_state.current_image, top_k=5)
                        st.session_state.last_results = results
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {str(e)}"
                })

        st.rerun()


with col_results:
    st.subheader("🎯 Top Matches")

    if st.session_state.last_results:
        results = st.session_state.last_results
        results_container = st.container(height=500)
        with results_container:
            for i, r in enumerate(results, 1):
                with st.container(border=True):
                    cols = st.columns([1, 2])
                    with cols[0]:
                        if os.path.exists(r["image_path"]):
                            st.image(r["image_path"])
                    with cols[1]:
                        st.markdown(f"**#{i} · {r['name'][:50]}**")
                        st.markdown(
                            f"<span class='score-badge'>{r['score']:.1%} match</span>",
                            unsafe_allow_html=True
                        )
                        st.caption(f"SKU: {r['sku']}")
                        st.caption(f"💰 ₹{r['discounted_price']:.0f} (MRP ₹{r['price']:.0f})")
                        with st.expander("Score Breakdown"):
                            st.write(f"🎨 Visual (CLIP): {r['clip_score']:.1%}")
                            st.write(f"🌈 Color: {r['color_score']:.1%}")
                            st.write(f"🧵 Texture: {r['texture_score']:.1%}")
                        if r.get('website_link'):
                            st.link_button("View on Website", r['website_link'])
    else:
        st.info("👈 Upload an image and ask me to find similar sarees!")


# ==================== FOOTER ====================
st.divider()
st.caption("Powered by Fashion-CLIP + ChromaDB + LangChain + Groq (Llama 3.1) | Hybrid re-ranking for fine-grained matches")