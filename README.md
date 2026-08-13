# 🥻 Saree Similarity Search AI Agent

An AI-powered chatbot that finds visually similar sarees from a fashion catalog using **hybrid image similarity search** (Fashion-CLIP embeddings + color histograms + texture features) and a **LangChain agent** with natural conversation.

## 🎯 Live Demo

- **App URL:** [https://huggingface.co/spaces/YOUR_USERNAME/saree-similarity-agent](#)

## ✨ Features

- 💬 **Natural chat interface** — talks conversationally, understands intent
- 📸 **Image input** — upload file OR paste image URL (auto-extracts from webpages)
- 🧠 **Fashion-CLIP embeddings** — fine-tuned on fashion data (much better than vanilla CLIP)
- 🌈 **Hybrid re-ranking** — combines visual + color + texture similarity
- 🎯 **Fine-grained matching** — designed for subtle differences between sarees
- ⚡ **Fast search** — ChromaDB vector store with cosine similarity
- 📊 **Transparent scoring** — see CLIP, color, texture breakdown for every match

## 🏗️ Architecture

```
User Input (Image + Chat)
        ↓
Streamlit UI
        ↓
LangChain Agent (Groq LLM: Llama 3.1 8B)
        ↓  (calls tool when image similarity requested)
Hybrid Search Tool
        ├─ Fashion-CLIP embedding (semantic features)
        ├─ HSV color histogram (color palette)
        └─ LBP texture (fabric/weave pattern)
        ↓
ChromaDB Vector Search (top-25 candidates)
        ↓
Re-ranking (weighted hybrid score)
        ↓
Top-5 Results with Scores
```

## 🛠️ Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| **Embedding Model** | [Fashion-CLIP](https://huggingface.co/patrickjohncyh/fashion-clip) | CLIP fine-tuned on 800K fashion images — significantly better than vanilla CLIP for garments |
| **Vector Database** | ChromaDB (persistent) | Free, embeddable, no external service needed |
| **Agent Framework** | LangChain (tool calling) | Clean tool schema + LLM function calling |
| **LLM** | Groq — Llama 3.1 8B Instant | Free tier, fast inference, supports tool calling |
| **Color Features** | OpenCV HSV Histogram (32×32×32 bins) | HSV captures color combinations better than RGB |
| **Texture Features** | scikit-image LBP (Local Binary Patterns) | Captures fabric weave/pattern texture |
| **Frontend** | Streamlit | Fast to build, native chat UI, easy deployment |
| **Deployment** | Hugging Face Spaces | Free, persistent storage, no cold-start |

## 🚀 What I Did to Improve Search Quality

The assignment explicitly warned: *"A basic embedding search will return loose, generic results on this dataset."* Here's how I addressed that:

### 1. Fashion-Specific Embedding Model
Used `patrickjohncyh/fashion-clip` instead of OpenAI CLIP. This model is fine-tuned on fashion product images, so its embedding space distinguishes fabrics, prints, and cuts much better.

### 2. Multi-Modal Feature Extraction
For every image, I extract THREE feature types:
- **Fashion-CLIP embedding (512-dim)** — captures overall semantic look
- **HSV color histogram (32×32×32)** — captures color palette combinations (crucial for sarees where color combos define identity)
- **LBP texture histogram** — captures fabric weave (silk vs cotton vs organza have different micro-patterns)

### 3. Two-Stage Retrieval with Re-ranking
- **Stage 1 (fast):** Query top-25 candidates from ChromaDB using CLIP cosine similarity
- **Stage 2 (accurate):** Re-rank candidates using weighted hybrid score:

```
final_score = 0.55 × CLIP_similarity 
            + 0.30 × color_histogram_intersection 
            + 0.15 × texture_histogram_intersection
```

Weights were chosen based on saree-specific reasoning:
- CLIP 0.55 → overall visual match is primary
- Color 0.30 → sarees often defined by their color palette (pink+gold, red+black, etc.)
- Texture 0.15 → subtle but important for fabric matching

### 4. Clean Tool Schema (LangChain)
The agent has a well-defined tool with strict Pydantic schema:

```python
class SareeSearchInput(BaseModel):
    image_reference: str  # Reference to uploaded image
    top_k: int = Field(default=5, ge=1, le=10)
```

The LLM only calls this tool when appropriate (image is uploaded + search intent detected), and receives structured, informative output to relay to the user.

## 📂 Project Structure

```
saree-similarity-agent/
├── app.py                    # Streamlit chat UI
├── requirements.txt
├── README.md
├── .env                      # GROQ_API_KEY (not committed)
├── src/
│   ├── config.py             # All config & constants
│   ├── embeddings.py         # Fashion-CLIP wrapper
│   ├── feature_extractor.py  # Color histogram + LBP texture
│   ├── vector_store.py       # ChromaDB operations
│   ├── search_engine.py      # Hybrid search + re-ranking
│   └── agent.py              # LangChain agent + tool schema
├── scripts/
│   └── build_index.py        # One-time indexing pipeline
└── data/
    ├── images/               # Downloaded catalog (gitignored)
    └── chroma_db/            # Persistent vector index (gitignored)
```

## 🖥️ Local Setup

### Prerequisites
- Python 3.11+
- Free Groq API key from [console.groq.com](https://console.groq.com/keys)

### Installation

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/saree-similarity-agent.git
cd saree-similarity-agent

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add Groq API key
echo GROQ_API_KEY=gsk_your_key_here > .env

# 5. Place the source CSV in project root
# (byrappa_tejas_31july.csv with columns: Name, SKU, image_url, Website Link, etc.)

# 6. Build the index (downloads images + generates embeddings)
python scripts/build_index.py

# 7. Run the app
streamlit run app.py
```

## 📊 Dataset

- **Source:** Byrappa Silks fashion catalog CSV (1074 rows)
- **Category:** Sarees (single category, fine-grained variation)
- **Indexed:** 651 unique sarees (after deduplication)
- **Attributes:** Name, SKU, price, image URL, product page

## ⚙️ Configuration

Adjust `src/config.py` to tune the system:

```python
INITIAL_SEARCH_K = 25       # Candidates from vector DB before re-ranking
FINAL_RESULTS_K = 5         # Final results shown to user

WEIGHT_CLIP = 0.55          # Visual similarity weight
WEIGHT_COLOR = 0.30         # Color match weight  
WEIGHT_TEXTURE = 0.15       # Texture match weight
```

## 🎨 Sample Queries

Try uploading:
- A saree image from another site
- A stock image of Indian traditional wear
- One of the catalog images (should return itself as top match)

The agent handles both:
- **Search intent:** *"find similar sarees"*, *"show me matches like this"*
- **General chat:** *"hi"*, *"what can you do?"* (no tool call)

## 🎯 Assumptions & Trade-offs

| Decision | Trade-off |
|----------|-----------|
| Fashion-CLIP over vanilla CLIP | +Better fashion embeddings, -600MB model download |
| ChromaDB local persistent store | +Free, no infrastructure, -limited to single-node scale |
| Groq Llama 3.1 8B | +Free & fast, -less capable than GPT-4 for complex reasoning |
| CPU inference | +Runs anywhere, -~2s per query embedding |
| Weighted hybrid (0.55/0.30/0.15) | +Balances visual + color + texture, -weights tuned empirically |
| One-time offline indexing | +Fast queries, -re-run needed when catalog changes |

## 🐛 Known Limitations

- Some product URLs in the source CSV return 404 (data issue, not application issue)
- CPU inference is slower than GPU (~2s vs ~200ms per query)
- ~157 duplicate SKUs in source CSV were deduplicated (first occurrence kept)

## 📄 License

MIT

## 🙏 Acknowledgements

- Fashion-CLIP by [Patrick John Chia](https://huggingface.co/patrickjohncyh/fashion-clip)
- Byrappa Silks for the catalog data
- Assignment by Internshala
