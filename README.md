# 🧠 Detection of Big Five Personality Traits from X Posts using ML

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

Dự án Web Mining capstone - phát hiện tính cách Big Five (OCEAN) từ các bài đăng mạng xã hội, tích hợp:
- **Information Retrieval (IR)** - Truy xuất bằng chứng với BM25
- **Opinion Mining** - Phân tích cảm xúc & emotion với CardiffNLP
- **Recommendation Systems** - Gợi ý hashtag dựa trên tính cách
- **RAG (LLM)** - Giải thích dự đoán với retrieval-augmented generation
- **Academic Benchmarking** - Đánh giá trên PAN15 Author Profiling dataset

---

## 📋 Mục lục

1. [Giới thiệu Big Five](#-big-five-personality-traits-ocean)
2. [Cấu trúc Project](#-cấu-trúc-project-chi-tiết)
3. [Cài đặt môi trường](#-cài-đặt-môi-trường)
4. [Chuẩn bị dữ liệu](#-chuẩn-bị-dữ-liệu)
5. [Chạy Demo nhanh](#-chạy-demo-nhanh)
6. [Pipeline đầy đủ](#-pipeline-đầy-đủ)
7. [Hướng dẫn Debug](#-hướng-dẫn-debug)
8. [Chi tiết các Module](#-chi-tiết-các-module)
9. [Chi tiết các Script](#-chi-tiết-các-script)
10. [Streamlit Application](#️-streamlit-application)
11. [Đánh giá & Metrics](#-đánh-giá--metrics)
12. [Docker Deployment](#-docker-deployment)
13. [Limitations & Ethics](#️-limitations--ethics)
14. [References](#-references)

---

## 🎯 Big Five Personality Traits (OCEAN)

| Trait | Tiếng Việt | Description |
|-------|------------|-------------|
| **O**penness | Cởi mở | Sáng tạo, tò mò, thích trải nghiệm mới |
| **C**onscientiousness | Tận tâm | Có tổ chức, trách nhiệm, hướng mục tiêu |
| **E**xtraversion | Hướng ngoại | Hòa đồng, năng động, cảm xúc tích cực |
| **A**greeableness | Dễ chịu | Hợp tác, đồng cảm, tin tưởng |
| **S**tability | Ổn định | Ổn định cảm xúc (ngược với Neuroticism) |

> **Lưu ý**: Chúng tôi dùng "Stable" (ổn định cảm xúc) thay vì "Neurotic" theo quy ước PAN15. Nếu cần Neuroticism: `N = 1 - Stable`.

---

## 📁 Cấu trúc Project chi tiết

```
web-mining-bigfive/
├── README.md                 # Tài liệu này
├── requirements.txt          # Dependencies Python
├── .env.example              # Template biến môi trường
├── Dockerfile                # Container config
│
├── app/                      # 🖥️ Streamlit Web Application
│   ├── __init__.py
│   └── streamlit_app.py      # Main app (781 lines) - UI demo đầy đủ
│
├── src/                      # 📦 Source Code Modules
│   ├── __init__.py
│   ├── config.py             # Cấu hình toàn cục (paths, models, constants)
│   │
│   ├── data/                 # 📊 Data Processing
│   │   ├── pan15_parser.py   # Parse XML data từ PAN15
│   │   └── build_splits.py   # Tạo train/dev/test splits
│   │
│   ├── models/               # 🤖 ML Models
│   │   ├── tfidf_ridge.py    # TF-IDF + Ridge Baseline (~8.6KB)
│   │   └── transformer_regressor.py  # Transformer model (~26.7KB)
│   │
│   ├── opinion/              # 💭 Opinion Mining
│   │   └── features.py       # Sentiment & Emotion extraction
│   │
│   ├── ir/                   # 🔍 Information Retrieval
│   │   ├── bm25.py           # BM25 indexing & search
│   │   ├── chroma_store.py   # ChromaDB vector store
│   │   ├── evidence.py       # Evidence retrieval per trait
│   │   └── ir_eval.py        # IR evaluation (P@k, nDCG@k)
│   │
│   ├── recsys/               # 🏷️ Recommendation System
│   │   ├── hashtag_recsys.py # Main RecSys logic (~27KB)
│   │   ├── metrics.py        # Precision@k, Recall@k, MAP@k
│   │   ├── gnn_recsys.py     # LightGCN, Personality-enhanced GCN
│   │   ├── sasrec.py         # Sequential recommendation (SASRec)
│   │   └── advanced_models.py# KGE, Hyperbolic GCN (~18KB)
│   │
│   ├── rag/                  # 🧠 RAG Explainer
│   │   ├── explain.py        # LLM/Rule-based explanations
│   │   └── prompts.py        # Prompt templates
│   │
│   └── utils/                # 🔧 Utilities
│       ├── io.py             # File I/O, logging setup
│       ├── seed.py           # Random seed management
│       └── text.py           # Text preprocessing
│
├── scripts/                  # 🚀 Executable Scripts
│   ├── download_pan15.sh     # Download dataset từ Zenodo
│   ├── preprocess_pan15.py   # Parse & create splits
│   ├── train_eval_baseline_tfidf.py  # Train TF-IDF baseline
│   ├── train_eval_transformer.py     # Train Transformer model
│   ├── opinion_features.py   # Extract sentiment/emotion
│   ├── build_ir_index.py     # Build BM25 index
│   ├── retrieve_evidence.py  # Retrieve evidence tweets
│   ├── build_chroma_db.py    # Build ChromaDB
│   ├── recsys_eval.py        # Evaluate RecSys (963 lines!)
│   ├── build_recsys_dataset.py # Build RecSys evaluation dataset
│   ├── ir_label_tool.py      # Manual IR labeling tool
│   ├── run_full_experiment.py# Run full pipeline
│   ├── consolidate_new.py    # Consolidate results
│   ├── plot_from_logs.py     # Plot training curves
│   └── regenerate_plots.py   # Regenerate visualizations
│
├── data/                     # 📁 Data Directory
│   ├── raw/                  # Downloaded raw PAN15 data
│   │   ├── pan15_train/
│   │   ├── pan15_test/
│   │   └── pan15_train_en/
│   ├── processed/            # Processed parquet/pkl files
│   │   ├── pan15_en.parquet
│   │   ├── pan15_es.parquet
│   │   ├── chroma_db/
│   │   ├── ir_bm25.pkl
│   │   └── evidence_topk.parquet
│   └── splits/               # User ID splits
│       ├── en/
│       ├── es/
│       ├── it/
│       └── nl/
│
├── models/                   # 💾 Saved Model Checkpoints
│   ├── baseline_en.joblib
│   └── transformer_en.pt
│
├── results/                  # 📈 Evaluation Results
│   ├── metrics_baseline_en.csv
│   └── metrics_transformer_en.csv
│
├── tests/                    # 🧪 Test Files
│   └── __init__.py
│
├── report.md                 # 📄 Báo cáo RecSys (tiếng Việt)
└── report_recsys_summary.md  # 📄 Tóm tắt RecSys
```

---

## 🛠️ Cài đặt môi trường

### Yêu cầu hệ thống

- **Python**: 3.10+ (khuyến nghị 3.11)
- **RAM**: Tối thiểu 8GB, khuyến nghị 16GB
- **GPU**: Optional, cần cho Transformer training (CUDA 11.8+)
- **Disk**: ~5GB cho data + models

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd web-mining-bigfive
```

### Bước 2: Tạo Virtual Environment

```bash
# Sử dụng Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Verify Python version
python --version  # Phải là 3.10+
```

### Bước 3: Cài đặt Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Danh sách dependencies chính:**

| Package | Version | Mục đích |
|---------|---------|----------|
| `pandas` | ≥2.0.0 | Data manipulation |
| `numpy` | ≥1.24.0 | Numerical computing |
| `scikit-learn` | ≥1.3.0 | TF-IDF, Ridge, metrics |
| `torch` | ≥2.0.0 | Deep learning |
| `transformers` | ≥4.35.0 | HuggingFace models |
| `sentence-transformers` | ≥2.2.0 | Sentence embeddings |
| `rank-bm25` | ≥0.2.2 | BM25 indexing |
| `chromadb` | ≥0.4.0 | Vector database |
| `streamlit` | ≥1.28.0 | Web UI |
| `matplotlib` | ≥3.7.0 | Plotting |
| `plotly` | ≥5.18.0 | Interactive charts |
| `openai` | ≥1.0.0 | RAG explanations |

### Bước 4: Cấu hình Environment Variables

```bash
cp .env.example .env
```

Chỉnh sửa `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here  # Optional - cho RAG
OPENAI_MODEL=gpt-3.5-turbo               # Optional
```

> **Lưu ý**: OpenAI API key chỉ cần cho tính năng RAG explanations. Demo vẫn chạy được mà không cần.

### Troubleshooting cài đặt

<details>
<summary>❌ <b>Lỗi: torch không cài được trên GPU</b></summary>

```bash
# Cài PyTorch với CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"
```
</details>

<details>
<summary>❌ <b>Lỗi: chromadb build fails</b></summary>

```bash
# Cài build tools
sudo apt-get install build-essential python3-dev

# Hoặc dùng pre-built wheel
pip install chromadb --prefer-binary
```
</details>

<details>
<summary>❌ <b>Lỗi: ModuleNotFoundError</b></summary>

```bash
# Đảm bảo PYTHONPATH đúng
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Hoặc chạy từ root directory
cd /path/to/web-mining-bigfive
python scripts/your_script.py
```
</details>

---

## 📥 Chuẩn bị dữ liệu

### Download PAN15 Dataset

```bash
# Download từ Zenodo (khoảng 500MB)
bash scripts/download_pan15.sh
```

Script sẽ:
1. Download file zip từ Zenodo
2. Extract vào `data/raw/`
3. Tạo các thư mục cần thiết

### Preprocess Data

```bash
# Parse XML và tạo train/dev/test splits
python scripts/preprocess_pan15.py
```

**Output:**
- `data/processed/pan15_{lang}.parquet` - Dữ liệu đã xử lý
- `data/splits/{lang}/train.txt` - User IDs cho training
- `data/splits/{lang}/dev.txt` - User IDs cho validation
- `data/splits/{lang}/test.txt` - User IDs cho testing

**Ngôn ngữ hỗ trợ:** `en` (English), `es` (Spanish), `it` (Italian), `nl` (Dutch)

---

## 🚀 Chạy Demo nhanh

### Option 1: Demo với dữ liệu mẫu (Nhanh nhất)

```bash
# 1. Preprocess (nếu chưa có data)
python scripts/preprocess_pan15.py

# 2. Train baseline model với sample nhỏ
python scripts/train_eval_baseline_tfidf.py --sample_size 200

# 3. Build BM25 index cho evidence
python scripts/build_ir_index.py --sample_size 200

# 4. Chạy Streamlit app
streamlit run app/streamlit_app.py
```

Mở browser tại: **http://localhost:8501**

### Option 2: Demo với model đầy đủ

```bash
# Train full baseline
python scripts/train_eval_baseline_tfidf.py --lang en

# Chạy app
streamlit run app/streamlit_app.py
```

---

## 🔄 Pipeline đầy đủ

Thực hiện tuần tự các bước sau:

### 1. Baseline Training

```bash
# TF-IDF + Ridge Regression
python scripts/train_eval_baseline_tfidf.py --lang en --alpha 1.0
```

### 2. Opinion Mining Features

```bash
# Extract sentiment/emotion features
python scripts/opinion_features.py

# Train với opinion features
python scripts/train_eval_baseline_tfidf.py --with_opinion --lang en
```

### 3. Information Retrieval

```bash
# Build BM25 index
python scripts/build_ir_index.py

# Retrieve evidence tweets per trait
python scripts/retrieve_evidence.py
```

### 4. Vector Database (cho RAG)

```bash
# Build ChromaDB với sentence embeddings
python scripts/build_chroma_db.py
```

### 5. Recommendation System Evaluation

```bash
# Build RecSys dataset
python scripts/build_recsys_dataset.py

# Evaluate RecSys (comprehensive)
python scripts/recsys_eval.py --k 10
```

### 6. Transformer Training (GPU required)

```bash
# Train với Twitter-RoBERTa (English)
python scripts/train_eval_transformer.py \
    --lang en \
    --epochs 50 \
    --batch_size 8 \
    --lr 2e-5 \
    --early_stopping 10

# Train với XLM-RoBERTa (Multilingual)
python scripts/train_eval_transformer.py \
    --lang es \
    --model_name xlm-roberta-base \
    --epochs 30
```

### 7. Launch Application

```bash
streamlit run app/streamlit_app.py
```

---

## 🐛 Hướng dẫn Debug

### Debug với logging

Mọi script đều có logging tự động:

```bash
# Xem logs chi tiết
python scripts/train_eval_baseline_tfidf.py --lang en 2>&1 | tee training.log
```

### Debug step-by-step trong Python

```python
import sys
sys.path.insert(0, '/path/to/web-mining-bigfive')

# Load config
from src.config import *
print(f"Data dir: {DATA_DIR}")
print(f"Models dir: {MODELS_DIR}")

# Load data
from src.utils.io import load_parquet, load_splits
df = load_parquet(PROCESSED_DIR / "pan15_en.parquet")
print(f"Loaded {len(df)} users")

# Load model
from src.models.tfidf_ridge import TfidfRidgeModel
model = TfidfRidgeModel.load(MODELS_DIR / "baseline_en.joblib")

# Test prediction
sample_text = "I love trying new things and meeting new people!"
pred = model.predict([sample_text])
print(f"Predictions: {pred}")
```

### Debug Streamlit App

```bash
# Chạy với hot-reload và logs
streamlit run app/streamlit_app.py --logger.level=debug

# Hoặc với Python debugger
python -m pdb -c continue app/streamlit_app.py
```

### Debug RecSys

```bash
# Chạy với verbose output
python scripts/recsys_eval.py --k 10 2>&1 | tee recsys_debug.log

# Kiểm tra từng method
python -c "
from src.recsys.hashtag_recsys import HashtagRecommender
from src.config import PROCESSED_DIR

# Load recommender
rec = HashtagRecommender()
rec.fit(PROCESSED_DIR / 'pan15_en.parquet')

# Test
recs = rec.recommend(user_tags=['happy', 'travel'], k=5)
print(recs)
"
```

### Kiểm tra GPU/CUDA

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

### Common Issues & Solutions

<details>
<summary>❌ <b>"Model file not found"</b></summary>

Chạy training trước:
```bash
python scripts/train_eval_baseline_tfidf.py --lang en
```
</details>

<details>
<summary>❌ <b>"Data file not found"</b></summary>

Chạy preprocessing:
```bash
bash scripts/download_pan15.sh
python scripts/preprocess_pan15.py
```
</details>

<details>
<summary>❌ <b>"CUDA out of memory"</b></summary>

Giảm batch size:
```bash
python scripts/train_eval_transformer.py --batch_size 4
```
</details>

<details>
<summary>❌ <b>"Streamlit connection refused"</b></summary>

```bash
# Kiểm tra port
lsof -i :8501

# Đổi port nếu cần
streamlit run app/streamlit_app.py --server.port 8502
```
</details>

---

## 📦 Chi tiết các Module

### `src/config.py` - Configuration

```python
# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Constants
TRAIT_NAMES = ["open", "conscientious", "extroverted", "agreeable", "stable"]
SEED = 42

# Models
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "cardiffnlp/twitter-roberta-base-emotion"
ENCODER_MODEL = "cardiffnlp/twitter-roberta-base"
MULTILINGUAL_ENCODER_MODEL = "xlm-roberta-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

---

### `src/models/` - ML Models

#### `tfidf_ridge.py` - Baseline Model

```python
from src.models.tfidf_ridge import TfidfRidgeModel

# Train
model = TfidfRidgeModel(alpha=1.0)
model.fit(train_texts, train_labels)

# Predict
predictions = model.predict(test_texts)  # shape: (n_samples, 5)

# Evaluate
metrics = model.evaluate(test_texts, test_labels)
# Returns: {'rmse_avg': 0.15, 'rmse_open': 0.14, ...}

# Save/Load
model.save("model.joblib")
model = TfidfRidgeModel.load("model.joblib")
```

#### `transformer_regressor.py` - Transformer Model

```python
from src.models.transformer_regressor import TransformerTrainer

trainer = TransformerTrainer(
    model_name="cardiffnlp/twitter-roberta-base",
    learning_rate=2e-5,
    batch_size=8,
    max_length=512
)

trainer.fit(
    train_texts, train_targets,
    val_texts, val_targets,
    epochs=50,
    early_stopping_patience=10
)

predictions = trainer.predict(test_texts)
```

---

### `src/ir/` - Information Retrieval

#### `bm25.py` - BM25 Index

```python
from src.ir.bm25 import BM25Index

# Build index
index = BM25Index()
index.fit(documents)  # List of strings

# Search
results = index.search("creative ideas", top_k=5)
# Returns: [(doc_idx, score), ...]

# Save/Load
index.save("bm25.pkl")
index = BM25Index.load("bm25.pkl")
```

#### `chroma_store.py` - Vector Database

```python
from src.ir.chroma_store import ChromaStore

store = ChromaStore(persist_dir="data/processed/chroma_db")
store.add_users(user_ids, embeddings, metadata)

# Find similar users
similar = store.query(query_embedding, k=5)
```

#### `evidence.py` - Evidence Retrieval

```python
from src.ir.evidence import EvidenceRetriever

retriever = EvidenceRetriever(bm25_index)
evidence = retriever.get_evidence_for_traits(k=5)
# Returns: {trait: [(tweet, score), ...]}
```

---

### `src/recsys/` - Recommendation System

#### `hashtag_recsys.py` - Main RecSys

```python
from src.recsys.hashtag_recsys import HashtagRecommender

rec = HashtagRecommender()
rec.fit(data_path)

# Methods available:
# - popularity(): Global popularity baseline
# - content_based(user_profile): Embedding similarity
# - personality_aware(user_traits): Trait matching
# - hybrid(user_profile, user_traits): Combined approach

recs = rec.recommend(
    user_tags=['happy', 'travel'],
    user_traits=[0.7, 0.6, 0.8, 0.5, 0.6],
    method='hybrid',
    k=10
)
```

#### `gnn_recsys.py` - Graph Neural Networks

```python
from src.recsys.gnn_recsys import PersonalityLightGCN

model = PersonalityLightGCN(
    n_users=1000,
    n_items=500,
    embedding_dim=64,
    n_layers=3
)
```

#### `metrics.py` - Evaluation Metrics

```python
from src.recsys.metrics import precision_at_k, recall_at_k, map_at_k

p = precision_at_k(recommendations, ground_truth, k=10)
r = recall_at_k(recommendations, ground_truth, k=10)
m = map_at_k(recommendations, ground_truth, k=10)
```

---

### `src/opinion/` - Opinion Mining

```python
from src.opinion.features import OpinionFeatureExtractor

extractor = OpinionFeatureExtractor()
features = extractor.extract(texts)
# Returns: np.array with sentiment/emotion features
# - Sentiment probabilities (neg, neu, pos)
# - Emotion probabilities (anger, joy, ...)
# - Entropy, positive/negative rates
```

---

### `src/rag/` - RAG Explainer

```python
from src.rag.explain import PersonalityExplainer

explainer = PersonalityExplainer(api_key="...")

# With OpenAI
explanation = explainer.explain(
    traits={'open': 0.8, 'extroverted': 0.7, ...},
    evidence={'open': ["I love art!", ...], ...}
)

# Fallback (rule-based, no API key)
explanation = explainer.explain_rule_based(traits)
```

---

## 📜 Chi tiết các Script

### `scripts/train_eval_baseline_tfidf.py`

Train và evaluate TF-IDF + Ridge baseline.

```bash
python scripts/train_eval_baseline_tfidf.py \
    --lang en                    # Language: en, es, it, nl
    --with_opinion               # Include sentiment features
    --sample_size 200            # Limit samples (for testing)
    --alpha 1.0                  # Ridge alpha parameter
    --seed 42                    # Random seed
    --results_dir ./results      # Output directory
```

**Output:**
- `models/baseline_{lang}.joblib` - Trained model
- `results/metrics_baseline_{lang}.csv` - Evaluation metrics

---

### `scripts/train_eval_transformer.py`

Train Transformer regressor.

```bash
python scripts/train_eval_transformer.py \
    --lang en                    # Language code
    --model_name cardiffnlp/twitter-roberta-base  # HF model
    --epochs 50                  # Max epochs
    --batch_size 8               # Batch size (reduce if OOM)
    --lr 2e-5                    # Learning rate
    --max_length 512             # Max sequence length
    --early_stopping 10          # Patience for early stopping
    --warmup_epochs 2            # Warmup epochs
    --no_cosine                  # Disable cosine scheduler
    --sample_size 100            # Limit samples
    --seed 42                    # Random seed
    --results_dir ./results      # Output directory
```

**Output:**
- `models/transformer_{lang}.pt` - Trained model
- `results/metrics_transformer_{lang}.csv` - Metrics

---

### `scripts/recsys_eval.py`

Comprehensive RecSys evaluation (963 lines).

```bash
python scripts/recsys_eval.py \
    --k 10                       # Top-k recommendations
    --lang en                    # Language
```

**Evaluates:**
- Popularity baseline
- Content-based filtering
- Personality-aware filtering
- Hybrid methods
- LightGCN
- SASRec (Sequential)
- KGE, Hyperbolic GCN

---

### `scripts/build_recsys_dataset.py`

Build dataset for RecSys evaluation.

```bash
python scripts/build_recsys_dataset.py
```

**Creates:**
- `data/processed/recsys_train.parquet`
- `data/processed/recsys_test.parquet`

---

### `scripts/opinion_features.py`

Extract sentiment/emotion features.

```bash
python scripts/opinion_features.py
```

**Creates:**
- `data/processed/opinion_features.parquet`

---

### `scripts/build_ir_index.py`

Build BM25 index for evidence retrieval.

```bash
python scripts/build_ir_index.py \
    --sample_size 1000           # Optional: limit documents
```

**Creates:**
- `data/processed/ir_bm25.pkl`

---

### `scripts/retrieve_evidence.py`

Retrieve evidence tweets for each trait.

```bash
python scripts/retrieve_evidence.py
```

**Creates:**
- `data/processed/evidence_topk.parquet`

---

### `scripts/build_chroma_db.py`

Build ChromaDB vector store.

```bash
python scripts/build_chroma_db.py
```

**Creates:**
- `data/processed/chroma_db/`

---

## 🖥️ Streamlit Application

### Features

1. **📝 Text Input**: Paste text hoặc upload file
2. **🔮 Personality Prediction**: Dự đoán 5 traits với confidence
3. **📊 Visualization**: Radar chart tương tác (Plotly)
4. **📑 Evidence**: Tweets liên quan đến từng trait
5. **💡 AI Explanation**: Giải thích dựa trên RAG/Rules
6. **🏷️ Hashtag Recommendations**: Gợi ý hashtag cá nhân hóa

### Configuration trong App

```python
# app/streamlit_app.py

# Màu sắc cho từng trait
TRAIT_COLORS = {
    "open": "#f59e0b",        # Amber
    "conscientious": "#10b981",# Emerald
    "extroverted": "#f43f5e", # Rose
    "agreeable": "#3b82f6",   # Blue
    "stable": "#8b5cf6",      # Purple
}

# Icons
TRAIT_ICONS = {
    "open": "🎨",
    "conscientious": "📋",
    "extroverted": "🎉",
    "agreeable": "🤝",
    "stable": "🧘",
}
```

### Customization

Chỉnh sửa `app/streamlit_app.py`:

```python
# Thay đổi page config
st.set_page_config(
    page_title="Your Title",
    page_icon="🧠",
    layout="wide",
)

# Thay đổi sidebar
with st.sidebar:
    st.title("Your Sidebar")
```

---

## 📊 Đánh giá & Metrics

### Personality Prediction

| Metric | Description | Formula |
|--------|-------------|---------|
| **RMSE** | Root Mean Squared Error | $\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}$ |
| **MAE** | Mean Absolute Error | $\frac{1}{n}\sum|y - \hat{y}|$ |
| **Avg RMSE** | Average across 5 traits | $\frac{1}{5}\sum RMSE_i$ |

### Information Retrieval

| Metric | Description |
|--------|-------------|
| **P@k** | Precision at k |
| **nDCG@k** | Normalized DCG |

### Recommendation System

| Metric | Description |
|--------|-------------|
| **Precision@k** | Relevant items in top-k / k |
| **Recall@k** | Relevant items in top-k / total relevant |
| **MAP@k** | Mean Average Precision at k |

### Sample Results

| Model | RMSE (avg) | Notes |
|-------|------------|-------|
| TF-IDF + Ridge | ~0.15 | Fast, baseline |
| TF-IDF + Opinion | ~0.14 | +Sentiment features |
| Twitter-RoBERTa | ~0.12 | Transformer, GPU |

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t bigfive-analyzer .
```

### Run Container

```bash
docker run -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -e OPENAI_API_KEY=your_key \
    bigfive-analyzer
```

### Docker Compose (Optional)

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

---

## ⚠️ Limitations & Ethics

### Limitations

1. **Dataset Bias**: PAN15 data từ 2015, có thể không phản ánh ngôn ngữ hiện đại
2. **English Focus**: Model chính train trên English
3. **Aggregation**: Dự đoán user-level từ limited posts
4. **Temporal**: Personality có thể thay đổi theo thời gian

### Ethical Considerations

1. **Privacy**: KHÔNG sử dụng cho profiling trái phép
2. **Consent**: Chỉ analyze content có consent
3. **Not Diagnostic**: Đây là ML predictions, KHÔNG phải clinical assessment
4. **Bias Awareness**: Models có thể perpetuate training data biases
5. **Transparency**: Luôn disclose khi apply personality analysis

### Data Usage

- Training chỉ dùng public benchmark datasets (PAN15)
- Không real-time X/Twitter API crawling
- Demo input là user-provided only

---

## 📚 References

1. **PAN15 Author Profiling Task**
   - Rangel, F., et al. "Overview of the 3rd Author Profiling Task at PAN 2015"
   - [PAN @ CLEF 2015](https://pan.webis.de/clef15/pan15-web/author-profiling.html)

2. **CardiffNLP Twitter Models**
   - https://github.com/cardiffnlp/twitter-models
   - Twitter-RoBERTa for sentiment, emotion, etc.

3. **Big Five Personality Model**
   - Costa, P. T., & McCrae, R. R. (1992). "Revised NEO Personality Inventory"

4. **Personality-Aware Recommendation**
   - Tkalcic, M., & Chen, L. (2015). "Personality and recommender systems"

5. **Hybrid Recommender Systems**
   - Burke, R. (2002). "Hybrid recommender systems: Survey and experiments"

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 👥 Contributors

**Course**: Web Mining  
**Capstone Project**: Detection of Big Five Personality Traits from Social Media

---

## 🆘 Support

Nếu gặp vấn đề:

1. Check [Troubleshooting](#troubleshooting-cài-đặt) section
2. Review logs trong `*.log` files
3. Mở issue trên GitHub với:
   - Error message đầy đủ
   - Python version (`python --version`)
   - OS information
   - Steps to reproduce

---

> **Last Updated**: January 2026
