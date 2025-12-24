# Detection of Big Five Personality Traits from X Posts using ML

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Web Mining capstone project for detecting Big Five personality traits (OCEAN) from social media posts, combining:
- **Information Retrieval (IR)** - BM25-based evidence retrieval
- **Opinion Mining** - Sentiment & emotion analysis using CardiffNLP models
- **Recommendation Systems** - Personality-aware hashtag recommendations
- **RAG (LLM)** - Explainable predictions using retrieval-augmented generation
- **Academic Benchmarking** - Evaluation on PAN15 Author Profiling dataset

## 🎯 Big Five Personality Traits (OCEAN)

| Trait | Description |
|-------|-------------|
| **O**penness | Creativity, curiosity, openness to new experiences |
| **C**onscientiousness | Organization, responsibility, goal-oriented behavior |
| **E**xtraversion | Sociability, energy, positive emotions |
| **A**greeableness | Cooperation, empathy, trust |
| **S**tability | Emotional stability (inverse of Neuroticism) |

> **Note**: We use "Stable" (emotional stability) instead of "Neurotic" following the PAN15 benchmark convention. If Neuroticism is needed: `N = 1 - Stable` or simply report Stability scores.

## 📁 Project Structure

```
webmining-bigfive/
├── README.md
├── requirements.txt
├── .env.example
├── Dockerfile
├── data/
│   ├── raw/              # Downloaded PAN15 data
│   ├── processed/        # Preprocessed parquet files
│   └── splits/           # Train/dev/test user IDs
├── src/
│   ├── config.py         # Configuration constants
│   ├── utils/            # Utility functions (seed, io, text)
│   ├── data/             # Data parsing and splitting
│   ├── models/           # ML models (TF-IDF, Transformer)
│   ├── opinion/          # Opinion mining features
│   ├── ir/               # Information retrieval (BM25, ChromaDB)
│   ├── recsys/           # Recommendation system
│   └── rag/              # RAG explainer
├── scripts/              # Executable pipeline scripts
├── app/                  # Streamlit application
├── results/              # Evaluation metrics
└── models/               # Saved model checkpoints
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env to add your OpenAI API key (optional, for RAG explanations)
```

### 3. Download & Preprocess Data

```bash
# Download PAN15 dataset from Zenodo
bash scripts/download_pan15.sh

# Preprocess and create train/dev/test splits
python scripts/preprocess_pan15.py
```

### 4. Train Baseline Model

```bash
# Train TF-IDF + Ridge baseline
python scripts/train_eval_baseline_tfidf.py
```

### 5. Run Full Pipeline (Optional)

```bash
# Extract opinion features (sentiment/emotion)
python scripts/opinion_features.py

# Train with opinion features (ablation)
python scripts/train_eval_baseline_tfidf.py --with_opinion

# Build IR index and retrieve evidence
python scripts/build_ir_index.py
python scripts/retrieve_evidence.py

# Evaluate recommendation system
python scripts/recsys_eval.py

# Build ChromaDB for user similarity (used by RAG)
python scripts/build_chroma_db.py

# Train transformer model (requires GPU, takes longer)
python scripts/train_eval_transformer.py --epochs 1 --sample_size 100
```

### 6. Launch Web Application

```bash
streamlit run app/streamlit_app.py
```

## 📊 Benchmark Dataset: PAN15 Author Profiling

**Source**: [PAN15 Author Profiling Task](https://pan.webis.de/clef15/pan15-web/author-profiling.html)

**Download**: [Zenodo](https://zenodo.org/records/3745945)

| Split | Users | Description |
|-------|-------|-------------|
| Train | ~70% | Model training |
| Dev   | ~10% | Hyperparameter tuning |
| Test  | ~20% | Final evaluation |

**Labels**: 5 continuous traits (0.0 - 1.0 scale):
- `y_open`, `y_conscientious`, `y_extroverted`, `y_agreeable`, `y_stable`

## 📈 Evaluation Metrics

### Personality Prediction
- **RMSE** (Root Mean Squared Error) per trait
- **MAE** (Mean Absolute Error) per trait
- **Average RMSE** across all traits

### Information Retrieval
- **P@k** (Precision at k)
- **nDCG@k** (Normalized Discounted Cumulative Gain)

### Recommendation System
- **Precision@k**
- **Recall@k**
- **MAP@k** (Mean Average Precision)

## 🔬 Model Architecture

### Baseline: TF-IDF + Ridge Regression
- Character n-grams (3-5) + Word n-grams (1-2)
- FeatureUnion for combined features
- Multi-output Ridge regression

### Transformer: Twitter-RoBERTa
- Encoder: `cardiffnlp/twitter-roberta-base`
- User-level pooling: Mean of tweet embeddings
- Linear regression head for 5 traits

### Opinion Features
- Sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Emotion: `cardiffnlp/twitter-roberta-base-emotion`
- Features: Mean probabilities, entropy, positive/negative rates

## 🔍 Components

### Information Retrieval
- **BM25 Index**: Tweet-level indexing with rank-bm25
- **Evidence Retrieval**: Top-k tweets per trait using keyword queries
- **Evaluation**: P@5, nDCG@5 with manual relevance labels

### Vector Database (ChromaDB)
- **User Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Similar User Retrieval**: For RAG context augmentation

### RAG Explainer
- **LLM Mode**: OpenAI GPT for detailed explanations
- **Rule-based Fallback**: Template-based explanations when no API key

### Recommendation System
- **Baseline**: Global popularity
- **Content-based**: Embedding similarity
- **Personality-aware**: Re-ranking with trait similarity

## 🖥️ Streamlit Application

Features:
- 📝 Text input (paste or upload)
- 📊 Radar chart visualization
- 📑 Evidence tweets per trait
- 💡 AI-generated explanations
- 🏷️ Personality-aware hashtag recommendations

## ⚠️ Limitations & Ethics

### Limitations
1. **Dataset Bias**: PAN15 data from 2015 may not reflect current language patterns
2. **English Only**: Model trained on English tweets only
3. **Aggregation**: User-level predictions from limited post samples
4. **Temporal**: Personality may evolve; snapshots may not be representative

### Ethical Considerations
1. **Privacy**: Do not use for unauthorized profiling
2. **Consent**: Only analyze content with proper consent
3. **Not Diagnostic**: Results are ML predictions, not clinical assessments
4. **Bias Awareness**: Models may perpetuate training data biases
5. **Transparency**: Always disclose when personality analysis is applied

### Data Usage
- Training uses only publicly released benchmark datasets (PAN15)
- No real-time X/Twitter API crawling
- Demo input is user-provided only

## 📝 End-to-End Commands

```bash
# Complete pipeline (copy/paste friendly)
bash scripts/download_pan15.sh
python scripts/preprocess_pan15.py
python scripts/train_eval_baseline_tfidf.py
python scripts/opinion_features.py
python scripts/train_eval_baseline_tfidf.py --with_opinion
python scripts/build_ir_index.py
python scripts/retrieve_evidence.py
python scripts/recsys_eval.py
python scripts/build_chroma_db.py
streamlit run app/streamlit_app.py
```

### Quick Demo Mode (faster, fewer samples)
```bash
python scripts/preprocess_pan15.py
python scripts/train_eval_baseline_tfidf.py --sample_size 200
python scripts/build_ir_index.py --sample_size 200
streamlit run app/streamlit_app.py
```

## 📚 References

1. PAN @ CLEF 2015 Author Profiling Task
2. Rangel, F., et al. "Overview of the 3rd Author Profiling Task at PAN 2015"
3. CardiffNLP Twitter Models: https://github.com/cardiffnlp/twitter-models
4. Big Five Personality Model: Costa & McCrae (1992)

## 📄 License

MIT License - See LICENSE file for details.

---

**Course**: Web Mining  
**Capstone Project**: Detection of Big Five Personality Traits from Social Media
