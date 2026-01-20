# System Architecture Mapping

This document maps each layer of the system architecture to its corresponding implementation files in the project.

---

## Architecture Overview

![System Architecture](/home/team_cv/.gemini/antigravity/brain/21333524-9fea-4eb6-aced-db38c1fc1207/system_architecture_diagram_1768874828600.png)

---

## Layer 1: Data Ingestion & Splitting

| Component | File Path | Description |
|---|---|---|
| **Dataset Parsing** | [pan15_parser.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/data/pan15_parser.py) | Parses PAN 2015 XML files, extracts tweets and personality labels |
| **Data Splitting** | [build_splits.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/data/build_splits.py) | Implements User-based Train/Val/Test splitting (70/10/20) |
| **Preprocessing Script** | [preprocess_pan15.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/scripts/preprocess_pan15.py) | Text cleaning pipeline: URL normalization, mention anonymization |
| **Download Script** | [download_pan15.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/scripts/download_pan15.py) | Downloads PAN 2015 dataset from Zenodo |

---

## Layer 2: Profiling Layer

### User Profiling (Big Five Personality Prediction)

| Component | File Path | Description |
|---|---|---|
| **Transformer Regressor** | [transformer_regressor.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/models/transformer_regressor.py) | `PersonalityRegressor` class with Hierarchical Chunking & LLRD |
| **TF-IDF Baseline** | [tfidf_ridge.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/models/tfidf_ridge.py) | TF-IDF + Ridge baseline for personality prediction |
| **Training Script** | [train_eval_transformer.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/scripts/train_eval_transformer.py) | Main training script for personality models |
| **Opinion Features** | [features.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/opinion/features.py) | Auxiliary features extraction for opinion mining |

### Item Profiling (Hashtag Personality)

| Component | File Path | Description |
|---|---|---|
| **Hashtag Recommender** | [hashtag_recsys.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/recsys/hashtag_recsys.py) | `HashtagRecommender.fit()` computes `hashtag_personalities` by aggregating user traits |

---

## Layer 3: Model Layer

### Content-based Filter

| Component | File Path | Description |
|---|---|---|
| **Content Similarity** | [hashtag_recsys.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/recsys/hashtag_recsys.py) | `HashtagRecommender.recommend_content()` uses SentenceTransformers |

### Co-occurrence Rules (Association Rules)

| Component | File Path | Description |
|---|---|---|
| **1st & 2nd Order Co-occurrence** | [hashtag_recsys.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/recsys/hashtag_recsys.py) | `HashtagRecommender.recommend_enhanced_cooc()` with transitive rules |
| **Popularity Baseline** | [hashtag_recsys.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/recsys/hashtag_recsys.py) | `HashtagRecommender.recommend_popularity()` |

### Personality-Enhanced LightGCN

| Component | File Path | Description |
|---|---|---|
| **GNN Model** | [gnn_recsys.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/recsys/gnn_recsys.py) | `PersonalityLightGCN` class with personality embedding injection |
| **GNN Trainer** | [gnn_recsys.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/recsys/gnn_recsys.py) | `GNNTrainer` class with BPR loss and adjacency matrix construction |
| **SimGCL (Advanced)** | [gnn_recsys.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/recsys/gnn_recsys.py) | `PersonalitySimGCL` contrastive learning variant |

### Hybrid Re-ranking

| Component | File Path | Description |
|---|---|---|
| **Hybrid Scoring** | [hashtag_recsys.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/recsys/hashtag_recsys.py) | `HashtagRecommender.recommend_personality_aware()` combines all signals |
| **RRF Ensemble** | [hashtag_recsys.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/recsys/hashtag_recsys.py) | `HashtagRecommender.reciprocal_rank_fusion()` and `recommend_rrf_ensemble()` |
| **MMR Diversity** | [hashtag_recsys.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/recsys/hashtag_recsys.py) | `HashtagRecommender._mmr_rerank()` for diverse recommendations |

### Evaluation Metrics

| Component | File Path | Description |
|---|---|---|
| **Metrics** | [metrics.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/recsys/metrics.py) | MAP@K, Precision@K, Recall@K implementations |
| **Evaluation Script** | [recsys_eval.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/scripts/recsys_eval.py) | Full evaluation pipeline |

---

## Layer 4: Serving Layer

| Component | File Path | Description |
|---|---|---|
| **Streamlit Web App** | [streamlit_app.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/app/streamlit_app.py) | User interface for personality prediction & hashtag recommendation |

---

## Supporting Modules

### Evidence Retrieval (IR)

| Component | File Path | Description |
|---|---|---|
| **BM25 Retriever** | [bm25.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/ir/bm25.py) | BM25Okapi implementation for evidence extraction |
| **Evidence Module** | [evidence.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/ir/evidence.py) | Retrieves tweets that justify personality predictions |
| **ChromaDB Store** | [chroma_store.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/ir/chroma_store.py) | Vector store for semantic search |
| **IR Evaluation** | [ir_eval.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/ir/ir_eval.py) | P@5, nDCG@5 evaluation metrics |

### Configuration

| Component | File Path | Description |
|---|---|---|
| **Config** | [config.py](file:///home/team_cv/nhdang/Workspace/Research/web-mining-bigfive/src/config.py) | Global constants: model names, paths, trait columns |

---

## Key Entry Points

| Task | Command |
|---|---|
| **Full Experiment** | `python scripts/run_full_experiment.py` |
| **Train Personality Model** | `python scripts/train_eval_transformer.py` |
| **Evaluate RecSys** | `python scripts/recsys_eval.py` |
| **Run Web App** | `streamlit run app/streamlit_app.py` |
