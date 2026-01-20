#!/usr/bin/env python3
"""
Train and Save Personality-Enhanced LightGCN Model for Streamlit App.
"""
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, MODELS_DIR, SEED, TRAIT_NAMES
from src.utils.io import setup_logging, load_parquet
from src.utils.seed import set_seed
from src.recsys.hashtag_recsys import HashtagRecommender
from src.recsys.gnn_recsys import PersonalityLightGCN, GNNTrainer

def main():
    logger = setup_logging("train_save_gnn")
    set_seed(SEED)
    
    # 1. Load Data
    logger.info("Loading RecSys dataset and profiles...")
    dataset_path = PROCESSED_DIR / "recsys_dataset.pkl"
    profiles_path = PROCESSED_DIR / "hashtag_profiles.pkl"
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return

    df = load_parquet(dataset_path)
    profiles_df = load_parquet(profiles_path)
    
    # Load vocabulary from base recommender
    recommender = HashtagRecommender()
    recommender.load_profiles(profiles_df)
    filtered_hashtags = recommender.filtered_hashtags
    
    logger.info(f"Training on {len(df)} users and {len(filtered_hashtags)} items")

    # 2. Prepare Mappings
    user_to_idx = {uid: i for i, uid in enumerate(df["user_id"].unique())}
    hashtag_to_idx = {h: i for i, h in enumerate(filtered_hashtags)}
    
    # Save mappings for inference
    mapping_path = MODELS_DIR / "gnn_mappings.pkl"
    with open(mapping_path, "wb") as f:
        pickle.dump({
            "user_to_idx": user_to_idx,
            "hashtag_to_idx": hashtag_to_idx,
            "idx_to_hashtag": {i: h for h, i in hashtag_to_idx.items()}
        }, f)
    logger.info(f"Saved mappings to {mapping_path}")

    # 3. Prepare Interactions
    train_interactions = []
    all_users_meta = {} 
    
    for _, row in df.iterrows():
        uid = row["user_id"]
        u_idx = user_to_idx[uid]
        traits = [row.get(f"y_{t}", 0.5) for t in TRAIT_NAMES]
        all_users_meta[u_idx] = traits
        
        for h in row["train_hashtags"]:
            h_lower = h.lower()
            if h_lower in hashtag_to_idx:
                train_interactions.append({
                    "user_mapping": u_idx,
                    "item_mapping": hashtag_to_idx[h_lower]
                })
                
    train_data_df = pd.DataFrame(train_interactions)
    logger.info(f"GNN Training Data: {len(train_data_df)} interactions")
    
    # 4. Prepare Features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # User Features (Personality)
    user_feats = torch.zeros(len(user_to_idx), 5)
    for u_idx, traits in all_users_meta.items():
        user_feats[u_idx] = torch.tensor(traits)
    user_feats = user_feats.to(device)
    
    # Item Features (Content Embeddings)
    item_feats = None
    if profiles_df is not None and "embedding" in profiles_df.columns:
        try:
            # Need to align embeddings with hashtag_to_idx
            # profiles_df index is hashtag name? Check load_profiles logic
            # Usually profiles_df is indexed by hashtag or has column 'hashtag'
            # HashtagRecommender.load_profiles sets self.hashtag_embeddings
            # Let's use that if available, it's already aligned with filtered_hashtags?
            # Recommender.hashtag_embeddings is list aligned with self.filtered_hashtags
            if recommender.hashtag_embeddings is not None:
                item_feats = torch.tensor(recommender.hashtag_embeddings, dtype=torch.float32).to(device)
                logger.info(f"Loaded item content features: {item_feats.shape}")
        except Exception as e:
            logger.warning(f"Could not load item content features: {e}")

    # 5. Initialize & Train Model
    gnn_model = PersonalityLightGCN(
        num_users=len(user_to_idx),
        num_items=len(hashtag_to_idx),
        embedding_dim=64,
        n_layers=3,
        user_personality_features=user_feats,
        item_content_features=item_feats
    ).to(device)
    
    optimizer = optim.Adam(gnn_model.parameters(), lr=0.005, weight_decay=1e-4)
    trainer = GNNTrainer(gnn_model, optimizer, device)
    
    gnn_adj = trainer.create_adj_matrix(
        train_data_df["user_mapping"].values,
        train_data_df["item_mapping"].values,
        len(user_to_idx),
        len(hashtag_to_idx)
    )
    
    logger.info("Training GNN...")
    epochs = 50 
    for epoch in range(epochs):
        loss = trainer.train_epoch(gnn_adj, train_data_df)
        if (epoch+1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
            
    gnn_model.eval()
    with torch.no_grad():
        final_u_emb, final_i_emb = gnn_model(gnn_adj)
        
    # 7. Save Inference Artifacts (to avoid building graph in Streamlit)
    # We need:
    # - final_item_embeddings (N_items, dim) -> to dot product with user
    # - mean_user_base_embedding (dim) -> to represent "unknown user ID" part
    # - personality_projection (linear layer) -> from model state dict
    
    # Calculate mean base embedding (ID only, before personality)
    mean_user_base = torch.mean(gnn_model.user_embedding.weight, dim=0)
    
    inference_path = MODELS_DIR / "gnn_inference.pt"
    torch.save({
        "final_item_embeddings": final_i_emb.cpu(),
        "mean_user_base_embedding": mean_user_base.cpu(),
        "config": {
            "num_items": len(hashtag_to_idx),
            "embedding_dim": 64,
            "use_personality": True,
            "use_content": True
        }
    }, inference_path)
    logger.info(f"Saved inference artifacts to {inference_path}")

if __name__ == "__main__":
    main()
