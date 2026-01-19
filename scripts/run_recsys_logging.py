#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, RESULTS_DIR, SEED, TRAIT_NAMES
from src.utils.io import setup_logging, load_parquet
from src.utils.seed import set_seed
from src.recsys.gnn_recsys import PersonalityLightGCN, GNNTrainer
from src.recsys.metrics import map_at_k

def evaluate_validation(model, adj_matrix, val_users_df, user_to_idx, hashtag_to_idx, k=10):
    """
    Compute MAP@K for a validation set of users.
    """
    model.eval()
    with torch.no_grad():
        final_u_emb, final_i_emb = model(adj_matrix)
        
    recommended_lists = []
    relevant_lists = []
    
    idx_to_hashtag = {i: h for h, i in hashtag_to_idx.items()}
    
    # Process only validation users
    for _, row in val_users_df.iterrows():
        uid = row["user_id"]
        if uid not in user_to_idx:
            continue
            
        u_idx = user_to_idx[uid]
        u_emb = final_u_emb[u_idx] # (dim,)
        
        # Calculate scores for all items
        scores = torch.matmul(u_emb, final_i_emb.T).cpu().numpy()
        
        # Exclude history (train_hashtags)
        user_history = row["train_hashtags"]
        excluded_indices = {hashtag_to_idx[h.lower()] for h in user_history if h.lower() in hashtag_to_idx}
        
        # Mask excluded
        # Note: In efficient impl, we might just filter after sorting, but masking is safer
        for ex_idx in excluded_indices:
            scores[ex_idx] = -np.inf
            
        # Top K
        top_k_indices = np.argsort(scores)[::-1][:k]
        recs = [idx_to_hashtag[idx] for idx in top_k_indices if idx in idx_to_hashtag]
        
        recommended_lists.append(recs)
        relevant_lists.append(set(row["test_hashtags"]))
        
    if not recommended_lists:
        return 0.0
        
    return map_at_k(recommended_lists, relevant_lists, k)

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--personality", type=str, choices=["oracle", "predicted"], default="oracle",
                        help="Personality source: 'oracle' (ground truth) or 'predicted' (model output)")
    args = parser.parse_args()
    
    run_name = f"recsys_learning_curve_{args.personality}"
    logger = setup_logging(run_name)
    set_seed(SEED)
    
    logger.info(f"Starting RecSys training with {args.personality} personality...")
    
    # 1. Load Data
    # For predicted, we need the file with predictions
    if args.personality == "predicted":
        dataset_path = PROCESSED_DIR / "recsys_dataset_with_preds.parquet"
        if not dataset_path.exists():
             # Fallback to pkl if needed
             dataset_path = PROCESSED_DIR / "recsys_dataset_with_preds.pkl"
    else:
        dataset_path = PROCESSED_DIR / "recsys_dataset.parquet"
        
    df = load_parquet(dataset_path)
    profiles_path = PROCESSED_DIR / "hashtag_profiles.parquet"
    profiles_df = load_parquet(profiles_path)
    
    # 2. Setup GNN Data
    # Create Mappings
    user_to_idx = {uid: i for i, uid in enumerate(df["user_id"].unique())}
    
    if profiles_df is not None and not profiles_df.empty:
        filtered_hashtags = profiles_df["hashtag"].tolist()
    else:
        all_hs = []
        for hs in df["train_hashtags"]: all_hs.extend(hs)
        filtered_hashtags = list(set(all_hs))
        
    hashtag_to_idx = {h.lower(): i for i, h in enumerate(filtered_hashtags)}
    
    # Build Train Interactions
    train_interactions = []
    all_users_meta = {}
    
    # Determine trait columns
    if args.personality == "oracle":
        trait_cols = [f"y_{t}" for t in TRAIT_NAMES]
    else:
        trait_cols = [f"pred_{t}" for t in TRAIT_NAMES]
        
    logger.info(f"Using trait columns: {trait_cols}")
    
    for _, row in df.iterrows():
        uid = row["user_id"]
        u_idx = user_to_idx[uid]
        
        # Load traits based on selection
        traits = [row.get(c, 0.5) for c in trait_cols]
        all_users_meta[u_idx] = traits
        
        for h in row["train_hashtags"]:
            h_lower = h.lower()
            if h_lower in hashtag_to_idx:
                train_interactions.append({
                    "user_mapping": u_idx,
                    "item_mapping": hashtag_to_idx[h_lower]
                })
                
    train_data_df = pd.DataFrame(train_interactions)
    logger.info(f"Interaction Data: {len(train_data_df)} edges")
    
    # Validation subset
    val_users_df = df.sample(frac=0.1, random_state=SEED)
    logger.info(f"Validation set size: {len(val_users_df)} users")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Features
    user_feats = torch.zeros(len(user_to_idx), 5)
    for u_idx, traits in all_users_meta.items():
        user_feats[u_idx] = torch.tensor(traits)
    user_feats = user_feats.to(device)
    
    item_feats = None
    if profiles_df is not None and "embedding" in profiles_df.columns:
        try:
            embs = np.stack(profiles_df["embedding"].values)
            if len(embs) == len(hashtag_to_idx):
                 item_feats = torch.tensor(embs, dtype=torch.float32).to(device)
        except:
            pass
            
    # Model
    model = PersonalityLightGCN(
        num_users=len(user_to_idx),
        num_items=len(hashtag_to_idx),
        embedding_dim=64,
        n_layers=3,
        user_personality_features=user_feats,
        item_content_features=item_feats
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    trainer = GNNTrainer(model, optimizer, device)
    
    adj_matrix = trainer.create_adj_matrix(
        train_data_df["user_mapping"].values,
        train_data_df["item_mapping"].values,
        len(user_to_idx),
        len(hashtag_to_idx)
    )
    
    # Training Loop
    epochs = 30
    history = []
    
    # logger.info("Starting training with logging...")
    for epoch in tqdm(range(epochs)):
        loss = trainer.train_epoch(adj_matrix, train_data_df)
        val_map = evaluate_validation(model, adj_matrix, val_users_df, user_to_idx, hashtag_to_idx, k=10)
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": loss,
            "val_map@10": val_map
        })
        
    # Save Results
    results_df = pd.DataFrame(history)
    csv_path = RESULTS_DIR / f"{run_name}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved logs to {csv_path}")
    
    # Plot (Comparison Logic could be separate, but here we just plot individual run)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BPR Loss', color=color)
    ax1.plot(results_df['epoch'], results_df['train_loss'], color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Validation MAP@10', color=color)  
    ax2.plot(results_df['epoch'], results_df['val_map@10'], color=color, marker='s', label='Val MAP@10')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'RecSys Training ({args.personality}): BPR Loss vs MAP@10')
    fig.tight_layout()
    
    png_path = RESULTS_DIR / f"{run_name}.png"
    plt.savefig(png_path, dpi=150)
    logger.info(f"Saved plot to {png_path}")

if __name__ == "__main__":
    main()
