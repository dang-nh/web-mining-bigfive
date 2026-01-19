#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, SEED, TRAIT_NAMES, TRAIT_COLS, SPLITS_DIR
from src.utils.io import setup_logging, load_parquet, load_splits
from src.models.transformer_regressor import TransformerTrainer

# Set plotting style
# plt.style.use('seaborn-v0_8-whitegrid') # Optional if seaborn style not available

def analyze_personality_errors(logger):
    """Analyze personality prediction errors."""
    logger.info("Analyzing Personality Errors...")
    
    # Check for trained models
    # We look for 'transformer_{lang}_twitter.pt' or similar
    languages = ["en", "es", "it", "nl"]
    
    error_stats = []
    
    for lang in languages:
        # Try various naming patterns
        candidates = [
            MODELS_DIR / f"transformer_{lang}_twitter.pt",
            MODELS_DIR / f"transformer_{lang}_twitter_xlm.pt",
            MODELS_DIR / f"transformer_{lang}_xlm.pt"
        ]
        
        model_path = None
        for cand in candidates:
            if cand.exists():
                model_path = cand
                break
            
        if model_path is None:
            logger.warning(f"No model found for {lang}. Skipping.")
            continue
            
        logger.info(f"Loading model for {lang} from {model_path}...")
        try:
            # Force load to CPU with weights_only=False (PyTorch 2.6+ compat)
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            trainer = TransformerTrainer(
                model_name=checkpoint["model_name"],
                learning_rate=checkpoint["learning_rate"],
                batch_size=checkpoint["batch_size"],
                max_length=checkpoint["max_length"],
                device="cpu"
            )
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            trainer.model.to("cpu")
            trainer.device = torch.device("cpu")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            continue
            
        # Load Data
        df_path = PROCESSED_DIR / f"pan15_{lang}.pkl"
        if not df_path.exists():
             logger.warning(f"Data file not found: {df_path}")
             continue
             
        df = load_parquet(df_path)
        
        # Load Splits for this language
        lang_splits_dir = SPLITS_DIR / lang
        if not lang_splits_dir.exists():
             logger.warning(f"Splits dir not found: {lang_splits_dir}")
             continue
             
        splits = load_splits(lang_splits_dir)
        if "test" not in splits:
             logger.warning(f"No test split for {lang}")
             continue
             
        test_uids = set(splits["test"])
        test_df = df[df["user_id"].isin(test_uids)]
        
        if test_df.empty:
            logger.warning(f"No test data for {lang}")
            continue
            
        texts = test_df["text_concat"].tolist()
        targets = test_df[TRAIT_COLS].values
        
        # Predict
        logger.info(f"Predicting for {len(texts)} users in {lang}...")
        preds = trainer.predict(texts)
        
        # Calculate Errors
        # diff = pred - true
        diffs = preds - targets
        abs_errors = np.abs(diffs)
        
        # 1. Per Trait MAE
        trait_maes = np.mean(abs_errors, axis=0)
        for i, trait in enumerate(TRAIT_NAMES):
            error_stats.append({
                "Language": lang,
                "Trait": trait,
                "MAE": trait_maes[i]
            })
            
        # 2. Identify Hardest Examples (Top 3 highest mean error)
        mean_user_error = np.mean(abs_errors, axis=1)
        top_error_indices = np.argsort(mean_user_error)[-3:][::-1]
        
        logger.info(f"--- Top Semantic Errors for {lang} ---")
        for idx in top_error_indices:
            user_text = texts[idx]
            # Truncate text for display
            disp_text = user_text[:200] + "..." if len(user_text) > 200 else user_text
            ground_truth = targets[idx]
            prediction = preds[idx]
            err = mean_user_error[idx]
            
            logger.info(f"User Idx {idx} (Error={err:.4f}):")
            logger.info(f"  Text: {disp_text}")
            logger.info(f"  True: {ground_truth}")
            logger.info(f"  Pred: {prediction}")
            
    # Save Error Stats
    if error_stats:
        err_df = pd.DataFrame(error_stats)
        csv_path = RESULTS_DIR / "personality_error_analysis.csv"
        err_df.to_csv(csv_path, index=False)
        
        # Plot Heatmap of MAE using matshow
        plt.figure(figsize=(8, 6))
        pivot_df = err_df.pivot(index="Language", columns="Trait", values="MAE")
        
        plt.imshow(pivot_df.values, cmap="Reds", aspect='auto')
        plt.colorbar(label="MAE")
        
        # Set ticks
        plt.xticks(range(len(pivot_df.columns)), pivot_df.columns)
        plt.yticks(range(len(pivot_df.index)), pivot_df.index)
        
        # Annotate
        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                plt.text(j, i, f"{pivot_df.values[i, j]:.3f}", 
                         ha="center", va="center", color="black")
                         
        plt.title("Personality Prediction MAE by Language and Trait")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "personality_error_heatmap.png")
        logger.info("Saved personality error analysis.")

def analyze_recsys_errors(logger):
    """Analyze RecSys errors: Cold Start & Popularity."""
    logger.info("Analyzing RecSys Errors...")
    
    dataset_path = PROCESSED_DIR / "recsys_dataset.pkl"
    if not dataset_path.exists():
        # Fallback check
        dataset_path = PROCESSED_DIR / "recsys_dataset.parquet"
        if not dataset_path.exists() and not PROCESSED_DIR.joinpath("recsys_dataset.pkl").exists():
            logger.warning("RecSys dataset not found")
            return
        
    df = load_parquet(dataset_path)
    
    # ... Copying basic setup from logging script ...
    from src.recsys.gnn_recsys import PersonalityLightGCN, GNNTrainer
    
    # Mappings from dataset
    # We need to reuse the same mappings to be meaningful, but for analysis we can just rebuild.
    all_hashtags = []
    for hs in df["train_hashtags"]: all_hashtags.extend(hs)
    hashtag_counts = pd.Series(all_hashtags).value_counts()
    
    filtered_hashtags = list(set(all_hashtags)) # Or use threshold
    user_to_idx = {uid: i for i, uid in enumerate(df["user_id"].unique())}
    hashtag_to_idx = {h.lower(): i for i, h in enumerate(filtered_hashtags)}
    
    # Train simplified model interaction data
    train_interactions = []
    for _, row in df.iterrows():
        uid = row["user_id"]
        u_idx = user_to_idx[uid]
        for h in row["train_hashtags"]:
            h_lower = h.lower()
            if h_lower in hashtag_to_idx:
                train_interactions.append({
                    "user_mapping": u_idx,
                    "item_mapping": hashtag_to_idx[h_lower]
                })
    train_data_df = pd.DataFrame(train_interactions)
    
    device = torch.device("cpu")
    model = PersonalityLightGCN(
        num_users=len(user_to_idx),
        num_items=len(hashtag_to_idx),
        embedding_dim=64,
        n_layers=2 # Fast
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = GNNTrainer(model, optimizer, device)
    adj = trainer.create_adj_matrix(
        train_data_df["user_mapping"].values, 
        train_data_df["item_mapping"].values,
        len(user_to_idx), len(hashtag_to_idx)
    )
    
    # Quick Train 
    logger.info("Quick training RecSys for analysis...")
    for _ in range(10): # 10 epochs
        trainer.train_epoch(adj, train_data_df)
        
    # Evaluate Subgroups
    model.eval()
    with torch.no_grad():
        u_emb, i_emb = model(adj)
        
    results = []
    idx_to_hashtag = {i: h for h, i in hashtag_to_idx.items()}
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Users"):
        uid = row["user_id"]
        if uid not in user_to_idx: continue
        u_idx = user_to_idx[uid]
        
        # History Length
        hist_len = len(row["train_hashtags"])
        
        # True Test Items
        true_test = set(row["test_hashtags"])
        if not true_test: continue
        
        # Predict
        ue = u_emb[u_idx]
        scores = torch.matmul(ue, i_emb.T).cpu().numpy()
        
        # Exclude train
        train_indices = [hashtag_to_idx[h.lower()] for h in row["train_hashtags"] if h.lower() in hashtag_to_idx]
        scores[train_indices] = -np.inf
        
        top_k = np.argsort(scores)[::-1][:10]
        recs = [idx_to_hashtag[i] for i in top_k if i in idx_to_hashtag]
        
        # Compute Recall@10
        hits = sum(1 for r in recs if r in true_test)
        recall = hits / len(true_test)
        
        results.append({
            "user_id": uid,
            "hist_len": hist_len,
            "recall@10": recall
        })
        
    res_df = pd.DataFrame(results)
    
    # Binning History Length
    res_df["length_bin"] = pd.cut(res_df["hist_len"], bins=[0, 5, 20, 100, 1000], labels=["Cold (1-5)", "Low (6-20)", "Medium (21-100)", "High (>100)"])
    
    avg_recall_by_bin = res_df.groupby("length_bin")["recall@10"].mean().reset_index()
    logger.info("Recall by History Length:")
    logger.info(avg_recall_by_bin)
    
    # Save
    res_df.to_csv(RESULTS_DIR / "recsys_error_analysis.csv", index=False)
    
    # Plot Barplot using bar
    plt.figure(figsize=(8, 5))
    
    # Since bin labels might be categorical, let's cast them to string for plotting
    x_labels = avg_recall_by_bin["length_bin"].astype(str).tolist()
    y_values = avg_recall_by_bin["recall@10"].tolist()
    
    plt.bar(x_labels, y_values, color="skyblue")
    
    plt.title("RecSys Performance (Recall@10) vs User History Length")
    plt.ylabel("Recall@10")
    plt.xlabel("History Length")
    plt.savefig(RESULTS_DIR / "recsys_cold_start.png")
    
def main():
    logger = setup_logging("error_analysis")
    
    analyze_personality_errors(logger)
    analyze_recsys_errors(logger)
    
    logger.info("Error analysis complete.")

if __name__ == "__main__":
    main()
