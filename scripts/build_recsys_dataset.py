#!/usr/bin/env python3
"""
Build RecSys dataset and item profiles.
1. Load processed PAN15 data.
2. Split users into Train/Test using existing splits.
3. Use Train users to build Item Profiles (Content Embedding + Aggregated Personality).
4. Save Item Profiles.
5. Create Train/Test Hashtag splits for each user (80% train, 20% test).
6. Save User Tasks (RecSys Dataset).
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    PROCESSED_DIR, 
    SPLITS_DIR, 
    SEED, 
    TRAIT_COLS,
    TRAIT_NAMES
)
from src.utils.io import setup_logging, load_parquet, load_splits, save_parquet
from src.utils.seed import set_seed
from src.recsys.hashtag_recsys import HashtagRecommender, prepare_user_hashtags

def main():
    parser = argparse.ArgumentParser(description="Build RecSys Dataset")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--min_freq", type=int, default=5, help="Minimum hashtag frequency in TRAIN set to be included")
    args = parser.parse_args()

    logger = setup_logging("build_recsys_dataset")
    set_seed(args.seed)

    # 1. Load Data
    logger.info("Loading PAN15 data...")
    df = load_parquet(PROCESSED_DIR / "pan15_en.parquet")
    splits = load_splits(SPLITS_DIR / "en") # Load English splits explicitly or generic
    
    # If generic splits dir, it might be a dict with train/dev/test lists
    # Check if load_splits returns what we expect
    if not splits and (SPLITS_DIR / "en").exists():
         splits = load_splits(SPLITS_DIR / "en")

    train_uids = set(splits.get("train", []))
    # Combine dev and test into "test" for RecSys eval if we want more evaluation data, 
    # OR strictly follow the personality prediction splits. 
    # Task description says: "Split per user: items_train 80%, items_test 20%".
    # And "Use Train-only to build item profiles".
    # So we should use the official Train users to learn Item Profiles.
    # But for the output dataset, we can generate rows for ALL users (Train + Test), 
    # so we can evaluate on Test users (Unseen users) or Train users (Seen users).
    
    train_df = df[df["user_id"].isin(train_uids)]
    logger.info(f"Train Users for Item Profiling: {len(train_df)}")

    # 2. Build Item Profiles (Train Users Only)
    logger.info("Building Item Profiles from Train Users...")
    recommender = HashtagRecommender()
    
    # We use Ground Truth traits for building profiles as per instructions:
    # "Nếu chưa có predicted personality: tạm dùng ground truth ở train để build p_h"
    recommender.fit(
        train_df, 
        trait_cols=TRAIT_COLS, 
        min_freq=args.min_freq
    )
    
    # Export profiles
    profiles_df = recommender.get_profiles_df()
    logger.info(f"Learned profiles for {len(profiles_df)} hashtags")
    
    profiles_path = PROCESSED_DIR / "hashtag_profiles.parquet"
    save_parquet(profiles_df, profiles_path)
    logger.info(f"Saved profiles to {profiles_path}")

    # 3. Create User Splits (All Users)
    # We want to create the evaluation task for everyone.
    # For each user, we split their hashtags 80/20.
    logger.info("Creating User Hashtag Splits (80/20)...")
    
    # We need to preserve the user's personality traits in the dataset
    # "Lưu: data/processed/recsys_dataset.parquet"
    # Expected columns: user_id, train_hashtags, test_hashtags, y_open, y_conscientious...
    
    # We pass the full DF to prepare_user_hashtags
    # But we should only consider recommendable hashtags (those in profiles_df)
    # to avoid recommending unknown items? 
    # Actually, `prepare_user_hashtags` re-computes global counts. 
    # We should probably pass the counts from the recommender (trained on TRAIN)
    # BUT, if a user in TEST uses a hashtag that wasn't in TRAIN, we can't recommend it anyway (cold start item).
    # So valid items are only those in `recommender.filtered_hashtags`.
    
    # Re-using logic from prepare_user_hashtags but customized constraints
    user_rows = []
    
    valid_hashtags = set(recommender.filtered_hashtags)
    
    for _, row in df.iterrows():
        user_id = row["user_id"]
        
        # Extract hashtags
        if isinstance(row["tweets"], list):
            raw_hashtags = []
            for t in row["tweets"]:
                from src.utils.text import extract_hashtags
                raw_hashtags.extend(extract_hashtags(t))
        else:
            from src.utils.text import extract_hashtags
            raw_hashtags = extract_hashtags(row["tweets"])
            
        # Filter to valid hashtags (known from training)
        user_valid_hashtags = [h for h in raw_hashtags if h.lower() in valid_hashtags]
        unique_valid = list(set(user_valid_hashtags)) # Unique set? Or sequence? 
        # Usually RecSys evaluates on Set Recall.
        # User request: "items = set(hashtags). Split per user: items_train 80%, items_test 20%"
        
        if len(unique_valid) < 5: # Skip users with too few hashtags
            continue
            
        np.random.shuffle(unique_valid)
        split_idx = int(len(unique_valid) * 0.8)
        
        train_items = unique_valid[:split_idx]
        test_items = unique_valid[split_idx:]
        
        if not test_items: # If 80/20 results in 0 test items (e.g. len=4 -> split=3, left=1. len=2 -> split=1, left=1)
             # Enforce at least 1 test item if possible
             if len(train_items) > 0:
                 test_items = [train_items.pop()]
        
        if len(test_items) == 0:
            continue
            
        record = {
            "user_id": user_id,
            "train_hashtags": train_items,
            "test_hashtags": test_items,
            "all_hashtags": unique_valid,
            "text_concat": row["text_concat"] if "text_concat" in row else "", # Pass text for content baseline
            "split": "train" if user_id in train_uids else "test"
        }
        
        # Add personality traits
        for t_col in TRAIT_COLS:
            if t_col in row:
                record[t_col] = row[t_col]
                
        user_rows.append(record)
        
    recsys_df = pd.DataFrame(user_rows)
    logger.info(f"Built RecSys dataset with {len(recsys_df)} users.")
    
    output_path = PROCESSED_DIR / "recsys_dataset.parquet"
    save_parquet(recsys_df, output_path)
    logger.info(f"Saved RecSys dataset to {output_path}")

if __name__ == "__main__":
    main()
