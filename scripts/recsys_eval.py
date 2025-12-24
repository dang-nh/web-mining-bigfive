#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import PROCESSED_DIR, SPLITS_DIR, RESULTS_DIR, TRAIT_COLS, SEED
from src.utils.io import setup_logging, load_parquet, load_splits
from src.utils.seed import set_seed
from src.recsys.hashtag_recsys import HashtagRecommender, prepare_user_hashtags
from src.recsys.metrics import evaluate_recommender


def main():
    parser = argparse.ArgumentParser(description="Evaluate recommendation systems")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k recommendations")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for testing")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    logger = setup_logging("recsys_eval")
    set_seed(args.seed)

    logger.info("Loading data...")
    df = load_parquet(PROCESSED_DIR / "pan15_en.parquet")
    splits = load_splits(SPLITS_DIR)

    if args.sample_size and args.sample_size < len(df):
        logger.info(f"Sampling {args.sample_size} users")
        df = df.sample(n=args.sample_size, random_state=args.seed)

    test_users = [uid for uid in splits.get("test", []) if uid in df["user_id"].values]
    test_df = df[df["user_id"].isin(test_users)]

    logger.info(f"Test users: {len(test_df)}")

    logger.info("Preparing user hashtag splits...")
    hashtag_splits = prepare_user_hashtags(df, holdout_ratio=0.2, seed=args.seed)
    test_hashtag_splits = hashtag_splits[hashtag_splits["user_id"].isin(test_users)]

    logger.info(f"Users with hashtags for evaluation: {len(test_hashtag_splits)}")

    if len(test_hashtag_splits) == 0:
        logger.warning("No users with enough hashtags for evaluation. Creating synthetic evaluation.")
        test_hashtag_splits = hashtag_splits.head(50)

    logger.info("Training recommender on full dataset...")
    recommender = HashtagRecommender()
    recommender.fit(df)

    logger.info(f"Total unique hashtags: {len(recommender.all_hashtags)}")

    results = []

    logger.info("\n=== Evaluating Popularity Baseline ===")
    pop_recs = []
    pop_relevant = []
    for _, row in test_hashtag_splits.iterrows():
        recs = recommender.recommend_popularity(
            exclude_hashtags=row["train_hashtags"],
            top_k=args.top_k,
        )
        pop_recs.append(recs)
        pop_relevant.append(set(row["test_hashtags"]))

    pop_metrics = evaluate_recommender(pop_recs, pop_relevant, k_values=[5, 10])
    pop_metrics["model"] = "popularity"
    results.append(pop_metrics)

    for k, v in pop_metrics.items():
        if k != "model":
            logger.info(f"  {k}: {v:.4f}")

    logger.info("\n=== Evaluating Content-Based ===")
    content_recs = []
    content_relevant = []

    for _, row in test_hashtag_splits.iterrows():
        user_df = df[df["user_id"] == row["user_id"]]
        if len(user_df) == 0:
            continue

        user_text = user_df.iloc[0]["text_concat"]
        recs = recommender.recommend_content(
            user_text,
            exclude_hashtags=row["train_hashtags"],
            top_k=args.top_k,
        )
        content_recs.append(recs)
        content_relevant.append(set(row["test_hashtags"]))

    if content_recs:
        content_metrics = evaluate_recommender(content_recs, content_relevant, k_values=[5, 10])
        content_metrics["model"] = "content"
        results.append(content_metrics)

        for k, v in content_metrics.items():
            if k != "model":
                logger.info(f"  {k}: {v:.4f}")

    logger.info("\n=== Evaluating Personality-Aware ===")
    personality_recs = []
    personality_relevant = []

    for _, row in test_hashtag_splits.iterrows():
        user_df = df[df["user_id"] == row["user_id"]]
        if len(user_df) == 0:
            continue

        user_row = user_df.iloc[0]
        user_text = user_row["text_concat"]
        user_traits = {
            "open": user_row.get("y_open", 0.5),
            "conscientious": user_row.get("y_conscientious", 0.5),
            "extroverted": user_row.get("y_extroverted", 0.5),
            "agreeable": user_row.get("y_agreeable", 0.5),
            "stable": user_row.get("y_stable", 0.5),
        }

        recs = recommender.recommend_personality_aware(
            user_text,
            user_traits,
            exclude_hashtags=row["train_hashtags"],
            top_k=args.top_k,
        )
        personality_recs.append(recs)
        personality_relevant.append(set(row["test_hashtags"]))

    if personality_recs:
        personality_metrics = evaluate_recommender(
            personality_recs, personality_relevant, k_values=[5, 10]
        )
        personality_metrics["model"] = "personality_aware"
        results.append(personality_metrics)

        for k, v in personality_metrics.items():
            if k != "model":
                logger.info(f"  {k}: {v:.4f}")

    results_df = pd.DataFrame(results)
    output_path = RESULTS_DIR / "metrics_recsys.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved results to {output_path}")

    logger.info("\nRecommendation evaluation complete!")


if __name__ == "__main__":
    main()

