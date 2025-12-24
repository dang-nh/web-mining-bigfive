#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import (
    PROCESSED_DIR,
    SPLITS_DIR,
    RESULTS_DIR,
    TRAIT_COLS,
    SEED,
)
from src.utils.io import setup_logging, load_parquet, load_splits
from src.utils.seed import set_seed
from src.models.tfidf_ridge import TfidfRidgeModel, TfidfRidgeWithOpinion


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate TF-IDF Ridge baseline")
    parser.add_argument("--with_opinion", action="store_true", help="Include opinion features")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for quick testing")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha parameter")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    logger = setup_logging("train_tfidf")
    set_seed(args.seed)

    logger.info("Loading data...")
    df = load_parquet(PROCESSED_DIR / "pan15_en.parquet")
    splits = load_splits(SPLITS_DIR)

    if args.sample_size and args.sample_size < len(df):
        logger.info(f"Sampling {args.sample_size} users for quick testing")
        df = df.sample(n=args.sample_size, random_state=args.seed)
        splits = {
            "train": [uid for uid in splits["train"] if uid in df["user_id"].values][:int(args.sample_size * 0.7)],
            "dev": [uid for uid in splits["dev"] if uid in df["user_id"].values][:int(args.sample_size * 0.1)],
            "test": [uid for uid in splits["test"] if uid in df["user_id"].values][:int(args.sample_size * 0.2)],
        }

    train_df = df[df["user_id"].isin(splits["train"])]
    dev_df = df[df["user_id"].isin(splits["dev"])]
    test_df = df[df["user_id"].isin(splits["test"])]

    logger.info(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    if args.with_opinion:
        logger.info("Loading opinion features...")
        opinion_path = PROCESSED_DIR / "opinion_features.parquet"
        if not opinion_path.exists():
            logger.error("Opinion features not found. Run scripts/opinion_features.py first.")
            sys.exit(1)

        opinion_df = load_parquet(opinion_path)
        opinion_cols = [c for c in opinion_df.columns if c != "user_id"]

        train_opinion = train_df.merge(opinion_df, on="user_id")[opinion_cols]
        dev_opinion = dev_df.merge(opinion_df, on="user_id")[opinion_cols]
        test_opinion = test_df.merge(opinion_df, on="user_id")[opinion_cols]

        train_df = train_df[train_df["user_id"].isin(opinion_df["user_id"])]
        dev_df = dev_df[dev_df["user_id"].isin(opinion_df["user_id"])]
        test_df = test_df[test_df["user_id"].isin(opinion_df["user_id"])]

        logger.info("Training TF-IDF Ridge with opinion features...")
        model = TfidfRidgeWithOpinion(alpha=args.alpha)
        model.fit_with_opinion(
            train_df["text_concat"],
            train_opinion,
            train_df[TRAIT_COLS],
        )

        logger.info("Evaluating on dev set...")
        dev_metrics = model.evaluate_with_opinion(
            dev_df["text_concat"],
            dev_opinion,
            dev_df[TRAIT_COLS],
        )

        logger.info("Evaluating on test set...")
        test_metrics = model.evaluate_with_opinion(
            test_df["text_concat"],
            test_opinion,
            test_df[TRAIT_COLS],
        )

        model.save()
        output_file = "metrics_text_opinion.csv"
    else:
        logger.info("Training TF-IDF Ridge baseline (text only)...")
        model = TfidfRidgeModel(alpha=args.alpha)
        model.fit(train_df["text_concat"], train_df[TRAIT_COLS])

        logger.info("Evaluating on dev set...")
        dev_metrics = model.evaluate(dev_df["text_concat"], dev_df[TRAIT_COLS])

        logger.info("Evaluating on test set...")
        test_metrics = model.evaluate(test_df["text_concat"], test_df[TRAIT_COLS])

        model.save()
        output_file = "metrics_baseline.csv"

    logger.info("=== Dev Set Metrics ===")
    for k, v in dev_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info("=== Test Set Metrics ===")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    results = []
    for split_name, metrics in [("dev", dev_metrics), ("test", test_metrics)]:
        row = {"split": split_name}
        row.update(metrics)
        results.append(row)

    results_df = pd.DataFrame(results)
    output_path = RESULTS_DIR / output_file
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved metrics to {output_path}")

    if args.with_opinion:
        ablation_path = RESULTS_DIR / "metrics_ablation.csv"
        baseline_path = RESULTS_DIR / "metrics_baseline.csv"

        if baseline_path.exists():
            baseline_df = pd.read_csv(baseline_path)
            baseline_df["model"] = "text_only"
            results_df["model"] = "text_opinion"
            ablation_df = pd.concat([baseline_df, results_df], ignore_index=True)
            ablation_df.to_csv(ablation_path, index=False)
            logger.info(f"Saved ablation comparison to {ablation_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

