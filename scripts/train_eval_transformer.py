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
    ENCODER_MODEL,
)
from src.utils.io import setup_logging, load_parquet, load_splits
from src.utils.seed import set_seed
from src.models.transformer_regressor import TransformerTrainer


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Transformer regressor")
    parser.add_argument("--model_name", type=str, default=ENCODER_MODEL, help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for quick testing")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    logger = setup_logging("train_transformer")
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

    logger.info(f"Initializing transformer trainer with {args.model_name}...")
    trainer = TransformerTrainer(
        model_name=args.model_name,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    logger.info("Training transformer regressor...")
    trainer.fit(
        train_texts=train_df["text_concat"].tolist(),
        train_targets=train_df[TRAIT_COLS].values,
        val_texts=dev_df["text_concat"].tolist(),
        val_targets=dev_df[TRAIT_COLS].values,
        epochs=args.epochs,
    )

    logger.info("Evaluating on dev set...")
    dev_metrics = trainer.evaluate(dev_df["text_concat"].tolist(), dev_df[TRAIT_COLS])

    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_df["text_concat"].tolist(), test_df[TRAIT_COLS])

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
    output_path = RESULTS_DIR / "metrics_transformer.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved metrics to {output_path}")

    trainer.save()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

