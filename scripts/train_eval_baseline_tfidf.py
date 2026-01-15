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
    MODELS_DIR,
    TRAIT_COLS,
    SEED,
)
from src.utils.io import setup_logging, load_parquet, load_splits
from src.utils.seed import set_seed
from src.models.tfidf_ridge import TfidfRidgeModel, TfidfRidgeWithOpinion


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate TF-IDF Ridge baseline")
    parser.add_argument("--lang", type=str, default="en", help="Language (en, es, it, nl)")
    parser.add_argument("--with_opinion", action="store_true", help="Include opinion features")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for quick testing")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha parameter")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs (for consistent interface, though Ridge matches closed form)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (ignored)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (ignored)")
    parser.add_argument("--max_length", type=int, default=None, help="Max sequence length (ignored)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to save results")
    args = parser.parse_args()

    logger = setup_logging(f"train_tfidf_{args.lang}")
    set_seed(args.seed)

    logger.info(f"Loading data for language: {args.lang}...")
    
    # Load data for specific language
    data_path = PROCESSED_DIR / f"pan15_{args.lang}.parquet"
    # load_parquet handles .pkl fallback
    try:
        df = load_parquet(data_path)
    except FileNotFoundError:
        logger.error(f"Data file {data_path} (or .pkl) not found. Run scripts/preprocess_pan15.py first.")
        sys.exit(1)
    
    # Load splits for specific language
    lang_splits_dir = SPLITS_DIR / args.lang
    if not lang_splits_dir.exists():
         # Fallback to main splits dir if lang subfolder doesn't exist (e.g. for 'en' backward compatibility)
         if args.lang == 'en' and (SPLITS_DIR / 'train.txt').exists():
             lang_splits_dir = SPLITS_DIR
         else:
             logger.error(f"Splits directory {lang_splits_dir} not found.")
             sys.exit(1)
             
    splits = load_splits(lang_splits_dir)

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
        # Opinion features logic (simplified for now as it's not main focus yet)
        # Needs updates to handle language-specific opinion files if we go there
        logger.warning("Opinion features support is pending multi-language update. Using text only.")
        args.with_opinion = False

    logger.info("Training TF-IDF Ridge baseline (text only)...")
    model = TfidfRidgeModel(alpha=args.alpha)
    model.fit(train_df["text_concat"], train_df[TRAIT_COLS])

    logger.info("Evaluating on dev set...")
    dev_metrics = model.evaluate(dev_df["text_concat"], dev_df[TRAIT_COLS])

    logger.info("Evaluating on test set...")
    test_metrics = model.evaluate(test_df["text_concat"], test_df[TRAIT_COLS])

    model_path = MODELS_DIR / f"baseline_{args.lang}.joblib"
    model.save(model_path)
    
    output_file = f"metrics_baseline_{args.lang}.csv"

    logger.info("=== Dev Set Metrics ===")
    for k, v in dev_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info("=== Test Set Metrics ===")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    results = []
    for split_name, metrics in [("dev", dev_metrics), ("test", test_metrics)]:
        row = {"split": split_name, "lang": args.lang}
        row.update(metrics)
        results.append(row)

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    output_path = results_dir / f"metrics_baseline_{args.lang}.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved metrics to {output_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

