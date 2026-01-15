#!/usr/bin/env python3
import argparse
import sys
import logging
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
    MULTILINGUAL_ENCODER_MODEL,
    MODELS_DIR,
)
from src.utils.io import setup_logging, load_parquet, load_splits
from src.utils.seed import set_seed
from src.models.transformer_regressor import TransformerTrainer


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Transformer regressor")
    parser.add_argument("--lang", type=str, default="en", help="Language (en, es, it, nl)")
    parser.add_argument("--model_name", type=str, default=None, help="HuggingFace model name (override)")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--early_stopping", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Number of warmup epochs")
    parser.add_argument("--no_cosine", action="store_true", help="Disable cosine annealing scheduler")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for quick testing")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--data_lang", type=str, default=None, help="Language code for data loading (if different from --lang)")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to save results")
    args = parser.parse_args()

    # Determine languages
    output_lang = args.lang
    input_lang = args.data_lang if args.data_lang else args.lang

    setup_logging()
    logger = logging.getLogger(f"train_transformer_{output_lang}")
    set_seed(args.seed)
    
    # Determine model name
    if args.model_name:
        model_name = args.model_name
    elif input_lang == "en":
        model_name = ENCODER_MODEL
    else:
        model_name = MULTILINGUAL_ENCODER_MODEL
    
    logger.info(f"Loading data for language: {input_lang} (Output: {output_lang})...")
    
    # Load data for specific language
    data_path = PROCESSED_DIR / f"pan15_{input_lang}.parquet"
    try:
        df = load_parquet(data_path)
    except FileNotFoundError:
        logger.error(f"Data file {data_path} (or .pkl) not found. Run scripts/preprocess_pan15.py first.")
        sys.exit(1)
    
    # Load splits for specific language
    lang_splits_dir = SPLITS_DIR / input_lang
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

    logger.info(f"Initializing transformer trainer with {model_name}...")
    logger.info(f"Training config: epochs={args.epochs}, lr={args.lr}, "
                f"early_stopping={args.early_stopping}, warmup={args.warmup_epochs}, "
                f"cosine_schedule={not args.no_cosine}")
    
    trainer = TransformerTrainer(
        model_name=model_name,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        results_dir=args.results_dir
    )

    logger.info("Training transformer regressor...")
    try:
        trainer.fit(
            train_texts=train_df["text_concat"].tolist(),
            train_targets=train_df[TRAIT_COLS].values,
            val_texts=dev_df["text_concat"].tolist(),
            val_targets=dev_df[TRAIT_COLS].values,
            epochs=args.epochs,
            early_stopping_patience=args.early_stopping,
            use_cosine_schedule=not args.no_cosine,
            warmup_epochs=args.warmup_epochs,
            save_suffix=args.lang,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

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
        row = {"split": split_name, "lang": args.lang}
        row.update(metrics)
        results.append(row)

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_path = results_dir / f"metrics_transformer_{output_lang}.csv"
    results_df.to_csv(results_path, index=False)
    
    logger.info(f"Saved metrics to {results_path}")

    model_path = MODELS_DIR / f"transformer_{args.lang}.pt"
    trainer.save(model_path)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

