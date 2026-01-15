#!/usr/bin/env python3
"""Preprocess PAN15 dataset - combines training and test English subsets."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    PAN15_TRAIN_EN_DIR,
    N_TWEETS_PER_USER,
    SEED,
)
from src.utils.io import setup_logging, save_parquet, save_splits
from src.utils.seed import set_seed
from src.data.pan15_parser import parse_pan15_dataset, find_xml_files, find_truth_file
from src.data.build_splits import create_splits


def find_language_dirs(raw_dir: Path, lang: str) -> list:
    """Find data directories for a specific language (both train and test)."""
    lang_dirs = []
    
    # Map common language codes to full names used in dataset
    lang_map = {
        "en": "english",
        "es": "spanish",
        "it": "italian",
        "nl": "dutch"
    }
    full_lang = lang_map.get(lang, lang)
    
    # Search patterns
    patterns = [
        f"*{full_lang}*",
        f"pan15_train_{lang}",
        f"pan15_test_{lang}"
    ]
    
    for pattern in patterns:
        for d in raw_dir.rglob(pattern):
            if d.is_dir() and find_xml_files(d):
                # Avoid nested duplicates if we found parent
                if not any(parent in d.parents for parent in lang_dirs):
                    lang_dirs.append(d)
    
    return sorted(list(set(lang_dirs)))


def process_language(lang: str, args, logger):
    """Process a single language."""
    logger.info(f"=== Processing language: {lang} ===")
    
    # Find data directories
    if args.data_dir:
        data_dirs = [Path(args.data_dir)]
    else:
        data_dirs = find_language_dirs(RAW_DIR, lang)
    
    if not data_dirs:
        logger.warning(f"No data found for language {lang}. Skipping.")
        return

    # Parse and combine
    all_dfs = []
    for data_dir in data_dirs:
        logger.info(f"  Found directory: {data_dir}")
        
        truth_file = find_truth_file(data_dir)
        if truth_file:
            logger.info(f"    Truth file: {truth_file.name}")
        
        # Determine language for preprocessing
        # Spanish/Italian/Dutch might need different preprocessing if we get strict
        # For now, default preprocessing works reasonably well for Western languages
        df = parse_pan15_dataset(data_dir, max_tweets=args.max_tweets, preprocess=True)
        logger.info(f"    Parsed {len(df)} users")
        
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        logger.warning(f"No valid data parsed for {lang}.")
        return

    # Combine
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["user_id"])
    df["lang"] = lang
    
    logger.info(f"  Total users for {lang}: {len(df)}")

    # Save data
    output_path = PROCESSED_DIR / f"pan15_{lang}.parquet"
    save_parquet(df, output_path)
    logger.info(f"  Saved data to {output_path}")

    # Create splits
    train_ids, dev_ids, test_ids = create_splits(df, seed=args.seed)
    
    # Save splits in language-specific folder
    lang_splits_dir = SPLITS_DIR / lang
    splits = {"train": train_ids, "dev": dev_ids, "test": test_ids}
    save_splits(splits, lang_splits_dir)
    
    logger.info(f"  Saved splits to {lang_splits_dir}")
    logger.info(f"  Train: {len(train_ids)}, Dev: {len(dev_ids)}, Test: {len(test_ids)}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess PAN15 dataset")
    parser.add_argument("--lang", type=str, default="all", help="Language to process (en, es, it, nl, or all)")
    parser.add_argument("--data_dir", type=str, default=None, help="Specific data directory override")
    parser.add_argument("--max_tweets", type=int, default=N_TWEETS_PER_USER, help="Max tweets per user")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    logger = setup_logging("preprocess_pan15")
    set_seed(args.seed)

    if args.lang == "all":
        languages = ["en", "es", "it", "nl"]
    else:
        languages = [args.lang]

    for lang in languages:
        process_language(lang, args, logger)

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()

