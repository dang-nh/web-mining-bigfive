#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="Preprocess PAN15 dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to PAN15 data directory",
    )
    parser.add_argument(
        "--max_tweets",
        type=int,
        default=N_TWEETS_PER_USER,
        help="Maximum tweets per user",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    logger = setup_logging("preprocess_pan15")
    set_seed(args.seed)

    data_dir = Path(args.data_dir) if args.data_dir else None

    if data_dir is None:
        possible_dirs = [
            PAN15_TRAIN_EN_DIR,
            RAW_DIR / "pan15_train_en",
            RAW_DIR / "pan15_train",
        ]
        for d in possible_dirs:
            if d.exists():
                xml_files = find_xml_files(d)
                if xml_files:
                    data_dir = d
                    break

    if data_dir is None or not data_dir.exists():
        logger.error(f"Data directory not found. Run scripts/download_pan15.sh first.")
        sys.exit(1)

    logger.info(f"Using data directory: {data_dir}")

    truth_file = find_truth_file(data_dir)
    if truth_file:
        logger.info(f"Found truth file: {truth_file}")
    else:
        logger.warning("No truth file found. Labels will be zeros.")

    logger.info("Parsing PAN15 dataset...")
    df = parse_pan15_dataset(data_dir, max_tweets=args.max_tweets, preprocess=True)
    logger.info(f"Parsed {len(df)} users")

    if df.empty:
        logger.error("No data parsed. Check data directory structure.")
        sys.exit(1)

    logger.info("Sample user data:")
    sample = df.iloc[0]
    logger.info(f"  User ID: {sample['user_id']}")
    logger.info(f"  Num tweets: {len(sample['tweets'])}")
    logger.info(f"  Traits: O={sample['y_open']:.2f}, C={sample['y_conscientious']:.2f}, "
                f"E={sample['y_extroverted']:.2f}, A={sample['y_agreeable']:.2f}, "
                f"S={sample['y_stable']:.2f}")

    output_path = PROCESSED_DIR / "pan15_en.parquet"
    save_parquet(df, output_path)
    logger.info(f"Saved processed data to {output_path}")

    logger.info("Creating train/dev/test splits...")
    train_ids, dev_ids, test_ids = create_splits(df, seed=args.seed)

    splits = {"train": train_ids, "dev": dev_ids, "test": test_ids}
    save_splits(splits, SPLITS_DIR)

    logger.info(f"Split sizes: train={len(train_ids)}, dev={len(dev_ids)}, test={len(test_ids)}")
    logger.info(f"Saved splits to {SPLITS_DIR}")

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()

