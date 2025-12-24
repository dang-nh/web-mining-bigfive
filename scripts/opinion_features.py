#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, SEED
from src.utils.io import setup_logging, load_parquet, save_parquet
from src.utils.seed import set_seed
from src.opinion.features import extract_opinion_features, OpinionExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract opinion features from tweets")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for quick testing")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    logger = setup_logging("opinion_features")
    set_seed(args.seed)

    logger.info("Loading data...")
    df = load_parquet(PROCESSED_DIR / "pan15_en.parquet")

    if args.sample_size and args.sample_size < len(df):
        logger.info(f"Sampling {args.sample_size} users for quick testing")
        df = df.sample(n=args.sample_size, random_state=args.seed)

    logger.info(f"Processing {len(df)} users...")
    logger.info("Initializing opinion extractor (loading models)...")

    extractor = OpinionExtractor(batch_size=args.batch_size)

    logger.info("Extracting opinion features...")
    opinion_df = extract_opinion_features(df, extractor=extractor)

    output_path = PROCESSED_DIR / "opinion_features.parquet"
    save_parquet(opinion_df, output_path)
    logger.info(f"Saved opinion features to {output_path}")

    logger.info("Feature summary:")
    for col in opinion_df.columns:
        if col != "user_id":
            logger.info(f"  {col}: mean={opinion_df[col].mean():.4f}, std={opinion_df[col].std():.4f}")

    logger.info("Opinion feature extraction complete!")


if __name__ == "__main__":
    main()

