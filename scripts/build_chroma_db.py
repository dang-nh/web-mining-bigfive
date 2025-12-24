#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, SPLITS_DIR, CHROMA_DIR, SEED
from src.utils.io import setup_logging, load_parquet, load_splits
from src.utils.seed import set_seed
from src.ir.chroma_store import build_chroma_store


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB vector store for user similarity")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for quick testing")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    logger = setup_logging("build_chroma_db")
    set_seed(args.seed)

    logger.info("Loading data...")
    df = load_parquet(PROCESSED_DIR / "pan15_en.parquet")
    splits = load_splits(SPLITS_DIR)

    train_ids = splits.get("train", [])
    train_df = df[df["user_id"].isin(train_ids)]

    if args.sample_size and args.sample_size < len(train_df):
        logger.info(f"Sampling {args.sample_size} users")
        train_df = train_df.sample(n=args.sample_size, random_state=args.seed)

    logger.info(f"Building ChromaDB with {len(train_df)} users...")
    store = build_chroma_store(train_df, persist_dir=CHROMA_DIR)

    logger.info(f"ChromaDB saved to {CHROMA_DIR}")

    logger.info("Testing similarity search...")
    test_text = "I love going to parties and meeting new people. Social events are so fun!"
    similar = store.get_similar_users(test_text, top_n=3)

    logger.info("Top 3 similar users to test query:")
    for u in similar:
        logger.info(f"  User: {u['user_id']}, Distance: {u['distance']:.4f}")
        if u.get("traits"):
            traits_str = ", ".join(f"{k}={v:.2f}" for k, v in u["traits"].items())
            logger.info(f"    Traits: {traits_str}")

    logger.info("ChromaDB build complete!")


if __name__ == "__main__":
    main()

