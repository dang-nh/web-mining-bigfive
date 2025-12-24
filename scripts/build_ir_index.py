#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, BM25_INDEX_PATH, SEED
from src.utils.io import setup_logging, load_parquet
from src.utils.seed import set_seed
from src.ir.bm25 import build_tweet_index


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index for tweets")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for quick testing")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    logger = setup_logging("build_ir_index")
    set_seed(args.seed)

    logger.info("Loading data...")
    df = load_parquet(PROCESSED_DIR / "pan15_en.parquet")

    if args.sample_size and args.sample_size < len(df):
        logger.info(f"Sampling {args.sample_size} users")
        df = df.sample(n=args.sample_size, random_state=args.seed)

    logger.info(f"Building index for {len(df)} users...")
    index = build_tweet_index(df)

    total_docs = len(index.doc_mapping)
    logger.info(f"Indexed {total_docs} tweets")

    index.save()
    logger.info(f"Saved index to {BM25_INDEX_PATH}")

    logger.info("Testing search...")
    test_results = index.search("creative artistic imaginative", top_k=5)
    logger.info("Sample results for 'creative artistic imaginative':")
    for doc, score in test_results[:3]:
        logger.info(f"  [{score:.4f}] {doc['text'][:80]}...")

    logger.info("Index building complete!")


if __name__ == "__main__":
    main()

