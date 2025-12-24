#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, EVIDENCE_PATH, TOP_K_EVIDENCE, SEED
from src.utils.io import setup_logging, load_parquet, save_parquet
from src.utils.seed import set_seed
from src.ir.bm25 import BM25Index
from src.ir.evidence import retrieve_all_evidence


def main():
    parser = argparse.ArgumentParser(description="Retrieve evidence tweets for users")
    parser.add_argument("--top_k", type=int, default=TOP_K_EVIDENCE, help="Top-k evidence per trait")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for quick testing")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    logger = setup_logging("retrieve_evidence")
    set_seed(args.seed)

    logger.info("Loading data...")
    df = load_parquet(PROCESSED_DIR / "pan15_en.parquet")

    if args.sample_size and args.sample_size < len(df):
        logger.info(f"Sampling {args.sample_size} users")
        df = df.sample(n=args.sample_size, random_state=args.seed)

    logger.info("Loading BM25 index...")
    index = BM25Index()
    index.load()

    user_ids = df["user_id"].tolist()
    logger.info(f"Retrieving evidence for {len(user_ids)} users...")

    evidence_df = retrieve_all_evidence(index, user_ids, top_k=args.top_k)

    save_parquet(evidence_df, EVIDENCE_PATH)
    logger.info(f"Saved evidence to {EVIDENCE_PATH}")

    logger.info("Evidence summary:")
    logger.info(f"  Total records: {len(evidence_df)}")
    logger.info(f"  Users: {evidence_df['user_id'].nunique()}")
    logger.info(f"  Traits: {evidence_df['trait'].unique().tolist()}")

    logger.info("\nSample evidence:")
    sample_user = user_ids[0]
    sample = evidence_df[evidence_df["user_id"] == sample_user]
    for trait in sample["trait"].unique()[:2]:
        trait_sample = sample[sample["trait"] == trait].head(2)
        logger.info(f"\n  {trait.upper()}:")
        for _, row in trait_sample.iterrows():
            logger.info(f"    [{row['score']:.4f}] {row['tweet'][:60]}...")

    logger.info("\nEvidence retrieval complete!")


if __name__ == "__main__":
    main()

