#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.config import (
    PROCESSED_DIR, 
    RESULTS_DIR, 
    EVIDENCE_PATH, 
    LABELS_DIR,
    SPLITS_DIR,
    BM25_INDEX_PATH,
    SEED,
    TOP_K_EVIDENCE,
)
from src.utils.io import setup_logging, load_parquet, load_splits
from src.utils.seed import set_seed
from src.ir.ir_eval import evaluate_ir
from src.ir.bm25 import BM25Index
from src.ir.evidence import retrieve_evidence_for_user


def cli_labeling(template_df: pd.DataFrame) -> pd.DataFrame:
    labels = template_df.copy()

    users = labels["user_id"].unique()
    traits = labels["trait"].unique()

    print("\n=== IR Relevance Labeling Tool ===")
    print("For each tweet, enter 1 if relevant to the trait, 0 otherwise.")
    print("Enter 'q' to quit and save progress.\n")

    for user_id in users:
        print(f"\n{'='*60}")
        print(f"User: {user_id}")
        print(f"{'='*60}")

        for trait in traits:
            print(f"\n--- Trait: {trait.upper()} ---")
            user_trait = labels[
                (labels["user_id"] == user_id) & (labels["trait"] == trait)
            ].sort_values("rank")

            for idx, row in user_trait.iterrows():
                print(f"\n[Rank {row['rank']}] {row['tweet']}")
                while True:
                    response = input("Relevant? (1/0/q): ").strip().lower()
                    if response == "q":
                        return labels
                    elif response in ["0", "1"]:
                        labels.loc[idx, "relevant"] = int(response)
                        break
                    else:
                        print("Invalid input. Enter 1, 0, or q.")

    return labels


def create_template_from_test_set(
    test_user_ids: list,
    index: BM25Index,
    n_users: int = 25,
    top_k: int = TOP_K_EVIDENCE,
) -> pd.DataFrame:
    """Create labeling template by retrieving evidence for test set users."""
    # Sample random users from test set
    np.random.seed(SEED)
    sampled_users = np.random.choice(test_user_ids, size=min(n_users, len(test_user_ids)), replace=False)
    
    records = []
    for user_id in sampled_users:
        evidence = retrieve_evidence_for_user(index, user_id, top_k=top_k)
        for trait, tweets in evidence.items():
            for rank, item in enumerate(tweets):
                records.append({
                    "user_id": user_id,
                    "trait": trait,
                    "rank": rank + 1,
                    "tweet": item["tweet"],
                    "score": item["score"],
                    "relevant": 0,  # Initialize as 0 (to be labeled)
                })
    
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="IR labeling tool for evidence evaluation")
    parser.add_argument("--n_users", type=int, default=25, help="Number of users to label (20-30 recommended)")
    parser.add_argument("--mode", choices=["create", "label", "eval"], default="create")
    parser.add_argument("--input_file", type=str, default=None, help="Input labels file for eval mode")
    parser.add_argument("--lang", type=str, default="en", help="Language code (en, es, it, nl)")
    args = parser.parse_args()

    logger = setup_logging("ir_label_tool")
    set_seed(SEED)

    labels_path = LABELS_DIR / "ir_labels.csv"

    if args.mode == "create":
        logger.info("Loading test split...")
        
        # Try language-specific splits directory first
        lang_splits_dir = SPLITS_DIR / args.lang
        if not lang_splits_dir.exists():
            # Fallback for backward compatibility (if splits are in root splits dir)
            if args.lang == "en" and (SPLITS_DIR / "test.txt").exists():
                logger.info("Using splits from root splits directory (backward compatibility)")
                lang_splits_dir = SPLITS_DIR
            else:
                # Try to create splits from data if splits don't exist
                logger.warning(f"Splits directory not found: {lang_splits_dir}")
                logger.info("Attempting to create splits from data...")
                
                # Load data and create splits on the fly
                # load_parquet handles both .parquet and .pkl formats
                data_path = PROCESSED_DIR / f"pan15_{args.lang}.parquet"
                pkl_path = data_path.with_suffix(".pkl")
                
                # Check if data file exists before attempting to load
                if not data_path.exists() and not pkl_path.exists():
                    logger.error(f"Data file not found!")
                    logger.error(f"  Expected at: {data_path}")
                    logger.error(f"  Or at: {pkl_path}")
                    logger.error(f"\nPlease run: python scripts/preprocess_pan15.py --lang {args.lang}")
                    logger.error(f"  This will create the data file and splits.")
                    sys.exit(1)
                
                try:
                    from src.data.build_splits import create_splits
                    from src.utils.io import load_parquet, save_splits
                    
                    logger.info(f"Loading data from {data_path}...")
                    df = load_parquet(data_path)
                    
                    logger.info(f"Loaded {len(df)} users")
                    
                    logger.info("Creating splits...")
                    train_ids, dev_ids, test_ids = create_splits(df, seed=SEED)
                    splits = {"train": train_ids, "dev": dev_ids, "test": test_ids}
                    
                    logger.info(f"Saving splits to {lang_splits_dir}...")
                    save_splits(splits, lang_splits_dir)
                    logger.info(f"  Train: {len(train_ids)}, Dev: {len(dev_ids)}, Test: {len(test_ids)}")
                except FileNotFoundError as e:
                    logger.error(f"Failed to load data file!")
                    logger.error(f"  Error: {e}")
                    logger.error(f"  Expected at: {data_path} or {pkl_path}")
                    logger.error(f"\nPlease run: python scripts/preprocess_pan15.py --lang {args.lang}")
                    sys.exit(1)
                except Exception as e:
                    logger.error(f"Error creating splits from data: {e}")
                    logger.error(f"\nPlease ensure data file exists at: {data_path}")
                    sys.exit(1)
        
        # Load splits
        if 'splits' not in locals():
            splits = load_splits(lang_splits_dir)
        
        test_user_ids = splits.get("test", [])
        
        if not test_user_ids:
            logger.error(f"No test users found. Splits may be empty.")
            logger.error(f"Please check: {lang_splits_dir}")
            sys.exit(1)
        
        logger.info(f"Found {len(test_user_ids)} test users. Sampling {args.n_users}...")
        
        logger.info("Loading BM25 index...")
        if not BM25_INDEX_PATH.exists():
            logger.error(f"BM25 index not found at {BM25_INDEX_PATH}. Run scripts/build_ir_index.py first.")
            sys.exit(1)
        
        index = BM25Index.load()
        
        logger.info(f"Retrieving top-{TOP_K_EVIDENCE} evidence for each trait per user...")
        template = create_template_from_test_set(test_user_ids, index, n_users=args.n_users)

        template.to_csv(labels_path, index=False)
        logger.info(f"Created template with {len(template)} records at {labels_path}")
        logger.info(f"  Users: {template['user_id'].nunique()}")
        logger.info(f"  Traits: {sorted(template['trait'].unique())}")
        logger.info("\nEdit this file manually or use --mode label for interactive labeling.")

    elif args.mode == "label":
        if not labels_path.exists():
            logger.error(f"Labels file not found: {labels_path}. Run --mode create first.")
            sys.exit(1)
        
        logger.info(f"Loading template from {labels_path}...")
        template = pd.read_csv(labels_path)
        labels = cli_labeling(template)

        labels.to_csv(labels_path, index=False)
        logger.info(f"Saved labels to {labels_path}")

    elif args.mode == "eval":
        input_path = Path(args.input_file) if args.input_file else labels_path

        if not input_path.exists():
            logger.error(f"Labels file not found: {input_path}")
            sys.exit(1)

        logger.info(f"Loading labels from {input_path}...")
        labels_df = pd.read_csv(input_path)

        labeled = labels_df[labels_df["relevant"].notna() & (labels_df["relevant"] != "")]
        if len(labeled) == 0:
            logger.error("No labeled records found. Please label some tweets first.")
            sys.exit(1)
        
        labeled["relevant"] = labeled["relevant"].astype(int)
        logger.info(f"Evaluating {len(labeled)} labeled records...")

        metrics = evaluate_ir(labeled, k=5)

        logger.info("\n=== IR Evaluation Results ===")
        logger.info("Per trait:")
        for trait in ["open", "conscientious", "extroverted", "agreeable", "stable"]:
            p_key = f"p@5_{trait}"
            n_key = f"ndcg@5_{trait}"
            if p_key in metrics:
                logger.info(f"  {trait}: P@5={metrics[p_key]:.4f}, nDCG@5={metrics[n_key]:.4f}")
        
        logger.info("\nOverall:")
        logger.info(f"  avg P@5: {metrics.get('avg_p@5', 0):.4f}")
        logger.info(f"  avg nDCG@5: {metrics.get('avg_ndcg@5', 0):.4f}")

        results_df = pd.DataFrame([metrics])
        output_path = RESULTS_DIR / "ir_eval.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()

