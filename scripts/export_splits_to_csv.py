#!/usr/bin/env python3
"""Export dataset splits to CSV for report artifacts."""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, SPLITS_DIR, SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "data" / "artifacts"


def load_split_ids(lang: str, split_name: str) -> list:
    """Load user IDs from split file."""
    split_file = SPLITS_DIR / lang / f"{split_name}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def export_lang_splits(lang: str):
    """Export splits for a specific language."""
    logger.info(f"Exporting splits for language: {lang}")
    
    # Load processed data
    data_file = PROCESSED_DIR / f"pan15_{lang}.parquet"
    if not data_file.exists():
        # Fallback to pickle
        data_file = PROCESSED_DIR / f"pan15_{lang}.pkl"
        if not data_file.exists():
            logger.error(f"Processed data file not found (parquet or pkl): {data_file}")
            return
        df = pd.read_pickle(data_file)
    else:
        df = pd.read_parquet(data_file)
    
    logger.info(f"Loaded {len(df)} users from {data_file.name}")

    # Output dir
    output_dir = ARTIFACTS_DIR / f"split_{lang}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Metadata
    metadata = {
        "dataset": "PAN 2015 Author Profiling",
        "language": lang,
        "seed": SEED,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "generated_by": "export_splits_to_csv.py"
    }

    # Process each split
    for split in ["train", "dev", "test"]:
        try:
            user_ids = load_split_ids(lang, split)
            split_df = df[df["user_id"].isin(user_ids)].copy()
            
            # Select relevant columns for report artifact
            # We want to show structure without dumping full text if too large, 
            # but usually for 'dataset link' full content or at least IDs + labels is expected.
            # Let's keep it compact: ID, labels, token_count
            
            # Check what cols we have
            # Assuming 'user_id', 'gender', 'age_group', 'extroverted', etc.
            
            cols_to_export = ["user_id"]
            traits = ["extroverted", "stable", "agreeable", "conscientious", "open"]
            if all(c in split_df.columns for c in traits):
                cols_to_export.extend(traits)
            
            if "gender" in split_df.columns:
                cols_to_export.append("gender")
            if "age_group" in split_df.columns:
                cols_to_export.append("age_group")
                
            final_df = split_df[cols_to_export]
            
            output_file = output_dir / f"{split}_users.csv"
            final_df.to_csv(output_file, index=False)
            
            logger.info(f"  Saved {split}: {len(final_df)} users to {output_file.name}")
            metadata[f"{split}_count"] = len(final_df)
            
        except FileNotFoundError:
            logger.warning(f"  Split {split} not found for {lang}. Skipping.")

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {output_dir / 'metadata.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="all", help="Language code (en, es, it, nl, or all)")
    args = parser.parse_args()

    if args.lang == "all":
        languages = ["en", "es", "it", "nl"]
    else:
        languages = [args.lang]

    for lang in languages:
        export_lang_splits(lang)


if __name__ == "__main__":
    main()
