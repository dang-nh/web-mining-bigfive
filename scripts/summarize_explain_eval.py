#!/usr/bin/env python3
"""
Summarize explanation evaluation results.
Computes mean/std by criteria and optionally inter-rater agreement.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from src.config import LABELS_DIR, RESULTS_DIR
from src.utils.io import setup_logging


def compute_inter_rater_agreement(df: pd.DataFrame, criterion: str) -> dict:
    """Compute inter-rater agreement for a criterion."""
    # Filter to samples with multiple raters
    sample_ratings = {}
    for sample_id in df["sample_id"].unique():
        sample_df = df[df["sample_id"] == sample_id]
        ratings = sample_df[criterion].dropna().astype(float).tolist()
        if len(ratings) > 1:
            sample_ratings[sample_id] = ratings
    
    if not sample_ratings:
        return {
            "mean_std": None,
            "cohen_kappa": None,
            "n_samples_multi_rater": 0,
        }
    
    # Compute mean pairwise agreement
    pairwise_agreements = []
    for ratings in sample_ratings.values():
        if len(ratings) >= 2:
            # Compute pairwise differences (inverse of disagreement)
            for i in range(len(ratings)):
                for j in range(i + 1, len(ratings)):
                    diff = abs(ratings[i] - ratings[j])
                    agreement = 1 - (diff / 4.0)  # Normalize by max possible diff (1-5 scale)
                    pairwise_agreements.append(agreement)
    
    # For Cohen's Kappa, we need categorical labels
    # Convert to agreement/disagreement (within 1 point = agreement)
    agreements = []
    for ratings in sample_ratings.values():
        if len(ratings) >= 2:
            for i in range(len(ratings)):
                for j in range(i + 1, len(ratings)):
                    if abs(ratings[i] - ratings[j]) <= 1:
                        agreements.append(1)
                    else:
                        agreements.append(0)
    
    mean_agreement = np.mean(pairwise_agreements) if pairwise_agreements else None
    
    # Simple percentage agreement
    pct_agreement = np.mean(agreements) if agreements else None
    
    return {
        "mean_pairwise_agreement": mean_agreement,
        "percentage_agreement": pct_agreement,
        "n_samples_multi_rater": len(sample_ratings),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize explanation evaluation results")
    parser.add_argument("--input_file", type=str, default=None, help="Input ratings CSV (default: data/labels/explain_ratings.csv)")
    parser.add_argument("--compute_agreement", action="store_true", help="Compute inter-rater agreement")
    args = parser.parse_args()

    logger = setup_logging("summarize_explain_eval")

    # Load ratings
    input_path = Path(args.input_file) if args.input_file else LABELS_DIR / "explain_ratings.csv"
    
    if not input_path.exists():
        logger.error(f"Ratings file not found: {input_path}")
        logger.info("Run scripts/create_explain_eval_set.py first to create the rating template.")
        sys.exit(1)
    
    logger.info(f"Loading ratings from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Filter to rated samples
    criteria = ["groundedness", "helpfulness", "consistency"]
    rated_df = df[df[criteria[0]].notna() & (df[criteria[0]] != "")].copy()
    
    if len(rated_df) == 0:
        logger.error("No rated samples found. Please fill in the ratings first.")
        sys.exit(1)
    
    # Convert ratings to float
    for criterion in criteria:
        rated_df[criterion] = pd.to_numeric(rated_df[criterion], errors="coerce")
    
    rated_df = rated_df[rated_df[criteria].notna().all(axis=1)]
    
    logger.info(f"Found {len(rated_df)} rated samples")
    logger.info(f"  Unique samples: {rated_df['sample_id'].nunique()}")
    logger.info(f"  Raters: {rated_df['rater_name'].nunique() if 'rater_name' in rated_df else 'N/A'}")
    
    # Compute summary statistics
    summary_records = []
    
    for criterion in criteria:
        ratings = rated_df[criterion].values
        
        summary_records.append({
            "criterion": criterion,
            "mean": float(np.mean(ratings)),
            "std": float(np.std(ratings)),
            "median": float(np.median(ratings)),
            "min": float(np.min(ratings)),
            "max": float(np.max(ratings)),
            "n_samples": len(ratings),
        })
        
        logger.info(f"\n{criterion.capitalize()}:")
        logger.info(f"  Mean: {np.mean(ratings):.3f} ± {np.std(ratings):.3f}")
        logger.info(f"  Median: {np.median(ratings):.3f}")
        logger.info(f"  Range: [{np.min(ratings):.1f}, {np.max(ratings):.1f}]")
        logger.info(f"  N: {len(ratings)}")
    
    # Overall (average across criteria)
    all_ratings = rated_df[criteria].values.flatten()
    summary_records.append({
        "criterion": "overall",
        "mean": float(np.mean(all_ratings)),
        "std": float(np.std(all_ratings)),
        "median": float(np.median(all_ratings)),
        "min": float(np.min(all_ratings)),
        "max": float(np.max(all_ratings)),
        "n_samples": len(all_ratings),
    })
    
    logger.info(f"\nOverall (all criteria):")
    logger.info(f"  Mean: {np.mean(all_ratings):.3f} ± {np.std(all_ratings):.3f}")
    
    # Inter-rater agreement (if requested and multiple raters per sample)
    agreement_results = {}
    if args.compute_agreement:
        logger.info("\n=== Inter-Rater Agreement ===")
        for criterion in criteria:
            agreement = compute_inter_rater_agreement(rated_df, criterion)
            agreement_results[criterion] = agreement
            
            if agreement["n_samples_multi_rater"] > 0:
                logger.info(f"\n{criterion.capitalize()}:")
                logger.info(f"  Samples with multiple raters: {agreement['n_samples_multi_rater']}")
                if agreement["mean_pairwise_agreement"] is not None:
                    logger.info(f"  Mean pairwise agreement: {agreement['mean_pairwise_agreement']:.3f}")
                if agreement["percentage_agreement"] is not None:
                    logger.info(f"  Percentage agreement (±1): {agreement['percentage_agreement']:.3f}")
            else:
                logger.info(f"\n{criterion.capitalize()}: No samples with multiple raters")
    
    # Save summary
    summary_df = pd.DataFrame(summary_records)
    output_path = RESULTS_DIR / "explain_eval_summary.csv"
    summary_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved summary to {output_path}")
    
    # Save detailed breakdown if agreement was computed
    if agreement_results:
        agreement_df = pd.DataFrame([
            {"criterion": k, **v} for k, v in agreement_results.items()
        ])
        agreement_path = RESULTS_DIR / "explain_eval_agreement.csv"
        agreement_df.to_csv(agreement_path, index=False)
        logger.info(f"Saved agreement analysis to {agreement_path}")


if __name__ == "__main__":
    main()
