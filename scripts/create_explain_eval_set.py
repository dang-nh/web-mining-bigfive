#!/usr/bin/env python3
"""
Create explanation evaluation dataset.
Selects 50 samples from test set, generates predictions, evidence, and explanations.
"""
import argparse
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.config import (
    PROCESSED_DIR,
    LABELS_DIR,
    SPLITS_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    BM25_INDEX_PATH,
    SEED,
    TRAIT_NAMES,
    TOP_K_EVIDENCE,
)
from src.utils.io import setup_logging, load_parquet, load_splits
from src.utils.seed import set_seed
from src.models.tfidf_ridge import TfidfRidgeModel
from src.models.transformer_regressor import TransformerTrainer
from src.ir.bm25 import BM25Index
from src.ir.evidence import retrieve_evidence_for_user
from src.rag.explain import PersonalityExplainer


def main():
    parser = argparse.ArgumentParser(description="Create explanation evaluation dataset")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to generate")
    parser.add_argument("--lang", type=str, default="en", help="Language code")
    parser.add_argument("--model_type", type=str, default="tfidf", choices=["tfidf", "transformer"], help="Model type")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model file")
    args = parser.parse_args()

    logger = setup_logging("create_explain_eval")
    set_seed(SEED)

    # 1. Load test split
    logger.info("Loading test split...")
    lang_splits_dir = SPLITS_DIR / args.lang
    if not lang_splits_dir.exists():
        if args.lang == "en" and (SPLITS_DIR / "test.txt").exists():
            lang_splits_dir = SPLITS_DIR
        else:
            logger.error(f"Splits directory not found: {lang_splits_dir}")
            sys.exit(1)
    
    splits = load_splits(lang_splits_dir)
    test_user_ids = splits.get("test", [])
    
    if not test_user_ids:
        logger.error(f"No test users found")
        sys.exit(1)
    
    # 2. Load data
    logger.info("Loading data...")
    data_path = PROCESSED_DIR / f"pan15_{args.lang}.parquet"
    df = load_parquet(data_path)
    
    # Filter to test users
    test_df = df[df["user_id"].isin(test_user_ids)].copy()
    
    # Sample n_samples users
    np.random.seed(SEED)
    sampled_user_ids = np.random.choice(
        test_df["user_id"].unique(), 
        size=min(args.n_samples, len(test_df["user_id"].unique())), 
        replace=False
    )
    sampled_df = test_df[test_df["user_id"].isin(sampled_user_ids)].copy()
    
    logger.info(f"Selected {len(sampled_user_ids)} users for evaluation")
    
    # 3. Load model
    logger.info(f"Loading {args.model_type} model...")
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        if args.model_type == "tfidf":
            model_path = MODELS_DIR / f"baseline_{args.lang}.joblib"
        else:
            model_path = MODELS_DIR / f"transformer_{args.lang}.pt"
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run training script first.")
        sys.exit(1)
    
    if args.model_type == "tfidf":
        model = TfidfRidgeModel.load(model_path)
    else:
        # Load transformer model
        # Force CPU to avoid CUDA OOM during inference if GPU is busy or limited
        device = "cpu" 
        model = TransformerTrainer.load(model_path)
        model.device = torch.device(device)
        model.model.to(device)
    
    # 4. Load IR index
    logger.info("Loading BM25 index...")
    if not BM25_INDEX_PATH.exists():
        logger.error(f"BM25 index not found. Run scripts/build_ir_index.py first.")
        sys.exit(1)
    
    index = BM25Index.load()
    
    # 5. Load explainer
    logger.info("Initializing explainer...")
    explainer = PersonalityExplainer(use_openai=False)  # Use rule-based for consistency
    
    # 6. Generate predictions, evidence, and explanations
    logger.info("Generating predictions, evidence, and explanations...")
    samples = []
    
    for user_id in sampled_user_ids:
        user_row = sampled_df[sampled_df["user_id"] == user_id].iloc[0]
        
        # Get user text (concatenate tweets)
        if isinstance(user_row.get("tweets"), list):
            user_text = " ".join(user_row["tweets"])
        elif "text_concat" in user_row:
            user_text = user_row["text_concat"]
        else:
            user_text = str(user_row.get("tweets", ""))
        
        # Predict traits
        predictions = model.predict(pd.Series([user_text]))[0]
        predicted_traits = {
            trait: float(np.clip(predictions[i], 0, 1))
            for i, trait in enumerate(TRAIT_NAMES)
        }
        
        # Retrieve evidence
        evidence = retrieve_evidence_for_user(index, user_id, top_k=TOP_K_EVIDENCE)
        
        # Generate explanation
        explanation = explainer.explain(predicted_traits, evidence)
        
        # Store sample
        sample = {
            "sample_id": len(samples) + 1,
            "user_id": user_id,
            "user_text": user_text[:500],  # Truncate for CSV readability
            "predicted_traits": predicted_traits,
            "evidence": evidence,
            "explanation": explanation,
        }
        samples.append(sample)
        
        if (len(samples) % 10) == 0:
            logger.info(f"  Processed {len(samples)}/{len(sampled_user_ids)} samples")
    
    # 7. Save JSON file with full data
    json_path = LABELS_DIR / "explain_samples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved full samples to {json_path}")
    
    # 8. Create CSV template for rating
    logger.info("Creating CSV template for ratings...")
    rating_records = []
    
    for sample in samples:
        # Extract explanation summary (for CSV readability)
        explanation_summary = ""
        if "overall_summary" in sample["explanation"]:
            explanation_summary = sample["explanation"]["overall_summary"]
        elif "trait_explanations" in sample["explanation"]:
            trait_exps = sample["explanation"]["trait_explanations"]
            explanation_summary = " | ".join([
                f"{t}: {exp[:50]}" 
                for t, exp in trait_exps.items()
            ])
        
        # Create one row per sample
        rating_records.append({
            "sample_id": sample["sample_id"],
            "user_id": sample["user_id"],
            "user_text_preview": sample["user_text"][:200] + "...",
            "predicted_traits": json.dumps(sample["predicted_traits"]),
            "explanation_summary": explanation_summary[:300],
            "groundedness": "",  # To be filled by raters (1-5)
            "helpfulness": "",   # To be filled by raters (1-5)
            "consistency": "",   # To be filled by raters (1-5)
            "rater_name": "",   # To be filled by raters
            "notes": "",        # Optional notes
        })
    
    ratings_df = pd.DataFrame(rating_records)
    ratings_path = LABELS_DIR / "explain_ratings.csv"
    ratings_df.to_csv(ratings_path, index=False)
    logger.info(f"Created rating template at {ratings_path}")
    logger.info(f"  Total samples: {len(ratings_df)}")
    logger.info(f"  Expected: ~12-13 samples per rater (4 raters for 50 samples)")
    
    logger.info("\n=== Next Steps ===")
    logger.info("1. Share explain_samples.json and explain_ratings.csv with raters")
    logger.info("2. Each rater should rate 12-13 samples in explain_ratings.csv")
    logger.info("3. Rating criteria:")
    logger.info("   - Groundedness (1-5): Does explanation reference evidence?")
    logger.info("   - Helpfulness (1-5): Is explanation clear and useful?")
    logger.info("   - Consistency (1-5): Does explanation align with predicted traits?")
    logger.info("4. After all ratings collected, run scripts/summarize_explain_eval.py")


if __name__ == "__main__":
    main()
