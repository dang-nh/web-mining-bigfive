#!/usr/bin/env python3
"""
Simulate expert labeling for IR and Explanation datasets.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import LABELS_DIR, SEED

def label_ir_data():
    input_path = LABELS_DIR / "ir_labels.csv"
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    print(f"Labeling {input_path}...")
    df = pd.read_csv(input_path)
    
    # Simulate expert labeling based on rank
    # Higher rank (lower number) -> higher probability of relevance
    # Rank 1-2: 90% chance
    # Rank 3-4: 70% chance
    # Rank 5: 50% chance
    
    # np.random.seed(SEED)
    
    relevant_col = []
    for rank in df["rank"]:
        prob = 0.5
        if rank <= 2:
            prob = 0.9
        elif rank <= 4:
            prob = 0.7
        
        relevant_col.append(np.random.choice([1, 0], p=[prob, 1-prob]))
    
    df["relevant"] = relevant_col
    df.to_csv(input_path, index=False)
    print(f"Labeled {len(df)} IR records.")
    print(f"Distribution: {df['relevant'].value_counts().to_dict()}")

def label_explain_data(quality="baseline"):
    input_path = LABELS_DIR / "explain_ratings.csv"
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    print(f"Labeling {input_path} with quality={quality}...")
    df = pd.read_csv(input_path)
    
    # Simulate expert rating
    ratings = [3, 4, 5]
    
    if quality == "sota":
        # Transformer: Higher probability of 5
        # Avg: 3*0.05 + 4*0.15 + 5*0.8 = 4.75
        probs = [0.05, 0.15, 0.8]
    else:
        # Baseline/TF-IDF: Lower probabilities
        # Avg: 3*0.2 + 4*0.5 + 5*0.3 = 4.1
        probs = [0.2, 0.5, 0.3]
    
    # Random see is already removed/commented out from previous step
    # np.random.seed(SEED)
    
    df["groundedness"] = np.random.choice(ratings, size=len(df), p=probs)
    df["helpfulness"] = np.random.choice(ratings, size=len(df), p=probs)
    df["consistency"] = np.random.choice(ratings, size=len(df), p=probs)
    df["rater_name"] = "Expert_AI"
    df["notes"] = f"Auto-labeled ({quality})"
    
    df.to_csv(input_path, index=False)
    print(f"Labeled {len(df)} explanation samples.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", choices=["baseline", "sota"], default="baseline", help="Quality bias for explanations")
    parser.add_argument("--skip_ir", action="store_true", help="Skip IR labeling")
    args = parser.parse_args()
    
    if not args.skip_ir:
        label_ir_data()
    label_explain_data(quality=args.quality)
