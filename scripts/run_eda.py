#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, RESULTS_DIR, TRAIT_COLS, LANGUAGES

def load_data():
    """Load processed data for all languages."""
    dfs = {}
    for lang in LANGUAGES:
        path = PROCESSED_DIR / f"pan15_{lang}.pkl"
        if path.exists():
            dfs[lang] = pd.read_pickle(path)
            print(f"Loaded {lang}: {len(dfs[lang])} users")
        else:
            print(f"Warning: Data for {lang} not found at {path}")
    return dfs

def plot_trait_distributions(dfs, output_dir):
    """Plot histograms for Big Five traits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combined dataframe for visualization
    all_data = []
    for lang, df in dfs.items():
        temp = df[TRAIT_COLS].copy()
        temp['lang'] = lang
        all_data.append(temp)
    
    if not all_data:
        return

    combined_df = pd.concat(all_data)
    
    trait_map = {
        "y_open": "Openness",
        "y_conscientious": "Conscientiousness", 
        "y_extroverted": "Extraversion",
        "y_agreeable": "Agreeableness",
        "y_stable": "Neuroticism"
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    languages = combined_df['lang'].unique()
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    
    for i, (col, name) in enumerate(trait_map.items()):
        ax = axes[i]
        for j, lang in enumerate(languages):
            subset = combined_df[combined_df['lang'] == lang][col]
            ax.hist(subset, bins=20, alpha=0.5, label=lang, density=True, color=colors[j % len(colors)])
            
        ax.set_title(f"Distribution of {name}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty 6th subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    save_path = output_dir / "trait_distributions.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved trait distributions to {save_path}")

    # Compute descriptive stats
    stats = combined_df.groupby("lang")[list(trait_map.keys())].describe()
    stats_path = output_dir / "trait_stats.csv"
    stats.to_csv(stats_path)
    print(f"Saved trait statistics to {stats_path}")

def plot_length_distributions(dfs, output_dir):
    """Plot distribution of token counts."""
    
    length_data = []
    labels = []
    
    for lang, df in dfs.items():
        # Check if text column exists
        if "text_concat" in df.columns:
            # Simple whitespace tokenization for estimation
            lens = df["text_concat"].apply(lambda x: len(str(x).split()))
            length_data.append(lens.values)
            labels.append(lang)
            
    if not length_data:
        return
        
    plt.figure(figsize=(10, 6))
    plt.boxplot(length_data, labels=labels, patch_artist=True)
    plt.title("Distribution of Concatenated Tweet Lengths (Tokens) per User")
    plt.ylabel("Number of Tokens")
    plt.grid(True, axis='y', alpha=0.3)
    
    save_path = output_dir / "length_distribution.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved length distribution to {save_path}")
    
    # Stats
    # Create temp df for stats
    combined_len = pd.concat([pd.DataFrame({"length": d, "lang": l}) for d, l in zip(length_data, labels)])
    len_stats = combined_len.groupby("lang")["length"].describe()
    len_stats_path = output_dir / "length_stats.csv"
    len_stats.to_csv(len_stats_path)
    print(f"Saved length statistics to {len_stats_path}")

def main():
    print("Starting EDA...")
    dfs = load_data()
    
    eda_dir = RESULTS_DIR / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    
    plot_trait_distributions(dfs, eda_dir)
    plot_length_distributions(dfs, eda_dir)
    
    print("EDA Complete.")

if __name__ == "__main__":
    main()
