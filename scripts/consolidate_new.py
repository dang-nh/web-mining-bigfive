
import pandas as pd
from pathlib import Path
import sys

RESULTS_DIR = Path("results")

def main():
    if not RESULTS_DIR.exists():
        print(f"Directory {RESULTS_DIR} not found.")
        return

    all_results = []

    # 1. TF-IDF
    for f in RESULTS_DIR.glob("metrics_baseline_*.csv"):
        # File: metrics_baseline_{lang}.csv
        try:
            df = pd.read_csv(f)
            # Row has: split, lang, rmse_open, ..., avg_rmse
            test_row = df[df["split"] == "test"].iloc[0]
            all_results.append({
                "Model": "TF-IDF Baseline",
                "Language": test_row["lang"],
                "RMSE": test_row["avg_rmse"],
                "MAE": test_row["avg_mae"],
                "Accuracy": test_row["avg_acc"],
                "Pearson": test_row["avg_pearson"]
            })
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # 2. Transformer
    for f in RESULTS_DIR.glob("metrics_transformer_*.csv"):
        # File: metrics_transformer_{lang}.csv or {lang}_{suffix}.csv
        # Suffix handling: metrics_transformer_en_xlm.csv -> Lang: en, Model: XLM-R (Chunked)
        name = f.stem.replace("metrics_transformer_", "")
        
        try:
            df = pd.read_csv(f)
            # Typically has split='test' or just one row if predict?
            # train_eval_transformer saves list of dicts.
            # We assume last row is Test if available, or check 'split' column?
            # My script saves: row = {"split": "test", ...}
            if "split" in df.columns:
                test_row = df[df["split"] == "test"]
                if not test_row.empty:
                    test_row = test_row.iloc[0]
                else:
                    test_row = df.iloc[-1]
            else:
                test_row = df.iloc[-1]

            # Determine Model Name from filename
            model_name = "Transformer"
            lang = name
            if "xlm" in name:
                model_name = "XLM-RoBERTa (Chunked)"
                lang = name.replace("_xlm", "")
            elif "twitter" in name:
                model_name = "Twitter-RoBERTa (Chunked)"
                lang = name.replace("_twitter", "")
            
            all_results.append({
                "Model": model_name,
                "Language": lang,
                "RMSE": test_row["avg_rmse"],
                "MAE": test_row["avg_mae"],
                "Accuracy": test_row["avg_acc"],
                "Pearson": test_row["avg_pearson"]
            })
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_results:
        print("No results found.")
        return

    final_df = pd.DataFrame(all_results)
    final_df = final_df.sort_values(by=["Language", "Model"])
    
    output_path = RESULTS_DIR / "consolidated_final.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Saved consolidated results to {output_path}")
    print(final_df)

if __name__ == "__main__":
    main()
