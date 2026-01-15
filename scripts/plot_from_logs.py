
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import RESULTS_DIR

def parse_logs_and_plot(log_file, suffix):
    print(f"Parsing {log_file}...")
    data = []
    with open(log_file, "r") as f:
        for line in f:
            if "Epoch " in line and "eval_rmse=" in line:
                # Format: Epoch 4: train_loss=0.0242, eval_loss=0.0214, eval_rmse=0.1464, eval_mae=0.1179, eval_acc=0.5000, pearson_r=0.6000, lr=2.00e-05
                try:
                    epoch = int(re.search(r"Epoch (\d+):", line).group(1))
                    train_loss = float(re.search(r"train_loss=([0-9.]+)", line).group(1))
                    eval_loss = float(re.search(r"eval_loss=([0-9.]+)", line).group(1))
                    eval_rmse = float(re.search(r"eval_rmse=([0-9.]+)", line).group(1))
                    eval_max = float(re.search(r"eval_mae=([0-9.]+)", line).group(1))
                    eval_acc = float(re.search(r"eval_acc=([0-9.]+)", line).group(1))
                    pearson_r = float(re.search(r"pearson_r=([0-9.-]+)", line).group(1))
                    
                    data.append({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "eval_loss": eval_loss,
                        "eval_rmse": eval_rmse,
                        "eval_acc": eval_acc,
                        "eval_pearson": pearson_r
                    })
                except Exception as e:
                    print(f"Skipping line: {line.strip()} | Error: {e}")

    if not data:
        print("No data found.")
        return

    df = pd.DataFrame(data)
    csv_path = RESULTS_DIR / f"learning_curve_{suffix}.csv"
    df.to_csv(csv_path, index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = df["epoch"].values
    
    # Loss
    ax1 = axes[0]
    ax1.plot(epochs, df["train_loss"], "b-", label="Train Loss", marker="o")
    ax1.plot(epochs, df["eval_loss"], "r-", label="Eval Loss", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title(f"Loss Curves ({suffix})")
    ax1.grid(True, alpha=0.3)
    
    # Metrics
    ax2 = axes[1]
    ax2.plot(epochs, df["eval_rmse"], "g-", label="RMSE", marker="^")
    ax2.plot(epochs, df["eval_acc"], "m-", label="Accuracy", marker="d")
    ax2.plot(epochs, df["eval_pearson"], "c-", label="Pearson", marker="x")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.set_title(f"Metrics ({suffix})")
    ax2.grid(True, alpha=0.3)
    
    png_path = RESULTS_DIR / f"learning_curve_{suffix}.png"
    plt.savefig(png_path, bbox_inches="tight")
    print(f"Saved {png_path}")

if __name__ == "__main__":
    parse_logs_and_plot("logs_twitter.txt", "en_chunk_twitter")
    parse_logs_and_plot("logs_xlm.txt", "en_chunk_xlm")
