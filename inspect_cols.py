
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.utils.io import load_parquet
from src.config import PROCESSED_DIR

def main():
    try:
        df = load_parquet(PROCESSED_DIR / "recsys_dataset.parquet")
        print("Columns:", df.columns.tolist())
        print("Sample:", df.iloc[0])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
