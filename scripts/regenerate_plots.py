
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer_regressor import TransformerTrainer
from src.config import MODELS_DIR

languages = ["en", "es", "it", "nl"]

def main():
    print("Regenerating learning curve plots...")
    
    for lang in languages:
        model_path = MODELS_DIR / f"transformer_{lang}.pt"
        if model_path.exists():
            print(f"Loading {lang} model from {model_path}...")
            try:
                # Load trainer (instantiation inside load needs model name, which is in checkpoint)
                trainer = TransformerTrainer.load(model_path)
                
                # Check history
                if not trainer.training_history:
                    print(f"  Warning: No training history found for {lang}.")
                    continue
                
                # Plot
                trainer._save_learning_curves(suffix=lang)
                print(f"  Saved plots for {lang}.")
                
            except Exception as e:
                print(f"  Error processing {lang}: {e}")
        else:
            print(f"Model for {lang} not found at {model_path}")

if __name__ == "__main__":
    main()
