
#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, MODELS_DIR, TRAIT_NAMES
from src.utils.io import setup_logging, load_parquet, save_parquet
from src.models.transformer_regressor import TransformerTrainer

def main():
    logger = setup_logging("generate_recsys_predictions")
    
    # 1. Load RecSys Dataset
    # Try parquet first, then pkl
    dataset_path = PROCESSED_DIR / "recsys_dataset.parquet"
    if not dataset_path.exists():
        dataset_path = PROCESSED_DIR / "recsys_dataset.pkl"
        
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return
        
    logger.info(f"Loading dataset from {dataset_path}...")
    df = load_parquet(dataset_path)
    
    # 2. Load Model (English Twitter)
    model_path = MODELS_DIR / "transformer_en_twitter.pt"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    logger.info(f"Loading model from {model_path}...")
    
    # Load checkpoint
    try:
        # Use CPU to avoid OOM if VRAM is tight, or small batch on GPU
        device = "cpu"
        # Reduce batch size for safety
        batch_size = 16 
        
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        trainer = TransformerTrainer(
            model_name=checkpoint["model_name"],
            learning_rate=checkpoint["learning_rate"],
            batch_size=batch_size,
            max_length=checkpoint["max_length"],
            device=device
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.to(trainer.device)
        trainer.model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 3. Predict with Explicit Batching
    logger.info("Generating predictions...")
    if "text_concat" not in df.columns:
        logger.error("Column 'text_concat' missing from dataset")
        return
        
    texts = df["text_concat"].tolist()
    
    # Process in chunks to avoid any internal OOM in tokenization or result aggregation
    all_preds = []
    chunk_size = 128 # Process 128 texts at a time (trainer handles internal batching too, but this is safer)
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        chunk = texts[i : i + chunk_size]
        try:
            # trainer.predict typically handles a list, but let's feed smaller lists
            chunk_preds = trainer.predict(chunk)
            all_preds.append(chunk_preds)
        except Exception as e:
            logger.error(f"Error predicting chunk {i}: {e}")
            # Fallback/Empty or stop
            return 
            
    preds = np.vstack(all_preds)
    
    # 4. Add to DataFrame
    logger.info("Adding predictions to dataframe...")
    for i, trait in enumerate(TRAIT_NAMES):
        col_name = f"pred_{trait}"
        df[col_name] = preds[:, i]
        df[col_name] = df[col_name].clip(0.0, 1.0)
        
    # 5. Save
    output_path = PROCESSED_DIR / "recsys_dataset_with_preds.parquet"
    save_parquet(df, output_path)
    logger.info(f"Saved dataset with predictions to {output_path}")
    
    # Show comparison for first user
    row = df.iloc[0]
    logger.info("Sample comparison (User 0):")
    for trait in TRAIT_NAMES:
        true_val = row.get(f"y_{trait}", 0.0)
        pred_val = row.get(f"pred_{trait}", 0.0)
        logger.info(f"  {trait}: True={true_val:.4f}, Pred={pred_val:.4f}")

if __name__ == "__main__":
    main()
