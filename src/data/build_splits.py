from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TEST_SIZE, DEV_SIZE, SEED


def create_splits(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    dev_size: float = DEV_SIZE,
    seed: int = SEED,
) -> Tuple[List[str], List[str], List[str]]:
    user_ids = df["user_id"].unique().tolist()

    train_dev_ids, test_ids = train_test_split(
        user_ids,
        test_size=test_size,
        random_state=seed,
    )

    dev_ratio = dev_size / (1 - test_size)
    train_ids, dev_ids = train_test_split(
        train_dev_ids,
        test_size=dev_ratio,
        random_state=seed,
    )

    return train_ids, dev_ids, test_ids


def get_split_dataframes(
    df: pd.DataFrame,
    train_ids: List[str],
    dev_ids: List[str],
    test_ids: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["user_id"].isin(train_ids)].copy()
    dev_df = df[df["user_id"].isin(dev_ids)].copy()
    test_df = df[df["user_id"].isin(test_ids)].copy()
    return train_df, dev_df, test_df

