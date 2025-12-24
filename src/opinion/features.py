from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import entropy
from tqdm import tqdm

from src.config import SENTIMENT_MODEL, EMOTION_MODEL


class OpinionExtractor:
    def __init__(
        self,
        sentiment_model: str = SENTIMENT_MODEL,
        emotion_model: str = EMOTION_MODEL,
        device: str = None,
        batch_size: int = 32,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model)
        self.sentiment_model.to(self.device)
        self.sentiment_model.eval()

        self.emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model)
        self.emotion_model.to(self.device)
        self.emotion_model.eval()

        self.sentiment_labels = ["negative", "neutral", "positive"]
        self.emotion_labels = ["anger", "joy", "optimism", "sadness"]

    def _get_probs(
        self,
        texts: List[str],
        tokenizer,
        model,
    ) -> np.ndarray:
        all_probs = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def extract_sentiment(self, tweets: List[str]) -> Dict[str, float]:
        if not tweets:
            return {
                "sent_negative": 0.0,
                "sent_neutral": 0.0,
                "sent_positive": 0.0,
                "sent_entropy": 0.0,
                "sent_pos_rate": 0.0,
                "sent_neg_rate": 0.0,
            }

        probs = self._get_probs(tweets, self.sentiment_tokenizer, self.sentiment_model)
        mean_probs = probs.mean(axis=0)

        preds = probs.argmax(axis=1)
        pos_rate = (preds == 2).mean()
        neg_rate = (preds == 0).mean()

        return {
            "sent_negative": float(mean_probs[0]),
            "sent_neutral": float(mean_probs[1]),
            "sent_positive": float(mean_probs[2]),
            "sent_entropy": float(entropy(mean_probs + 1e-10)),
            "sent_pos_rate": float(pos_rate),
            "sent_neg_rate": float(neg_rate),
        }

    def extract_emotion(self, tweets: List[str]) -> Dict[str, float]:
        if not tweets:
            return {f"emo_{label}": 0.0 for label in self.emotion_labels + ["entropy"]}

        probs = self._get_probs(tweets, self.emotion_tokenizer, self.emotion_model)
        mean_probs = probs.mean(axis=0)

        features = {}
        for i, label in enumerate(self.emotion_labels):
            features[f"emo_{label}"] = float(mean_probs[i])

        features["emo_entropy"] = float(entropy(mean_probs + 1e-10))

        return features

    def extract_all(self, tweets: List[str]) -> Dict[str, float]:
        features = {}
        features.update(self.extract_sentiment(tweets))
        features.update(self.extract_emotion(tweets))
        return features


def extract_opinion_features(
    df: pd.DataFrame,
    tweet_col: str = "tweets",
    extractor: OpinionExtractor = None,
) -> pd.DataFrame:
    if extractor is None:
        extractor = OpinionExtractor()

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting opinion features"):
        tweets = row[tweet_col]
        if isinstance(tweets, str):
            tweets = [tweets]

        features = extractor.extract_all(tweets)
        features["user_id"] = row["user_id"]
        records.append(features)

    return pd.DataFrame(records)

