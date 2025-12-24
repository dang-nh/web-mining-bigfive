import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd

from src.config import N_TWEETS_PER_USER
from src.utils.text import preprocess_tweets


def parse_user_xml(xml_path: Path) -> Tuple[str, List[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    user_id = root.attrib.get("id", xml_path.stem)
    tweets = []
    for doc in root.findall(".//document"):
        text = doc.text
        if text:
            tweets.append(text.strip())
    return user_id, tweets


def parse_truth_file(truth_path: Path) -> Dict[str, Dict[str, float]]:
    labels = {}
    with open(truth_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(":::")
            if len(parts) >= 7:
                user_id = parts[0]
                labels[user_id] = {
                    "gender": parts[1],
                    "age_group": parts[2],
                    "y_extroverted": float(parts[3]),
                    "y_stable": float(parts[4]),
                    "y_agreeable": float(parts[5]),
                    "y_conscientious": float(parts[6]),
                    "y_open": float(parts[7]) if len(parts) > 7 else 0.0,
                }
    return labels


def find_xml_files(data_dir: Path) -> List[Path]:
    xml_files = list(data_dir.rglob("*.xml"))
    xml_files = [f for f in xml_files if "truth" not in f.name.lower()]
    return xml_files


def find_truth_file(data_dir: Path) -> Optional[Path]:
    for pattern in ["truth.txt", "truth*.txt", "*truth*"]:
        matches = list(data_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def parse_pan15_dataset(
    data_dir: Path,
    max_tweets: int = N_TWEETS_PER_USER,
    preprocess: bool = True,
) -> pd.DataFrame:
    xml_files = find_xml_files(data_dir)
    truth_path = find_truth_file(data_dir)

    labels = {}
    if truth_path:
        labels = parse_truth_file(truth_path)

    records = []
    for xml_path in xml_files:
        user_id, tweets = parse_user_xml(xml_path)
        if preprocess:
            tweets = preprocess_tweets(tweets)
        tweets = tweets[:max_tweets]
        if not tweets:
            continue

        record = {
            "user_id": user_id,
            "lang": "en",
            "tweets": tweets,
            "text_concat": " ".join(tweets),
        }

        if user_id in labels:
            record.update(labels[user_id])
        else:
            record.update({
                "y_open": 0.0,
                "y_conscientious": 0.0,
                "y_extroverted": 0.0,
                "y_agreeable": 0.0,
                "y_stable": 0.0,
            })

        records.append(record)

    return pd.DataFrame(records)

