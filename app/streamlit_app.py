#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    TRAIT_NAMES,
    TRAIT_COLS,
    TOP_K_EVIDENCE,
    TOP_K_RECS,
)
from src.utils.text import preprocess_tweets
from src.models.tfidf_ridge import TfidfRidgeModel
from src.ir.bm25 import BM25Index, build_tweet_index
from src.ir.evidence import retrieve_evidence_for_user, TRAIT_QUERIES
from src.ir.chroma_store import ChromaUserStore
from src.rag.explain import get_explainer
from src.recsys.hashtag_recsys import HashtagRecommender

st.set_page_config(
    page_title="Big Five Personality Analyzer",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .trait-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .evidence-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #6366f1;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_tfidf_model():
    model_path = MODELS_DIR / "tfidf_ridge.pkl"
    if model_path.exists():
        return TfidfRidgeModel.load(model_path)
    return None


@st.cache_resource
def load_chroma_store():
    try:
        store = ChromaUserStore()
        store.load_collection()
        return store
    except Exception:
        return None


@st.cache_resource
def load_recommender():
    try:
        df = pd.read_parquet(PROCESSED_DIR / "pan15_en.parquet")
        recommender = HashtagRecommender()
        recommender.fit(df)
        return recommender
    except Exception:
        return None


def create_radar_chart(traits: dict):
    categories = list(traits.keys())
    values = list(traits.values())
    values += values[:1]

    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, values, 'o-', linewidth=2, color='#6366f1')
    ax.fill(angles, values, alpha=0.25, color='#6366f1')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([t.capitalize() for t in categories], size=12)
    ax.set_ylim(0, 1)

    ax.set_title("Big Five Personality Profile", size=14, weight='bold', pad=20)

    return fig


def build_temp_index(tweets: list) -> BM25Index:
    documents = [
        {"doc_id": f"temp_{i}", "user_id": "temp_user", "text": t, "tweet_idx": i}
        for i, t in enumerate(tweets)
    ]
    index = BM25Index()
    index.build(documents, text_key="text")
    return index


def retrieve_temp_evidence(index: BM25Index, top_k: int = 5) -> dict:
    evidence = {}
    for trait in TRAIT_NAMES:
        query = TRAIT_QUERIES[trait]
        results = index.search(query, top_k=top_k, user_id="temp_user")
        evidence[trait] = [
            {"tweet": doc["text"], "score": score}
            for doc, score in results
        ]
    return evidence


def main():
    st.markdown('<h1 class="main-header">🧠 Big Five Personality Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("Analyze personality traits from social media posts using ML + RAG")

    with st.sidebar:
        st.header("⚙️ Settings")

        model_type = st.selectbox(
            "Prediction Model",
            ["TF-IDF + Ridge (Fast)", "Transformer (if available)"],
            index=0,
        )

        top_k_evidence = st.slider("Evidence per trait", 3, 10, TOP_K_EVIDENCE)
        top_k_recs = st.slider("Recommendations", 5, 20, TOP_K_RECS)

        st.divider()
        st.markdown("### About")
        st.markdown("""
        This app predicts **Big Five** personality traits:
        - **O**penness
        - **C**onscientiousness
        - **E**xtraversion
        - **A**greeableness
        - **S**tability (Emotional)

        *Note: 'Stable' = Emotional Stability (inverse of Neuroticism)*
        """)

    tab1, tab2 = st.tabs(["📝 Paste Text", "📁 Upload File"])

    with tab1:
        st.markdown("### Enter posts (one per line)")
        input_text = st.text_area(
            "Posts",
            height=200,
            placeholder="Enter tweets or posts here, one per line...\n\nExample:\nJust finished reading an amazing book about philosophy!\nHad a great time at the party with friends tonight 🎉\nNeed to organize my schedule for next week...",
        )

    with tab2:
        uploaded_file = st.file_uploader(
            "Upload a text file (.txt) or CSV",
            type=["txt", "csv"],
        )
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                file_df = pd.read_csv(uploaded_file)
                if "text" in file_df.columns:
                    input_text = "\n".join(file_df["text"].dropna().tolist())
                else:
                    input_text = "\n".join(file_df.iloc[:, 0].dropna().tolist())
            else:
                input_text = uploaded_file.read().decode("utf-8")

    if st.button("🔍 Analyze Personality", type="primary", use_container_width=True):
        if not input_text or not input_text.strip():
            st.error("Please enter some text or upload a file.")
            return

        tweets = [t.strip() for t in input_text.strip().split("\n") if t.strip()]
        tweets = preprocess_tweets(tweets)

        if len(tweets) < 1:
            st.error("Please provide at least one post.")
            return

        with st.spinner("Analyzing personality..."):
            model = load_tfidf_model()
            if model is None:
                st.error("Model not found. Please run training scripts first.")
                return

            text_concat = " ".join(tweets)
            predictions = model.predict(pd.Series([text_concat]))[0]
            predicted_traits = {
                trait: float(np.clip(predictions[i], 0, 1))
                for i, trait in enumerate(TRAIT_NAMES)
            }

            temp_index = build_temp_index(tweets)
            evidence = retrieve_temp_evidence(temp_index, top_k=top_k_evidence)

            chroma_store = load_chroma_store()
            similar_users = None
            if chroma_store:
                try:
                    similar_users = chroma_store.get_similar_users(text_concat, top_n=3)
                except Exception:
                    pass

            explainer = get_explainer()
            explanation = explainer.explain(
                predicted_traits=predicted_traits,
                evidence=evidence,
                similar_users=similar_users,
                user_text=text_concat,
            )

            recommender = load_recommender()
            recommendations = []
            if recommender:
                try:
                    recommendations = recommender.recommend_personality_aware(
                        text_concat,
                        predicted_traits,
                        top_k=top_k_recs,
                    )
                except Exception:
                    recommendations = recommender.recommend_popularity(top_k=top_k_recs)

        st.success("Analysis complete!")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Personality Profile")
            fig = create_radar_chart(predicted_traits)
            st.pyplot(fig)

            st.subheader("📈 Trait Scores")
            for trait, score in predicted_traits.items():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(score)
                with col_b:
                    st.write(f"**{trait.capitalize()}**: {score:.2f}")

        with col2:
            st.subheader("📑 Evidence by Trait")

            for trait in TRAIT_NAMES:
                with st.expander(f"**{trait.upper()}** (score: {predicted_traits[trait]:.2f})"):
                    trait_evidence = evidence.get(trait, [])
                    if trait_evidence:
                        for i, item in enumerate(trait_evidence[:3]):
                            st.markdown(f"""
                            <div class="evidence-card">
                                <strong>#{i+1}</strong> (relevance: {item['score']:.3f})<br>
                                {item['tweet']}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No evidence found for this trait.")

        st.divider()

        col3, col4 = st.columns([1, 1])

        with col3:
            st.subheader("💡 AI Explanation")

            if "trait_explanations" in explanation:
                for trait, exp in explanation["trait_explanations"].items():
                    st.markdown(f"**{trait.capitalize()}:** {exp}")

            if "overall_summary" in explanation:
                st.info(explanation["overall_summary"])

            if "caveats" in explanation:
                with st.expander("⚠️ Caveats"):
                    for caveat in explanation["caveats"]:
                        st.markdown(f"- {caveat}")

        with col4:
            st.subheader("🏷️ Recommended Hashtags")

            if recommendations:
                hashtag_cols = st.columns(2)
                for i, (hashtag, score) in enumerate(recommendations[:top_k_recs]):
                    with hashtag_cols[i % 2]:
                        st.markdown(f"`#{hashtag}` ({score:.3f})")
            else:
                st.info("No recommendations available. Train the recommender first.")

        with st.expander("🔧 Raw JSON Output"):
            st.json(explanation)


if __name__ == "__main__":
    main()

