#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, RESULTS_DIR, SEED, TRAIT_NAMES
from src.utils.io import setup_logging, load_parquet
from src.utils.seed import set_seed
from src.recsys.hashtag_recsys import HashtagRecommender
from src.recsys.metrics import evaluate_recommender
from src.recsys.gnn_recsys import PersonalityLightGCN, GNNTrainer, PersonalitySimGCL, SimGCLTrainer
from src.recsys.sasrec import SASRec, SASRecTrainer, SequentialRecommender
from src.recsys.advanced_models import HashtagKGE, KGETrainer, HyperbolicGCN, HyperbolicGCNTrainer
import torch
import torch.optim as optim

def main():
    parser = argparse.ArgumentParser(description="Evaluate recommendation systems")
    parser.add_argument("--method", type=str, required=True, 
                        choices=["popularity", "content", "personality_rerank", "hybrid_cooc", 
                                 "enhanced_cooc", "rrf_ensemble",
                                 "gnn", "gnn_personality", "gnn_ensemble",
                                 "simgcl", "simgcl_personality", "simgcl_cooc_hybrid",
                                 "sasrec", "meta_ensemble",
                                 "kge", "hyperbolic_gcn", "ultimate_ensemble"],
                        help="Recommendation method to evaluate")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20], help="K values for evaluation")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for testing")
    parser.add_argument("--alpha", type=float, default=0.4, help="Alpha for personality rerank")
    parser.add_argument("--cooc_weight", type=float, default=0.0, help="Co-occurrence weight")
    args = parser.parse_args()

    logger = setup_logging(f"recsys_eval_{args.method}")
    set_seed(args.seed)

    # 1. Load Data
    logger.info("Loading RecSys dataset and profiles...")
    dataset_path = PROCESSED_DIR / "recsys_dataset.parquet"
    profiles_path = PROCESSED_DIR / "hashtag_profiles.parquet"
    
    # Checks handled by load_parquet
    pass

    df = load_parquet(dataset_path)
    profiles_df = load_parquet(profiles_path)
    
    # Filter for Test set? 
    # Usually we evaluate on Test set (unseen interactions or unseen users).
    # The dataset has "train_hashtags" and "test_hashtags" for each user.
    # We can evaluate on all users provided in the file (since they all have test splits).
    # But usually we report on Test Split Users separately? 
    # The request says: "Output results/metrics_recsys.csv". 
    # Let's use all users in the file (which might be filtered to only valid ones).
    # Optionally filter by split if desired, but "recsys_dataset" implies ready-to-test.
    
    if args.sample_size and args.sample_size < len(df):
        logger.info(f"Sampling {args.sample_size} users")
        df = df.sample(n=args.sample_size, random_state=args.seed)
    
    logger.info(f"Evaluating on {len(df)} users")

    # 2. Setup Recommender
    recommender = HashtagRecommender()
    recommender.load_profiles(profiles_df)
    
    # 3. Evaluation Loop
    results = []
    
    recommended_list = []
    relevant_list = []
    
    # Global counts needed for Popularity? 
    # load_profiles initializes global_hashtag_counts with dummy counts if needed.
    # But for "popularity" baseline, we might want the REAL counts from Training set.
    # The current `load_profiles` just counts the filtered_hashtags presence (which is 1 each), 
    # which breaks popularity baseline.
    # We should probably save/load global counts ??
    # OR re-compute from profiles_df if we saved count there? We didn't.
    # Workaround: For popularity, we need the counts.
    # Let's fix this by reloading PAN15 train or saving counts.
    # Checking `build_recsys_dataset`: it didn't save counts.
    # But wait, `p_h` (profiles) don't include counts.
    # Maybe we can just load the raw data to get counts again? Or we update build script to save counts.
    # As a quick fix for pure Popularity, we can re-compute global counts from `train_hashtags` column of ALL users 
    # in the dataset (or just Train users).
    # Better: Recalculate global counts from `train_hashtags` column in this dataframe.
    # This simulates "Popularity in the observed training data".
    
    all_train_hashtags = []
    for _, row in df.iterrows():
        all_train_hashtags.extend(row["train_hashtags"])
    
    from collections import Counter
    recommender.global_hashtag_counts = Counter(all_train_hashtags)
    # Also update `filtered_hashtags` to match loaded profiles 
    # (already done in load_profiles, but ensure consistency)
    
    # Build Co-occurrence Matrices for enhanced_cooc and rrf_ensemble methods
    logger.info("Building Co-occurrence Matrices...")
    pair_counts = {}
    item_counts = {}
    vocab = set(recommender.filtered_hashtags)
    
    for _, row in df.iterrows():
        user_tags = [h.lower() for h in row["train_hashtags"] if h.lower() in vocab]
        unique_tags = list(set(user_tags))
        
        for h in unique_tags:
            item_counts[h] = item_counts.get(h, 0) + 1
            
        for i in range(len(unique_tags)):
            for j in range(len(unique_tags)):
                if i == j: continue
                h_a = unique_tags[i]
                h_b = unique_tags[j]
                if h_a not in pair_counts: pair_counts[h_a] = {}
                pair_counts[h_a][h_b] = pair_counts[h_a].get(h_b, 0) + 1
    
    # First-order: P(B|A)
    recommender.cooccurrence_probs = {}
    for h_a, targets in pair_counts.items():
        recommender.cooccurrence_probs[h_a] = {}
        count_a = item_counts[h_a]
        for h_b, count_ab in targets.items():
            recommender.cooccurrence_probs[h_a][h_b] = count_ab / count_a
    
    # Second-order: P(C|A) = sum_B P(C|B) * P(B|A)
    recommender.second_order_cooc = {}
    for h_a in recommender.cooccurrence_probs:
        recommender.second_order_cooc[h_a] = {}
        for h_b, p_b_a in recommender.cooccurrence_probs[h_a].items():
            if h_b in recommender.cooccurrence_probs:
                for h_c, p_c_b in recommender.cooccurrence_probs[h_b].items():
                    if h_c != h_a and h_c != h_b:
                        transitive_prob = p_b_a * p_c_b
                        recommender.second_order_cooc[h_a][h_c] = recommender.second_order_cooc[h_a].get(h_c, 0) + transitive_prob
    
    logger.info(f"Built 1st-order co-occurrence for {len(recommender.cooccurrence_probs)} items, 2nd-order for {len(recommender.second_order_cooc)} items")
    
    # === GNN SETUP & TRAINING (if applicable) ===
    gnn_model = None
    gnn_adj = None
    user_to_idx = {}
    hashtag_to_idx = {}
    
    if args.method in ["gnn", "gnn_personality", "gnn_ensemble"]:
         # 1. Prepare Data
         user_to_idx = {uid: i for i, uid in enumerate(df["user_id"].unique())}
         hashtag_to_idx = {h: i for i, h in enumerate(recommender.filtered_hashtags)}
         
         all_users_meta = {} 
         
         train_interactions = []
         for _, row in df.iterrows():
             uid = row["user_id"]
             u_idx = user_to_idx[uid]
             if args.method == "gnn_personality":
                 traits = [row.get(f"y_{t}", 0.5) for t in TRAIT_NAMES]
                 all_users_meta[u_idx] = traits
             
             for h in row["train_hashtags"]:
                 h_lower = h.lower()
                 if h_lower in hashtag_to_idx:
                     train_interactions.append({
                         "user_mapping": u_idx,
                         "item_mapping": hashtag_to_idx[h_lower]
                     })
                     
         train_data_df = pd.DataFrame(train_interactions)
         logger.info(f"GNN Training Data: {len(train_data_df)} interactions")
         
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
         # Features
         user_feats = None
         if args.method == "gnn_personality":
             user_feats = torch.zeros(len(user_to_idx), 5)
             for u_idx, traits in all_users_meta.items():
                 user_feats[u_idx] = torch.tensor(traits)
             user_feats = user_feats.to(device)
         
         item_feats = None
         if profiles_df is not None and not profiles_df.empty and "embedding" in profiles_df.columns:
             try:
                 # Check if the first element is a list or string (parquet sometimes odd)
                 sample = profiles_df.iloc[0]["embedding"]
                 if isinstance(sample, np.ndarray) or isinstance(sample, list):
                      embs = np.stack(profiles_df["embedding"].values)
                 else:
                      # Falback if needed
                      embs = np.zeros((len(hashtag_to_idx), 64)) 
                 
                 item_feats = torch.tensor(embs, dtype=torch.float32).to(device)
             except Exception as e:
                 logger.warning(f"Could not load item content features: {e}")

         gnn_model = PersonalityLightGCN(
             num_users=len(user_to_idx),
             num_items=len(hashtag_to_idx),
             embedding_dim=64,
             n_layers=3,
             user_personality_features=user_feats,
             item_content_features=item_feats
         ).to(device)
         
         optimizer = optim.Adam(gnn_model.parameters(), lr=0.005, weight_decay=1e-4) # Slightly higher LR
         trainer = GNNTrainer(gnn_model, optimizer, device)
         
         gnn_adj = trainer.create_adj_matrix(
             train_data_df["user_mapping"].values,
             train_data_df["item_mapping"].values,
             len(user_to_idx),
             len(hashtag_to_idx)
         )
         
         logger.info("Training GNN...")
         epochs = 50 
         for epoch in range(epochs):
             loss = trainer.train_epoch(gnn_adj, train_data_df)
             if (epoch+1) % 10 == 0:
                 logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
                 
         gnn_model.eval()
         gnn_model.eval()
         with torch.no_grad():
             final_u_emb, final_i_emb = gnn_model(gnn_adj)

    # === CO-OCCURRENCE SETUP (for Ensemble) ===
    cooc_probs = {}
    if args.method == "gnn_ensemble":
        logger.info("Building Co-occurrence Matrix...")
        pair_counts = {}
        item_counts = {}
        
        # Use filtered vocabulary
        vocab = set(recommender.filtered_hashtags)
        
        for _, row in df.iterrows():
            user_tags = [h.lower() for h in row["train_hashtags"] if h.lower() in vocab]
            unique_tags = list(set(user_tags))
            
            for h in unique_tags:
                item_counts[h] = item_counts.get(h, 0) + 1
                
            for i in range(len(unique_tags)):
                for j in range(len(unique_tags)):
                    if i == j: continue
                    h_a = unique_tags[i]
                    h_b = unique_tags[j]
                    if h_a not in pair_counts: pair_counts[h_a] = {}
                    pair_counts[h_a][h_b] = pair_counts[h_a].get(h_b, 0) + 1
                    
        # Normalize P(B|A)
        for h_a, targets in pair_counts.items():
            cooc_probs[h_a] = {}
            count_a = item_counts[h_a]
            for h_b, count_ab in targets.items():
                cooc_probs[h_a][h_b] = count_ab / count_a
        logger.info(f"Co-occurrence Matrix built. Covered {len(cooc_probs)} items.")

    # === SimGCL SETUP (for simgcl methods) ===
    simgcl_model = None
    simgcl_u_emb = None
    simgcl_i_emb = None
    
    if args.method in ["simgcl", "simgcl_personality", "simgcl_cooc_hybrid", "meta_ensemble"]:
        logger.info("Setting up SimGCL model...")
        
        # Reuse user/item mappings from GNN block
        if not user_to_idx:
            user_to_idx = {uid: i for i, uid in enumerate(df["user_id"].unique())}
            hashtag_to_idx = {h: i for i, h in enumerate(recommender.filtered_hashtags)}
        
        # Prepare training data if not already done
        if 'train_data_df' not in dir() or train_data_df is None or train_data_df.empty:
            train_interactions = []
            all_users_meta = {}
            for _, row in df.iterrows():
                uid = row["user_id"]
                u_idx = user_to_idx[uid]
                traits = [row.get(f"y_{t}", 0.5) for t in TRAIT_NAMES]
                all_users_meta[u_idx] = traits
                for h in row["train_hashtags"]:
                    h_lower = h.lower()
                    if h_lower in hashtag_to_idx:
                        train_interactions.append({
                            "user_mapping": u_idx,
                            "item_mapping": hashtag_to_idx[h_lower]
                        })
            train_data_df = pd.DataFrame(train_interactions)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # User features (personality)
        user_feats = None
        if args.method == "simgcl_personality" or args.method == "meta_ensemble":
            user_feats = torch.zeros(len(user_to_idx), 5)
            for u_idx, traits in all_users_meta.items():
                user_feats[u_idx] = torch.tensor(traits)
            user_feats = user_feats.to(device)
        
        # Item features (content embeddings)
        item_feats = None
        if profiles_df is not None and "embedding" in profiles_df.columns:
            try:
                embs = np.stack(profiles_df["embedding"].values)
                item_feats = torch.tensor(embs, dtype=torch.float32).to(device)
            except Exception as e:
                logger.warning(f"Could not load item content features: {e}")
        
        simgcl_model = PersonalitySimGCL(
            num_users=len(user_to_idx),
            num_items=len(hashtag_to_idx),
            embedding_dim=64,
            n_layers=3,
            eps=0.1,
            user_personality_features=user_feats,
            item_content_features=item_feats
        ).to(device)
        
        optimizer = optim.Adam(simgcl_model.parameters(), lr=0.005, weight_decay=1e-4)
        trainer = SimGCLTrainer(simgcl_model, optimizer, device, tau=0.2, cl_weight=0.1)
        
        # Reuse or create adjacency matrix
        if gnn_adj is None:
            gnn_adj = trainer.create_adj_matrix(
                train_data_df["user_mapping"].values,
                train_data_df["item_mapping"].values,
                len(user_to_idx),
                len(hashtag_to_idx)
            )
        
        logger.info("Training SimGCL...")
        epochs = 50
        for epoch in range(epochs):
            loss, bpr_loss, cl_loss = trainer.train_epoch(gnn_adj, train_data_df)
            if (epoch+1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f} (BPR: {bpr_loss:.4f}, CL: {cl_loss:.4f})")
        
        simgcl_model.eval()
        with torch.no_grad():
            simgcl_u_emb, simgcl_i_emb = simgcl_model(gnn_adj, perturb=False)

    # === SASRec SETUP (for sasrec method) ===
    sasrec_recommender = None
    
    if args.method in ["sasrec", "meta_ensemble"]:
        logger.info("Setting up SASRec model...")
        
        if not hashtag_to_idx:
            hashtag_to_idx = {h: i for i, h in enumerate(recommender.filtered_hashtags)}
        
        # Prepare user sequences
        user_sequences = {}
        for _, row in df.iterrows():
            uid = row["user_id"]
            # Convert hashtags to indices
            seq = []
            for h in row["train_hashtags"]:
                h_lower = h.lower()
                if h_lower in hashtag_to_idx:
                    seq.append(hashtag_to_idx[h_lower] + 1)  # +1 for padding
            if len(seq) >= 2:
                user_sequences[uid] = seq
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        sasrec_model = SASRec(
            num_items=len(hashtag_to_idx),
            embedding_dim=64,
            max_seq_len=50,
            n_heads=2,
            n_layers=2,
            dropout=0.2
        ).to(device)
        
        optimizer = optim.Adam(sasrec_model.parameters(), lr=0.001)
        trainer = SASRecTrainer(sasrec_model, optimizer, device, max_seq_len=50)
        
        # Prepare training data
        inputs, targets, masks = trainer.prepare_sequences(user_sequences)
        
        logger.info(f"Training SASRec with {len(inputs)} sequences...")
        epochs = 30
        for epoch in range(epochs):
            loss = trainer.train_epoch(inputs, targets, masks)
            if (epoch+1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        sasrec_model.eval()
        sasrec_recommender = SequentialRecommender(hashtag_to_idx, sasrec_model, device)

    # === KGE SETUP (Knowledge Graph Embeddings) ===
    kge_model = None
    
    if args.method in ["kge", "ultimate_ensemble"]:
        logger.info("Setting up Knowledge Graph Embeddings...")
        
        if not user_to_idx:
            user_to_idx = {uid: i for i, uid in enumerate(df["user_id"].unique())}
            hashtag_to_idx = {h: i for i, h in enumerate(recommender.filtered_hashtags)}
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        kge_model = HashtagKGE(
            num_users=len(user_to_idx),
            num_hashtags=len(hashtag_to_idx),
            embedding_dim=64,
            num_relations=2,  # user-hashtag, hashtag-hashtag
            margin=1.0,
            use_rotation=True
        ).to(device)
        
        optimizer = optim.Adam(kge_model.parameters(), lr=0.001)
        trainer = KGETrainer(kge_model, optimizer, device)
        
        # Build user-hashtag interactions
        user_hashtag_interactions = {}
        for _, row in df.iterrows():
            uid = row["user_id"]
            if uid in user_to_idx:
                u_idx = user_to_idx[uid]
                user_hashtag_interactions[u_idx] = []
                for h in row["train_hashtags"]:
                    h_lower = h.lower()
                    if h_lower in hashtag_to_idx:
                        user_hashtag_interactions[u_idx].append(hashtag_to_idx[h_lower])
        
        # Build co-occurrence as KG relations (hashtag_idx -> hashtag_idx -> prob)
        cooc_matrix_idx = {}
        for h_a, targets in recommender.cooccurrence_probs.items():
            if h_a in hashtag_to_idx:
                a_idx = hashtag_to_idx[h_a]
                cooc_matrix_idx[a_idx] = {}
                for h_b, prob in targets.items():
                    if h_b in hashtag_to_idx:
                        cooc_matrix_idx[a_idx][hashtag_to_idx[h_b]] = prob
        
        # Build triplets
        triplets = trainer.build_kg_triplets(user_hashtag_interactions, cooc_matrix_idx, threshold=0.1)
        logger.info(f"Built {len(triplets)} KG triplets")
        
        logger.info("Training KGE...")
        epochs = 50
        for epoch in range(epochs):
            loss = trainer.train_epoch(triplets)
            if (epoch+1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        kge_model.eval()

    # === HYPERBOLIC GCN SETUP ===
    hgcn_model = None
    hgcn_u_emb = None
    hgcn_i_emb = None
    
    if args.method in ["hyperbolic_gcn", "ultimate_ensemble"]:
        logger.info("Setting up Hyperbolic GCN...")
        
        if not user_to_idx:
            user_to_idx = {uid: i for i, uid in enumerate(df["user_id"].unique())}
            hashtag_to_idx = {h: i for i, h in enumerate(recommender.filtered_hashtags)}
        
        # Prepare training data
        if 'train_data_df' not in dir() or train_data_df is None or len(train_data_df) == 0:
            train_interactions = []
            all_users_meta = {}
            for _, row in df.iterrows():
                uid = row["user_id"]
                u_idx = user_to_idx[uid]
                traits = [row.get(f"y_{t}", 0.5) for t in TRAIT_NAMES]
                all_users_meta[u_idx] = traits
                for h in row["train_hashtags"]:
                    h_lower = h.lower()
                    if h_lower in hashtag_to_idx:
                        train_interactions.append({
                            "user_mapping": u_idx,
                            "item_mapping": hashtag_to_idx[h_lower]
                        })
            train_data_df = pd.DataFrame(train_interactions)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Personality features
        user_feats = torch.zeros(len(user_to_idx), 5)
        for u_idx, traits in all_users_meta.items():
            user_feats[u_idx] = torch.tensor(traits)
        user_feats = user_feats.to(device)
        
        hgcn_model = HyperbolicGCN(
            num_users=len(user_to_idx),
            num_items=len(hashtag_to_idx),
            embedding_dim=64,
            n_layers=2,
            curvature=1.0,
            user_personality_features=user_feats
        ).to(device)
        
        optimizer = optim.Adam(hgcn_model.parameters(), lr=0.001)
        trainer = HyperbolicGCNTrainer(hgcn_model, optimizer, device, margin=0.5)
        
        # Create or reuse adjacency matrix
        if gnn_adj is None:
            gnn_adj = trainer.create_adj_matrix(
                train_data_df["user_mapping"].values,
                train_data_df["item_mapping"].values,
                len(user_to_idx),
                len(hashtag_to_idx)
            )
        
        logger.info("Training Hyperbolic GCN...")
        epochs = 50
        for epoch in range(epochs):
            loss = trainer.train_epoch(gnn_adj, train_data_df)
            if (epoch+1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        hgcn_model.eval()
        with torch.no_grad():
            hgcn_u_emb, hgcn_i_emb = hgcn_model(gnn_adj)

    logger.info(f"Running method: {args.method}")
    
    for _, row in df.iterrows():
        # Helpers
        user_history = row["train_hashtags"]
        target_set = set(row["test_hashtags"])
        
        user_text = row.get("text_concat", "") 
        # If text is empty (legacy), might need to fetch or skip. 
        # But build script puts it there.

        recs = []
        
        if args.method == "popularity":
            recs = recommender.recommend_popularity(
                exclude_hashtags=user_history,
                top_k=max(args.k)
            )
            
        elif args.method == "content":
            recs = recommender.recommend_content(
                user_text=user_text,
                user_history_hashtags=user_history,
                exclude_hashtags=user_history,
                top_k=max(args.k)
            )
            
        elif args.method == "personality_rerank":
            user_traits = {t: row.get(f"y_{t}", 0.5) for t in TRAIT_NAMES}
            recs = recommender.recommend_personality_aware(
                user_text=user_text,
                user_traits=user_traits,
                user_history_hashtags=user_history,
                exclude_hashtags=user_history,
                top_k=max(args.k),
                alpha=args.alpha,
                popularity_weight=0.2,
                keyword_weight=0.0,
                cooccurrence_weight=args.cooc_weight if args.cooc_weight > 0 else 0.0,
                use_mmr=True
            ) 
            
        elif args.method == "gnn" or args.method == "gnn_personality":
            # Prediction for the current user
            current_user_id = row["user_id"]
            if current_user_id not in user_to_idx:
                logger.warning(f"User {current_user_id} not found in GNN user mapping. Skipping.")
                recs = []
            else:
                u_idx = user_to_idx[current_user_id]
                
                # Get user embedding
                u_emb = final_u_emb[u_idx]
                
                # Calculate scores for all items
                scores = torch.matmul(u_emb, final_i_emb.T)
                
                # Convert scores to numpy and get top_k
                scores_np = scores.cpu().numpy()
                
                # Map item indices back to hashtag strings
                idx_to_hashtag = {i: h for h, i in hashtag_to_idx.items()}
                
                # Exclude hashtags from user history
                excluded_indices = {hashtag_to_idx[h.lower()] for h in user_history if h.lower() in hashtag_to_idx}
                
                # Set scores of excluded items to a very low value
                for ex_idx in excluded_indices:
                    scores_np[ex_idx] = -np.inf
                
                # Get top K item indices
                top_k_indices = np.argsort(scores_np)[::-1][:max(args.k)]
                
                # Convert indices to hashtags
                recs = [idx_to_hashtag[idx] for idx in top_k_indices if idx in idx_to_hashtag]

        elif args.method == "gnn_ensemble":
            # 1. Prediction for GNN part (Personality-Enhanced assumed trained)
            # Check if user exists
            current_user_id = row["user_id"]
            if current_user_id not in user_to_idx:
                recs = []
            else:
                u_idx = user_to_idx[current_user_id]
                u_emb = final_u_emb[u_idx]
                gnn_scores = torch.matmul(u_emb, final_i_emb.T).cpu().numpy()
                
                # Normalize GNN scores to [0, 1] for this user to make compatible with Probabilities
                # Robust Min-Max
                min_s = gnn_scores.min()
                max_s = gnn_scores.max()
                if max_s > min_s:
                    gnn_scores_norm = (gnn_scores - min_s) / (max_s - min_s)
                else:
                    gnn_scores_norm = gnn_scores
                
                # 2. Co-occurrence part
                # Score = sum P(item | hist_item)
                cooc_vector = np.zeros_like(gnn_scores)
                
                # Pre-map hashtag strings to indices
                # hashtag_to_idx is available from GNN block
                
                active_hist = [h.lower() for h in user_history if h.lower() in cooc_probs]
                
                for h_hist in active_hist:
                    # Get targets
                    targets = cooc_probs[h_hist]
                    for h_target, prob in targets.items():
                        if h_target in hashtag_to_idx:
                            idx = hashtag_to_idx[h_target]
                            cooc_vector[idx] += prob
                            
                # Average? Or Sum?
                # "Summation rewards more evidence". 
                # If we have 10 history items, max score could be 10.
                # GNN is 0..1.
                # Let's average cooc to keep it 0..1 range approximately?
                # Or just let alpha handle it.
                if len(active_hist) > 0:
                    cooc_vector /= len(active_hist)
                    
                # 3. Ensemble
                final_scores = gnn_scores_norm + (args.cooc_weight * cooc_vector)
                
                # Filter excluded
                excluded_indices = [hashtag_to_idx[h.lower()] for h in user_history if h.lower() in hashtag_to_idx]
                final_scores[excluded_indices] = -np.inf
                
                # Top K
                top_k_indices = np.argsort(final_scores)[::-1][:max(args.k)]
                idx_to_hashtag = {i: h for h, i in hashtag_to_idx.items()}
                recs = [idx_to_hashtag[idx] for idx in top_k_indices if idx in idx_to_hashtag]

        elif args.method == "hybrid_cooc":
             # Re-fit recommender on train data to get cooccurrence matrix
             user_traits = {t: row.get(f"y_{t}", 0.5) for t in TRAIT_NAMES}
             recs = recommender.recommend_personality_aware(
                 user_text=user_text,
                 user_traits=user_traits,
                 user_history_hashtags=user_history,
                 exclude_hashtags=user_history,
                 top_k=max(args.k),
                 alpha=args.alpha,
                 cooccurrence_weight=args.cooc_weight if args.cooc_weight > 0 else 1.0,
                 use_mmr=True
             )

        elif args.method == "enhanced_cooc":
            # Enhanced co-occurrence with 2nd order transitions
            recs = recommender.recommend_enhanced_cooc(
                user_history_hashtags=user_history,
                exclude_hashtags=user_history,
                top_k=max(args.k),
                second_order_weight=0.3
            )

        elif args.method == "rrf_ensemble":
            # Reciprocal Rank Fusion across multiple models
            user_traits = {t: row.get(f"y_{t}", 0.5) for t in TRAIT_NAMES}
            recs = recommender.recommend_rrf_ensemble(
                user_text=user_text,
                user_traits=user_traits,
                user_history_hashtags=user_history,
                exclude_hashtags=user_history,
                top_k=max(args.k),
                models=["content", "cooc", "enhanced_cooc", "personality"]
            )

        elif args.method in ["simgcl", "simgcl_personality"]:
            # SimGCL evaluation (using pre-trained model from GNN block if available)
            current_user_id = row["user_id"]
            if current_user_id not in user_to_idx:
                recs = []
            elif simgcl_model is None:
                recs = []
            else:
                u_idx = user_to_idx[current_user_id]
                u_emb = simgcl_u_emb[u_idx]
                scores = torch.matmul(u_emb, simgcl_i_emb.T)
                scores_np = scores.cpu().numpy()
                idx_to_hashtag = {i: h for h, i in hashtag_to_idx.items()}
                excluded_indices = {hashtag_to_idx[h.lower()] for h in user_history if h.lower() in hashtag_to_idx}
                for ex_idx in excluded_indices:
                    scores_np[ex_idx] = -np.inf
                top_k_indices = np.argsort(scores_np)[::-1][:max(args.k)]
                recs = [idx_to_hashtag[idx] for idx in top_k_indices if idx in idx_to_hashtag]

        elif args.method == "sasrec":
            # Sequential recommendation
            if sasrec_recommender is not None:
                recs = sasrec_recommender.recommend(
                    user_history=user_history,
                    exclude_hashtags=user_history,
                    top_k=max(args.k)
                )
            else:
                recs = []

        elif args.method == "meta_ensemble":
            # Meta-ensemble: RRF of GNN + SimGCL + Enhanced Cooc + SASRec
            rankings = []
            
            # 1. Enhanced Cooc ranking
            cooc_recs = recommender.recommend_enhanced_cooc(
                user_history_hashtags=user_history,
                exclude_hashtags=user_history,
                top_k=max(args.k) * 3
            )
            if cooc_recs:
                rankings.append([r[0] for r in cooc_recs])
            
            # 2. GNN ranking (if available)
            if gnn_model is not None and row["user_id"] in user_to_idx:
                u_idx = user_to_idx[row["user_id"]]
                u_emb = final_u_emb[u_idx]
                gnn_scores = torch.matmul(u_emb, final_i_emb.T).cpu().numpy()
                idx_to_ht = {i: h for h, i in hashtag_to_idx.items()}
                excluded = {hashtag_to_idx[h.lower()] for h in user_history if h.lower() in hashtag_to_idx}
                gnn_scores[list(excluded)] = -np.inf
                top_indices = np.argsort(gnn_scores)[::-1][:max(args.k) * 3]
                rankings.append([idx_to_ht[i] for i in top_indices if i in idx_to_ht])
            
            # 3. SimGCL ranking (if available)
            if simgcl_model is not None and row["user_id"] in user_to_idx:
                u_idx = user_to_idx[row["user_id"]]
                u_emb = simgcl_u_emb[u_idx]
                sim_scores = torch.matmul(u_emb, simgcl_i_emb.T).cpu().numpy()
                idx_to_ht = {i: h for h, i in hashtag_to_idx.items()}
                excluded = {hashtag_to_idx[h.lower()] for h in user_history if h.lower() in hashtag_to_idx}
                sim_scores[list(excluded)] = -np.inf
                top_indices = np.argsort(sim_scores)[::-1][:max(args.k) * 3]
                rankings.append([idx_to_ht[i] for i in top_indices if i in idx_to_ht])
            
            # 4. SASRec ranking (if available)
            if sasrec_recommender is not None:
                seq_recs = sasrec_recommender.recommend(
                    user_history=user_history,
                    exclude_hashtags=user_history,
                    top_k=max(args.k) * 3
                )
                if seq_recs:
                    rankings.append([r[0] for r in seq_recs])
            
            # RRF Fusion
            if rankings:
                fused = HashtagRecommender.reciprocal_rank_fusion(rankings)
                recs = fused[:max(args.k)]
            else:
                recs = []

        elif args.method == "simgcl_cooc_hybrid":
            # SimGCL + Enhanced Co-occurrence Hybrid (weighted sum like GNN ensemble)
            current_user_id = row["user_id"]
            if current_user_id not in user_to_idx or simgcl_model is None:
                recs = []
            else:
                u_idx = user_to_idx[current_user_id]
                u_emb = simgcl_u_emb[u_idx]
                
                # SimGCL scores
                simgcl_scores = torch.matmul(u_emb, simgcl_i_emb.T).cpu().numpy()
                
                # Normalize to 0-1
                min_s = simgcl_scores.min()
                max_s = simgcl_scores.max()
                if max_s > min_s:
                    simgcl_scores_norm = (simgcl_scores - min_s) / (max_s - min_s)
                else:
                    simgcl_scores_norm = simgcl_scores
                
                # Enhanced Co-occurrence scores (1st + 2nd order)
                cooc_vector = np.zeros_like(simgcl_scores)
                second_order_vector = np.zeros_like(simgcl_scores)
                
                for h_hist in user_history:
                    h_hist_lower = h_hist.lower()
                    # 1st order
                    if h_hist_lower in recommender.cooccurrence_probs:
                        for h_target, prob in recommender.cooccurrence_probs[h_hist_lower].items():
                            if h_target in hashtag_to_idx:
                                cooc_vector[hashtag_to_idx[h_target]] += prob
                    # 2nd order
                    if h_hist_lower in recommender.second_order_cooc:
                        for h_target, prob in recommender.second_order_cooc[h_hist_lower].items():
                            if h_target in hashtag_to_idx:
                                second_order_vector[hashtag_to_idx[h_target]] += prob
                
                # Normalize
                if len(user_history) > 0:
                    cooc_vector /= len(user_history)
                    second_order_vector /= len(user_history)
                
                # Combine: SimGCL + 1st-order + 0.3*2nd-order
                # args.cooc_weight controls the strength
                cooc_weight = args.cooc_weight if args.cooc_weight > 0 else 2.0
                enhanced_cooc = cooc_vector + 0.3 * second_order_vector
                final_scores = simgcl_scores_norm + (cooc_weight * enhanced_cooc)
                
                # Exclude history
                excluded_indices = [hashtag_to_idx[h.lower()] for h in user_history if h.lower() in hashtag_to_idx]
                final_scores[excluded_indices] = -np.inf
                
                # Top K
                top_k_indices = np.argsort(final_scores)[::-1][:max(args.k)]
                idx_to_hashtag = {i: h for h, i in hashtag_to_idx.items()}
                recs = [idx_to_hashtag[idx] for idx in top_k_indices if idx in idx_to_hashtag]

        elif args.method == "kge":
            # Knowledge Graph Embeddings
            current_user_id = row["user_id"]
            if current_user_id not in user_to_idx or kge_model is None:
                recs = []
            else:
                u_idx = user_to_idx[current_user_id]
                scores = kge_model.get_user_hashtag_scores(u_idx, relation_idx=0)
                
                idx_to_hashtag = {i: h for h, i in hashtag_to_idx.items()}
                excluded_indices = {hashtag_to_idx[h.lower()] for h in user_history if h.lower() in hashtag_to_idx}
                
                for ex_idx in excluded_indices:
                    scores[ex_idx] = -np.inf
                
                top_k_indices = np.argsort(scores)[::-1][:max(args.k)]
                recs = [idx_to_hashtag[idx] for idx in top_k_indices if idx in idx_to_hashtag]

        elif args.method == "hyperbolic_gcn":
            # Hyperbolic GCN
            current_user_id = row["user_id"]
            if current_user_id not in user_to_idx or hgcn_model is None:
                recs = []
            else:
                u_idx = user_to_idx[current_user_id]
                
                # Use hyperbolic distance for scoring
                from src.recsys.advanced_models import HyperbolicMath
                with torch.no_grad():
                    u_emb = hgcn_u_emb[u_idx:u_idx+1]  # (1, dim)
                    scores = -HyperbolicMath.hyperbolic_distance(
                        u_emb.unsqueeze(1),
                        hgcn_i_emb.unsqueeze(0),
                        hgcn_model.c
                    ).squeeze()
                    scores_np = scores.cpu().numpy()
                
                idx_to_hashtag = {i: h for h, i in hashtag_to_idx.items()}
                excluded_indices = {hashtag_to_idx[h.lower()] for h in user_history if h.lower() in hashtag_to_idx}
                
                for ex_idx in excluded_indices:
                    scores_np[ex_idx] = -np.inf
                
                top_k_indices = np.argsort(scores_np)[::-1][:max(args.k)]
                recs = [idx_to_hashtag[idx] for idx in top_k_indices if idx in idx_to_hashtag]

        elif args.method == "ultimate_ensemble":
            # Ultimate ensemble: All models combined with RRF
            rankings = []
            idx_to_ht = {i: h for h, i in hashtag_to_idx.items()}
            excluded = {hashtag_to_idx[h.lower()] for h in user_history if h.lower() in hashtag_to_idx}
            
            # 1. Enhanced Co-occurrence
            cooc_recs = recommender.recommend_enhanced_cooc(
                user_history_hashtags=user_history,
                exclude_hashtags=user_history,
                top_k=max(args.k) * 3
            )
            if cooc_recs:
                rankings.append([r[0] for r in cooc_recs])
            
            # 2. SimGCL (if available)
            if simgcl_model is not None and row["user_id"] in user_to_idx:
                u_idx = user_to_idx[row["user_id"]]
                sim_scores = torch.matmul(simgcl_u_emb[u_idx], simgcl_i_emb.T).cpu().numpy()
                sim_scores[list(excluded)] = -np.inf
                top_indices = np.argsort(sim_scores)[::-1][:max(args.k) * 3]
                rankings.append([idx_to_ht[i] for i in top_indices if i in idx_to_ht])
            
            # 3. KGE (if available)
            if kge_model is not None and row["user_id"] in user_to_idx:
                u_idx = user_to_idx[row["user_id"]]
                kge_scores = kge_model.get_user_hashtag_scores(u_idx, relation_idx=0)
                kge_scores[list(excluded)] = -np.inf
                top_indices = np.argsort(kge_scores)[::-1][:max(args.k) * 3]
                rankings.append([idx_to_ht[i] for i in top_indices if i in idx_to_ht])
            
            # 4. Hyperbolic GCN (if available)
            if hgcn_model is not None and row["user_id"] in user_to_idx:
                u_idx = user_to_idx[row["user_id"]]
                from src.recsys.advanced_models import HyperbolicMath
                with torch.no_grad():
                    u_emb = hgcn_u_emb[u_idx:u_idx+1]
                    hgcn_scores = -HyperbolicMath.hyperbolic_distance(
                        u_emb.unsqueeze(1),
                        hgcn_i_emb.unsqueeze(0),
                        hgcn_model.c
                    ).squeeze().cpu().numpy()
                hgcn_scores[list(excluded)] = -np.inf
                top_indices = np.argsort(hgcn_scores)[::-1][:max(args.k) * 3]
                rankings.append([idx_to_ht[i] for i in top_indices if i in idx_to_ht])
            
            # RRF Fusion
            if rankings:
                fused = HashtagRecommender.reciprocal_rank_fusion(rankings)
                recs = fused[:max(args.k)]
            else:
                recs = []


        recommended_list.append(recs)
        relevant_list.append(target_set)

    # 4. Compute Metrics
    metrics = evaluate_recommender(recommended_list, relevant_list, k_values=args.k)
    metrics["method"] = args.method
    if args.method == "personality_rerank":
        metrics["alpha"] = args.alpha
        
    # Print and Save
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

    # Append to CSV
    output_path = RESULTS_DIR / "metrics_recsys.csv"
    results_df = pd.DataFrame([metrics])
    
    if output_path.exists():
        results_df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(output_path, index=False)
        
    logger.info(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()


