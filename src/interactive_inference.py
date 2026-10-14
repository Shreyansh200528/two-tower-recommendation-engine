"""
Interactive Recommendation Inference Script with Production Cold-Start Fallback.
Handles:
1. Known Users: Full Two-Stage Pipeline (Two-Tower User Tower + FAISS HNSW + LightGBM Re-Ranker)
2. Unknown Users with Demographics (Tier 1): Demographic-Cluster Vector Retrieval
3. Anonymous Unknown Users (Tier 2): Bayesian Popularity Fallback (IMDb Weighted Rating)

Usage:
    # Known user
    python src/interactive_inference.py --user_id 42

    # Cold-start Tier 1 (Demographic Warm Fallback)
    python src/interactive_inference.py --user_id 9999 --gender F --age 18-24

    # Cold-start Tier 2 (Pure Anonymous Bayesian Fallback)
    python src/interactive_inference.py --user_id 9999
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import faiss
import lightgbm as lgb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import MovieLens1MPreprocessor
from two_tower_model import TwoTowerModel
from reranker import LightGBMReranker
from cold_start import BayesianPopularityRanker, DemographicColdStartHandler

def main():
    parser = argparse.ArgumentParser(description="Query recommendations with Cold-Start Fallback.")
    parser.add_argument("--user_id", type=int, default=42, help="User ID (e.g., 1 to 6040, or 9999 for cold start)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of final recommendations")
    parser.add_argument("--gender", type=str, default=None, help="Demographic gender: 'M' or 'F' (for cold-start)")
    parser.add_argument("--age", type=str, default=None, help="Demographic age group: '<18', '18-24', '25-34', '35-44', etc.")
    parser.add_argument("--occupation", type=int, default=0, help="Demographic occupation code (0 to 20)")
    parser.add_argument("--allow_seen", action="store_true", help="Allow recommending already-watched movies (default: False)")
    args = parser.parse_args()

    prep = MovieLens1MPreprocessor("data/ml-1m").process()
    train_df, test_df = prep.temporal_train_test_split(test_ratio=0.2)

    # Initialize Bayesian Ranker
    bayesian_ranker = BayesianPopularityRanker(train_df, prep.movies_meta)

    # -------------------------------------------------------------
    # CASE A: Unknown User (Cold-Start Fallback Handlers)
    # -------------------------------------------------------------
    if args.user_id not in prep.user2idx:
        print("=" * 75)
        print(f" [TRAFFIC FALLBACK TRIGGERED] User ID {args.user_id} not found in historical database.")

        # Check if cold-start demographics are supplied (Tier 1)
        if args.gender is not None or args.age is not None:
            g_val = args.gender if args.gender is not None else "M"
            a_val = args.age if args.age is not None else "25-34"
            print(" MODE: [Tier 1 Cold-Start: Demographic Cluster Retrieval]")
            print(f" Supplied Demographics: Gender: {g_val}, Age: {a_val}, Occupation Code: {args.occupation}")
            print("=" * 75)

            # Load model and FAISS index
            model = TwoTowerModel(prep.num_users, prep.num_movies, num_genres=18, embedding_dim=64)
            model.load_state_dict(torch.load("models/two_tower.pt", map_location="cpu"))
            model.eval()
            faiss_index = faiss.read_index("models/faiss_hnsw.index")

            handler = DemographicColdStartHandler(
                user_tower=model.user_tower,
                faiss_index=faiss_index,
                movies_meta=prep.movies_meta,
                bayesian_ranker=bayesian_ranker
            )
            recs = handler.recommend(gender_str=g_val, age_str=a_val, occupation=args.occupation, top_k=args.top_k)

            print(f"\n[Top-{args.top_k} Demographic Cluster Recommendations (Two-Tower Mean User + FAISS)]")
            print(f"{'Rank':<5} | {'Movie Title':<42} | {'Genres':<22} | {'Score':<10}")
            print("-" * 88)
            for r in recs:
                print(f"{r['rank']:<5} | {r['title'][:40]:<42} | {r['genres'][:20]:<22} | {r['score']:<10.4f}")
            print("=" * 75)
            return

        # Pure anonymous cold-start (Tier 2)
        else:
            print(" MODE: [Tier 2 Cold-Start: Pure Anonymous Bayesian Popularity Fallback]")
            print(" Formula: IMDb Bayesian Weighted Rating (m = 80th percentile minimum votes)")
            print("=" * 75)

            recs = bayesian_ranker.get_top_k(top_k=args.top_k)
            print(f"\n[Top-{args.top_k} Universal Classics (Bayesian Popularity Fallback)]")
            print(f"{'Rank':<5} | {'Movie Title':<42} | {'Genres':<22} | {'Bayesian Score':<15}")
            print("-" * 88)
            for r in recs:
                print(f"{r['rank']:<5} | {r['title'][:40]:<42} | {r['genres'][:20]:<22} | {r['score']:<15.4f}")
            print("=" * 75)
            return

    # -------------------------------------------------------------
    # CASE B: Known User (Standard Two-Stage Pipeline)
    # -------------------------------------------------------------
    user_idx = prep.user2idx[args.user_id]
    u_feat = prep.user_features[user_idx]
    
    gender = "Female" if u_feat[1].item() == 1 else "Male"
    age_groups = ["<18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
    age_str = age_groups[u_feat[2].item()]

    print("=" * 65)
    print(f" USER PROFILE (ID: {args.user_id} | Index: {user_idx})")
    print(f" Demographics: {gender}, Age Group: {age_str}")
    print("=" * 65)

    user_seen_movies = set(train_df[train_df["user_idx"] == user_idx]["movie_idx"].values)
    user_history = train_df[train_df["user_idx"] == user_idx].sort_values("rating", ascending=False).head(5)
    user_history = user_history.merge(prep.movies_meta, on="movie_idx")
    print(f"\n[User's Top 5 Historical Favorites in Training Data ({len(user_seen_movies)} total watched)]")
    for _, r in user_history.iterrows():
        print(f"  * {r['title']} [{r['genres']}] - {r['rating']} Stars")

    model = TwoTowerModel(prep.num_users, prep.num_movies, num_genres=18, embedding_dim=64)
    model.load_state_dict(torch.load("models/two_tower.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        u_emb = model.user_tower(u_feat.unsqueeze(0)).cpu().numpy().astype(np.float32)

    index = faiss.read_index("models/faiss_hnsw.index")
    fetch_k = 100 + (len(user_seen_movies) if not args.allow_seen else 0)
    sim_scores, cand_indices = index.search(u_emb, min(fetch_k, prep.num_movies))
    cand_indices = cand_indices[0]
    sim_scores = sim_scores[0]

    # Deduplication / Seen Filter: Exclude already-watched movies
    if not args.allow_seen:
        unseen_mask = [m_idx not in user_seen_movies for m_idx in cand_indices]
        cand_indices = [m for m, keep in zip(cand_indices, unseen_mask) if keep]
        sim_scores = [s for s, keep in zip(sim_scores, unseen_mask) if keep]

    reranker = LightGBMReranker()
    reranker.extract_historical_stats(train_df, prep.movies_meta)
    reranker.load("models/lightgbm_reranker.txt")

    # Use unified reranker.rerank() method
    final_indices, final_probs = reranker.rerank(user_idx, cand_indices, sim_scores)

    print(f"\n[Final Top-{args.top_k} Personalized Unseen Recommendations (Two-Stage Pipeline)]")
    print(f"{'Rank':<5} | {'Movie Title':<40} | {'Genres':<25} | {'Re-Rank Score':<12}")
    print("-" * 88)
    for rank in range(min(args.top_k, len(final_indices))):
        idx = final_indices[rank]
        prob = final_probs[rank]
        m_row = prep.movies_meta[prep.movies_meta["movie_idx"] == idx].iloc[0]
        print(f"{rank+1:<5} | {m_row['title'][:38]:<40} | {m_row['genres'][:23]:<25} | {prob:<12.4f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
