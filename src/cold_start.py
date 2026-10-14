"""
Production Cold-Start Fallback Module.
Implements:
1. Tier 1: Demographic Warm Cold-Start (Population-mean User vector + Demographics)
2. Tier 2: Pure Cold-Start (Bayesian Weighted Rating / Dirichlet smoothed Popularity)
"""

import numpy as np
import pandas as pd
import torch

class BayesianPopularityRanker:
    """
    Computes IMDb-style Bayesian Weighted Ratings across items:
        WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
        v = number of ratings for the movie
        m = minimum votes required to be considered credible (e.g., 80th percentile)
        R = average rating of the movie
        C = mean rating across the entire training catalog
    """
    def __init__(self, train_df: pd.DataFrame, movies_meta: pd.DataFrame, min_percentile=80):
        self.movies_meta = movies_meta
        
        # Calculate item stats
        item_grp = train_df.groupby("movie_idx")["rating"]
        counts = item_grp.count()
        means = item_grp.mean()
        
        self.global_mean = float(train_df["rating"].mean())
        self.min_votes = float(np.percentile(counts.values, min_percentile))
        
        # Compute Bayesian Weighted Rating
        scores = {}
        for m_idx in movies_meta["movie_idx"].values:
            v = counts.get(m_idx, 0)
            r = means.get(m_idx, self.global_mean)
            if v == 0:
                wr = self.global_mean
            else:
                wr = (v / (v + self.min_votes)) * r + (self.min_votes / (v + self.min_votes)) * self.global_mean
            scores[m_idx] = float(wr)
            
        self.scores = scores
        # Pre-sort all movie indices by Bayesian score descending
        self.ranked_movie_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    def get_top_k(self, top_k=10):
        results = []
        for rank, m_idx in enumerate(self.ranked_movie_indices[:top_k], 1):
            m_row = self.movies_meta[self.movies_meta["movie_idx"] == m_idx].iloc[0]
            results.append({
                "rank": rank,
                "movie_idx": int(m_idx),
                "movie_id": int(m_row["movie_id"]),
                "title": str(m_row["title"]),
                "genres": str(m_row["genres"]),
                "score": self.scores[m_idx]
            })
        return results


class DemographicColdStartHandler:
    """
    Handles new users with known demographics (Gender, Age group, Occupation)
    but zero interaction history.
    """
    def __init__(self, user_tower, faiss_index, movies_meta, bayesian_ranker=None):
        self.user_tower = user_tower
        self.faiss_index = faiss_index
        self.movies_meta = movies_meta
        self.bayesian_ranker = bayesian_ranker

        self.age_map = {
            "under 18": 0, "<18": 0, "1": 0,
            "18-24": 1, "18": 1,
            "25-34": 2, "25": 2,
            "35-44": 3, "35": 3,
            "45-49": 4, "45": 4,
            "50-55": 5, "50": 5,
            "56+": 6, "56": 6
        }

    def parse_demographics(self, gender_str="M", age_str="25-34", occupation=0):
        # Gender: 'M'/'Male' -> 0, 'F'/'Female' -> 1
        g_str = str(gender_str).strip().upper()
        gender_idx = 1 if g_str in ["F", "FEMALE"] else 0

        # Age mapping
        a_str = str(age_str).strip().lower()
        age_idx = self.age_map.get(a_str, 2)  # default: 25-34 (idx 2)

        # Occupation mapping (0 to 20)
        try:
            occ_idx = int(occupation)
            if not (0 <= occ_idx <= 20):
                occ_idx = 0
        except ValueError:
            occ_idx = 0

        return gender_idx, age_idx, occ_idx

    def recommend(self, gender_str="M", age_str="25-34", occupation=0, top_k=10):
        """
        Retrieves candidates for a demographic cluster.
        """
        gender_idx, age_idx, occ_idx = self.parse_demographics(gender_str, age_str, occupation)
        
        self.user_tower.eval()
        with torch.no_grad():
            cluster_emb = self.user_tower.forward_demographic(gender_idx, age_idx, occ_idx)
            cluster_vec = cluster_emb.cpu().numpy().astype(np.float32)

        # Retrieve top candidates via FAISS (fetch 3x candidate pool for quality re-ranking)
        fetch_k = top_k * 3
        sim_scores, cand_indices = self.faiss_index.search(cluster_vec, fetch_k)
        cand_indices = cand_indices[0]
        sim_scores = sim_scores[0]

        # Score ALL retrieved candidates combining semantic similarity and Bayesian quality
        results = []
        for m_idx, sim in zip(cand_indices, sim_scores):
            m_row = self.movies_meta[self.movies_meta["movie_idx"] == m_idx].iloc[0]
            b_score = self.bayesian_ranker.scores.get(m_idx, 3.5) if self.bayesian_ranker else 3.5
            
            # Blended score: cosine similarity * normalized Bayesian rating
            composite_score = float(sim * (b_score / 5.0))
            
            results.append({
                "movie_idx": int(m_idx),
                "movie_id": int(m_row["movie_id"]),
                "title": str(m_row["title"]),
                "genres": str(m_row["genres"]),
                "score": composite_score,
                "cosine_sim": float(sim),
                "bayesian_rating": float(b_score)
            })
            
        # Re-sort all 3x candidates by composite score descending, then take top_k
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        for rank, item in enumerate(results, 1):
            item["rank"] = rank
            
        return results
