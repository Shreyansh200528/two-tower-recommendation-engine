"""
Stage 2 Re-Ranking Module using LightGBM.
Enhances top candidate retrieval by computing granular cross-features:
- Retrieval similarity score
- User historical statistics (activity, average rating, rating variance)
- Item historical statistics (popularity, average rating, rating variance)
- Explicit Genre Affinity (Jaccard / intersection over user top preferred genres)
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

class LightGBMReranker:
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.user_stats = {}
        self.item_stats = {}
        self.user_top_genres = {}

    def extract_historical_stats(self, train_df, movies_df):
        """
        Computes historical user/item statistics strictly from the training set.
        """
        # User Stats
        u_grp = train_df.groupby("user_idx")["rating"]
        self.user_stats = pd.DataFrame({
            "user_rating_count": u_grp.count(),
            "user_rating_mean": u_grp.mean(),
            "user_rating_std": u_grp.std().fillna(0.0)
        }).to_dict(orient="index")

        # Item Stats
        i_grp = train_df.groupby("movie_idx")["rating"]
        self.item_stats = pd.DataFrame({
            "item_rating_count": i_grp.count(),
            "item_rating_mean": i_grp.mean(),
            "item_rating_std": i_grp.std().fillna(0.0)
        }).to_dict(orient="index")

        # Build movie genres mapping
        movie_genre_map = {}
        for _, row in movies_df.iterrows():
            movie_genre_map[row["movie_idx"]] = set(str(row["genres"]).split("|"))
        self.movie_genre_map = movie_genre_map

        # Build user top preferred genres
        train_with_genres = train_df.merge(movies_df[["movie_idx", "genres"]], on="movie_idx", how="left")
        for u_idx, group in train_with_genres.groupby("user_idx"):
            genre_counts = {}
            for g_str in group["genres"].dropna():
                for g in str(g_str).split("|"):
                    genre_counts[g] = genre_counts.get(g, 0) + 1
            # Top 3 genres for this user
            top_g = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            self.user_top_genres[u_idx] = set([g[0] for g in top_g])

    def compute_features_for_pairs(self, user_indices, movie_indices, retrieval_scores=None):
        """
        Constructs the feature matrix for (user, item) candidate pairs.
        """
        features = []
        default_u = {"user_rating_count": 0, "user_rating_mean": 3.5, "user_rating_std": 0.0}
        default_i = {"item_rating_count": 0, "item_rating_mean": 3.5, "item_rating_std": 0.0}

        for idx, (u, m) in enumerate(zip(user_indices, movie_indices)):
            u_stat = self.user_stats.get(u, default_u)
            i_stat = self.item_stats.get(m, default_i)
            
            u_genres = self.user_top_genres.get(u, set())
            m_genres = self.movie_genre_map.get(m, set())
            genre_overlap = len(u_genres.intersection(m_genres)) / (len(m_genres) + 1e-6)

            sim_score = retrieval_scores[idx] if retrieval_scores is not None else 0.5
            rating_diff = u_stat["user_rating_mean"] - i_stat["item_rating_mean"]

            row = [
                sim_score,
                u_stat["user_rating_count"],
                u_stat["user_rating_mean"],
                u_stat["user_rating_std"],
                i_stat["item_rating_count"],
                i_stat["item_rating_mean"],
                i_stat["item_rating_std"],
                rating_diff,
                genre_overlap
            ]
            features.append(row)

        cols = [
            "retrieval_sim_score", "user_count", "user_mean", "user_std",
            "item_count", "item_mean", "item_std", "rating_diff", "genre_overlap"
        ]
        return pd.DataFrame(features, columns=cols)

    def train(self, train_features_df, train_labels):
        print(f"Training LightGBM Re-Ranker on {len(train_features_df)} samples...")
        self.model.fit(train_features_df, train_labels)
        preds = self.model.predict_proba(train_features_df)[:, 1]
        auc = roc_auc_score(train_labels, preds)
        print(f"LightGBM Re-Ranker Training ROC-AUC: {auc:.4f}")
        return self

    def save(self, model_path="models/lightgbm_reranker.txt"):
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        if hasattr(self.model, "booster_"):
            self.model.booster_.save_model(model_path)
        elif hasattr(self.model, "save_model"):
            self.model.save_model(model_path)
        print(f"Saved LightGBM model to {model_path}")

    def load(self, model_path="models/lightgbm_reranker.txt"):
        self.model = lgb.Booster(model_file=model_path)
        return self

    def rerank(self, user_idx, candidate_movie_indices, retrieval_scores):
        """
        Re-ranks top candidate movie indices for a given user.
        Supports both LGBMClassifier (training) and lgb.Booster (inference).
        Returns:
            reranked_movie_indices: np.ndarray sorted by Re-Ranker probability score
            reranked_scores: corresponding probabilities
        """
        if len(candidate_movie_indices) == 0:
            return np.array([]), np.array([])

        user_list = [user_idx] * len(candidate_movie_indices)
        feat_df = self.compute_features_for_pairs(user_list, candidate_movie_indices, retrieval_scores)

        # Support both LGBMClassifier and raw lgb.Booster
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feat_df)[:, 1]
        else:
            probs = self.model.predict(feat_df)
        
        # Sort descending by re-ranking probability
        ranked_order = np.argsort(-probs)
        candidate_arr = np.array(candidate_movie_indices)
        return candidate_arr[ranked_order], probs[ranked_order]
