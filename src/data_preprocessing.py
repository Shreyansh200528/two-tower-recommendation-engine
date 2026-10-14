"""
Data Preprocessing Module for MovieLens-1M Two-Tower Recommendation System.
Handles loading, ID re-indexing, feature encoding, temporal train/test splitting,
and negative sampling.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MovieLens1MPreprocessor:
    def __init__(self, data_dir="data/ml-1m"):
        self.data_dir = data_dir
        self.all_genres = [
            "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
            "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        self.genre2idx = {g: i for i, g in enumerate(self.all_genres)}
        
    def load_data(self):
        # 1. Load Movies
        movies_path = os.path.join(self.data_dir, "movies.dat")
        movies_df = pd.read_csv(
            movies_path,
            sep="::",
            names=["movie_id", "title", "genres"],
            engine="python",
            encoding="latin-1"
        )
        
        # 2. Load Users
        users_path = os.path.join(self.data_dir, "users.dat")
        users_df = pd.read_csv(
            users_path,
            sep="::",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
            engine="python",
            encoding="latin-1"
        )
        
        # 3. Load Ratings
        ratings_path = os.path.join(self.data_dir, "ratings.dat")
        ratings_df = pd.read_csv(
            ratings_path,
            sep="::",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python"
        )
        
        return users_df, movies_df, ratings_df

    def process(self):
        users_df, movies_df, ratings_df = self.load_data()
        
        # Map user_id and movie_id to continuous zero-indexed integers
        unique_users = ratings_df["user_id"].unique()
        unique_movies = movies_df["movie_id"].unique()
        
        self.user2idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx2user = {i: uid for uid, i in self.user2idx.items()}
        
        self.movie2idx = {mid: i for i, mid in enumerate(unique_movies)}
        self.idx2movie = {i: mid for mid, i in self.movie2idx.items()}
        
        ratings_df["user_idx"] = ratings_df["user_id"].map(self.user2idx)
        ratings_df["movie_idx"] = ratings_df["movie_id"].map(self.movie2idx)
        # Drop ratings for movies not in movies_df if any
        ratings_df = ratings_df.dropna(subset=["movie_idx"]).copy()
        ratings_df["movie_idx"] = ratings_df["movie_idx"].astype(int)
        
        # Encode User Features
        users_df["user_idx"] = users_df["user_id"].map(self.user2idx)
        users_df = users_df.dropna(subset=["user_idx"]).copy()
        users_df["user_idx"] = users_df["user_idx"].astype(int)
        
        # Gender: 'M' -> 0, 'F' -> 1
        users_df["gender_idx"] = (users_df["gender"] == "F").astype(int)
        
        # Age mapping
        age_map = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
        users_df["age_idx"] = users_df["age"].map(age_map).fillna(0).astype(int)
        
        # Occupation (0 to 20)
        users_df["occupation_idx"] = users_df["occupation"].astype(int)
        
        # Store user feature lookup table ordered by user_idx
        users_df = users_df.sort_values("user_idx").reset_index(drop=True)
        self.user_features = torch.tensor(
            users_df[["user_idx", "gender_idx", "age_idx", "occupation_idx"]].values,
            dtype=torch.long
        )
        
        # Encode Item (Movie) Multi-Hot Genres
        movies_df["movie_idx"] = movies_df["movie_id"].map(self.movie2idx)
        movies_df = movies_df.dropna(subset=["movie_idx"]).copy()
        movies_df["movie_idx"] = movies_df["movie_idx"].astype(int)
        movies_df = movies_df.sort_values("movie_idx").reset_index(drop=True)
        
        num_movies = len(self.movie2idx)
        genre_matrix = np.zeros((num_movies, len(self.all_genres)), dtype=np.float32)
        for _, row in movies_df.iterrows():
            m_idx = row["movie_idx"]
            genres = str(row["genres"]).split("|")
            for g in genres:
                if g in self.genre2idx:
                    genre_matrix[m_idx, self.genre2idx[g]] = 1.0
                    
        self.item_genres = torch.tensor(genre_matrix, dtype=torch.float32)
        self.movies_meta = movies_df
        self.users_meta = users_df
        self.ratings_df = ratings_df
        
        self.num_users = len(self.user2idx)
        self.num_movies = len(self.movie2idx)
        
        return self

    def temporal_train_test_split(self, test_ratio=0.2):
        """
        Chronological split per user: earlier 80% ratings for training,
        latest 20% ratings for testing (strictly prevents future data leakage).
        """
        # Sort by user and timestamp
        sorted_df = self.ratings_df.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)
        
        train_dfs = []
        test_dfs = []
        
        for _, group in sorted_df.groupby("user_idx"):
            n = len(group)
            if n < 5:
                train_dfs.append(group)
                continue
            split_point = int(n * (1 - test_ratio))
            train_dfs.append(group.iloc[:split_point])
            test_dfs.append(group.iloc[split_point:])
            
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True) if len(test_dfs) > 0 else pd.DataFrame()
        
        return train_df, test_df


class RecSysDataset(Dataset):
    """
    Dataset for Two-Tower training with explicit Negative Sampling.
    For each observed positive interaction, pairs with random unobserved items.
    """
    def __init__(self, ratings_df, user_features, item_genres, num_items, num_negatives=4, exclude_items_dict=None):
        self.ratings_df = ratings_df
        self.user_features = user_features
        self.item_genres = item_genres
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # Build set of watched items per user for negative sampling.
        # Can pass complete user history (train + test) to avoid sampling future test movies as negatives.
        if exclude_items_dict is not None:
            self.user_positive_items = exclude_items_dict
        else:
            self.user_positive_items = ratings_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
        
        self.pos_pairs = ratings_df[["user_idx", "movie_idx"]].values

    def __len__(self):
        return len(self.pos_pairs) * (1 + self.num_negatives)

    def __getitem__(self, idx):
        pos_idx = idx // (1 + self.num_negatives)
        is_neg = (idx % (1 + self.num_negatives)) != 0
        
        user_idx, pos_movie_idx = self.pos_pairs[pos_idx]
        
        if not is_neg:
            movie_idx = pos_movie_idx
            label = 1.0
        else:
            # Sample random negative item not interacted with by this user (bounded retry)
            user_pos = self.user_positive_items.get(user_idx, set())
            neg_movie_idx = np.random.randint(0, self.num_items)
            retries = 0
            while neg_movie_idx in user_pos and retries < 100:
                neg_movie_idx = np.random.randint(0, self.num_items)
                retries += 1
            movie_idx = neg_movie_idx
            label = 0.0
            
        u_feat = self.user_features[user_idx]
        i_genres = self.item_genres[movie_idx]
        
        return (
            u_feat,                   # [user_idx, gender_idx, age_idx, occ_idx]
            torch.tensor(movie_idx, dtype=torch.long),
            i_genres,                 # multi-hot [18]
            torch.tensor(label, dtype=torch.float32)
        )
