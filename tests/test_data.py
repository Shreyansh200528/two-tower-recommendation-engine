"""
Unit tests for data preprocessing integrity, chronological train-test split,
and negative sampling correctness.
"""

import pytest
import pandas as pd
import numpy as np
from data_preprocessing import RecSysDataset

def test_temporal_split_zero_data_leakage(train_test_data):
    """
    Chronological verification:
    Assert that for ANY user in the test set, min(test_timestamps) >= max(train_timestamps)
    to guarantee zero future data leakage.
    """
    train_df, test_df = train_test_data
    
    max_train_ts = train_df.groupby("user_idx")["timestamp"].max()
    min_test_ts = test_df.groupby("user_idx")["timestamp"].min()
    
    # Evaluate across all users present in the test set
    common_users = min_test_ts.index.intersection(max_train_ts.index)
    assert len(common_users) > 0, "No common users found between train and test sets"
    
    violations = min_test_ts.loc[common_users] < max_train_ts.loc[common_users]
    leakage_count = int(violations.sum())
    
    assert leakage_count == 0, (
        f"Data leakage detected! {leakage_count} users have test timestamps earlier than train timestamps."
    )


def test_negative_sampling_disjoint_from_positives(preprocessor, train_test_data):
    """
    Negative sampling verification:
    Assert that sampled negative movies for a user NEVER intersect with movies
    that the user positively interacted with in the training set.
    """
    train_df, _ = train_test_data
    sample_train = train_df.head(500).copy()
    
    num_negatives = 4
    dataset = RecSysDataset(
        ratings_df=sample_train,
        user_features=preprocessor.user_features,
        item_genres=preprocessor.item_genres,
        num_items=preprocessor.num_movies,
        num_negatives=num_negatives
    )
    
    total_checks = min(2000, len(dataset))
    negative_sample_count = 0
    
    for i in range(total_checks):
        u_feat, movie_idx_tensor, _, label_tensor = dataset[i]
        user_idx = u_feat[0].item()
        movie_idx = movie_idx_tensor.item()
        label = label_tensor.item()
        
        pos_items_for_user = dataset.user_positive_items.get(user_idx, set())
        
        if label == 0.0:
            negative_sample_count += 1
            assert movie_idx not in pos_items_for_user, (
                f"Negative sample collision! Movie {movie_idx} was sampled as negative for user {user_idx}, "
                f"but exists in user's positive interactions: {pos_items_for_user}"
            )
        else:
            assert movie_idx in pos_items_for_user, (
                f"Positive sample mismatch! Movie {movie_idx} missing from positive set for user {user_idx}"
            )
            
    assert negative_sample_count > 0, "No negative samples were tested"


def test_feature_indices_continuity(preprocessor):
    """
    Verify mapped user and movie indices are strictly continuous and zero-indexed.
    """
    assert len(preprocessor.user2idx) == preprocessor.num_users
    assert len(preprocessor.movie2idx) == preprocessor.num_movies
    
    user_indices = set(preprocessor.user2idx.values())
    movie_indices = set(preprocessor.movie2idx.values())
    
    assert user_indices == set(range(preprocessor.num_users))
    assert movie_indices == set(range(preprocessor.num_movies))
