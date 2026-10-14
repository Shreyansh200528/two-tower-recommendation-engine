"""
End-to-end integration tests for LightGBM Re-Ranker and complete Recommendation Pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import torch

def test_feature_engineering_schema_and_types(reranker, preprocessor):
    """
    Verify feature engineering generates candidate rows with the exact column schema
    and numeric data types expected by the trained LightGBM booster.
    """
    sample_users = [0, 1, 2]
    sample_movies = [10, 20, 30]
    sample_sims = [0.85, 0.72, 0.65]
    
    features_df = reranker.compute_features_for_pairs(
        user_indices=sample_users,
        movie_indices=sample_movies,
        retrieval_scores=sample_sims
    )
    
    expected_columns = [
        "retrieval_sim_score", "user_count", "user_mean", "user_std",
        "item_count", "item_mean", "item_std", "rating_diff", "genre_overlap"
    ]
    
    # 1. Exact column list match
    assert list(features_df.columns) == expected_columns, (
        f"Schema mismatch! Expected {expected_columns}, got {list(features_df.columns)}"
    )
    
    # 2. LightGBM booster feature names match
    booster_feature_names = reranker.model.feature_name()
    assert list(features_df.columns) == booster_feature_names, (
        f"Feature mismatch with booster: {booster_feature_names}"
    )
    
    # 3. All column dtypes must be numeric (no nulls or object columns)
    assert features_df.isnull().sum().sum() == 0, "Null values detected in feature matrix"
    for col in features_df.columns:
        assert np.issubdtype(features_df[col].dtype, np.number), (
            f"Column {col} has non-numeric dtype: {features_df[col].dtype}"
        )


def test_reranker_probability_bounds(reranker, preprocessor):
    """
    Assert that all predicted re-ranking probability scores fall strictly
    within valid probability bounds [0.0, 1.0].
    """
    user_indices = [42] * 50
    movie_indices = list(range(50))
    sim_scores = [0.5] * 50
    
    feat_df = reranker.compute_features_for_pairs(user_indices, movie_indices, sim_scores)
    probs = reranker.model.predict(feat_df)
    
    assert len(probs) == 50
    assert (probs >= 0.0).all(), f"Found probability score < 0.0: min is {probs.min()}"
    assert (probs <= 1.0).all(), f"Found probability score > 1.0: max is {probs.max()}"


def test_end_to_end_recommendation_pipeline(two_tower_model, faiss_index, reranker, preprocessor, train_test_data):
    """
    Full end-to-end integration test:
    Queries recommendations for a dummy user and verifies that the payload:
    1. Contains the requested top_k items.
    2. Excludes already-watched movies from training history (seen-item filter).
    3. Contains valid, non-empty movie titles and genre strings.
    4. Has scores sorted in strictly descending order.
    """
    train_df, _ = train_test_data
    dummy_raw_user_id = 42
    top_k = 10
    retrieval_k = 100
    
    user_idx = preprocessor.user2idx[dummy_raw_user_id]
    u_feat = preprocessor.user_features[user_idx]
    user_seen_movies = set(train_df[train_df["user_idx"] == user_idx]["movie_idx"].values)
    
    # Step 1: User Tower Vector Generation
    with torch.no_grad():
        u_emb = two_tower_model.user_tower(u_feat.unsqueeze(0)).cpu().numpy().astype(np.float32)
        
    assert u_emb.shape == (1, 64)
    
    # Step 2: FAISS HNSW Retrieval
    sim_scores, cand_indices = faiss_index.search(u_emb, retrieval_k)
    cand_indices = cand_indices[0]
    sim_scores = sim_scores[0]
    
    assert len(cand_indices) == retrieval_k

    # Step 3: Deduplication / Seen-Item Filter
    unseen_mask = [m not in user_seen_movies for m in cand_indices]
    cand_indices = [m for m, keep in zip(cand_indices, unseen_mask) if keep]
    sim_scores = [s for s, keep in zip(sim_scores, unseen_mask) if keep]
    
    # Step 4: LightGBM Re-Ranking via unified rerank() method
    final_indices, final_probs = reranker.rerank(user_idx, cand_indices, sim_scores)
    final_indices = final_indices[:top_k]
    final_probs = final_probs[:top_k]
    
    # Build returned payload
    payload = []
    for rank, (m_idx, score) in enumerate(zip(final_indices, final_probs), 1):
        m_row = preprocessor.movies_meta[preprocessor.movies_meta["movie_idx"] == m_idx].iloc[0]
        payload.append({
            "rank": rank,
            "movie_idx": int(m_idx),
            "movie_id": int(m_row["movie_id"]),
            "title": str(m_row["title"]),
            "genres": str(m_row["genres"]),
            "score": float(score)
        })
        
    # Payload Assertions
    assert len(payload) == top_k, f"Expected {top_k} recommendations, got {len(payload)}"
    
    for item in payload:
        assert item["title"].strip() != "", f"Empty title found in recommendation: {item}"
        assert item["genres"].strip() != "", f"Empty genres found in recommendation: {item}"
        assert 0.0 <= item["score"] <= 1.0, f"Score out of bounds: {item['score']}"
        assert item["movie_id"] > 0, f"Invalid movie_id: {item['movie_id']}"
        assert item["movie_idx"] not in user_seen_movies, f"Seen item {item['title']} was not filtered out!"
        
    # Monotonic descending score assertion
    for i in range(len(payload) - 1):
        assert payload[i]["score"] >= payload[i + 1]["score"], (
            f"Recommendations are not sorted in descending order: "
            f"Rank {payload[i]['rank']} score ({payload[i]['score']}) < "
            f"Rank {payload[i+1]['rank']} score ({payload[i+1]['score']})"
        )


def test_cold_start_tier2_bayesian_fallback(preprocessor, train_test_data):
    """
    Test Pure Cold-Start Fallback (Tier 2):
    Verifies that unknown/anonymous users receive valid Bayesian-weighted recommendations
    sorted in strictly descending order.
    """
    from cold_start import BayesianPopularityRanker
    train_df, _ = train_test_data

    ranker = BayesianPopularityRanker(train_df, preprocessor.movies_meta)
    top_10 = ranker.get_top_k(top_k=10)

    assert len(top_10) == 10, f"Expected 10 items, got {len(top_10)}"

    # Scores must be sorted descending
    for i in range(len(top_10) - 1):
        assert top_10[i]["score"] >= top_10[i + 1]["score"], (
            f"Bayesian fallback scores not descending: {top_10[i]['score']} < {top_10[i+1]['score']}"
        )

    # Sanity checks on ratings scale and titles
    for item in top_10:
        assert 3.5 <= item["score"] <= 5.0, f"Bayesian score out of bounds: {item['score']}"
        assert item["title"].strip() != ""
        assert item["movie_id"] > 0


def test_cold_start_tier1_demographic_cluster(two_tower_model, faiss_index, preprocessor, train_test_data):
    """
    Test Demographic Cold-Start Fallback (Tier 1):
    Verifies that new users with known demographics (e.g. Female, Age 18-24)
    receive demographic-cluster vector recommendations via FAISS.
    """
    from cold_start import BayesianPopularityRanker, DemographicColdStartHandler
    train_df, _ = train_test_data

    bayesian = BayesianPopularityRanker(train_df, preprocessor.movies_meta)
    handler = DemographicColdStartHandler(
        user_tower=two_tower_model.user_tower,
        faiss_index=faiss_index,
        movies_meta=preprocessor.movies_meta,
        bayesian_ranker=bayesian
    )

    recs = handler.recommend(gender_str="F", age_str="18-24", occupation=0, top_k=10)

    assert len(recs) == 10
    unique_ids = set(r["movie_id"] for r in recs)
    assert len(unique_ids) == 10, "Duplicate items in demographic recommendations"

    for i in range(len(recs) - 1):
        assert recs[i]["score"] >= recs[i + 1]["score"], (
            f"Demographic scores not sorted descending: {recs[i]['score']} < {recs[i+1]['score']}"
        )

