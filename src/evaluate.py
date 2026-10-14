"""
Evaluation Module for Two-Tower & Two-Stage Recommendation Pipeline.
Computes:
- HitRate@K (HR@K)
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
"""

import numpy as np
import math
from tqdm import tqdm

def compute_ndcg_at_k(recommended_items, ground_truth_set, k=10):
    """
    recommended_items: ordered list of item indices
    ground_truth_set: set of relevant items
    """
    dcg = 0.0
    idcg = 0.0
    
    # Calculate DCG
    for i, item in enumerate(recommended_items[:k]):
        if item in ground_truth_set:
            # Relevance = 1
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) is 0
            
    # Calculate Ideal DCG (IDCG)
    num_hits = min(len(ground_truth_set), k)
    for i in range(num_hits):
        idcg += 1.0 / math.log2(i + 2)
        
    return (dcg / idcg) if idcg > 0 else 0.0


def compute_hit_rate_at_k(recommended_items, ground_truth_set, k=10):
    for item in recommended_items[:k]:
        if item in ground_truth_set:
            return 1.0
    return 0.0


def compute_recall_at_k(recommended_items, ground_truth_set, k=10):
    if len(ground_truth_set) == 0:
        return 0.0
    hits = sum(1 for item in recommended_items[:k] if item in ground_truth_set)
    return hits / len(ground_truth_set)


def evaluate_pipeline(test_user_ground_truth, retrieval_fn, rerank_fn=None, k=10, num_eval_users=500, user_train_history=None):
    """
    Evaluates recommendation quality across test users.
    retrieval_fn: fn(user_idx, top_k) -> candidate_indices, scores
    rerank_fn: Optional fn(user_idx, candidate_indices, scores) -> reranked_indices, reranked_scores
    user_train_history: Optional dict(user_idx -> set of seen movie_idx) to filter already-rated items.
    """
    hr_list = []
    ndcg_list = []
    recall_list = []

    user_ids = list(test_user_ground_truth.keys())[:num_eval_users]
    desc = "Evaluating (Two-Stage)" if rerank_fn is not None else "Evaluating (Stage 1)"

    for u_idx in tqdm(user_ids, desc=desc, unit="user", leave=False):
        gt_items = test_user_ground_truth[u_idx]
        if len(gt_items) == 0:
            continue

        # Stage 1: Retrieval (e.g. top 100)
        cand_items, cand_scores = retrieval_fn(u_idx, top_k=100)

        # Deduplication / Seen Filter: Exclude training history items so they don't displace unseen test items
        if user_train_history is not None:
            seen_items = user_train_history.get(u_idx, set())
            unseen_indices = [i for i, m in enumerate(cand_items) if m not in seen_items]
            cand_items = [cand_items[i] for i in unseen_indices]
            cand_scores = [cand_scores[i] for i in unseen_indices]

        # Stage 2: Re-ranking (if provided)
        if rerank_fn is not None and len(cand_items) > 0:
            final_items, _ = rerank_fn(u_idx, cand_items, cand_scores)
        else:
            final_items = cand_items

        hr = compute_hit_rate_at_k(final_items, gt_items, k=k)
        ndcg = compute_ndcg_at_k(final_items, gt_items, k=k)
        rec = compute_recall_at_k(final_items, gt_items, k=k)

        hr_list.append(hr)
        ndcg_list.append(ndcg)
        recall_list.append(rec)

    return {
        f"HitRate@{k}": float(np.mean(hr_list)),
        f"NDCG@{k}": float(np.mean(ndcg_list)),
        f"Recall@{k}": float(np.mean(recall_list)),
        "users_evaluated": len(hr_list)
    }
