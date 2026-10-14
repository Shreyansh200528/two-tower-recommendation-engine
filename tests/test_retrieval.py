"""
Unit tests for FAISS vector retrieval integrity, uniqueness, and latency SLAs.
"""

import time
import pytest
import numpy as np
import torch

def test_faiss_index_total_items(faiss_index, preprocessor):
    """
    Assert that index.ntotal matches the exact number of unique items in the catalog.
    """
    assert faiss_index.ntotal == preprocessor.num_movies, (
        f"FAISS index count ({faiss_index.ntotal}) does not match catalog ({preprocessor.num_movies})"
    )


def test_top_k_uniqueness_and_validity(faiss_index, two_tower_model, preprocessor):
    """
    Assert that querying for top_k=50 returns strictly 50 unique, valid candidate IDs.
    """
    top_k = 50
    user_feature = preprocessor.user_features[0:1]
    
    with torch.no_grad():
        user_emb = two_tower_model.user_tower(user_feature).cpu().numpy().astype(np.float32)
        
    scores, indices = faiss_index.search(user_emb, top_k)
    retrieved_ids = indices[0]
    
    # 1. Exactly top_k items returned
    assert len(retrieved_ids) == top_k, f"Expected {top_k} items, got {len(retrieved_ids)}"
    
    # 2. Strictly unique IDs
    unique_ids = set(retrieved_ids)
    assert len(unique_ids) == top_k, f"Duplicates detected in top_{top_k} retrieval results"
    
    # 3. Valid candidate indices within [0, num_movies - 1]
    for idx in retrieved_ids:
        assert 0 <= idx < preprocessor.num_movies, (
            f"Candidate ID {idx} out of bounds [0, {preprocessor.num_movies - 1}]"
        )


def test_retrieval_latency_sla(faiss_index, two_tower_model, preprocessor):
    """
    Performance SLA test:
    Assert that average retrieval latency per query across 100 test queries
    is strictly under 1.5 ms.
    """
    num_queries = 100
    top_k = 50
    
    sample_users = preprocessor.user_features[:num_queries]
    with torch.no_grad():
        query_vectors = two_tower_model.user_tower(sample_users).cpu().numpy().astype(np.float32)
        
    latencies_ms = []
    
    # Warm-up query
    faiss_index.search(query_vectors[0:1], top_k)
    
    for i in range(num_queries):
        single_vec = query_vectors[i:i+1]
        t0 = time.perf_counter()
        faiss_index.search(single_vec, top_k)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        
    avg_latency_ms = float(np.mean(latencies_ms))
    p99_latency_ms = float(np.percentile(latencies_ms, 99))
    
    print(f"\n[SLA Benchmark] Average Latency: {avg_latency_ms:.4f} ms | P99: {p99_latency_ms:.4f} ms")
    
    # Strict SLA assertions
    assert avg_latency_ms < 1.5, (
        f"Retrieval SLA violated: average latency was {avg_latency_ms:.3f} ms (expected < 1.5 ms)"
    )
    assert p99_latency_ms < 5.0, (
        f"Retrieval P99 SLA violated: {p99_latency_ms:.3f} ms (expected < 5.0 ms)"
    )
