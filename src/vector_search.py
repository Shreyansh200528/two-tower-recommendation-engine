"""
FAISS Vector Search Indexing & Candidate Retrieval Module.
Builds and benchmarks HNSW graph indexes for sub-millisecond candidate retrieval.
"""

import time
import numpy as np
import faiss
import torch

class VectorSearchEngine:
    def __init__(self, embedding_dim=64, hnsw_m=32, ef_construction=64, ef_search=64):
        """
        hnsw_m: Number of bidirectional links created for each new element in HNSW graph.
        ef_construction: Size of the dynamic candidate list for construction.
        ef_search: Size of the dynamic candidate list for search (controls speed/recall tradeoff).
        """
        self.embedding_dim = embedding_dim
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        self.index_hnsw = None
        self.index_exact = None
        self.item_embeddings = None

    def build_index(self, item_embeddings: np.ndarray):
        """
        item_embeddings: np.ndarray of shape [num_items, embedding_dim], L2-normalized.
        """
        self.item_embeddings = item_embeddings.astype(np.float32)
        num_items = self.item_embeddings.shape[0]

        # 1. Exact Flat Inner-Product Index (Ground Truth baseline)
        self.index_exact = faiss.IndexFlatIP(self.embedding_dim)
        self.index_exact.add(self.item_embeddings)

        # 2. HNSW Graph Index (Approximate Nearest Neighbors)
        # Using METRIC_INNER_PRODUCT because vectors are L2-normalized (equivalent to cosine similarity)
        self.index_hnsw = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self.index_hnsw.hnsw.efConstruction = self.ef_construction
        self.index_hnsw.hnsw.efSearch = self.ef_search
        self.index_hnsw.add(self.item_embeddings)

        print(f"FAISS Index built successfully with {num_items} items.")
        return self

    def query(self, user_vector: np.ndarray, top_k=100, use_hnsw=True):
        """
        user_vector: np.ndarray [1, embedding_dim] or [embedding_dim]
        Returns:
            scores: [top_k] similarity scores
            indices: [top_k] retrieved item indices
            latency_ms: time taken in milliseconds
        """
        if user_vector.ndim == 1:
            user_vector = user_vector.reshape(1, -1)
        user_vector = user_vector.astype(np.float32)

        index = self.index_hnsw if use_hnsw else self.index_exact
        
        start_time = time.perf_counter()
        scores, indices = index.search(user_vector, top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return scores[0], indices[0], latency_ms

    def benchmark_latency(self, test_user_vectors: np.ndarray, top_k=100, num_runs=500):
        """
        Benchmarks average query latency (in ms) over multiple queries.
        """
        test_vectors = test_user_vectors[:num_runs].astype(np.float32)
        n = len(test_vectors)

        # Benchmark HNSW
        start = time.perf_counter()
        for i in range(n):
            self.index_hnsw.search(test_vectors[i:i+1], top_k)
        hnsw_avg_ms = ((time.perf_counter() - start) / n) * 1000.0

        # Benchmark Exact Flat
        start = time.perf_counter()
        for i in range(n):
            self.index_exact.search(test_vectors[i:i+1], top_k)
        exact_avg_ms = ((time.perf_counter() - start) / n) * 1000.0

        return {
            "hnsw_avg_ms": hnsw_avg_ms,
            "exact_avg_ms": exact_avg_ms,
            "speedup": exact_avg_ms / (hnsw_avg_ms + 1e-9)
        }

    def save(self, filepath="models/faiss_hnsw.index"):
        faiss.write_index(self.index_hnsw, filepath)

    def load(self, filepath="models/faiss_hnsw.index"):
        self.index_hnsw = faiss.read_index(filepath)
