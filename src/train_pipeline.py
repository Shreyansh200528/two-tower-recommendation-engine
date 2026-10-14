"""
End-to-End Two-Tower + FAISS + LightGBM Re-Ranking Pipeline (Production Configuration).
Optimized parameters: 5 epochs, 64-dim embeddings, top_k=100 candidate retrieval,
seen-item filtering, bounded negative sampling, and real-time tqdm progress reporting.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load environment configurations if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import MovieLens1MPreprocessor, RecSysDataset
from two_tower_model import TwoTowerModel
from vector_search import VectorSearchEngine
from reranker import LightGBMReranker
from evaluate import evaluate_pipeline

def main():
    print("=" * 65, flush=True)
    print(" TWO-TOWER RECOMMENDATION SYSTEM (PRODUCTION PIPELINE)", flush=True)
    print("=" * 65, flush=True)
    
    device = torch.device("cpu")
    print(f"Executing on: {device}", flush=True)

    # Read configuration from environment or use production defaults
    data_dir = os.getenv("DATA_DIR", "data/ml-1m")
    models_dir = os.getenv("MODELS_DIR", "models")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "64"))
    epochs = int(os.getenv("EPOCHS", "5"))
    learning_rate = float(os.getenv("LEARNING_RATE", "0.002"))
    batch_size = int(os.getenv("BATCH_SIZE", "2048"))
    top_k_retrieval = int(os.getenv("TOP_K_RETRIEVAL", "100"))
    
    # -------------------------------------------------------------
    # 1. Data Loading & Preprocessing
    # -------------------------------------------------------------
    print(f"\n[Phase 1/5] Loading & Preprocessing MovieLens-1M from {data_dir}...", flush=True)
    prep = MovieLens1MPreprocessor(data_dir)
    prep.process()
    
    train_df, test_df = prep.temporal_train_test_split(test_ratio=0.2)
    print(f"Users: {prep.num_users:,} | Movies: {prep.num_movies:,}", flush=True)
    print(f"Train Interactions: {len(train_df):,} | Test Interactions: {len(test_df):,}", flush=True)
    
    # All historical interactions to avoid sampling future test movies as negatives
    user_all_items = prep.ratings_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    user_train_history = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    train_dataset = RecSysDataset(
        train_df,
        prep.user_features,
        prep.item_genres,
        prep.num_movies,
        num_negatives=4,
        exclude_items_dict=user_all_items
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # -------------------------------------------------------------
    # 2. Train Two-Tower Neural Network
    # -------------------------------------------------------------
    print(f"\n[Phase 2/5] Training PyTorch Two-Tower Model ({epochs} Epochs, Dim={embedding_dim}, LR={learning_rate})...", flush=True)
    model = TwoTowerModel(prep.num_users, prep.num_movies, num_genres=18, embedding_dim=embedding_dim)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start_epoch = time.time()
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{epochs:02d}",
            unit="batch",
            file=sys.stdout,
            dynamic_ncols=True
        )
        for u_feat, m_idx, i_genres, label in pbar:
            optimizer.zero_grad()
            logits, _, _ = model(u_feat, m_idx, i_genres)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            running_avg = total_loss / (pbar.n + 1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{running_avg:.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        epoch_sec = time.time() - start_epoch
        print(f"  --> Completed Epoch {epoch:02d}/{epochs:02d} | Avg BCE Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s", flush=True)
        
    os.makedirs(models_dir, exist_ok=True)
    two_tower_path = os.path.join(models_dir, "two_tower.pt")
    torch.save(model.state_dict(), two_tower_path)
    print(f"  Saved Two-Tower weights to {two_tower_path}", flush=True)

    # -------------------------------------------------------------
    # 3. Generate Item Embeddings & Build Production FAISS HNSW Index
    # -------------------------------------------------------------
    print("\n[Phase 3/5] Extracting Item Embeddings & Building FAISS HNSW Graph...", flush=True)
    model.eval()
    with torch.no_grad():
        all_item_indices = torch.arange(prep.num_movies, dtype=torch.long)
        all_item_genres = prep.item_genres
        all_item_embs = model.item_tower(all_item_indices, all_item_genres).cpu().numpy()
        all_user_embs = model.user_tower(prep.user_features).cpu().numpy()
        
    engine = VectorSearchEngine(embedding_dim=embedding_dim, hnsw_m=32, ef_construction=64, ef_search=64)
    engine.build_index(all_item_embs)
    faiss_path = os.path.join(models_dir, "faiss_hnsw.index")
    engine.save(faiss_path)
    
    # Latency Benchmarking (Candidate pool: top_k)
    print(f"  Benchmarking FAISS HNSW search latency across 500 test queries (top_k={top_k_retrieval})...", flush=True)
    bench = engine.benchmark_latency(all_user_embs, top_k=top_k_retrieval, num_runs=500)
    print(f"  >>> HNSW Average Latency: {bench['hnsw_avg_ms']:.3f} ms per query <<<", flush=True)
    print(f"  >>> Exact Flat Latency:   {bench['exact_avg_ms']:.3f} ms per query <<<", flush=True)
    print(f"  >>> HNSW Speedup:         {bench['speedup']:.1f}x faster than brute-force <<<", flush=True)

    # -------------------------------------------------------------
    # 4. Train Stage 2 LightGBM Re-Ranker (100k samples)
    # -------------------------------------------------------------
    print("\n[Phase 4/5] Training Stage 2 LightGBM Re-Ranker on 100,000 samples...", flush=True)
    reranker = LightGBMReranker()
    reranker.extract_historical_stats(train_df, prep.movies_meta)
    
    sample_train = train_df.sample(n=min(100000, len(train_df)), random_state=42).copy()
    user_idxs = sample_train["user_idx"].values
    movie_idxs = sample_train["movie_idx"].values
    labels = (sample_train["rating"] >= 4).astype(int).values
    
    u_vecs = all_user_embs[user_idxs]
    m_vecs = all_item_embs[movie_idxs]
    sim_scores = (u_vecs * m_vecs).sum(axis=-1)
    
    train_feat_df = reranker.compute_features_for_pairs(user_idxs, movie_idxs, sim_scores)
    reranker.train(train_feat_df, labels)
    reranker_path = os.path.join(models_dir, "lightgbm_reranker.txt")
    reranker.save(reranker_path)

    # -------------------------------------------------------------
    # 5. Comparative Evaluation on Held-Out Test Set (with Seen Filter)
    # -------------------------------------------------------------
    print("\n[Phase 5/5] Comparative Evaluation on Held-Out Test Set (with Seen-Item Deduplication)...", flush=True)
    test_pos = test_df[test_df["rating"] >= 4]
    test_ground_truth = test_pos.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    def retrieval_fn(u_idx, top_k=top_k_retrieval):
        u_vec = all_user_embs[u_idx]
        scores, indices, _ = engine.query(u_vec, top_k=top_k, use_hnsw=True)
        return indices, scores

    def rerank_fn(u_idx, cand_indices, cand_scores):
        return reranker.rerank(u_idx, cand_indices, cand_scores)

    eval_users = 300
    print(f"  Evaluating over {eval_users} test users...", flush=True)
    metrics_stage1 = evaluate_pipeline(
        test_ground_truth,
        retrieval_fn,
        rerank_fn=None,
        k=10,
        num_eval_users=eval_users,
        user_train_history=user_train_history
    )
    metrics_twostage = evaluate_pipeline(
        test_ground_truth,
        retrieval_fn,
        rerank_fn=rerank_fn,
        k=10,
        num_eval_users=eval_users,
        user_train_history=user_train_history
    )

    print("\n" + "=" * 65, flush=True)
    print(" PERFORMANCE BENCHMARK SUMMARY (PRODUCTION RUN)", flush=True)
    print("=" * 65, flush=True)
    print(f"{'Metric':<15} | {'Stage 1 (FAISS Only)':<22} | {'Two-Stage (FAISS + LightGBM)':<28}", flush=True)
    print("-" * 72, flush=True)
    for m in ["HitRate@10", "NDCG@10", "Recall@10"]:
        print(f"{m:<15} | {metrics_stage1[m]:<22.4f} | {metrics_twostage[m]:<28.4f}", flush=True)
    
    ndcg_gain = ((metrics_twostage['NDCG@10'] - metrics_stage1['NDCG@10']) / (metrics_stage1['NDCG@10'] + 1e-9)) * 100
    print(f"\n>>> Re-ranking NDCG@10 Relative Improvement: +{ndcg_gain:.1f}% <<<", flush=True)

    # Load Run 2 ablation study for comparison if exists
    run2_summary_path = "models_ablation_run2/run2_summary.json"
    if os.path.exists(run2_summary_path):
        with open(run2_summary_path, "r") as f:
            r2 = json.load(f)
        print("\n" + "=" * 65, flush=True)
        print(" ABLATION COMPARISON: PRODUCTION (5 EPOCHS) vs ABLATION (10 EPOCHS)", flush=True)
        print("=" * 65, flush=True)
        print(f"{'Metric / Parameter':<30} | {'Production (Run 1)':<18} | {'Ablation (Run 2)':<18}", flush=True)
        print("-" * 72, flush=True)
        r2_ndcg = r2["metrics"]["two_stage_pipeline"]["NDCG@10"]
        r2_gain = r2["metrics"]["ndcg_relative_improvement"]
        r2_lat = r2["faiss"]["avg_latency_ms"]

        r1_ndcg = metrics_twostage['NDCG@10']
        r1_gain_str = f"+{ndcg_gain:.1f}%"
        r1_lat_str = f"{bench['hnsw_avg_ms']:.3f} ms"
        r2_lat_str = f"{r2_lat:.3f} ms"

        print(f"{'Epochs':<30} | {'5':<18} | {'10':<18}", flush=True)
        print(f"{'Candidate Pool (top_k)':<30} | {'100':<18} | {'150':<18}", flush=True)
        print(f"{'HNSW Graph m / efSearch':<30} | {'32 / 64':<18} | {'48 / 128':<18}", flush=True)
        print(f"{'LightGBM Train Samples':<30} | {'100,000':<18} | {'200,000':<18}", flush=True)
        print(f"{'Two-Stage NDCG@10':<30} | {r1_ndcg:<18.4f} | {r2_ndcg:<18.4f}", flush=True)
        print(f"{'NDCG Relative Gain':<30} | {r1_gain_str:<18} | {r2_gain:<18}", flush=True)
        print(f"{'HNSW Query Latency':<30} | {r1_lat_str:<18} | {r2_lat_str:<18}", flush=True)
        print("=" * 65, flush=True)

    # -------------------------------------------------------------
    # Live Demonstration for a Sample User (with Seen Filter)
    # -------------------------------------------------------------
    demo_user = 42
    print(f"\nINFERENCE DEMO FOR USER IDX: {demo_user}", flush=True)
    user_seen = user_train_history.get(demo_user, set())
    user_history = train_df[train_df["user_idx"] == demo_user].sort_values("rating", ascending=False).head(5)
    user_history = user_history.merge(prep.movies_meta, on="movie_idx")
    print(f"User's Top Historical Favorites in Training Data ({len(user_seen)} total watched):", flush=True)
    for _, r in user_history.iterrows():
        print(f"  * {r['title']} ({r['genres']}) - Rated: {r['rating']} stars", flush=True)

    # Retrieve extra candidates to guarantee top_k after seen filter
    cand_indices, cand_scores = retrieval_fn(demo_user, top_k=top_k_retrieval + len(user_seen))
    
    # Filter seen items
    unseen_mask = [m not in user_seen for m in cand_indices]
    cand_indices = [m for m, keep in zip(cand_indices, unseen_mask) if keep]
    cand_scores = [s for s, keep in zip(cand_scores, unseen_mask) if keep]
    
    final_indices, final_probs = rerank_fn(demo_user, cand_indices, cand_scores)

    print("\nTop-5 Unseen Recommendations from Stage 1 (FAISS Vector Retrieval):", flush=True)
    for rank in range(min(5, len(cand_indices))):
        idx = cand_indices[rank]
        m_row = prep.movies_meta[prep.movies_meta["movie_idx"] == idx].iloc[0]
        print(f"  {rank+1}. {m_row['title']} [{m_row['genres']}] (Cosine Sim: {cand_scores[rank]:.3f})", flush=True)

    print("\nTop-5 Unseen Recommendations after Stage 2 (LightGBM Re-Ranker):", flush=True)
    for rank in range(min(5, len(final_indices))):
        idx = final_indices[rank]
        m_row = prep.movies_meta[prep.movies_meta["movie_idx"] == idx].iloc[0]
        print(f"  {rank+1}. {m_row['title']} [{m_row['genres']}] (Re-rank Score: {final_probs[rank]:.4f})", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    main()
