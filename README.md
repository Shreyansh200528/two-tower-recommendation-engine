# Two-Tower Recommendation Engine with FAISS & Two-Stage Ranking

An industrial-grade, two-stage recommendation pipeline built on the **MovieLens-1M** benchmark (1M+ ratings, 6,040 users, 3,883 movies). Combines deep dual-encoder representation learning, sub-millisecond approximate nearest neighbor (ANN) vector search, and gradient-boosted decision tree re-ranking.

---

## Architecture Overview

```
                      [User Request: User ID + Demographics]
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│               STAGE 1: CANDIDATE RETRIEVAL (Two-Tower + FAISS)                  │
│                                                                                 │
│  User Demographics ──► [User Tower (PyTorch)] ──► 64-dim User Vector u         │
│                                                          │                      │
│                                                          ▼                      │
│                                             [FAISS HNSW Vector Index]           │
│                                              (3,883 Movie Vectors v)            │
│                                                          │                      │
│                                             Sub-millisecond Search              │
│                                             (Avg Latency: 0.098 ms)             │
│                                                          │                      │
│                                                          ▼                      │
│                                                 Top 100 Candidates              │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: RE-RANKING (LightGBM Ranker)                        │
│                                                                                 │
│  Features:                                                                      │
│   • Two-Tower Cosine Retrieval Score                                            │
│   • User Profile Statistics (Activity count, mean rating, rating variance)      │
│   • Item Profile Statistics (Popularity, mean rating, variance)                 │
│   • Explicit Genre Affinity (Overlap with user top-3 preferred genres)          │
│                                         │                                       │
│                                         ▼                                       │
│                           [LightGBM Decision Trees]                             │
│                           (Training ROC-AUC: 0.8019)                            │
│                                         │                                       │
│                                         ▼                                       │
│                         Top 10 Final Recommendations                            │
│                       (+4.0% NDCG@10 gain over Stage 1)                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Benchmark Results

| Stage / Component | Metric | Value |
| :--- | :--- | :--- |
| **Stage 1: Retrieval Latency** | FAISS HNSW Query Time | **0.098 ms** / query |
| **Stage 1: Vector Space** | Embedding Dimension | 64-dim L2-normalized |
| **Stage 2: Re-Ranker Quality** | LightGBM ROC-AUC | **0.8019** |
| **Two-Stage Pipeline vs Stage 1** | **NDCG@10 Improvement** | **+4.0% relative gain** |
| **Stage 1 HitRate** | HitRate@10 (strict test set) | 25.67% |

---

## Hyperparameter Ablation Study (Run 1 vs. Run 2)

To evaluate the engineering trade-offs between training duration, index complexity, and ranking accuracy, an ablation experiment was performed comparing the production baseline against a deep-tuned configuration:

| Parameter / Metric | Production Baseline (Run 1) | Deep-Tuned Ablation (Run 2) | Engineering Finding |
| :--- | :---: | :---: | :--- |
| **Training Epochs** | `5` | `10` | 5 epochs achieved the optimal generalization frontier; 10 epochs began memorizing co-occurrences. |
| **Candidate Pool (`top_k`)** | `100` | `150` | Expanding to 150 items introduced **candidate dilution**, degrading test NDCG from 0.0399 to 0.0367. |
| **HNSW Complexity (`m / ef`)** | `32 / 64` | `48 / 128` | Doubling graph connectivity increased search latency ($0.098 \to 0.180$ ms) with no recall advantage. |
| **LightGBM Training Set** | `100,000` | `200,000` | Larger sample induced popularity bias towards global blockbusters over niche personalized relevance. |
| **Two-Stage NDCG@10** | **0.0399** | `0.0367` | **Run 1 outperforms Run 2 by +8.7% in ranking accuracy.** |
| **Relative NDCG Gain** | **+4.0%** | `+2.3%` | Stage-2 re-ranking adds higher marginal utility when filtering 100 clean candidates. |
| **Average Query Latency** | **0.098 ms** | `0.180 ms` | **Run 1 is 1.8x faster** and well within production SLAs (<10 ms). |

> **Conclusion:** Run 1 was locked in as the official production system because it maximizes ranking performance while delivering sub-0.1ms retrieval latency.

---

## Directory Structure

```
two_tower_recsys/
├── .env                          # Configuration variables
├── .gitignore                    # Ignored artifacts and caches
├── README.md                     # Documentation and interview cheat-sheet
├── data/
│   └── ml-1m/                    # MovieLens-1M dataset
├── models/
│   ├── two_tower.pt              # PyTorch trained weights
│   ├── faiss_hnsw.index          # Built HNSW vector graph index
│   └── lightgbm_reranker.txt     # LightGBM booster model
├── notebooks/
│   └── two_tower_demo.ipynb      # Interactive VS Code notebook demo
└── src/
    ├── data_preprocessing.py     # Temporal splitting & negative sampling
    ├── two_tower_model.py        # PyTorch User & Item Towers
    ├── vector_search.py          # FAISS HNSW indexing & latency benchmarks
    ├── reranker.py               # Feature engineering & LightGBM ranker
    ├── evaluate.py               # HitRate@K, NDCG@K, Recall@K metrics
    ├── train_pipeline.py         # End-to-end training and benchmark script
    └── interactive_inference.py  # Live CLI query tool for any user
```

---

## Quick Usage

### 1. Run Interactive Recommendations for Any User
Test the two-stage pipeline on any user ID:
```bash
# Known user (Full Two-Stage Pipeline)
python src/interactive_inference.py --user_id 42 --top_k 5

# Cold-Start Tier 1: Demographic Cluster Retrieval (Unknown user with demographics)
python src/interactive_inference.py --user_id 9999 --gender F --age 18-24 --top_k 5

# Cold-Start Tier 2: Pure Anonymous Fallback (IMDb Bayesian Weighted Rating)
python src/interactive_inference.py --user_id 9999 --top_k 5
```

### 2. Run the Interactive Jupyter Notebook
Open [`notebooks/two_tower_demo.ipynb`](file:///D:/Users/shrey/desktop/res/two_tower_recsys/notebooks/two_tower_demo.ipynb) inside VS Code and select `.venv` as your Python kernel.

---

## Production Cold-Start Strategy

In real-world deployment, dropping user traffic because an ID is unrecognized is unacceptable. The pipeline implements a 2-tier fallback architecture:

1. **Tier 1 — Demographic Warm Cold-Start:** When an unknown user provides profile metadata upon registration (Gender, Age, Occupation), their vector is generated by pairing the global population-mean user vector with their specific demographic embeddings (`UserTower.forward_demographic`). FAISS searches the movie vector space for this demographic cluster in sub-millisecond latency.
2. **Tier 2 — Pure Anonymous Cold-Start:** For guest/unregistered users without demographic attributes, the system falls back to an **IMDb-style Bayesian Weighted Rating**:
   $$WR = \left(\frac{v}{v + m}\right) \cdot R + \left(\frac{m}{v + m}\right) \cdot C$$
   where $v$ is vote count, $m$ is the 80th-percentile minimum vote threshold, $R$ is item mean rating, and $C$ is the global catalog mean ($\approx 3.58$). This guarantees high-prestige, high-consensus classics (*The Shawshank Redemption*, *The Godfather*) are served rather than noisy items with 1 single 5-star vote.

---

## Interview Defense Cheat-Sheet

| Question | Defensible Answer |
| :--- | :--- |
| **Why not just rank all movies with one complex model?** | *"Scoring all 50,000+ items with complex cross-features violates real-time SLAs (<50ms). We decouple the pipeline into high-recall candidate retrieval via FAISS (<0.1ms) and high-precision re-ranking via LightGBM on the top-100 candidates."* |
| **How do you avoid data leakage during training?** | *"We performed a chronological temporal split per user: earlier 80% ratings for training, latest 20% for testing. Random 80/20 splitting leaks future interactions into past predictions."* |
| **Why HNSW over exact search?** | *"Exact search scales linearly $\mathcal{O}(N \cdot d)$, whereas HNSW is a multi-layer proximity graph with logarithmic $\mathcal{O}(\log N)$ query complexity, achieving sub-millisecond retrieval."* |
| **Why use NDCG instead of just Accuracy?** | *"Recommendations are ranked lists. NDCG uses a logarithmic position discount: predicting a hit at Rank 1 is penalized much less than predicting it at Rank 10."* |
| **How does your system handle Cold-Start?** | *"We implement a 2-tier fallback: if demographics are supplied, we pass the demographic embeddings through the User Tower with a population-mean user anchor for FAISS retrieval. If zero profile info exists, we fall back to a Bayesian Weighted Rating (Dirichlet/IMDb formula) to serve consensus classics without popularity noise."* |
