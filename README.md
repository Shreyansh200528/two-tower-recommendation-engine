# Two-Tower Recommendation Engine with FAISS HNSW & LightGBM Re-Ranking

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-HNSW_Vector_Search-008080)](https://github.com/facebookresearch/faiss)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-brightgreen)](https://github.com/microsoft/LightGBM)
[![Tests](https://img.shields.io/badge/Tests-16%20Passed-success)](tests/)

An industrial-grade, two-stage recommendation system built on the **MovieLens-1M** benchmark (1,000,209 ratings across 6,040 users and 3,883 movies). The architecture decouples large-scale candidate retrieval from high-precision personalized re-ranking, achieving **sub-0.1ms retrieval latency** and a **+4.0% relative improvement in NDCG@10**.

---

## Architecture Overview

Modern production recommendation engines (e.g., YouTube, Netflix, Instagram) cannot score an entire catalog of millions of items in real time. This system implements an industry-standard **two-stage funnel architecture**:

```
                       [Incoming Request: User ID / Demographics]
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│               STAGE 1: CANDIDATE RETRIEVAL (Two-Tower + FAISS HNSW)             │
│                                                                                 │
│  User Demographics ──► [User Tower (PyTorch)] ──► 64-dim Vector u               │
│                                                          │                      │
│                                                          ▼                      │
│                                             [FAISS HNSW Vector Index]           │
│                                              (3,883 Movie Vectors v)            │
│                                                          │                      │
│                                              Latency: 0.098 ms / query          │
│                                                          │                      │
│                                                          ▼                      │
│                                                 Top 100 Candidates              │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                        [Seen-Item Deduplication Filter]
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: RE-RANKING (LightGBM Ranker)                        │
│                                                                                 │
│  Feature Engineering:                                                           │
│   • Semantic Cosine Retrieval Similarity                                        │
│   • User Profile Statistics (Activity volume, historical mean rating & variance)│
│   • Item Profile Statistics (Popularity, historical mean rating & variance)     │
│   • Cross-Affinity: Rating differential & User-Top-3-Genre overlap              │
│                                         │                                       │
│                                         ▼                                       │
│                           [LightGBM Boosted Trees]                              │
│                           (Training ROC-AUC: 0.8019)                            │
│                                         │                                       │
│                                         ▼                                       │
│                          Top 10 Personalized Recommendations                    │
│                        (+4.0% NDCG@10 Gain over Retrieval)                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key System Features

* **Dual-Encoder Representation Learning:** Separate PyTorch neural towers for users (embedding demographic features: age, gender, occupation) and items (embedding item IDs and multi-hot genre vectors), projecting both into a shared, L2-normalized 64-dimensional space.
* **Sub-Millisecond Vector Search:** Approximate Nearest Neighbor (ANN) search powered by Meta FAISS using a **Hierarchical Navigable Small World (HNSW)** proximity graph ($\mathcal{O}(\log N)$ query complexity).
* **Strict Temporal Partitioning:** Users' interactions are partitioned chronologically (earliest 80% train, latest 20% test per user), strictly eliminating future data leakage.
* **Deduplication & Novelty Filtering:** Candidates previously interacted with by a user are dynamically filtered prior to re-ranking to guarantee 100% catalog novelty during inference.
* **Two-Tier Cold-Start Architecture:**
  * **Tier 1 (Demographic Warm Fallback):** New users with demographic metadata are assigned a population-mean user vector combined with their demographic embeddings, retrieving demographic-cluster favorites in sub-millisecond latency.
  * **Tier 2 (Pure Anonymous Fallback):** Anonymous guest requests default to an **IMDb-style Bayesian Weighted Rating**:
    $$WR = \left(\frac{v}{v + m}\right) R + \left(\frac{m}{v + m}\right) C$$
    where $m$ is the 80th-percentile minimum vote threshold ($\approx 250$ votes) and $C$ is the catalog mean ($\approx 3.58$), suppressing low-sample rating noise.

---

## Benchmark Results

Evaluated across held-out chronological test sets over 300 active users:

| Component | Metric | Performance |
| :--- | :--- | :--- |
| **Stage 1 Retrieval Latency** | FAISS HNSW (`M=32`, `efSearch=64`) | **0.098 ms** / query |
| **Stage 1 Exact Flat Latency** | Brute-force Inner Product | **0.127 ms** / query |
| **Stage 1 Representation** | Embedding Dimensionality | 64-dim (L2-normalized) |
| **Stage 2 Discriminative Power** | LightGBM Re-Ranker ROC-AUC | **0.8019** |
| **Stage 1 HitRate** | HitRate@10 (strict hold-out) | **25.67%** |
| **End-to-End Ranking Gain** | **NDCG@10 Improvement (Stage 2 vs. Stage 1)** | **+4.0% relative gain** |

---

## Hyperparameter Ablation Study

To evaluate architectural trade-offs between compute footprint, index complexity, and ranking accuracy, an ablation experiment was conducted comparing the production configuration against a higher-capacity setup:

| Parameter / Metric | Production Pipeline (Run 1) | High-Capacity Ablation (Run 2) | Analysis & Trade-Off |
| :--- | :---: | :---: | :--- |
| **Training Epochs** | `5` | `10` | 5 epochs achieved optimal generalization; 10 epochs induced slight co-occurrence overfitting. |
| **Candidate Pool (`top_k`)** | `100` | `150` | Expanding retrieval to 150 items introduced **candidate dilution**, degrading test NDCG from 0.0399 to 0.0367. |
| **HNSW Complexity (`M / ef`)** | `32 / 64` | `48 / 128` | Higher graph density increased latency ($0.098 \to 0.180$ ms) with no measurable recall gain on this catalog size. |
| **LightGBM Training Set** | `100,000` pairs | `200,000` pairs | Doubling samples increased popularity bias toward global blockbusters over personalized niche relevance. |
| **Two-Stage NDCG@10** | **0.0399** | `0.0367` | **Run 1 delivers an +8.7% higher ranking accuracy.** |
| **Relative NDCG Gain** | **+4.0%** | `+2.3%` | Stage 2 provides higher marginal utility when operating on cleaner, top-100 candidates. |
| **Retrieval Query Latency** | **0.098 ms** | `0.180 ms` | **Run 1 is 1.8x faster**, preserving headroom for downstream feature extraction. |

---

## Project Structure

```
two_tower_recsys/
├── data/
│   └── ml-1m/                    # MovieLens-1M dataset
├── models/
│   ├── two_tower.pt              # Trained PyTorch dual-tower weights
│   ├── faiss_hnsw.index          # Serialized HNSW graph index
│   └── lightgbm_reranker.txt     # Serialized LightGBM booster
├── models_baseline_run1/         # Archived Run 1 production artifacts & metrics
├── models_ablation_run2/         # Archived Run 2 ablation artifacts & metrics
├── notebooks/
│   └── two_tower_demo.ipynb      # Interactive step-by-step walkthrough
├── src/
│   ├── cold_start.py             # 2-Tier demographic & Bayesian fallback handlers
│   ├── data_preprocessing.py     # Temporal partitioning, ID mapping & negative sampling
│   ├── evaluate.py               # Vectorized HitRate@K, NDCG@K, Recall@K evaluators
│   ├── interactive_inference.py  # Production CLI inference entry-point
│   ├── reranker.py               # Feature extraction & LightGBM ranking engine
│   ├── train_pipeline.py         # End-to-end training, indexing & benchmarking script
│   ├── two_tower_model.py        # PyTorch User & Item Tower architectures
│   └── vector_search.py          # FAISS HNSW build, query & latency benchmarks
├── tests/
│   ├── test_data.py              # Temporal split zero-leakage & negative sampling tests
│   ├── test_models.py            # Tensor shapes, strict L2-normalization & boundary tests
│   ├── test_pipeline.py          # Feature schema, score bounds & cold-start tests
│   └── test_retrieval.py         # FAISS catalog integrity & <1.5ms latency SLA tests
├── .env                          # Pipeline configuration variables
├── .gitignore                    # Git tracking ignore rules
├── requirements.txt              # Pinned environment dependencies
└── README.md                     # System documentation
```

---

## Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/<YOUR_USERNAME>/two-tower-recsys.git
cd two-tower-recsys

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .\.venv\Scripts\Activate.ps1  # On Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 2. Interactive Recommendation Inference

```bash
# Known User (Full Two-Stage Pipeline with Seen-Item Deduplication)
python src/interactive_inference.py --user_id 42 --top_k 5

# Cold-Start Tier 1: Demographic Warm Retrieval (Unknown User with Demographics)
python src/interactive_inference.py --user_id 9999 --gender F --age 18-24 --top_k 5

# Cold-Start Tier 2: Pure Anonymous Fallback (IMDb Bayesian Weighted Classics)
python src/interactive_inference.py --user_id 9999 --top_k 5
```

### 3. Automated Test Suite

Run the full pytest test suite covering embedding invariants, latency SLAs, temporal leakage prevention, and pipeline integration:

```bash
pytest tests/ -v
```

All 16 unit and integration tests execute in $\sim 5$ seconds:

```text
tests/test_data.py::test_temporal_split_zero_data_leakage PASSED         [  6%]
tests/test_data.py::test_negative_sampling_disjoint_from_positives PASSED [ 12%]
tests/test_data.py::test_feature_indices_continuity PASSED               [ 18%]
tests/test_models.py::test_user_tower_output_shape PASSED                [ 25%]
tests/test_models.py::test_item_tower_output_shape PASSED                [ 31%]
tests/test_models.py::test_user_embeddings_l2_normalized PASSED          [ 37%]
tests/test_models.py::test_item_embeddings_l2_normalized PASSED          [ 43%]
tests/test_models.py::test_edge_case_user_and_movie_ids PASSED           [ 50%]
tests/test_pipeline.py::test_feature_engineering_schema_and_types PASSED [ 56%]
tests/test_pipeline.py::test_reranker_probability_bounds PASSED          [ 62%]
tests/test_pipeline.py::test_end_to_end_recommendation_pipeline PASSED   [ 68%]
tests/test_pipeline.py::test_cold_start_tier2_bayesian_fallback PASSED   [ 75%]
tests/test_pipeline.py::test_cold_start_tier1_demographic_cluster PASSED [ 81%]
tests/test_retrieval.py::test_faiss_index_total_items PASSED             [ 87%]
tests/test_retrieval.py::test_top_k_uniqueness_and_validity PASSED       [ 93%]
tests/test_retrieval.py::test_retrieval_latency_sla PASSED               [100%]
============================== 16 passed in 4.84s ==============================
```

---

## License

This project is licensed under the MIT License.
