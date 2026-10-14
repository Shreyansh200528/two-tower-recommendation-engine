"""
Pytest configuration and session-scoped fixtures for Two-Tower RecSys test suite.
"""

import os
import sys
import pytest
import torch
import faiss
import lightgbm as lgb

# Ensure src/ is importable
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_preprocessing import MovieLens1MPreprocessor
from two_tower_model import TwoTowerModel
from reranker import LightGBMReranker

@pytest.fixture(scope="session")
def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@pytest.fixture(scope="session")
def preprocessor(project_root):
    data_dir = os.path.join(project_root, "data", "ml-1m")
    prep = MovieLens1MPreprocessor(data_dir=data_dir).process()
    return prep

@pytest.fixture(scope="session")
def train_test_data(preprocessor):
    return preprocessor.temporal_train_test_split(test_ratio=0.2)

@pytest.fixture(scope="session")
def two_tower_model(preprocessor, project_root):
    model_path = os.path.join(project_root, "models", "two_tower.pt")
    model = TwoTowerModel(
        num_users=preprocessor.num_users,
        num_items=preprocessor.num_movies,
        num_genres=18,
        embedding_dim=64
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

@pytest.fixture(scope="session")
def faiss_index(project_root):
    index_path = os.path.join(project_root, "models", "faiss_hnsw.index")
    index = faiss.read_index(index_path)
    return index

@pytest.fixture(scope="session")
def reranker(preprocessor, train_test_data, project_root):
    train_df, _ = train_test_data
    reranker_obj = LightGBMReranker()
    reranker_obj.extract_historical_stats(train_df, preprocessor.movies_meta)
    model_path = os.path.join(project_root, "models", "lightgbm_reranker.txt")
    reranker_obj.model = lgb.Booster(model_file=model_path)
    return reranker_obj
