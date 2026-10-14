"""
Unit tests for Two-Tower PyTorch model integrity and embedding normalization.
"""

import pytest
import torch

def test_user_tower_output_shape(two_tower_model, preprocessor):
    """
    Verify User Tower outputs exact tensor shape [batch_size, embedding_dim].
    """
    batch_size = 16
    sample_features = preprocessor.user_features[:batch_size]
    
    with torch.no_grad():
        user_embs = two_tower_model.user_tower(sample_features)
        
    assert user_embs.shape == (batch_size, 64), (
        f"Expected shape ({batch_size}, 64), got {user_embs.shape}"
    )


def test_item_tower_output_shape(two_tower_model, preprocessor):
    """
    Verify Item Tower outputs exact tensor shape [batch_size, embedding_dim].
    """
    batch_size = 32
    item_indices = torch.arange(batch_size, dtype=torch.long)
    item_genres = preprocessor.item_genres[:batch_size]
    
    with torch.no_grad():
        item_embs = two_tower_model.item_tower(item_indices, item_genres)
        
    assert item_embs.shape == (batch_size, 64), (
        f"Expected shape ({batch_size}, 64), got {item_embs.shape}"
    )


def test_user_embeddings_l2_normalized(two_tower_model, preprocessor):
    """
    Assert that User Tower embeddings are strictly L2-normalized
    (||v||_2 == 1.0 within 1e-4 tolerance).
    """
    sample_features = preprocessor.user_features[:100]
    
    with torch.no_grad():
        user_embs = two_tower_model.user_tower(sample_features)
        
    norms = torch.norm(user_embs, p=2, dim=-1)
    expected_ones = torch.ones_like(norms)
    
    assert torch.allclose(norms, expected_ones, atol=1e-4), (
        f"User embedding norms deviate from 1.0: min={norms.min():.6f}, max={norms.max():.6f}"
    )


def test_item_embeddings_l2_normalized(two_tower_model, preprocessor):
    """
    Assert that Item Tower embeddings are strictly L2-normalized
    (||v||_2 == 1.0 within 1e-4 tolerance).
    """
    item_indices = torch.arange(preprocessor.num_movies, dtype=torch.long)
    item_genres = preprocessor.item_genres
    
    with torch.no_grad():
        item_embs = two_tower_model.item_tower(item_indices, item_genres)
        
    norms = torch.norm(item_embs, p=2, dim=-1)
    expected_ones = torch.ones_like(norms)
    
    assert torch.allclose(norms, expected_ones, atol=1e-4), (
        f"Item embedding norms deviate from 1.0: min={norms.min():.6f}, max={norms.max():.6f}"
    )


def test_edge_case_user_and_movie_ids(two_tower_model, preprocessor):
    """
    Verify behavior on boundary index edge cases:
    - Minimum valid user/movie index: 0
    - Maximum valid user index: num_users - 1
    - Maximum valid movie index: num_movies - 1
    """
    min_user_idx = 0
    max_user_idx = preprocessor.num_users - 1
    
    min_movie_idx = 0
    max_movie_idx = preprocessor.num_movies - 1
    
    # Boundary user inputs
    edge_user_features = torch.stack([
        preprocessor.user_features[min_user_idx],
        preprocessor.user_features[max_user_idx]
    ])
    
    # Boundary movie inputs
    edge_movie_indices = torch.tensor([min_movie_idx, max_movie_idx], dtype=torch.long)
    edge_movie_genres = torch.stack([
        preprocessor.item_genres[min_movie_idx],
        preprocessor.item_genres[max_movie_idx]
    ])
    
    with torch.no_grad():
        u_embs = two_tower_model.user_tower(edge_user_features)
        i_embs = two_tower_model.item_tower(edge_movie_indices, edge_movie_genres)
        logits, _, _ = two_tower_model(edge_user_features, edge_movie_indices, edge_movie_genres)
        
    # Shape and norm assertions for edge inputs
    assert u_embs.shape == (2, 64)
    assert i_embs.shape == (2, 64)
    assert logits.shape == (2,)
    
    u_norms = torch.norm(u_embs, p=2, dim=-1)
    i_norms = torch.norm(i_embs, p=2, dim=-1)
    assert torch.allclose(u_norms, torch.ones_like(u_norms), atol=1e-4)
    assert torch.allclose(i_norms, torch.ones_like(i_norms), atol=1e-4)
