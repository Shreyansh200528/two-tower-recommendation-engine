"""
Two-Tower Recommendation Model in PyTorch.
Consists of:
- UserTower: Embeds user ID and demographic features (gender, age, occupation).
- ItemTower: Embeds item ID and multi-hot genre features.
Outputs L2-normalized 64-dimensional dense vectors for vector similarity search.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, num_users, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, 32)
        self.gender_embedding = nn.Embedding(2, 4)
        self.age_embedding = nn.Embedding(7, 8)
        self.occ_embedding = nn.Embedding(21, 8)
        
        # Total input dimension = 32 + 4 + 8 + 8 = 52
        in_dim = 32 + 4 + 8 + 8
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, user_features):
        """
        user_features: [batch_size, 4] containing:
          [user_idx, gender_idx, age_idx, occ_idx]
        """
        u_emb = self.user_embedding(user_features[:, 0])
        g_emb = self.gender_embedding(user_features[:, 1])
        a_emb = self.age_embedding(user_features[:, 2])
        o_emb = self.occ_embedding(user_features[:, 3])
        
        x = torch.cat([u_emb, g_emb, a_emb, o_emb], dim=-1)
        out = self.mlp(x)
        # L2 Normalize for cosine similarity & FAISS Inner-Product search
        return F.normalize(out, p=2, dim=-1)

    def forward_demographic(self, gender_idx, age_idx, occ_idx):
        """
        Demographic cold-start inference: When user_idx is unknown, uses
        population-mean user embedding combined with explicit demographic embeddings.
        """
        device = self.user_embedding.weight.device
        mean_u_emb = self.user_embedding.weight.mean(dim=0, keepdim=True)
        
        g_emb = self.gender_embedding(torch.tensor([gender_idx], dtype=torch.long, device=device))
        a_emb = self.age_embedding(torch.tensor([age_idx], dtype=torch.long, device=device))
        o_emb = self.occ_embedding(torch.tensor([occ_idx], dtype=torch.long, device=device))
        
        x = torch.cat([mean_u_emb, g_emb, a_emb, o_emb], dim=-1)
        out = self.mlp(x)
        return F.normalize(out, p=2, dim=-1)


class ItemTower(nn.Module):
    def __init__(self, num_items, num_genres=18, embedding_dim=64):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, 32)
        self.genre_dense = nn.Linear(num_genres, 16)
        
        # Total input dimension = 32 + 16 = 48
        in_dim = 32 + 16
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, item_indices, item_genres):
        """
        item_indices: [batch_size]
        item_genres:  [batch_size, 18] multi-hot float tensor
        """
        i_emb = self.item_embedding(item_indices)
        g_emb = F.relu(self.genre_dense(item_genres))
        
        x = torch.cat([i_emb, g_emb], dim=-1)
        out = self.mlp(x)
        # L2 Normalize for cosine similarity & FAISS Inner-Product search
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, num_genres=18, embedding_dim=64):
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim)
        self.item_tower = ItemTower(num_items, num_genres, embedding_dim)
        # Learnable logit scale (temperature inverse)
        self.logit_scale = nn.Parameter(torch.ones([]) * 5.0)

    def forward(self, user_features, item_indices, item_genres):
        user_emb = self.user_tower(user_features)          # [B, 64]
        item_emb = self.item_tower(item_indices, item_genres) # [B, 64]
        
        # Cosine similarity is the dot product of L2-normalized vectors
        cosine_sim = (user_emb * item_emb).sum(dim=-1)     # [B]
        logits = cosine_sim * self.logit_scale
        return logits, user_emb, item_emb
