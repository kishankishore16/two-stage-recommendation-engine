"""
Two-Tower neural network for candidate retrieval.

* **UserTower** — embeds (user_id, user_features) → 64-d normalised vector.
* **ItemTower** — embeds (item_id, item_features) → 64-d normalised vector.
* **TwoTowerModel** — thin wrapper that exposes both towers and convenience
  encode helpers for inference / index-building.
* **infonce_loss** — InfoNCE (NT-Xent) loss using in-batch negatives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# User Tower
# --------------------------------------------------------------------------- #

class UserTower(nn.Module):
    """Maps (user_id, user_dense_features) → L2-normalised embedding.

    Architecture
    ------------
    Embedding(num_users, 32)  ─┐
                                ├─ concat → [40] → 256 → BN → ReLU → Drop
    8 dense features          ─┘                → 128 → BN → ReLU → Drop
                                                → 64  → L2-norm
    """

    def __init__(self, num_users: int, num_features: int = 8) -> None:
        super().__init__()
        id_emb_dim = 32
        input_dim = id_emb_dim + num_features  # 32 + 8 = 40

        self.id_embedding = nn.Embedding(num_users, id_emb_dim, padding_idx=0)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
        )

    def forward(self, user_ids: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """Return (batch_size, 64) L2-normalised user embedding."""
        id_emb = self.id_embedding(user_ids)                 # (B, 32)
        x = torch.cat([id_emb, user_features], dim=-1)       # (B, 40)
        x = self.mlp(x)                                      # (B, 64)
        return F.normalize(x, p=2, dim=-1)


# --------------------------------------------------------------------------- #
# Item Tower
# --------------------------------------------------------------------------- #

class ItemTower(nn.Module):
    """Maps (item_id, item_dense_features) → L2-normalised embedding.

    Architecture
    ------------
    Embedding(num_items, 32)  ─┐
                                ├─ concat → [39] → 256 → BN → ReLU → Drop
    7 dense features          ─┘                → 128 → BN → ReLU → Drop
                                                → 64  → L2-norm
    """

    def __init__(self, num_items: int, num_features: int = 7) -> None:
        super().__init__()
        id_emb_dim = 32
        input_dim = id_emb_dim + num_features  # 32 + 7 = 39

        self.id_embedding = nn.Embedding(num_items, id_emb_dim, padding_idx=0)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
        )

    def forward(self, item_ids: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """Return (batch_size, 64) L2-normalised item embedding."""
        id_emb = self.id_embedding(item_ids)                 # (B, 32)
        x = torch.cat([id_emb, item_features], dim=-1)       # (B, 39)
        x = self.mlp(x)                                      # (B, 64)
        return F.normalize(x, p=2, dim=-1)


# --------------------------------------------------------------------------- #
# Two-Tower wrapper
# --------------------------------------------------------------------------- #

class TwoTowerModel(nn.Module):
    """Dual-encoder retrieval model.

    Wraps a :class:`UserTower` and :class:`ItemTower` and exposes separate
    ``encode_user`` / ``encode_item`` methods for inference.
    """

    def __init__(self, num_users: int, num_items: int) -> None:
        super().__init__()
        self.user_tower = UserTower(num_users)
        self.item_tower = ItemTower(num_items)

    def forward(
        self,
        user_ids: torch.Tensor,
        user_features: torch.Tensor,
        item_ids: torch.Tensor,
        item_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(user_emb, item_emb)`` — each (B, 64) L2-normalised."""
        user_emb = self.user_tower(user_ids, user_features)
        item_emb = self.item_tower(item_ids, item_features)
        return user_emb, item_emb

    # ── Inference helpers -------------------------------------------------- #

    @torch.no_grad()
    def encode_user(
        self, user_ids: torch.Tensor, user_features: torch.Tensor
    ) -> torch.Tensor:
        """Encode users (inference mode, no grad)."""
        self.user_tower.eval()
        return self.user_tower(user_ids, user_features)

    @torch.no_grad()
    def encode_item(
        self, item_ids: torch.Tensor, item_features: torch.Tensor
    ) -> torch.Tensor:
        """Encode items (inference mode, no grad)."""
        self.item_tower.eval()
        return self.item_tower(item_ids, item_features)


# --------------------------------------------------------------------------- #
# InfoNCE loss (in-batch negatives)
# --------------------------------------------------------------------------- #

def infonce_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute InfoNCE (NT-Xent) loss with in-batch negatives.

    Parameters
    ----------
    user_emb : (B, D) L2-normalised user embeddings.
    item_emb : (B, D) L2-normalised item embeddings.
    temperature : float
        Softmax temperature (lower → sharper distribution).

    Returns
    -------
    torch.Tensor
        Scalar loss value.

    Notes
    -----
    The positive pairs are along the diagonal of the similarity matrix
    ``user_emb @ item_emb.T``.  Every off-diagonal entry acts as a negative.
    """
    # Cosine similarity matrix (since embeddings are L2-normalised)
    logits = torch.matmul(user_emb, item_emb.t()) / temperature  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)   # [0,1,..,B-1]
    return F.cross_entropy(logits, labels)
