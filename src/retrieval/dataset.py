"""
PyTorch Dataset and DataLoader utilities for the Two-Tower retrieval model.

Joins interaction rows with pre-computed user/item feature DataFrames and
exposes each sample as (user_id, user_features, item_id, item_features, label).
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class InteractionDataset(Dataset):
    """Dataset that pairs each user–item interaction with dense feature vectors.

    Parameters
    ----------
    interactions_df : pd.DataFrame
        Must contain at least ``user_id``, ``item_id``, and ``label`` columns.
    user_features_df : pd.DataFrame
        Dense user features indexed by ``user_id`` (8 float columns).
    item_features_df : pd.DataFrame
        Dense item features indexed by ``item_id`` (7 float columns).
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> None:
        super().__init__()

        self.user_ids = torch.tensor(
            interactions_df["user_id"].values, dtype=torch.long
        )
        self.item_ids = torch.tensor(
            interactions_df["item_id"].values, dtype=torch.long
        )
        self.labels = torch.tensor(
            interactions_df["label"].values, dtype=torch.float32
        )

        # Pre-compute feature tensors for fast lookup --------------------------
        # Ensure the index is user_id / item_id for .loc lookups
        if user_features_df.index.name != "user_id":
            if "user_id" in user_features_df.columns:
                user_features_df = user_features_df.set_index("user_id")

        if item_features_df.index.name != "item_id":
            if "item_id" in item_features_df.columns:
                item_features_df = item_features_df.set_index("item_id")

        self.num_user_features = len(user_features_df.columns)  # expected: 8
        self.num_item_features = len(item_features_df.columns)  # expected: 7

        # Build dense lookup tables keyed by ID for O(1) access ---------------
        self._user_feat_dict: dict[int, torch.Tensor] = {}
        for uid, row in user_features_df.iterrows():
            self._user_feat_dict[int(uid)] = torch.tensor(
                row.values, dtype=torch.float32
            )

        self._item_feat_dict: dict[int, torch.Tensor] = {}
        for iid, row in item_features_df.iterrows():
            self._item_feat_dict[int(iid)] = torch.tensor(
                row.values, dtype=torch.float32
            )

        # Zero-vector fallbacks for missing IDs
        self._user_feat_zeros = torch.zeros(self.num_user_features, dtype=torch.float32)
        self._item_feat_zeros = torch.zeros(self.num_item_features, dtype=torch.float32)

    # ---------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int):
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        label = self.labels[idx]

        user_features = self._user_feat_dict.get(
            user_id.item(), self._user_feat_zeros
        )
        item_features = self._item_feat_dict.get(
            item_id.item(), self._item_feat_zeros
        )

        return user_id, user_features, item_id, item_features, label


# --------------------------------------------------------------------------- #
# DataLoader factory
# --------------------------------------------------------------------------- #

def create_dataloaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    batch_size: int = 1024,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build train and test :class:`DataLoader` instances.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        Interaction DataFrames with ``user_id``, ``item_id``, ``label``.
    user_features, item_features : pd.DataFrame
        Dense feature DataFrames indexed (or with column) by ``user_id`` / ``item_id``.
    batch_size : int
        Mini-batch size.
    num_workers : int
        Number of DataLoader worker processes (0 for Windows compatibility).

    Returns
    -------
    tuple[DataLoader, DataLoader]
        ``(train_loader, test_loader)``
    """
    train_dataset = InteractionDataset(train_df, user_features, item_features)
    test_dataset = InteractionDataset(test_df, user_features, item_features)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, test_loader
