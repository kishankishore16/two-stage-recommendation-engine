"""
Training loop for the Two-Tower retrieval model.

* ``train_epoch``  — single training epoch.
* ``evaluate``     — Hit-Rate@K and NDCG@K evaluation.
* ``train_two_tower`` — end-to-end training entry point.
"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import get_config
from src.retrieval.dataset import InteractionDataset, create_dataloaders
from src.retrieval.model import TwoTowerModel, infonce_loss

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Single-epoch training
# --------------------------------------------------------------------------- #

def train_epoch(
    model: TwoTowerModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    cfg = get_config().retrieval

    for user_ids, user_feats, item_ids, item_feats, _labels in tqdm(
        dataloader, desc="  train", leave=False
    ):
        user_ids = user_ids.to(device)
        user_feats = user_feats.to(device)
        item_ids = item_ids.to(device)
        item_feats = item_feats.to(device)

        optimizer.zero_grad()
        user_emb, item_emb = model(user_ids, user_feats, item_ids, item_feats)
        loss = infonce_loss(user_emb, item_emb, temperature=cfg.temperature)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(
    model: TwoTowerModel,
    dataloader: DataLoader,
    all_item_embeddings: torch.Tensor,
    device: torch.device,
    k: int = 100,
) -> dict[str, float]:
    """Compute Hit-Rate@K and NDCG@K over the test set.

    For each test interaction the true positive item is known.  We score the
    user embedding against **all** item embeddings and check whether the
    true item appears in the top-*k*.

    Parameters
    ----------
    model : TwoTowerModel
    dataloader : DataLoader
        Test dataloader (only positive interactions used).
    all_item_embeddings : (num_items, 64) tensor — precomputed.
    device : torch.device
    k : int
        Cut-off for the ranking metrics.

    Returns
    -------
    dict with ``hit_rate_at_k`` and ``ndcg_at_k``.
    """
    model.eval()
    hits = 0
    ndcg_sum = 0.0
    total = 0

    # all_item_embeddings: (num_items, D) on device
    all_item_embeddings = all_item_embeddings.to(device)

    for user_ids, user_feats, item_ids, _item_feats, labels in tqdm(
        dataloader, desc="  eval ", leave=False
    ):
        # Only evaluate on positive interactions
        pos_mask = labels == 1.0
        if pos_mask.sum() == 0:
            continue

        user_ids = user_ids[pos_mask].to(device)
        user_feats = user_feats[pos_mask].to(device)
        true_item_ids = item_ids[pos_mask]  # keep on CPU for indexing

        user_emb = model.encode_user(user_ids, user_feats)  # (B', D)

        # Cosine similarity (embeddings are L2-normed)
        scores = torch.matmul(user_emb, all_item_embeddings.t())  # (B', num_items)
        _, topk_indices = scores.topk(k, dim=-1)  # (B', k) — item indices

        topk_indices = topk_indices.cpu()
        true_item_ids = true_item_ids.unsqueeze(1)  # (B', 1)

        # Hit-Rate: is the true item anywhere in top-k?
        hit_mask = (topk_indices == true_item_ids).any(dim=-1)  # (B',)
        hits += hit_mask.sum().item()

        # NDCG: 1/log2(rank+1) if hit, else 0
        match_positions = (topk_indices == true_item_ids).float()  # (B', k)
        ranks = match_positions.argmax(dim=-1).float() + 1  # 1-indexed
        # Only count if there actually was a hit
        ndcg_scores = hit_mask.float() / torch.log2(ranks + 1)
        ndcg_sum += ndcg_scores.sum().item()

        total += pos_mask.sum().item()

    hit_rate = hits / max(total, 1)
    ndcg = ndcg_sum / max(total, 1)

    return {"hit_rate_at_k": hit_rate, "ndcg_at_k": ndcg}


# --------------------------------------------------------------------------- #
# Pre-compute all item embeddings
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _build_all_item_embeddings(
    model: TwoTowerModel,
    item_features_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Encode every item in the catalogue.  Returns (num_items, 64) on *CPU*."""
    model.eval()

    if item_features_df.index.name != "item_id":
        if "item_id" in item_features_df.columns:
            item_features_df = item_features_df.set_index("item_id")

    num_items = item_features_df.index.max() + 1
    all_ids = torch.arange(num_items, dtype=torch.long)

    # Build a dense feature matrix (zero-filled for missing IDs)
    feat_cols = item_features_df.columns
    feat_matrix = np.zeros((num_items, len(feat_cols)), dtype=np.float32)
    for iid, row in item_features_df.iterrows():
        if 0 <= int(iid) < num_items:
            feat_matrix[int(iid)] = row.values.astype(np.float32)
    feat_tensor = torch.from_numpy(feat_matrix)

    embeddings = []
    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        ids_batch = all_ids[start:end].to(device)
        feats_batch = feat_tensor[start:end].to(device)
        emb = model.encode_item(ids_batch, feats_batch)  # (B, 64)
        embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)  # (num_items, 64)


# --------------------------------------------------------------------------- #
# Main training entry point
# --------------------------------------------------------------------------- #

def train_two_tower() -> TwoTowerModel:
    """End-to-end training of the Two-Tower retrieval model.

    1. Loads parquet data produced by Component 1.
    2. Creates DataLoaders.
    3. Trains with AdamW + CosineAnnealingLR.
    4. Early-stops on validation Hit-Rate@100.
    5. Saves the best checkpoint and returns the model.
    """
    cfg = get_config()
    paths = cfg.paths
    ret_cfg = cfg.retrieval

    # ── Load data ---------------------------------------------------------- #
    logger.info("Loading data …")
    train_df = pd.read_parquet(paths.train_file)
    test_df = pd.read_parquet(paths.test_file)
    user_features = pd.read_parquet(paths.user_features_file)
    item_features = pd.read_parquet(paths.item_features_file)

    num_users = max(train_df["user_id"].max(), test_df["user_id"].max()) + 1
    num_items = max(train_df["item_id"].max(), test_df["item_id"].max()) + 1
    logger.info(f"  users={num_users:,}  items={num_items:,}")
    logger.info(f"  train={len(train_df):,}  test={len(test_df):,}")

    # ── DataLoaders -------------------------------------------------------- #
    train_loader, test_loader = create_dataloaders(
        train_df,
        test_df,
        user_features,
        item_features,
        batch_size=ret_cfg.batch_size,
        num_workers=ret_cfg.num_workers,
    )

    # ── Model, optimiser, scheduler --------------------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = TwoTowerModel(num_users, num_items).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ret_cfg.learning_rate,
        weight_decay=ret_cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=ret_cfg.epochs)

    # ── Training loop ------------------------------------------------------ #
    best_hit_rate = -1.0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  Two-Tower Training  |  {ret_cfg.epochs} epochs  |  device={device}")
    print(f"{'='*60}\n")

    for epoch in range(1, ret_cfg.epochs + 1):
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        # Evaluate
        all_item_embs = _build_all_item_embeddings(model, item_features, device)
        metrics = evaluate(model, test_loader, all_item_embs, device, k=100)

        hit_rate = metrics["hit_rate_at_k"]
        ndcg = metrics["ndcg_at_k"]
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:>2}/{ret_cfg.epochs}  │  "
            f"loss={avg_loss:.4f}  │  "
            f"HR@100={hit_rate:.4f}  │  "
            f"NDCG@100={ndcg:.4f}  │  "
            f"lr={lr:.2e}"
        )

        # Early stopping / checkpointing
        if hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hit_rate": hit_rate,
                    "ndcg": ndcg,
                    "num_users": num_users,
                    "num_items": num_items,
                },
                paths.two_tower_model_file,
            )
            logger.info(f"  ✓ Saved best model (HR@100={hit_rate:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= ret_cfg.patience:
                print(f"\n  ⏹  Early stopping at epoch {epoch} "
                      f"(no improvement for {ret_cfg.patience} epochs)")
                break

    # ── Reload best checkpoint --------------------------------------------- #
    ckpt = torch.load(paths.two_tower_model_file, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"\n{'='*60}")
    print(f"  Training complete  │  Best HR@100={ckpt['hit_rate']:.4f}  "
          f"│  NDCG@100={ckpt['ndcg']:.4f}")
    print(f"  Checkpoint: {paths.two_tower_model_file}")
    print(f"{'='*60}\n")

    return model
