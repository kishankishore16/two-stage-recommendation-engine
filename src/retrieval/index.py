"""
FAISS index management for fast approximate nearest-neighbour retrieval.

* ``build_faiss_index`` — encode every item → build & save a FAISS index.
* ``load_faiss_index``  — reload index + id_map from disk.
* ``retrieve_candidates`` — query the index for top-*k* item candidates.
"""

import logging

import faiss
import numpy as np
import pandas as pd
import torch

from src.config import get_config
from src.retrieval.model import TwoTowerModel

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def build_faiss_index(
    model: TwoTowerModel,
    item_features_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 2048,
) -> faiss.Index:
    """Encode all items through the ItemTower and build a FAISS index.

    The index type is chosen automatically:
    * **≤ 100 000 items** → ``IndexFlatIP`` (exact inner product).
    * **> 100 000 items** → ``IndexIVFFlat`` with ``nlist`` clusters from
      :pyattr:`RetrievalConfig.faiss_nlist`.

    The index and a mapping array (FAISS position → real ``item_id``) are
    saved to the paths specified in :class:`PathConfig`.

    Parameters
    ----------
    model : TwoTowerModel
        Trained model (only the item tower is used).
    item_features_df : pd.DataFrame
        Dense item features indexed (or with column) ``item_id``.
    device : torch.device
    batch_size : int
        Encoding batch size.

    Returns
    -------
    faiss.Index
        The constructed (and saved) FAISS index.
    """
    cfg = get_config()
    paths = cfg.paths
    ret_cfg = cfg.retrieval

    model.eval()

    # ── Prepare feature matrix -------------------------------------------- #
    if item_features_df.index.name != "item_id":
        if "item_id" in item_features_df.columns:
            item_features_df = item_features_df.set_index("item_id")

    item_ids_array = np.array(item_features_df.index.tolist(), dtype=np.int64)
    num_items = len(item_ids_array)
    num_features = len(item_features_df.columns)

    logger.info(f"Building FAISS index for {num_items:,} items …")

    # ── Encode items in batches ------------------------------------------- #
    all_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            batch_ids = item_ids_array[start:end]

            ids_tensor = torch.tensor(batch_ids, dtype=torch.long, device=device)
            feats_np = item_features_df.iloc[start:end].values.astype(np.float32)
            feats_tensor = torch.from_numpy(feats_np).to(device)

            emb = model.encode_item(ids_tensor, feats_tensor)  # (B, 64)
            all_embeddings.append(emb.cpu().numpy().astype(np.float32))

    embeddings = np.vstack(all_embeddings).astype(np.float32)  # (N, 64)
    dim = embeddings.shape[1]

    # ── Build index -------------------------------------------------------- #
    if num_items > 100_000:
        nlist = ret_cfg.faiss_nlist
        quantiser = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        logger.info(f"  Using IndexIVFFlat (nlist={nlist})")

        # IVF requires training on a representative sample
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = ret_cfg.faiss_nprobe
    else:
        index = faiss.IndexFlatIP(dim)
        logger.info("  Using IndexFlatIP (exact)")
        index.add(embeddings)

    logger.info(f"  Index contains {index.ntotal:,} vectors of dim {dim}")

    # ── Persist ------------------------------------------------------------ #
    faiss.write_index(index, str(paths.faiss_index_file))
    np.save(str(paths.faiss_id_map_file), item_ids_array)

    logger.info(f"  Saved index  → {paths.faiss_index_file}")
    logger.info(f"  Saved id_map → {paths.faiss_id_map_file}")

    return index


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def load_faiss_index() -> tuple[faiss.Index, np.ndarray]:
    """Load the FAISS index and id_map from disk.

    Returns
    -------
    tuple[faiss.Index, np.ndarray]
        ``(index, id_map)`` where ``id_map[faiss_position]`` gives the real
        ``item_id``.
    """
    cfg = get_config()
    paths = cfg.paths

    index = faiss.read_index(str(paths.faiss_index_file))
    id_map = np.load(str(paths.faiss_id_map_file))

    logger.info(
        f"Loaded FAISS index ({index.ntotal:,} vectors) and id_map "
        f"({len(id_map):,} entries)"
    )
    return index, id_map


# --------------------------------------------------------------------------- #
# Retrieve
# --------------------------------------------------------------------------- #

def retrieve_candidates(
    index: faiss.Index,
    id_map: np.ndarray,
    user_embedding: np.ndarray,
    k: int = 200,
) -> tuple[list[int], list[float]]:
    """Search the FAISS index for top-*k* candidate items.

    Parameters
    ----------
    index : faiss.Index
        Pre-loaded FAISS index.
    id_map : np.ndarray
        Mapping from FAISS row position to real ``item_id``.
    user_embedding : np.ndarray
        (1, D) or (D,) float32 user embedding.
    k : int
        Number of candidates to retrieve.

    Returns
    -------
    tuple[list[int], list[float]]
        ``(item_ids, scores)`` — the real item IDs and their similarity scores,
        sorted by descending score.
    """
    # Ensure shape is (1, D) and dtype is float32
    if user_embedding.ndim == 1:
        user_embedding = user_embedding.reshape(1, -1)
    user_embedding = user_embedding.astype(np.float32)

    scores, indices = index.search(user_embedding, k)  # each (1, k)

    scores = scores[0]    # (k,)
    indices = indices[0]   # (k,)

    # Map FAISS positions → real item IDs, filtering invalid entries (-1)
    valid = indices >= 0
    item_ids = id_map[indices[valid]].tolist()
    item_scores = scores[valid].tolist()

    return item_ids, item_scores
