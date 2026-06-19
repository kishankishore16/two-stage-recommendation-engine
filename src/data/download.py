"""
Download the Amazon Clothing, Shoes & Jewelry 5-core review dataset.

Downloads both review data and product metadata from the UCSD McAuley Lab
repository. Files are saved as gzipped JSON to data/raw/.
"""

import os
import urllib.request
from pathlib import Path

from tqdm import tqdm

from src.config import get_config


# ── Custom urllib reporthook for tqdm ────────────────────────────────────────


class _TqdmDownloadHook:
    """Report hook that drives a tqdm progress bar during urllib downloads."""

    def __init__(self, desc: str = "Downloading"):
        self.pbar: tqdm | None = None
        self.desc = desc

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if self.pbar is None:
            self.pbar = tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=self.desc,
            )
        downloaded = block_size  # bytes received in this call
        self.pbar.update(downloaded)

    def close(self):
        if self.pbar is not None:
            self.pbar.close()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _file_ok(path: Path, min_bytes: int = 1024) -> bool:
    """Return True if *path* exists and is larger than *min_bytes*."""
    return path.is_file() and path.stat().st_size > min_bytes


def _download_file(url: str, dest: Path, desc: str) -> None:
    """Download *url* to *dest* with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    hook = _TqdmDownloadHook(desc=desc)
    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=hook)
    finally:
        hook.close()
    print(f"  ✓ Saved to {dest}  ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


# ── Public API ───────────────────────────────────────────────────────────────


def download_dataset() -> None:
    """Download the Amazon Clothing, Shoes & Jewelry review + metadata files.

    Skips any file that already exists on disk and passes a basic size check.
    """
    cfg = get_config()
    paths = cfg.paths
    data_cfg = cfg.data

    files_to_download = [
        (data_cfg.reviews_url, paths.reviews_file, "Reviews"),
        (data_cfg.metadata_url, paths.metadata_file, "Metadata"),
    ]

    for url, dest, label in files_to_download:
        if _file_ok(dest):
            print(f"  ⏭ {label} already exists ({dest.name}, "
                  f"{dest.stat().st_size / 1024 / 1024:.1f} MB) — skipping.")
            continue
        print(f"  ↓ Downloading {label} …")
        _download_file(url, dest, desc=label)

    print("\nAll dataset files are ready.")


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Amazon Clothing, Shoes & Jewelry — Dataset Download")
    print("=" * 60)
    download_dataset()
