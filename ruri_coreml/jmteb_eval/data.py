"""Loads JMTEB splits from the Hugging Face parquet mirror (the dataset itself
is script-based, which current `datasets` releases no longer execute).
"""

from __future__ import annotations

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from .constants import JMTEB_DATASET_ID, JMTEB_PARQUET_REVISION

_PARQUET_SUFFIX = ".parquet"


def load_split(config: str, split: str) -> pd.DataFrame:
    prefix = f"{config}/{split}/"
    files = sorted(
        path
        for path in HfApi().list_repo_files(
            JMTEB_DATASET_ID, repo_type="dataset", revision=JMTEB_PARQUET_REVISION
        )
        if path.startswith(prefix) and path.endswith(_PARQUET_SUFFIX)
    )
    if not files:
        raise FileNotFoundError(f"no parquet files for {config}/{split}")
    frames = [
        pd.read_parquet(
            hf_hub_download(
                JMTEB_DATASET_ID, path, repo_type="dataset", revision=JMTEB_PARQUET_REVISION
            )
        )
        for path in files
    ]
    return pd.concat(frames, ignore_index=True)
