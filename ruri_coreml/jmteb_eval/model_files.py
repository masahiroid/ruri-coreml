"""Downloads the published Core ML packages the evaluation needs."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from .constants import (
    COREML_PACKAGE_NAME_TEMPLATE,
    COREML_PRECISIONS,
    COREML_REPO_ID_TEMPLATE,
    LONG_SEQUENCE_LENGTH,
    SHORT_SEQUENCE_LENGTH,
)

_SINGLE_DOWNLOAD_STREAM = 1


def coreml_package_path(models_directory: Path, size: str, sequence_length: int, precision: str) -> Path:
    return models_directory / size / COREML_PACKAGE_NAME_TEMPLATE.format(
        size=size, sequence_length=sequence_length, precision=precision
    )


def download_coreml_packages(models_directory: Path, size: str) -> None:
    patterns = [
        COREML_PACKAGE_NAME_TEMPLATE.format(size=size, sequence_length=length, precision=precision) + "/*"
        for length in (SHORT_SEQUENCE_LENGTH, LONG_SEQUENCE_LENGTH)
        for precision in COREML_PRECISIONS
    ]
    snapshot_download(
        COREML_REPO_ID_TEMPLATE.format(size=size),
        allow_patterns=patterns,
        local_dir=models_directory / size,
        max_workers=_SINGLE_DOWNLOAD_STREAM,
    )
