"""Runs every (encoder variant, task) pair, skipping pairs that already have a
result, and writes one JSON file of scores per variant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ruri_coreml_convert.constants import RuriPromptPrefix

from .constants import COREML_PRECISIONS, ORIGINAL_MODEL_ID_TEMPLATE
from .embedding_cache import cached_embeddings
from .encoders import CoreMLEncoder, SentenceEncoder, TorchEncoder
from .model_files import coreml_package_path, download_coreml_packages
from .tasks import TASKS

_PYTORCH_VARIANT = "pytorch"
_COREML_VARIANT_TEMPLATE = "coreml-{precision}"


@dataclass
class EncoderVariant:
    """One model under test. Encoders are built lazily, one per sequence
    length, because each Core ML package is exported at a fixed length.
    """

    name: str
    build_encoder: Callable[[int], SentenceEncoder]

    def __post_init__(self):
        self._encoders: dict[int, SentenceEncoder] = {}

    def encoder(self, sequence_length: int) -> SentenceEncoder:
        if sequence_length not in self._encoders:
            self._encoders[sequence_length] = self.build_encoder(sequence_length)
        return self._encoders[sequence_length]


def variants_for_size(size: str, models_directory: Path) -> list[EncoderVariant]:
    model_id = ORIGINAL_MODEL_ID_TEMPLATE.format(size=size)
    variants = [
        EncoderVariant(f"{size}-{_PYTORCH_VARIANT}", lambda length: TorchEncoder(model_id, length))
    ]
    for precision in COREML_PRECISIONS:
        variants.append(
            EncoderVariant(
                f"{size}-{_COREML_VARIANT_TEMPLATE.format(precision=precision)}",
                lambda length, p=precision: CoreMLEncoder(
                    coreml_package_path(models_directory, size, length, p), model_id, length
                ),
            )
        )
    return variants


def run(sizes: list[str], work_directory: Path, results_directory: Path, row_limit: int | None) -> None:
    models_directory = work_directory / "models"
    cache_root = work_directory / ("cache" if row_limit is None else f"cache_limit{row_limit}")
    results_directory.mkdir(parents=True, exist_ok=True)

    for size in sizes:
        download_coreml_packages(models_directory, size)
        for variant in variants_for_size(size, models_directory):
            _run_variant(variant, cache_root, results_directory, row_limit)


def _run_variant(variant: EncoderVariant, cache_root: Path, results_directory: Path, row_limit: int | None) -> None:
    result_path = results_directory / f"{variant.name}.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else {}

    for task in TASKS:
        if task.name in results:
            continue

        def embed(texts: list[str], prefix: RuriPromptPrefix, sequence_length: int, role: str) -> np.ndarray:
            cache_directory = cache_root / variant.name / task.name / f"{role}_seq{sequence_length}"
            return cached_embeddings(
                cache_directory,
                [prefix.value + text for text in texts],
                variant.encoder(sequence_length).encode,
            )

        score = task.score(embed, row_limit)
        results[task.name] = {"metric": task.metric_name, "score": score}
        result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"{variant.name} {task.name} {task.metric_name}={score:.4f}", flush=True)
