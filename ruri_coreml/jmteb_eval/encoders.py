"""Sentence encoders under evaluation, behind one interface: the PyTorch
original (reference) and the published Core ML exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ruri_coreml_convert.constants import (
    COREML_ATTENTION_MASK_FEATURE_NAME,
    COREML_INPUT_IDS_FEATURE_NAME,
    COREML_OUTPUT_FEATURE_NAME,
)
from ruri_coreml_convert.pooling import MeanPoolingSentenceEmbedder

from .constants import COREML_BATCH_SIZE, COREML_COMPUTE_UNITS, TORCH_BATCH_SIZE


class SentenceEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Returns L2-normalized embeddings, one row per text."""


class TorchEncoder:
    """The original Hugging Face checkpoint with the same mean-pooling head
    the Core ML export bakes in, so differences come from conversion alone.
    """

    def __init__(self, model_id: str, sequence_length: int):
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = MeanPoolingSentenceEmbedder(AutoModel.from_pretrained(model_id)).eval()
        self._sequence_length = sequence_length

    def encode(self, texts: list[str]) -> np.ndarray:
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), TORCH_BATCH_SIZE):
                encoded = self._tokenizer(
                    texts[start : start + TORCH_BATCH_SIZE],
                    padding=True,
                    truncation=True,
                    max_length=self._sequence_length,
                    return_tensors="pt",
                )
                batches.append(self._model(encoded["input_ids"], encoded["attention_mask"]).numpy())
        return np.concatenate(batches).astype(np.float32)


class CoreMLEncoder:
    """A published `.mlpackage`, run through coremltools on this Mac's Neural
    Engine — the same backend the models use on iPhone. (CPU-only execution
    of these fp16 graphs returns NaN on the macOS beta this was built on.)
    """

    def __init__(self, package_path: Path, tokenizer_model_id: str, sequence_length: int):
        self._model = ct.models.MLModel(str(package_path), compute_units=COREML_COMPUTE_UNITS)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id)
        self._sequence_length = sequence_length

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for start in range(0, len(texts), COREML_BATCH_SIZE):
            encoded = self._tokenizer(
                texts[start : start + COREML_BATCH_SIZE],
                padding="max_length",
                truncation=True,
                max_length=self._sequence_length,
                return_tensors="np",
            )
            inputs = [
                {
                    COREML_INPUT_IDS_FEATURE_NAME: encoded["input_ids"][i : i + 1].astype(np.int32),
                    COREML_ATTENTION_MASK_FEATURE_NAME: encoded["attention_mask"][i : i + 1].astype(np.int32),
                }
                for i in range(len(encoded["input_ids"]))
            ]
            rows.extend(output[COREML_OUTPUT_FEATURE_NAME][0] for output in self._model.predict(inputs))
        embeddings = np.asarray(rows, dtype=np.float32)
        if np.isnan(embeddings).any():
            raise ValueError("Core ML returned NaN embeddings; scores would be meaningless")
        return embeddings
