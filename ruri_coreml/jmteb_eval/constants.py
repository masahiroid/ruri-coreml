"""Every named value the JMTEB evaluation depends on, in one place."""

from pathlib import Path

import coremltools as ct

JMTEB_DATASET_ID = "sbintuitions/JMTEB"
JMTEB_PARQUET_REVISION = "refs/convert/parquet"
EVALUATION_SPLIT = "test"
CORPUS_SPLIT = "corpus"

ORIGINAL_MODEL_ID_TEMPLATE = "cl-nagoya/ruri-v3-{size}"
COREML_REPO_ID_TEMPLATE = "masahiroid/ruri-v3-{size}-coreml"
COREML_PACKAGE_NAME_TEMPLATE = "ruri-v3-{size}_seq{sequence_length}_{precision}.mlpackage"
MODEL_SIZES = ("130m", "310m")
COREML_PRECISIONS = ("fp16", "int8")

SHORT_SEQUENCE_LENGTH = 128
LONG_SEQUENCE_LENGTH = 512

COREML_COMPUTE_UNITS = ct.ComputeUnit.CPU_AND_NE

TORCH_BATCH_SIZE = 16
COREML_BATCH_SIZE = 16
CACHE_CHUNK_SIZE = 256

RETRIEVAL_CUTOFF = 10
KMEANS_RESTARTS = 10
KMEANS_MAX_ITERATIONS = 300
KMEANS_TOLERANCE = 1e-6
RANDOM_SEED = 42

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_WORK_DIRECTORY = PACKAGE_ROOT.parent / "jmteb_work"
DEFAULT_RESULTS_DIRECTORY = PACKAGE_ROOT.parent.parent / "jmteb_results"
