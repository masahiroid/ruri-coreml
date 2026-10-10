"""Compares ruri-v3's PyTorch original against its Core ML fp16/int8 exports on
a subset of JMTEB, so conversion and quantization loss can be measured on a
public benchmark.

Modules:
- constants: every tunable value and identifier
- data: loads JMTEB splits from the Hub's parquet mirror
- encoders: PyTorch and Core ML sentence encoders behind one interface
- embedding_cache: resumable, chunked on-disk embedding cache
- metrics: Spearman, k-means + V-measure, nDCG@k (numpy only)
- tasks: the evaluated JMTEB tasks and how each is scored
- model_files: downloads the published Core ML packages
- runner: runs every (encoder, task) pair and writes results
- summary: renders the results as a Markdown table
"""
