# JMTEB subset: ruri-v3 PyTorch vs Core ML

Core ML packages evaluated with coremltools on a Mac (CPU + Neural Engine). Δ = Core ML score − PyTorch score for the same model size.

## ruri-v3-130m

| タスク | 指標 | PyTorch | Core ML fp16 (Δ) | Core ML int8 (Δ) |
|---|---|---|---|---|
| JSTS | spearman | 0.8345 | 0.8345 (+0.0001) | 0.8337 (-0.0008) |
| JSICK | spearman | 0.7394 | 0.7394 (-0.0000) | 0.7407 (+0.0013) |
| livedoor_news | v_measure | 0.4270 | 0.4261 (-0.0008) | 0.4212 (-0.0058) |
| MewsC-16-ja | v_measure | 0.2936 | 0.2945 (+0.0009) | 0.2952 (+0.0017) |
| NLP Journal title→abs | ndcg@10 | 0.9814 | 0.9812 (-0.0003) | 0.9804 (-0.0011) |
| JaGovFaqs-22k | ndcg@10 | 0.7657 | 0.7657 (+0.0000) | 0.7672 (+0.0016) |

## ruri-v3-310m

| タスク | 指標 | PyTorch | Core ML fp16 (Δ) | Core ML int8 (Δ) |
|---|---|---|---|---|
| JSTS | spearman | 0.8359 | 0.8360 (+0.0001) | 0.8368 (+0.0009) |
| JSICK | spearman | 0.7797 | 0.7797 (+0.0000) | 0.7798 (+0.0002) |
| livedoor_news | v_measure | 0.3615 | 0.3638 (+0.0023) | 0.3638 (+0.0023) |
| MewsC-16-ja | v_measure | 0.3171 | 0.2901 (-0.0270) | 0.3169 (-0.0002) |
| NLP Journal title→abs | ndcg@10 | 0.9787 | 0.9787 (+0.0000) | 0.9780 (-0.0007) |
| JaGovFaqs-22k | ndcg@10 | 0.7709 | 0.7716 (+0.0007) | 0.7718 (+0.0009) |
