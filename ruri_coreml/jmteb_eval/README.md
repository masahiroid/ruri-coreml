# jmteb_eval

Measures how much ruri-v3's Core ML exports (fp16 and int8) differ from the
original PyTorch checkpoint on a subset of [JMTEB](https://huggingface.co/datasets/sbintuitions/JMTEB),
so conversion/quantization loss is quantified on a public benchmark rather than
assumed.

## Tasks

STS (JSTS, JSICK), clustering (livedoor_news, MewsC-16-ja) and retrieval
(NLP Journal title→abs, JaGovFaqs-22k) — chosen to cover the task types that
matter for on-device search while fitting overnight runs on a laptop.

## Run

```bash
# full run, sizes 130m and 310m (resumable; embeddings are cached in chunks)
python -m jmteb_eval run --sizes 130m 310m

# quick smoke test on 40 rows per task
python -m jmteb_eval run --sizes 130m --limit 40

# render SUMMARY.md from finished result JSONs
python -m jmteb_eval summarize
```

Core ML packages are downloaded from the published Hugging Face repos and run
through coremltools on the Mac's Neural Engine (CPU-only execution of these
fp16 graphs returns NaN on the macOS beta this was developed on). Results and
the rendered table live in `../../jmteb_results/`.

## Results summary

See [`jmteb_results/SUMMARY.md`](../../jmteb_results/SUMMARY.md). In short, the
Core ML fp16 and int8 exports track the PyTorch original within a few
thousandths on almost every task — on-device quantization does not meaningfully
degrade retrieval/STS/clustering quality.
