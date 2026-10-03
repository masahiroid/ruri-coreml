"""Renders the Hugging Face model card (README.md) published alongside a
converted ruri-v3-coreml repository.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

_TEMPLATE = """---
license: apache-2.0
language:
- ja
tags:
- sentence-similarity
- feature-extraction
- coreml
- ios
- japanese
- ruri
- modernbert
base_model: cl-nagoya/ruri-v3-{model_size_label}
pipeline_tag: sentence-similarity
---

# ruri-v3-{model_size_label}-coreml

[cl-nagoya/ruri-v3-{model_size_label}](https://huggingface.co/cl-nagoya/ruri-v3-{model_size_label})(日本語汎用テキスト埋め込みモデル)を、iOS/macOS (Core ML) で直接動かせるように変換したものです。

- ベースモデル: [cl-nagoya/ruri-v3-{model_size_label}](https://huggingface.co/cl-nagoya/ruri-v3-{model_size_label})(ModernBERT-Ja, {parameter_count_description})
- 出力: **mean pooling + L2 normalize 済みの文埋め込みベクトル**({hidden_dimension}次元)。そのままコサイン類似度の代わりに内積(dot product)で類似度計算できます。
- 系列長ごとに固定長でモデルを分けています(CoreMLはグラフ内の可変長トレースに難があるため、実用的な長さでバケット化しています)。入力は `input_ids` / `attention_mask` を該当の長さまでパディング/切り詰めしてください。

## ファイル一覧

| ファイル | 系列長 | 精度 | サイズ目安 |
|---|---|---|---|
| `ruri-v3-{model_size_label}_seq128_fp16.mlpackage` | 128 | fp16 | {fp16_file_size_mb}MB |
| `ruri-v3-{model_size_label}_seq256_fp16.mlpackage` | 256 | fp16 | {fp16_file_size_mb}MB |
| `ruri-v3-{model_size_label}_seq512_fp16.mlpackage` | 512 | fp16 | {fp16_file_size_mb}MB |
| `ruri-v3-{model_size_label}_seq128_int8.mlpackage` | 128 | int8(重みのみ量子化) | {int8_file_size_mb}MB |
| `ruri-v3-{model_size_label}_seq256_int8.mlpackage` | 256 | int8(重みのみ量子化) | {int8_file_size_mb}MB |
| `ruri-v3-{model_size_label}_seq512_int8.mlpackage` | 512 | int8(重みのみ量子化) | {int8_file_size_mb}MB |

短いクエリやチャンクで検索するなら seq128、長めの文書チャンクなら seq256/512 を選んでください。モバイルアプリには **int8版** を推奨します。

## 動作確認

- **iPhone 17 Pro 実機(iOS 27)**: ruri-v3-130m の seq128 fp16版・int8版がともに GPU/Neural Engine(`computeUnits = .all`)で動作。定常状態の推論は1文あたり約5ms(Apple標準の `NLContextualEmbedding` は約9〜10ms)。fp16版とint8版の類似度スコア差は最大0.0042で、検索結果の順位はほぼ同一。
- **Mac(coremltools)**: 全ファイルでオリジナル(PyTorch)出力とのコサイン類似度0.9998以上を確認。

詳しい検証結果は [REPORT.md](https://github.com/masahiroid/ruri-coreml/blob/main/REPORT.md) を参照してください。

## プレフィックスについて(重要)

ruri-v3 は "1+3 prefix scheme" を採用しているため、埋め込み対象のテキストに応じて以下のプレフィックスを付けてからトークナイズしてください(元モデルと同じ挙動にするため必須です)。

- 空文字列: 意味的な類似度計算全般
- `トピック: `: 分類・クラスタリング用
- `検索クエリ: `: 検索クエリ側
- `検索文書: `: 検索対象の文書側

## Swift (Core ML) での使用例

ダウンロードしたモデルをXcodeプロジェクトに追加する手順・完全な実装例は [MANUAL.md](https://github.com/masahiroid/ruri-coreml/blob/main/RuriDemo/MANUAL.md) を参照してください。

```swift
import CoreML

let configuration = MLModelConfiguration()
configuration.computeUnits = .all
let model = try MLModel(
    contentsOf: Bundle.main.url(forResource: "ruri-v3-{model_size_label}_seq128_int8", withExtension: "mlmodelc")!,
    configuration: configuration
)

// トークナイザは別途用意する必要があります。huggingface/swift-transformers の
// Tokenizers、もしくは tokenizer.json を読み込める同等の実装を推奨します。
let inputIDs: MLMultiArray = ...       // shape [1, 128], Int32
let attentionMask: MLMultiArray = ...  // shape [1, 128], Int32

let input = try MLDictionaryFeatureProvider(dictionary: [
    "input_ids": MLFeatureValue(multiArray: inputIDs),
    "attention_mask": MLFeatureValue(multiArray: attentionMask),
])
let output = try model.prediction(from: input)
let embedding = output.featureValue(for: "sentence_embedding")!.multiArrayValue!
// [1, {hidden_dimension}] の L2 正規化済みベクトル(dtype は float16。Swift の Float16 として読み取る)
```

トークナイザは元モデルの `tokenizer.json`(ModernBERT-Ja / PLaMo系トークナイザ)をそのまま使う必要があります。

## Python での動作確認例

```python
import coremltools as ct
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("cl-nagoya/ruri-v3-{model_size_label}")
model = ct.models.MLModel("ruri-v3-{model_size_label}_seq128_int8.mlpackage")

text = "検索文書: 瑠璃色（るりいろ）は、紫みを帯びた濃い青のことである。"
encoded = tokenizer([text], return_tensors="np", padding="max_length", truncation=True, max_length=128)
output = model.predict({{
    "input_ids": encoded["input_ids"].astype(np.int32),
    "attention_mask": encoded["attention_mask"].astype(np.int32),
}})
print(output["sentence_embedding"].shape)  # (1, {hidden_dimension})
```

## 変換方法(再現手順)

このリポジトリのモデルは [ruri_coreml_convert](https://github.com/masahiroid/ruri-coreml/tree/main/ruri_coreml/ruri_coreml_convert) 変換ツールで生成されています。手順の概要:

1. `transformers` の `AutoModel` で ModernBERT ベースの ruri-v3 をロード
2. mean pooling + L2 normalize を行う `nn.Module` でラップ
3. `torch.jit.trace` でトレース(`transformers.masking_utils.and_masks`/`or_masks` が `Tensor.new_ones`/`new_zeros` を使っており Core ML コンバータが未対応のため、`torch.ones`/`torch.zeros` ベースの実装に差し替えてからトレース)
4. `coremltools.convert(..., convert_to="mlprogram")` で `.mlpackage` を生成
5. `coremltools.optimize.coreml.linear_quantize_weights` で int8 版も生成

## ライセンス

ベースモデル [cl-nagoya/ruri-v3-{model_size_label}](https://huggingface.co/cl-nagoya/ruri-v3-{model_size_label}) と同じ Apache License 2.0 です。オリジナルモデルの著作権は名古屋大学 conversational AI research group (cl-nagoya) に帰属します。本リポジトリは重みフォーマットの変換のみを行ったものです。

## 引用

```bibtex
@misc{{ruri2024,
  title={{{{Ruri}}: {{J}}apanese {{G}}eneral {{T}}ext {{E}}mbeddings}},
  author={{Hayato Tsukagoshi and Ryohei Sasano}},
  year={{2024}},
  eprint={{2409.07737}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.CL}},
}}
```
"""


@dataclass(frozen=True)
class ModelCardContext:
    model_size_label: str  # e.g. "130m"
    parameter_count_description: str  # e.g. "132M params, hidden=512"
    hidden_dimension: int
    fp16_file_size_mb: int
    int8_file_size_mb: int


def render_model_card(context: ModelCardContext) -> str:
    values = {field.name: getattr(context, field.name) for field in fields(context)}
    return _TEMPLATE.format(**values)


def write_model_card(context: ModelCardContext, output_path: Path) -> None:
    output_path.write_text(render_model_card(context), encoding="utf-8")
