"""Renders per-variant result files as one Markdown table, with each Core ML
variant's difference from its PyTorch original.
"""

from __future__ import annotations

import json
from pathlib import Path

from .constants import COREML_PRECISIONS, MODEL_SIZES
from .tasks import TASKS

_SCORE_FORMAT = "{:.4f}"
_DELTA_FORMAT = "{:+.4f}"
_MISSING = "—"
_SUMMARY_FILE_NAME = "SUMMARY.md"


def write_summary(results_directory: Path) -> Path:
    lines = [
        "# JMTEB subset: ruri-v3 PyTorch vs Core ML",
        "",
        "Core ML packages evaluated with coremltools on a Mac (CPU + Neural Engine). "
        "Δ = Core ML score − PyTorch score for the same model size.",
        "",
    ]
    for size in MODEL_SIZES:
        reference = _load(results_directory / f"{size}-pytorch.json")
        coreml = {p: _load(results_directory / f"{size}-coreml-{p}.json") for p in COREML_PRECISIONS}
        if not reference and not any(coreml.values()):
            continue
        header = ["タスク", "指標", "PyTorch"] + [f"Core ML {p} (Δ)" for p in COREML_PRECISIONS]
        lines += [f"## ruri-v3-{size}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for task in TASKS:
            base = reference.get(task.name, {}).get("score")
            cells = [task.name, task.metric_name, _format(base)]
            for p in COREML_PRECISIONS:
                value = coreml[p].get(task.name, {}).get("score")
                cells.append(_format(value) if value is None or base is None
                             else f"{_format(value)} ({_DELTA_FORMAT.format(value - base)})")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    path = results_directory / _SUMMARY_FILE_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _format(value) -> str:
    return _MISSING if value is None else _SCORE_FORMAT.format(value)
