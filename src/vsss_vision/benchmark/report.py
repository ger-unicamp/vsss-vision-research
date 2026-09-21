"""Exportação dos resultados em formatos que vão direto para o artigo.

CSV para análise e gráficos, Markdown para colar no texto. O JSON de
`RunResult` continua sendo a fonte canônica — estes formatos são derivados
dele e podem ser regerados a qualquer momento.

As colunas seguem a ordem em que os resultados são discutidos: precisão
espacial primeiro (é a pergunta da P1), depois confiabilidade de
identidade, depois latência (é a pergunta da P2).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence

from vsss_vision.benchmark.metrics import Summary
from vsss_vision.benchmark.runner import RunResult

SUMMARY_COLUMNS = [
    ("detector", "Detector"),
    ("condition", "Condicao"),
    ("frames", "Frames"),
    ("position_error_mean_cm", "Pos. media (cm)"),
    ("position_error_p95_cm", "Pos. p95 (cm)"),
    ("orientation_error_mean_deg", "Ori. media (graus)"),
    ("ball_error_mean_cm", "Bola (cm)"),
    ("miss_rate", "Perdidas"),
    ("id_swap_rate", "Trocas de id"),
    ("false_positive_rate", "Falsos pos."),
    ("f1", "F1"),
    ("latency_mean_ms", "Lat. media (ms)"),
    ("latency_p95_ms", "Lat. p95 (ms)"),
    ("latency_p99_ms", "Lat. p99 (ms)"),
    ("fps", "FPS"),
]


def result_rows(result: RunResult, split_by_condition: bool = True) -> list[dict]:
    """Uma linha agregada por execução, ou uma por nível de iluminância."""

    latency = _latency_columns(result)
    rows = [{"detector": result.name, "condition": "todas", **result.summary.as_dict(), **latency}]

    if split_by_condition:
        by_level = result.by_illuminance()
        if len(by_level) > 1:
            for level in sorted(by_level, key=lambda value: (value is None, value)):
                rows.append(
                    {
                        "detector": result.name,
                        "condition": "desconhecida" if level is None else f"{level:g} lux",
                        **by_level[level].as_dict(),
                        # A latência é da execução inteira, não da condição:
                        # iluminância não muda o custo computacional. Repetir
                        # o valor por linha seria sugerir uma medição que não
                        # foi feita.
                        **{key: None for key in latency},
                    }
                )
    return rows


def write_csv(results: Sequence[RunResult], path: Path | str, split_by_condition: bool = True) -> Path:
    """Uma linha por (detector, condição). É o arquivo que alimenta os gráficos."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for result in results for row in result_rows(result, split_by_condition)]
    keys = [key for key, _label in SUMMARY_COLUMNS]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})
    return path


def write_markdown(results: Sequence[RunResult], path: Path | str, split_by_condition: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_table(results, split_by_condition), encoding="utf-8")
    return path


def markdown_table(results: Sequence[RunResult], split_by_condition: bool = True) -> str:
    """Tabela pronta para colar no artigo."""

    rows = [row for result in results for row in result_rows(result, split_by_condition)]
    labels = [label for _key, label in SUMMARY_COLUMNS]
    keys = [key for key, _label in SUMMARY_COLUMNS]

    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in labels) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_value(key, row.get(key)) for key in keys) + " |")
    return "\n".join(lines) + "\n"


def latency_curve_rows(results: Iterable[RunResult]) -> list[dict]:
    """Pontos (latência, erro) da curva de compromisso da proposta P2."""

    rows = []
    for result in results:
        latency = _latency_columns(result)
        rows.append(
            {
                "detector": result.name,
                "imgsz": result.config_snapshot.get("yolo_imgsz"),
                "device": result.config_snapshot.get("yolo_device"),
                "latency_mean_ms": latency["latency_mean_ms"],
                "latency_p95_ms": latency["latency_p95_ms"],
                "latency_p99_ms": latency["latency_p99_ms"],
                "fps": latency["fps"],
                "position_error_mean_cm": result.summary.position_error_mean_cm,
                "orientation_error_mean_deg": result.summary.orientation_error_mean_deg,
                "f1": result.summary.f1,
            }
        )
    return rows


def write_latency_curve_csv(results: Sequence[RunResult], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = latency_curve_rows(results)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["detector"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    return path


def _latency_columns(result: RunResult) -> dict[str, Optional[float]]:
    total = next(
        (stage for stage in result.latency.get("stages", []) if stage["stage"] == "total"), None
    )
    if total is None:
        return {"latency_mean_ms": None, "latency_p95_ms": None, "latency_p99_ms": None, "fps": None}
    return {
        "latency_mean_ms": total["mean_ms"],
        "latency_p95_ms": total["p95_ms"],
        "latency_p99_ms": total["p99_ms"],
        "fps": result.latency.get("throughput_fps"),
    }


def _csv_value(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_value(key: str, value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if key in ("miss_rate", "id_swap_rate", "false_positive_rate", "ball_miss_rate"):
        return f"{value:.1%}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def summary_of(result: RunResult) -> Summary:
    return result.summary
