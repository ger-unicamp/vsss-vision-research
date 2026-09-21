"""Exportação de resultados para CSV e Markdown."""
import csv

import pytest

from vsss_vision.benchmark.metrics import FrameMetrics, aggregate
from vsss_vision.benchmark.report import latency_curve_rows, markdown_table, write_csv
from vsss_vision.benchmark.runner import RunResult


def _frame(scene_id, lux, error_cm):
    return FrameMetrics(
        scene_id=scene_id,
        position_errors_cm=[error_cm],
        orientation_errors_deg=[2.0],
        ground_truth_count=1,
        matched_count=1,
        illuminance_lux=lux,
    )


def _result(label="color", frames=None, imgsz=None):
    frames = frames or [_frame("a", 500, 1.0), _frame("b", 200, 4.0)]
    return RunResult(
        detector="color",
        dataset_root="datasets/vision",
        summary=aggregate(frames),
        latency={
            "throughput_fps": 50.0,
            "stages": [
                {"stage": "total", "samples": 2, "mean_ms": 20.0, "p50_ms": 19.0,
                 "p95_ms": 25.0, "p99_ms": 30.0, "max_ms": 31.0}
            ],
        },
        per_frame=frames,
        config_snapshot={"yolo_imgsz": imgsz, "yolo_device": "cpu"},
        label=label,
    )


def test_csv_has_one_row_per_condition_plus_the_aggregate(tmp_path):
    path = write_csv([_result()], tmp_path / "saida.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))

    conditions = [row["condition"] for row in rows]
    assert conditions[0] == "todas"
    assert set(conditions[1:]) == {"200 lux", "500 lux"}


def test_latency_is_not_repeated_per_condition(tmp_path):
    """Iluminancia nao muda custo computacional; repetir o numero sugeriria medicao que nao houve."""

    path = write_csv([_result()], tmp_path / "saida.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))

    assert rows[0]["latency_p99_ms"] == "30.0000"
    assert all(row["latency_p99_ms"] == "" for row in rows[1:])


def test_markdown_formats_rates_as_percentages():
    table = markdown_table([_result()], split_by_condition=False)

    assert table.startswith("| Detector |")
    assert "%" in table
    assert "color" in table


def test_missing_values_render_as_a_dash():
    frames = [FrameMetrics(scene_id="a", ground_truth_count=0)]
    table = markdown_table([_result(frames=frames)], split_by_condition=False)

    assert "—" in table


def test_latency_curve_carries_the_swept_axis():
    rows = latency_curve_rows([_result(label="yolov8n@320", imgsz=320)])

    assert rows[0]["detector"] == "yolov8n@320"
    assert rows[0]["imgsz"] == 320
    assert rows[0]["latency_p95_ms"] == pytest.approx(25.0)
    assert rows[0]["position_error_mean_cm"] == pytest.approx(2.5)
