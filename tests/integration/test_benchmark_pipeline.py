"""Ponta a ponta: cena anotada em disco -> detecção -> métricas em centímetros.

Gera o dataset sinteticamente (cv2, em tmp_path) em vez de depender de
imagens versionadas: o objetivo aqui é garantir que o esquema de anotação,
o carregamento, a conversão de unidades e o cálculo de métricas se
encaixam — não avaliar a qualidade do detector, que é trabalho dos
experimentos, não da suíte de testes.
"""
import json

import cv2
import numpy as np
import pytest

from vsss_vision.benchmark.dataset import load_dataset
from vsss_vision.benchmark.runner import format_summary, run_benchmark
from vsss_vision.vision.config import default_config

FRAME_SIZE = (480, 640)  # (height, width)
BODY_PX = (200, 200)
MARKER_PX = (230, 200)  # deslocado em +X -> theta esperado ~0 grau
BALL_PX = (400, 300)


def _mid(color_range):
    return tuple((low + high) // 2 for low, high in zip(color_range.lower, color_range.upper))


def _draw_hsv_circle(frame, center, radius, hsv_mid):
    hsv_pixel = np.uint8([[hsv_mid]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
    cv2.circle(frame, center, radius, tuple(int(channel) for channel in bgr_pixel), -1)


@pytest.fixture
def annotated_dataset(tmp_path):
    """Uma cena sintética com ground truth em pixel e iluminância anotada."""

    config = default_config()
    frame = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
    _draw_hsv_circle(frame, BODY_PX, 20, _mid(config.own_color))
    _draw_hsv_circle(frame, MARKER_PX, 6, _mid(config.markers[0]))
    _draw_hsv_circle(frame, BALL_PX, 10, _mid(config.ball_color))

    cv2.imwrite(str(tmp_path / "cena.png"), frame)
    (tmp_path / "cena.json").write_text(
        json.dumps(
            {
                "conditions": {"illuminance_lux": 500, "light_type": "cold", "uniform": True},
                "vision": {
                    "units": "px",
                    "robots": [{"id": 0, "team": "own", "x": BODY_PX[0], "y": BODY_PX[1], "theta_deg": 0.0}],
                    "ball": {"x": BALL_PX[0], "y": BALL_PX[1]},
                },
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_dataset_loads_scene_with_conditions(annotated_dataset):
    dataset = load_dataset(annotated_dataset)

    assert len(dataset) == 1
    scene = dataset.scenes[0]
    assert scene.scene_id == "cena"
    assert scene.conditions.illuminance_lux == 500
    assert scene.conditions.light_type == "cold"
    assert len(scene.ground_truth.robots) == 1
    assert scene.ground_truth.ball is not None


def test_media_without_annotation_is_ignored(annotated_dataset, tmp_path):
    cv2.imwrite(str(annotated_dataset / "sem_anotacao.png"), np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(load_dataset(annotated_dataset)) == 1


def test_benchmark_reports_small_error_on_a_clean_synthetic_scene(annotated_dataset):
    result = run_benchmark(annotated_dataset, detector_name="color", use_tracker=False)

    summary = result.summary
    assert summary.frames == 1
    # Cena sintética sem ruído: o erro deve ser sub-centimétrico. Um valor
    # grande aqui indica quebra na conversão px -> m, não no detector.
    assert summary.position_error_mean_cm < 1.0
    assert summary.orientation_error_mean_deg < 5.0
    assert summary.ball_error_mean_cm < 1.0
    assert summary.miss_rate == 0.0
    assert summary.id_swap_rate == 0.0


def test_benchmark_result_groups_by_illuminance_and_serializes(annotated_dataset, tmp_path):
    result = run_benchmark(annotated_dataset, detector_name="color", use_tracker=False)

    by_level = result.by_illuminance()
    assert list(by_level) == [500]
    assert by_level[500].frames == 1

    out = result.write_json(tmp_path / "resultado.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["detector"] == "color"
    assert payload["by_illuminance"]["500"]["frames"] == 1
    assert payload["config"]["use_tracker"] is False
    assert "total" in {stage["stage"] for stage in payload["latency"]["stages"]}

    assert "erro de posicao" in format_summary(result)


def test_unimplemented_detector_fails_loudly(annotated_dataset):
    with pytest.raises(NotImplementedError):
        run_benchmark(annotated_dataset, detector_name="yolo")
