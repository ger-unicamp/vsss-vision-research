"""Exportação para formato YOLO: caixas derivadas, keypoints e divisão por origem."""
import json
import math

import cv2
import numpy as np
import pytest

from vsss_vision.benchmark.dataset import load_dataset
from vsss_vision.benchmark.yolo_export import (
    BALL_DIAMETER_M,
    ROBOT_SIZE_M,
    class_index,
    class_names,
    export_dataset,
    split_by_source,
    team_color_of,
)
from vsss_vision.vision.config import default_config

FRAME_SIZE = (260, 300)  # (height, width) — 300 px / 1,50 m = 200 px/m em X


def _write_scene(directory, name, robots, ball=None, conditions=None):
    frame = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
    cv2.imwrite(str(directory / f"{name}.png"), frame)
    payload = {"vision": {"units": "m", "robots": robots}}
    if ball is not None:
        payload["vision"]["ball"] = ball
    if conditions:
        payload["conditions"] = conditions
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def dataset(tmp_path):
    _write_scene(
        tmp_path,
        "cena_a",
        robots=[{"id": 0, "team": "own", "x": 0.0, "y": 0.0, "theta_deg": 0.0}],
        ball={"x": 0.30, "y": 0.0},
    )
    _write_scene(
        tmp_path,
        "cena_b",
        robots=[{"id": 1, "team": "opponent", "x": -0.40, "y": 0.20}],
    )
    return load_dataset(tmp_path)


def test_class_names_come_from_the_config_map():
    config = default_config()
    assert class_names(config) == ["ball", "robot_yellow", "robot_blue"]
    assert class_index(config, "ball") == 0
    assert class_index(config, "robot", color="yellow") == 1
    assert class_index(config, "robot", color="blue") == 2


def test_team_color_follows_the_configured_side():
    config = default_config()
    config.team_color = "yellow"
    assert team_color_of("own", config) == "yellow"
    assert team_color_of("opponent", config) == "blue"

    config.team_color = "blue"
    assert team_color_of("own", config) == "blue"
    assert team_color_of("opponent", config) == "yellow"


def test_splits_never_share_a_source_file(dataset):
    splits = split_by_source(dataset, ratios=(0.5, 0.5, 0.0))

    sources = {
        split: {str(scene.source) for scene in scenes} for split, scenes in splits.items() if scenes
    }
    seen = [source for group in sources.values() for source in group]
    assert len(seen) == len(set(seen)), "um arquivo de origem apareceu em mais de um split"


def test_box_is_derived_from_the_physical_robot_size(dataset, tmp_path):
    config = default_config()
    out = tmp_path / "yolo"
    export_dataset(dataset, out, config, task="detect", ratios=(1.0, 0.0, 0.0))

    label = (out / "labels" / "train" / "cena_a.txt").read_text(encoding="utf-8").strip().splitlines()
    robot_line = next(line for line in label if line.startswith("1 "))  # robot_yellow
    _cls, cx, cy, w, h = robot_line.split()

    # Robo no centro do campo -> centro normalizado em (0.5, 0.5).
    assert float(cx) == pytest.approx(0.5, abs=1e-3)
    assert float(cy) == pytest.approx(0.5, abs=1e-3)
    # Largura = 7,5 cm * margem, em fracao do campo de 1,50 m.
    expected_w = ROBOT_SIZE_M * 1.15 / config.field_length_m
    assert float(w) == pytest.approx(expected_w, rel=1e-3)


def test_ball_box_uses_the_ball_diameter(dataset, tmp_path):
    config = default_config()
    out = tmp_path / "yolo"
    export_dataset(dataset, out, config, task="detect", ratios=(1.0, 0.0, 0.0))

    lines = (out / "labels" / "train" / "cena_a.txt").read_text(encoding="utf-8").strip().splitlines()
    ball_line = next(line for line in lines if line.startswith("0 "))
    _cls, _cx, _cy, w, _h = ball_line.split()

    assert float(w) == pytest.approx(BALL_DIAMETER_M * 1.15 / config.field_length_m, rel=1e-3)


def test_pose_keypoints_are_half_a_body_apart_along_the_annotated_angle(dataset, tmp_path):
    config = default_config()
    out = tmp_path / "yolo"
    export_dataset(dataset, out, config, task="pose", ratios=(1.0, 0.0, 0.0))

    lines = (out / "labels" / "train" / "cena_a.txt").read_text(encoding="utf-8").strip().splitlines()
    robot_line = next(line for line in lines if line.startswith("1 "))
    fields = robot_line.split()
    front_x, front_y, front_v = float(fields[5]), float(fields[6]), fields[7]
    back_x, back_y, back_v = float(fields[8]), float(fields[9]), fields[10]

    assert front_v == "2" and back_v == "2"
    # theta = 0 -> front a direita de back, mesma altura.
    assert front_x > back_x
    assert front_y == pytest.approx(back_y, abs=1e-6)
    # Separacao = um corpo inteiro (meio para cada lado), em fracao do campo.
    assert front_x - back_x == pytest.approx(ROBOT_SIZE_M / config.field_length_m, rel=1e-3)


def test_robot_without_annotated_angle_gets_invisible_keypoints(dataset, tmp_path):
    """Melhor visibilidade 0 do que um keypoint inventado: o Ultralytics ignora na perda."""

    config = default_config()
    out = tmp_path / "yolo"
    stats = export_dataset(dataset, out, config, task="pose", ratios=(0.0, 0.0, 1.0))

    lines = (out / "labels" / "test" / "cena_b.txt").read_text(encoding="utf-8").strip().splitlines()
    fields = lines[0].split()
    assert fields[7] == "0" and fields[10] == "0"
    assert stats.skipped_without_orientation == 1


def test_ball_keypoints_are_never_visible(dataset, tmp_path):
    """Bola e homogenea: anotar um eixo arbitrario ensina a rede a prever ruido."""

    config = default_config()
    out = tmp_path / "yolo"
    export_dataset(dataset, out, config, task="pose", ratios=(1.0, 0.0, 0.0))

    lines = (out / "labels" / "train" / "cena_a.txt").read_text(encoding="utf-8").strip().splitlines()
    ball_fields = next(line for line in lines if line.startswith("0 ")).split()
    assert ball_fields[7] == "0" and ball_fields[10] == "0"


def test_data_yaml_declares_classes_and_keypoint_shape(dataset, tmp_path):
    config = default_config()
    out = tmp_path / "yolo"
    export_dataset(dataset, out, config, task="pose")

    content = (out / "data.yaml").read_text(encoding="utf-8")
    assert "0: ball" in content
    assert "1: robot_yellow" in content
    assert "2: robot_blue" in content
    assert "kpt_shape: [2, 3]" in content


def test_detect_task_omits_keypoint_shape(dataset, tmp_path):
    config = default_config()
    out = tmp_path / "yolo"
    export_dataset(dataset, out, config, task="detect")

    assert "kpt_shape" not in (out / "data.yaml").read_text(encoding="utf-8")


def test_opponent_is_exported_as_the_other_team_color(dataset, tmp_path):
    """O modelo aprende cor de time; quem e 'adversario' e decidido em execucao."""

    config = default_config()
    config.team_color = "yellow"
    out = tmp_path / "yolo"
    export_dataset(dataset, out, config, task="detect", ratios=(0.0, 0.0, 1.0))

    line = (out / "labels" / "test" / "cena_b.txt").read_text(encoding="utf-8").strip()
    assert line.startswith("2 ")  # robot_blue
