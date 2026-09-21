"""Smoke tests do módulo de visão: ColorDetector + Tracker.

Usa imagens sintéticas geradas em memória (numpy/cv2, sem arquivos em
disco), então não depende do dataset em datasets/vision/ (que ainda
não tem ground truth em JSON). Cobre exatamente a lógica nova/crítica:
resolução de identidade por marcador e correção do bug de wraparound
angular da versão antiga.
"""
import math

import cv2
import numpy as np
import pytest

from vsss_vision.vision.config import default_config
from vsss_vision.vision.detectors.base import DetectedRobot, DetectionResult
from vsss_vision.vision.detectors.color import ColorDetector
from vsss_vision.vision.tracker import Tracker

FRAME_SIZE = (480, 640)  # (height, width)


def _blank_frame() -> np.ndarray:
    return np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)


def _draw_hsv_circle(frame: np.ndarray, center: tuple[int, int], radius: int, hsv_mid: tuple[int, int, int]) -> None:
    """Pinta um círculo BGR equivalente ao ponto médio de uma faixa HSV."""

    hsv_pixel = np.uint8([[hsv_mid]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
    cv2.circle(frame, center, radius, tuple(int(c) for c in bgr_pixel), -1)


def _mid(color_range) -> tuple[int, int, int]:
    return tuple((lo + hi) // 2 for lo, hi in zip(color_range.lower, color_range.upper))


def test_color_detector_detects_own_robot_and_ball():
    config = default_config()
    frame = _blank_frame()

    body_center = (200, 200)
    marker_center = (230, 200)  # deslocado em +X -> theta esperado próximo de 0
    ball_center = (400, 300)

    _draw_hsv_circle(frame, body_center, radius=20, hsv_mid=_mid(config.own_color))
    _draw_hsv_circle(frame, marker_center, radius=6, hsv_mid=_mid(config.markers[0]))
    _draw_hsv_circle(frame, ball_center, radius=10, hsv_mid=_mid(config.ball_color))

    result = ColorDetector(config).detect(frame)

    assert len(result.robots) == 1
    robot = result.robots[0]
    assert robot.team == "own"
    assert robot.robot_id == 0
    assert math.isclose(robot.x, body_center[0], abs_tol=3)
    assert math.isclose(robot.y, body_center[1], abs_tol=3)
    assert robot.theta is not None
    assert abs(robot.theta) < 0.2  # marcador alinhado no eixo X -> ~0 rad

    assert result.ball is not None
    assert math.isclose(result.ball.x, ball_center[0], abs_tol=3)
    assert math.isclose(result.ball.y, ball_center[1], abs_tol=3)


def test_color_detector_skips_robot_without_marker():
    config = default_config()
    frame = _blank_frame()
    _draw_hsv_circle(frame, (200, 200), radius=20, hsv_mid=_mid(config.own_color))
    # Nenhum marcador desenhado -> identidade não pode ser resolvida.

    result = ColorDetector(config).detect(frame)

    assert result.robots == []


def test_color_detector_orders_opponents_by_area():
    config = default_config()
    frame = _blank_frame()
    _draw_hsv_circle(frame, (150, 150), radius=12, hsv_mid=_mid(config.opponent_color))
    _draw_hsv_circle(frame, (450, 350), radius=25, hsv_mid=_mid(config.opponent_color))

    result = ColorDetector(config).detect(frame)

    opponents = [r for r in result.robots if r.team == "opponent"]
    assert len(opponents) == 2
    assert opponents[0].robot_id == 0
    assert math.isclose(opponents[0].x, 450, abs_tol=3)  # maior blob primeiro
    assert opponents[1].robot_id == 1
    assert math.isclose(opponents[1].x, 150, abs_tol=3)


def test_tracker_handles_angle_wraparound():
    """EMA linear ingênua salta ~180 graus na virada 359deg->1deg; a EMA
    circular usada em Tracker deve manter o ângulo suavizado sempre
    próximo do ângulo real observado."""

    tracker = Tracker(alpha_pos=0.0, alpha_angle=0.5, stale_timeout_s=5.0)
    frame_shape = FRAME_SIZE

    angles_deg = [355, 358, 1, 4, 7]
    for deg in angles_deg:
        theta = math.radians(deg)
        detection = DetectionResult(
            robots=[DetectedRobot(robot_id=0, team="own", x=100.0, y=100.0, theta=theta)],
            ball=None,
            frame_shape=frame_shape,
        )
        smoothed = tracker.update(detection)
        smoothed_theta = smoothed.robots[0].theta

        # distância angular circular entre o suavizado e o real observado
        diff = abs(math.atan2(math.sin(smoothed_theta - theta), math.cos(smoothed_theta - theta)))
        assert diff < math.radians(30), f"salto angular grande demais em {deg} graus: diff={math.degrees(diff)}"
