"""Conversão pixel <-> metro e diferença angular circular."""
import math

import pytest

from vsss_vision.vision.config import default_config
from vsss_vision.vision.detectors.base import DetectedBall, DetectedRobot, DetectionResult
from vsss_vision.vision.geometry import (
    angular_difference,
    meters_to_pixel,
    pixel_to_meters,
    pixels_per_meter,
    to_field_state,
)

FRAME_SHAPE = (300, 600)  # (height, width)
FIELD_LENGTH_M, FIELD_WIDTH_M = 1.50, 1.30


def test_frame_center_maps_to_field_origin():
    x_m, y_m = pixel_to_meters(300, 150, FRAME_SHAPE, FIELD_LENGTH_M, FIELD_WIDTH_M)
    assert x_m == pytest.approx(0.0)
    assert y_m == pytest.approx(0.0)


def test_top_left_pixel_maps_to_upper_left_corner():
    """Y do pixel cresce para baixo, Y do campo para cima: o topo vira +Y."""

    x_m, y_m = pixel_to_meters(0, 0, FRAME_SHAPE, FIELD_LENGTH_M, FIELD_WIDTH_M)
    assert x_m == pytest.approx(-FIELD_LENGTH_M / 2)
    assert y_m == pytest.approx(+FIELD_WIDTH_M / 2)


@pytest.mark.parametrize("point_px", [(0, 0), (123, 45), (599, 299), (300, 150)])
def test_meters_to_pixel_is_the_inverse(point_px):
    x_m, y_m = pixel_to_meters(*point_px, FRAME_SHAPE, FIELD_LENGTH_M, FIELD_WIDTH_M)
    back_px = meters_to_pixel(x_m, y_m, FRAME_SHAPE, FIELD_LENGTH_M, FIELD_WIDTH_M)
    assert back_px[0] == pytest.approx(point_px[0])
    assert back_px[1] == pytest.approx(point_px[1])


def test_pixels_per_meter_reports_both_axes():
    scale_x, scale_y = pixels_per_meter(FRAME_SHAPE, FIELD_LENGTH_M, FIELD_WIDTH_M)
    assert scale_x == pytest.approx(600 / 1.50)
    assert scale_y == pytest.approx(300 / 1.30)


def test_angular_difference_is_circular():
    """Subtração crua daria ~358 graus; a diferença circular deve dar 2."""

    difference = angular_difference(math.radians(1), math.radians(359))
    assert math.degrees(difference) == pytest.approx(2.0, abs=1e-6)


def test_to_field_state_converts_robots_and_ball():
    config = default_config()
    detection = DetectionResult(
        robots=[DetectedRobot(robot_id=1, team="own", x=600.0, y=150.0, theta=0.5)],
        ball=DetectedBall(x=300.0, y=150.0),
        frame_shape=FRAME_SHAPE,
    )

    state = to_field_state(detection, config)

    assert state.ball.x_m == pytest.approx(0.0)
    assert state.robots[0].x_m == pytest.approx(config.field_length_m / 2)
    assert state.robots[0].theta_rad == pytest.approx(0.5)
    assert state.robots[0].robot_id == 1
