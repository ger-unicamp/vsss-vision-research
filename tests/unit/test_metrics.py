"""Associação predição/ground truth e as quatro métricas reportadas."""
import math

import pytest

from vsss_vision.benchmark.metrics import aggregate, evaluate_frame, percentile
from vsss_vision.vision.geometry import FieldBall, FieldRobot, FieldState


def robot(robot_id, x_m, y_m, theta_deg=None, team="own"):
    return FieldRobot(
        robot_id=robot_id,
        team=team,
        x_m=x_m,
        y_m=y_m,
        theta_rad=None if theta_deg is None else math.radians(theta_deg),
    )


def test_position_error_reported_in_centimeters():
    truth = FieldState(robots=[robot(0, 0.0, 0.0)], ball=None)
    prediction = FieldState(robots=[robot(0, 0.03, 0.04)], ball=None)  # 5 cm

    metrics = evaluate_frame(prediction, truth)

    assert metrics.position_errors_cm == [pytest.approx(5.0)]
    assert metrics.missed_count == 0
    assert metrics.id_swap_count == 0


def test_orientation_error_does_not_blow_up_at_wraparound():
    truth = FieldState(robots=[robot(0, 0.0, 0.0, theta_deg=359)], ball=None)
    prediction = FieldState(robots=[robot(0, 0.0, 0.0, theta_deg=1)], ball=None)

    metrics = evaluate_frame(prediction, truth)

    assert metrics.orientation_errors_deg == [pytest.approx(2.0, abs=1e-6)]


def test_detection_beyond_match_radius_counts_as_missed_and_false_positive():
    truth = FieldState(robots=[robot(0, 0.0, 0.0)], ball=None)
    prediction = FieldState(robots=[robot(0, 0.5, 0.5)], ball=None)

    metrics = evaluate_frame(prediction, truth, match_radius_cm=10.0)

    assert metrics.missed_count == 1
    assert metrics.false_positive_count == 1
    assert metrics.position_errors_cm == []


def test_id_swap_counted_as_swap_not_as_miss():
    """Dois robôs detectados nas posições certas, com os ids trocados.

    É o caso que a associação por id esconderia: apareceria como duas
    detecções perdidas mais dois falsos positivos, e a troca de identidade
    nunca seria contada.
    """

    truth = FieldState(robots=[robot(0, -0.30, 0.0), robot(1, 0.30, 0.0)], ball=None)
    prediction = FieldState(robots=[robot(1, -0.30, 0.0), robot(0, 0.30, 0.0)], ball=None)

    metrics = evaluate_frame(prediction, truth)

    assert metrics.id_swap_count == 2
    assert metrics.missed_count == 0
    assert metrics.false_positive_count == 0
    assert metrics.matched_count == 2


def test_robots_of_different_teams_never_match_each_other():
    truth = FieldState(robots=[robot(0, 0.0, 0.0, team="own")], ball=None)
    prediction = FieldState(robots=[robot(0, 0.0, 0.0, team="opponent")], ball=None)

    metrics = evaluate_frame(prediction, truth)

    assert metrics.missed_count == 1
    assert metrics.false_positive_count == 1


def test_ball_error_and_ball_miss():
    truth = FieldState(robots=[], ball=FieldBall(0.0, 0.0))

    seen = evaluate_frame(FieldState(robots=[], ball=FieldBall(0.10, 0.0)), truth)
    assert seen.ball_error_cm == pytest.approx(10.0)
    assert seen.ball_missed is False

    missed = evaluate_frame(FieldState(robots=[], ball=None), truth)
    assert missed.ball_missed is True
    assert missed.ball_error_cm is None


def test_aggregate_rates_use_the_right_denominators():
    truth = FieldState(robots=[robot(0, -0.30, 0.0), robot(1, 0.30, 0.0)], ball=None)
    # Um robô certo, outro fora do raio: 1 perdida, 1 falso positivo, 0 trocas.
    prediction = FieldState(robots=[robot(0, -0.30, 0.0), robot(1, 0.30, 0.60)], ball=None)

    summary = aggregate([evaluate_frame(prediction, truth)])

    assert summary.frames == 1
    assert summary.miss_rate == pytest.approx(0.5)  # sobre o total anotado
    assert summary.false_positive_rate == pytest.approx(0.5)
    assert summary.id_swap_rate == pytest.approx(0.0)  # sobre o que foi associado


def test_aggregate_of_nothing_is_empty_not_a_crash():
    summary = aggregate([])
    assert summary.frames == 0
    assert math.isnan(summary.position_error_mean_cm)


def test_percentile_interpolates():
    assert percentile([0, 10], 50) == pytest.approx(5.0)
    assert percentile([1, 2, 3, 4], 100) == pytest.approx(4.0)
    assert math.isnan(percentile([], 95))
