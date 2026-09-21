"""YoloDetector com um modelo falso: mapa de classes, identidade, orientação.

Não depende de `ultralytics` nem de pesos nem de GPU — o detector aceita
um modelo injetado. O que está sob teste é a camada que traduz saída de
rede em `DetectionResult`, que é onde moram as decisões metodológicas
(esquema de classes, atribuição de time, resolução de identidade,
convenção de orientação); a rede em si é responsabilidade do Ultralytics.
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytest

from vsss_vision.vision.config import default_config
from vsss_vision.vision.detectors.yolo import YoloDetector

FRAME_SIZE = (480, 640)


@dataclass
class FakeBoxes:
    cls: list[int]
    conf: list[float]
    xyxy: list[list[float]]

    def __len__(self):
        return len(self.cls)


@dataclass
class FakeKeypoints:
    xy: list[list[list[float]]]


@dataclass
class FakeResult:
    boxes: FakeBoxes
    keypoints: FakeKeypoints | None = None
    speed: dict = field(default_factory=lambda: {"preprocess": 1.5, "inference": 8.0, "postprocess": 0.5})


@dataclass
class FakeModel:
    """Substituto de `ultralytics.YOLO` com saída fixa."""

    names: dict
    result: FakeResult
    calls: list = field(default_factory=list)

    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


TEAM_NAMES = {0: "ball", 1: "robot_yellow", 2: "robot_blue"}


def _box(center, size=40):
    x, y = center
    half = size / 2
    return [x - half, y - half, x + half, y + half]


def _frame():
    return np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)


def _draw_hsv_circle(frame, center, radius, hsv_mid):
    hsv_pixel = np.uint8([[hsv_mid]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
    cv2.circle(frame, center, radius, tuple(int(channel) for channel in bgr_pixel), -1)


def _mid(color_range):
    return tuple((low + high) // 2 for low, high in zip(color_range.lower, color_range.upper))


def test_team_color_classes_map_to_own_and_opponent():
    """A mesma rede serve os dois lados: trocar `team_color` inverte os papeis."""

    config = default_config()
    config.team_color = "yellow"
    config.yolo_identity = "positional"
    result = FakeResult(
        boxes=FakeBoxes(
            cls=[1, 2],
            conf=[0.9, 0.8],
            xyxy=[_box((100, 100)), _box((500, 300))],
        )
    )
    model = FakeModel(names=TEAM_NAMES, result=result)

    detection = YoloDetector(config, model=model).detect(_frame())
    teams = {robot.team for robot in detection.robots}
    assert teams == {"own", "opponent"}
    own = next(robot for robot in detection.robots if robot.team == "own")
    assert own.x == pytest.approx(100)

    config.team_color = "blue"
    detection = YoloDetector(config, model=FakeModel(names=TEAM_NAMES, result=result)).detect(_frame())
    own = next(robot for robot in detection.robots if robot.team == "own")
    assert own.x == pytest.approx(500)  # agora o azul e o proprio


def test_pose_keypoints_give_orientation_with_field_convention():
    """front a direita de back -> theta ~ 0, apesar de Y da imagem crescer para baixo."""

    config = default_config()
    config.yolo_identity = "positional"
    result = FakeResult(
        boxes=FakeBoxes(cls=[1], conf=[0.9], xyxy=[_box((200, 200))]),
        keypoints=FakeKeypoints(xy=[[[220.0, 200.0], [180.0, 200.0]]]),
    )

    detection = YoloDetector(config, model=FakeModel(names=TEAM_NAMES, result=result)).detect(_frame())

    assert detection.robots[0].theta == pytest.approx(0.0, abs=1e-6)


def test_keypoint_pointing_up_in_the_image_is_a_positive_angle():
    config = default_config()
    config.yolo_identity = "positional"
    result = FakeResult(
        boxes=FakeBoxes(cls=[1], conf=[0.9], xyxy=[_box((200, 200))]),
        # front acima de back na imagem (y menor) -> +90 graus no campo
        keypoints=FakeKeypoints(xy=[[[200.0, 180.0], [200.0, 220.0]]]),
    )

    detection = YoloDetector(config, model=FakeModel(names=TEAM_NAMES, result=result)).detect(_frame())

    assert math.degrees(detection.robots[0].theta) == pytest.approx(90.0, abs=1e-6)


def test_missing_keypoint_gives_no_orientation_instead_of_a_made_up_one():
    config = default_config()
    config.yolo_identity = "positional"
    result = FakeResult(
        boxes=FakeBoxes(cls=[1], conf=[0.9], xyxy=[_box((200, 200))]),
        keypoints=FakeKeypoints(xy=[[[0.0, 0.0], [180.0, 200.0]]]),  # ausente no Ultralytics
    )

    detection = YoloDetector(config, model=FakeModel(names=TEAM_NAMES, result=result)).detect(_frame())

    assert detection.robots[0].theta is None


def test_marker_inside_the_box_resolves_a_stable_robot_id():
    config = default_config()
    config.yolo_identity = "marker"
    config.yolo_task = "detect"

    frame = _frame()
    # Marcador do robo 1 dentro da caixa; o do robo 0 nao esta em campo.
    _draw_hsv_circle(frame, (210, 200), 6, _mid(config.markers[1]))

    result = FakeResult(boxes=FakeBoxes(cls=[1], conf=[0.9], xyxy=[_box((200, 200), size=80)]))
    detection = YoloDetector(config, model=FakeModel(names=TEAM_NAMES, result=result)).detect(frame)

    assert len(detection.robots) == 1
    assert detection.robots[0].robot_id == 1
    # Sem keypoints, a orientacao vem do vetor corpo -> marcador.
    assert detection.robots[0].theta == pytest.approx(0.0, abs=0.2)


def test_own_box_without_marker_is_dropped_not_given_an_invented_id():
    config = default_config()
    config.yolo_identity = "marker"
    result = FakeResult(boxes=FakeBoxes(cls=[1], conf=[0.9], xyxy=[_box((200, 200))]))

    detection = YoloDetector(config, model=FakeModel(names=TEAM_NAMES, result=result)).detect(_frame())

    assert detection.robots == []


def test_per_robot_class_scheme_carries_identity_in_the_class():
    """Esquema do trabalho de referencia, reproduzivel por configuracao."""

    config = default_config()
    config.yolo_class_map = {
        "ball": {"kind": "ball"},
        "robot0": {"kind": "robot", "color": "blue", "robot_id": 0},
        "robot1": {"kind": "robot", "color": "blue", "robot_id": 1},
    }
    config.team_color = "blue"
    names = {0: "ball", 1: "robot0", 2: "robot1"}
    result = FakeResult(
        boxes=FakeBoxes(cls=[2, 1], conf=[0.8, 0.9], xyxy=[_box((400, 100)), _box((100, 100))])
    )

    detection = YoloDetector(config, model=FakeModel(names=names, result=result)).detect(_frame())

    by_id = {robot.robot_id: robot for robot in detection.robots}
    assert set(by_id) == {0, 1}
    assert by_id[0].x == pytest.approx(100)
    assert by_id[1].x == pytest.approx(400)
    assert all(robot.team == "own" for robot in detection.robots)


def test_highest_confidence_ball_wins():
    config = default_config()
    config.yolo_identity = "positional"
    result = FakeResult(
        boxes=FakeBoxes(cls=[0, 0], conf=[0.3, 0.95], xyxy=[_box((50, 50)), _box((300, 300))])
    )

    detection = YoloDetector(config, model=FakeModel(names=TEAM_NAMES, result=result)).detect(_frame())

    assert detection.ball.x == pytest.approx(300)


def test_class_map_that_matches_nothing_fails_at_construction():
    """Mapa errado nao pode virar '100% de deteccoes perdidas' silencioso."""

    config = default_config()
    result = FakeResult(boxes=FakeBoxes(cls=[], conf=[], xyxy=[]))
    model = FakeModel(names={0: "bola", 1: "robo_amarelo"}, result=result)

    with pytest.raises(ValueError, match="yolo_class_map"):
        YoloDetector(config, model=model)


def test_profile_exposes_yolo_internal_stages():
    config = default_config()
    config.yolo_identity = "positional"
    result = FakeResult(boxes=FakeBoxes(cls=[], conf=[], xyxy=[]))
    detector = YoloDetector(config, model=FakeModel(names=TEAM_NAMES, result=result))

    detector.detect(_frame())

    assert detector.profile() == {
        "yolo_preprocess": 1.5,
        "yolo_inference": 8.0,
        "yolo_postprocess": 0.5,
    }


def test_inference_parameters_come_from_the_config():
    """imgsz e o parametro varrido na P2; precisa chegar ao modelo."""

    config = default_config()
    config.yolo_identity = "positional"
    config.yolo_imgsz = 320
    config.yolo_confidence = 0.4
    config.yolo_device = "cpu"
    model = FakeModel(names=TEAM_NAMES, result=FakeResult(boxes=FakeBoxes(cls=[], conf=[], xyxy=[])))

    YoloDetector(config, model=model).detect(_frame())

    assert model.calls[0]["imgsz"] == 320
    assert model.calls[0]["conf"] == 0.4
    assert model.calls[0]["device"] == "cpu"
