"""Conversão entre o domínio do detector (pixel) e o domínio físico (metro).

Derivado de `src/vision/protocol.py` do repositório de jogo
(`ger-unicamp/futebol-vsss`). Naquele repositório este arquivo também
serializava um protobuf `Environment` (vssproto) e o publicava por UDP
multicast para a estratégia. Aqui esse transporte foi **removido de
propósito**: este repositório é de pesquisa, e a saída de interesse é o
estado do campo em unidades físicas (metros/centímetros), que é o que as
métricas de erro exigem — não um pacote de rede para um time de jogo.

Consequência prática: nenhuma dependência de `vssproto`/`protobuf` aqui.
Para voltar a rodar em jogo real (necessário, por exemplo, para medir o
efeito da visão sobre o comportamento do robô na proposta P2), basta
acrescentar um módulo de saída que consuma `FieldState` e serialize —
nenhuma outra parte do pipeline precisa mudar.

Convenção de coordenadas
------------------------
Pixel: origem no canto superior-esquerdo, Y cresce para baixo.
Campo: origem no CENTRO, Y cresce para cima, X cresce para a direita —
a mesma convenção de `vssproto`/FIRASim, mantida para que os resultados
sejam comparáveis com o sistema de jogo e com a literatura.

Assume que o frame retificado (produzido por `camera/recorte.py`) cobre
exatamente o retângulo físico do campo, isto é, que o ROI calibrado em
`tools/camera_configurator.py` foi posicionado nos 4 cantos reais. Em
escala real isso é uma hipótese a ser **verificada**, não assumida: ver
`docs/research/p1-illumination.md`, seção de ground truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from vsss_vision.vision.config import VisionConfig
from vsss_vision.vision.detectors.base import DetectionResult


@dataclass(frozen=True)
class FieldRobot:
    """Robô em coordenadas físicas do campo (metros, origem no centro)."""

    robot_id: int
    team: Literal["own", "opponent"]
    x_m: float
    y_m: float
    theta_rad: Optional[float]


@dataclass(frozen=True)
class FieldBall:
    x_m: float
    y_m: float


@dataclass(frozen=True)
class FieldState:
    """Estado do campo em unidades físicas — unidade de comparação das métricas.

    É esta estrutura (não `DetectionResult`, em pixel) que
    `benchmark/metrics.py` compara com o ground truth: erro em pixel não
    é comparável entre resoluções nem entre montagens de câmera — uma das
    limitações declaradas do TCC de referência (Marques, 2025), ver
    `docs/research/related-work.md`.
    """

    robots: list[FieldRobot]
    ball: Optional[FieldBall]


def pixel_to_meters(
    x_px: float,
    y_px: float,
    frame_shape: tuple[int, int],
    field_length_m: float,
    field_width_m: float,
) -> tuple[float, float]:
    """Converte um ponto do frame retificado para metros no referencial do campo."""

    height, width = frame_shape
    x_m = (x_px / width) * field_length_m - field_length_m / 2
    y_m = field_width_m / 2 - (y_px / height) * field_width_m
    return x_m, y_m


def meters_to_pixel(
    x_m: float,
    y_m: float,
    frame_shape: tuple[int, int],
    field_length_m: float,
    field_width_m: float,
) -> tuple[float, float]:
    """Inverso de `pixel_to_meters` — usado para desenhar ground truth sobre o frame."""

    height, width = frame_shape
    x_px = (x_m + field_length_m / 2) / field_length_m * width
    y_px = (field_width_m / 2 - y_m) / field_width_m * height
    return x_px, y_px


def pixels_per_meter(
    frame_shape: tuple[int, int], field_length_m: float, field_width_m: float
) -> tuple[float, float]:
    """Escala (px/m) em X e Y. Valores muito diferentes indicam ROI fora de proporção com o campo."""

    height, width = frame_shape
    return width / field_length_m, height / field_width_m


def to_field_state(detection: DetectionResult, config: VisionConfig) -> FieldState:
    """Converte uma `DetectionResult` (pixel) para `FieldState` (metros)."""

    robots = []
    for robot in detection.robots:
        x_m, y_m = pixel_to_meters(
            robot.x, robot.y, detection.frame_shape,
            config.field_length_m, config.field_width_m,
        )
        robots.append(
            FieldRobot(robot_id=robot.robot_id, team=robot.team, x_m=x_m, y_m=y_m, theta_rad=robot.theta)
        )

    ball = None
    if detection.ball is not None:
        x_m, y_m = pixel_to_meters(
            detection.ball.x, detection.ball.y, detection.frame_shape,
            config.field_length_m, config.field_width_m,
        )
        ball = FieldBall(x_m=x_m, y_m=y_m)

    return FieldState(robots=robots, ball=ball)


def angular_difference(a_rad: float, b_rad: float) -> float:
    """Menor diferença angular com sinal entre dois ângulos, em radianos, no intervalo (-pi, pi].

    Necessária porque erro de orientação calculado por subtração crua
    reporta ~360 graus na transição 359 graus -> 1 grau — o mesmo bug de
    wraparound que `tracker.py` corrige na suavização.
    """

    return math.atan2(math.sin(a_rad - b_rad), math.cos(a_rad - b_rad))
