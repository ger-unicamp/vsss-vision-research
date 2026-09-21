"""Contrato comum entre implementações de detecção (Strategy Pattern).

Qualquer `Detector` recebe o frame já retificado (sem distorção de
perspectiva — isso é responsabilidade de `src/vsss_vision/camera/recorte.py`, feito
antes do frame chegar aqui) e devolve posições em PIXEL, não em metros.
A conversão para metros é feita depois, em `geometry.py`, então nenhum
Detector precisa conhecer as dimensões físicas do campo — e a mesma
conversão se aplica a todos, o que mantém a comparação entre detectores
livre de diferenças de infraestrutura.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass(frozen=True)
class DetectedRobot:
    """Um robô detectado em coordenadas de pixel do frame retificado.

    `robot_id` é único dentro do seu `team` (0..robots_per_team-1), nunca
    entre times. `theta` é `None` quando a orientação não pôde ser
    calculada neste frame (ex.: marcador não encontrado).
    """

    robot_id: int
    team: Literal["own", "opponent"]
    x: float
    y: float
    theta: Optional[float]


@dataclass(frozen=True)
class DetectedBall:
    x: float
    y: float


@dataclass
class DetectionResult:
    """Saída de um `Detector` (ou de `Tracker.update`, já suavizada) para um frame."""

    robots: list[DetectedRobot]
    ball: Optional[DetectedBall]
    frame_shape: tuple[int, int]  # (height, width) — usado em geometry.py para escala px->m


class Detector(ABC):
    """Interface plugável de detecção de robôs/bola em um frame retificado."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Detecta robôs e bola em `frame` (BGR, uint8) e devolve posições em pixel."""

    def profile(self) -> dict[str, float]:
        """Tempos internos do último `detect()`, em milissegundos.

        Vazio por padrão. Um detector que já mede as próprias etapas (o
        YOLO reporta pré-processamento, inferência e pós-processamento
        separadamente) devolve esses tempos aqui, e o `LatencyRecorder` os
        grava como estágios próprios. Sem isso, a proposta P2 só conseguiria
        dizer "a inferência é lenta", nunca *qual parte* dela é lenta — e a
        conclusão sobre qual variante usar muda se o custo estiver no
        pré-processamento e não na rede.
        """

        return {}

    def warmup(self) -> None:
        """Paga custos de primeira execução (alocação, carga de pesos, JIT).

        Chamado antes de qualquer medição. O padrão não faz nada.
        """
