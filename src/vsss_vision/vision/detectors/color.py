"""Detecção por segmentação de cor (HSV + contornos) — linha de base clássica.

É o método contra o qual as abordagens por rede neural são comparadas, e
o que a maioria das equipes de VSSS usa hoje. Para que a comparação seja
justa, este detector precisa estar **bem calibrado**, não numa versão
propositalmente fraca: as faixas HSV devem ser ajustadas com
`tools/vision_calibrator.py` para cada condição avaliada.

A posição de cada robô próprio é o centróide do CORPO (blob grande da cor
do time), não do marcador de identidade (blob pequeno): o corpo tem área
maior e oscila menos frame a frame. O marcador só decide "qual corpo é o
robô N" e dá o ângulo.

Limitação inerente ao método, não um defeito da implementação: sem um
marcador individual, o adversário não tem `robot_id` estável entre
frames — a única informação disponível é a posição, então o id é apenas a
ordem por área do blob (maior primeiro). Não há acesso físico ao robô
adversário para colar um marcador. Isso torna a **taxa de identificações
trocadas** (ver `benchmark/metrics.py`) uma das métricas em que se espera
que a detecção por rede neural leve vantagem.
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from vsss_vision.vision.config import ColorRange, VisionConfig
from vsss_vision.vision.detectors.base import DetectedBall, DetectedRobot, Detector, DetectionResult


class ColorDetector(Detector):
    """Detector HSV+contornos. Único detector implementado hoje."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config

    def detect(self, frame: np.ndarray) -> DetectionResult:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cfg = self.config

        own_mask = self._mask(hsv, cfg.own_color)
        opponent_mask = self._mask(hsv, cfg.opponent_color)
        ball_mask = self._mask(hsv, cfg.ball_color)
        marker_masks = [self._mask(hsv, marker) for marker in cfg.markers[: cfg.robots_per_team]]

        own_bodies = self._largest_centroids(own_mask, cfg.robots_per_team, cfg.min_area_robot)
        opponent_bodies = self._largest_centroids(opponent_mask, cfg.robots_per_team, cfg.min_area_robot)

        robots: list[DetectedRobot] = []
        robots.extend(self._resolve_own_robots(own_bodies, marker_masks))
        robots.extend(
            DetectedRobot(robot_id=i, team="opponent", x=x, y=y, theta=None)
            for i, (x, y, _area) in enumerate(opponent_bodies)
        )

        ball_center = self._largest_centroid(ball_mask, cfg.min_area_ball)
        ball = DetectedBall(*ball_center) if ball_center is not None else None

        return DetectionResult(robots=robots, ball=ball, frame_shape=frame.shape[:2])

    def _resolve_own_robots(
        self,
        own_bodies: list[tuple[float, float, float]],
        marker_masks: list[np.ndarray],
    ) -> list[DetectedRobot]:
        """Associa cada marcador ao corpo mais próximo para dar identidade estável.

        Um robô sem marcador correspondente encontrado (oclusão, marcador
        fora da faixa de cor calibrada, etc.) simplesmente não entra no
        resultado deste frame — `Tracker` é quem decide se mantém o último
        estado conhecido ou descarta após `stale_timeout_s`.
        """

        cfg = self.config
        resolved: list[DetectedRobot] = []
        for robot_id, marker_mask in enumerate(marker_masks):
            marker_center = self._largest_centroid(marker_mask, cfg.min_area_marker)
            if marker_center is None or not own_bodies:
                continue

            body_x, body_y = self._closest_point(marker_center, own_bodies)
            distance = math.hypot(marker_center[0] - body_x, marker_center[1] - body_y)
            if distance > cfg.max_marker_body_distance_px:
                continue

            # Y da imagem cresce para baixo; invertido aqui para que theta=0
            # aponte "para a direita" no sistema de coordenadas do campo
            # (eixo Y crescendo para cima), consistente com pixel_to_meters
            # em geometry.py.
            theta = math.atan2(-(marker_center[1] - body_y), marker_center[0] - body_x)
            resolved.append(DetectedRobot(robot_id=robot_id, team="own", x=body_x, y=body_y, theta=theta))

        return resolved

    @staticmethod
    def _mask(hsv: np.ndarray, color: ColorRange) -> np.ndarray:
        return cv2.inRange(hsv, np.array(color.lower), np.array(color.upper))

    @staticmethod
    def _largest_centroids(
        mask: np.ndarray, max_blobs: int, min_area: float
    ) -> list[tuple[float, float, float]]:
        """Até `max_blobs` maiores contornos da máscara, como (x, y, área), desc."""

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [(c, cv2.contourArea(c)) for c in contours]
        candidates = [(c, area) for c, area in candidates if area >= min_area]
        candidates.sort(key=lambda pair: pair[1], reverse=True)

        centroids: list[tuple[float, float, float]] = []
        for contour, area in candidates[:max_blobs]:
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            centroids.append((cx, cy, area))
        return centroids

    @classmethod
    def _largest_centroid(cls, mask: np.ndarray, min_area: float) -> Optional[tuple[float, float]]:
        centroids = cls._largest_centroids(mask, max_blobs=1, min_area=min_area)
        if not centroids:
            return None
        x, y, _area = centroids[0]
        return x, y

    @staticmethod
    def _closest_point(
        target: tuple[float, float], candidates: list[tuple[float, float, float]]
    ) -> tuple[float, float]:
        tx, ty = target
        best = min(candidates, key=lambda c: math.hypot(c[0] - tx, c[1] - ty))
        return best[0], best[1]
