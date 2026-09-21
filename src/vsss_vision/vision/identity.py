"""Atribuição de `robot_id` a caixas detectadas.

Um detector por rede neural treinado com classes por **cor de time**
(o esquema padrão deste repositório, ver `config.yolo_class_map`) diz
*que time* cada robô é, mas não *qual* robô. A identidade tem que vir de
outro lugar — e de onde ela vem é uma decisão metodológica, não um
detalhe:

- **`marker`** — o marcador de cor que a equipe já cola em cada robô,
  procurado **dentro da caixa detectada**. Restringir a busca à caixa é
  mais robusto que a busca global do `ColorDetector`: um falso positivo
  de marcador no meio do campo não tem caixa de robô ao redor e é
  descartado de graça.
- **`positional`** — ordem por área da caixa. Ids instáveis entre frames;
  é a única opção para o adversário, que não tem marcador (não há acesso
  físico ao robô adversário para colar um). Mesma limitação do
  `ColorDetector`, e é justamente o que a **taxa de identificações
  trocadas** mede.

Alternativa deliberadamente **não** adotada como padrão: treinar uma
classe por robô (`robot0`, `robot1`, `robot2`), como fez o trabalho de
referência. Amarra o modelo às camisas de uma equipe e a uma cor de time,
e as regras do VSSS mandam que a cor alterne entre partidas com etiqueta
destacável. Continua expressável via `yolo_class_map` para fins de
reprodução — ver `docs/research/methodology.md`.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np

from vsss_vision.vision.config import ColorRange, VisionConfig
from vsss_vision.vision.detectors.base import DetectedRobot

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2) em pixel


@dataclass(frozen=True)
class RobotCandidate:
    """Um robô detectado, ainda sem identidade atribuída."""

    x: float
    y: float
    box: Box
    team: Literal["own", "opponent"]
    confidence: float = 1.0
    theta: Optional[float] = None  # já resolvido (ex.: keypoints de um modelo pose)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return abs(x2 - x1) * abs(y2 - y1)


class IdentityResolver(ABC):
    """Atribui `robot_id` (e, quando possível, orientação) a candidatos."""

    @abstractmethod
    def resolve(self, frame: np.ndarray, candidates: list[RobotCandidate]) -> list[DetectedRobot]:
        """Devolve os candidatos já identificados; pode descartar os não resolvidos."""


class PositionalIdentityResolver(IdentityResolver):
    """Id = ordem por área da caixa (maior primeiro). Não é estável entre frames."""

    def resolve(self, frame: np.ndarray, candidates: list[RobotCandidate]) -> list[DetectedRobot]:
        robots: list[DetectedRobot] = []
        for team in ("own", "opponent"):
            same_team = sorted(
                (candidate for candidate in candidates if candidate.team == team),
                key=lambda candidate: candidate.area,
                reverse=True,
            )
            robots.extend(
                DetectedRobot(robot_id=index, team=candidate.team, x=candidate.x, y=candidate.y, theta=candidate.theta)
                for index, candidate in enumerate(same_team)
            )
        return robots


class MarkerIdentityResolver(IdentityResolver):
    """Id do time próprio pelo marcador de cor dentro da caixa detectada.

    O adversário não tem marcador, então cai no resolvedor posicional —
    a limitação é do cenário físico, não do método.

    Quando o candidato não traz orientação (modelo só de detecção, sem
    keypoints), o vetor centro-da-caixa -> centróide-do-marcador serve de
    estimativa, a mesma usada pelo `ColorDetector`. Essa estimativa é
    ruidosa quando o marcador está perto do centro do corpo, e isso deve
    ser reportado como tal, não misturado com a orientação vinda de
    keypoints: são fontes diferentes com erros diferentes.
    """

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._fallback = PositionalIdentityResolver()

    def resolve(self, frame: np.ndarray, candidates: list[RobotCandidate]) -> list[DetectedRobot]:
        own = [candidate for candidate in candidates if candidate.team == "own"]
        opponents = [candidate for candidate in candidates if candidate.team == "opponent"]

        robots = self._fallback.resolve(frame, opponents)
        if own:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            robots.extend(self._resolve_own(hsv, own))
        return robots

    def _resolve_own(self, hsv: np.ndarray, own: list[RobotCandidate]) -> list[DetectedRobot]:
        config = self.config
        markers = config.markers[: config.robots_per_team]

        # (robot_id, candidato, centróide do marcador, área) para todo par
        # marcador/caixa com sobreposição suficiente. A atribuição depois é
        # gulosa pela maior área: um marcador parcialmente ocluído não deve
        # roubar a caixa de um marcador plenamente visível.
        scored: list[tuple[float, int, RobotCandidate, tuple[float, float]]] = []
        for robot_id, marker in enumerate(markers):
            for candidate in own:
                found = self._marker_in_box(hsv, marker, candidate.box)
                if found is None:
                    continue
                centroid, area = found
                scored.append((area, robot_id, candidate, centroid))

        scored.sort(key=lambda entry: entry[0], reverse=True)

        robots: list[DetectedRobot] = []
        used_ids: set[int] = set()
        used_candidates: set[int] = set()
        for _area, robot_id, candidate, centroid in scored:
            if robot_id in used_ids or id(candidate) in used_candidates:
                continue
            used_ids.add(robot_id)
            used_candidates.add(id(candidate))

            theta = candidate.theta
            if theta is None:
                # Y da imagem cresce para baixo; invertido para que theta=0
                # aponte para a direita no referencial do campo, como em
                # geometry.pixel_to_meters.
                theta = math.atan2(-(centroid[1] - candidate.y), centroid[0] - candidate.x)

            robots.append(
                DetectedRobot(robot_id=robot_id, team="own", x=candidate.x, y=candidate.y, theta=theta)
            )

        # Caixa do time próprio sem marcador associado (oclusão, marcador
        # fora da faixa calibrada) fica de fora deste frame. O `Tracker`
        # decide se mantém o último estado conhecido — inventar um id aqui
        # produziria exatamente a troca de identidade que queremos medir.
        return robots

    def _marker_in_box(
        self, hsv: np.ndarray, marker: ColorRange, box: Box
    ) -> Optional[tuple[tuple[float, float], float]]:
        """Maior blob do marcador dentro da caixa; devolve (centróide global, área)."""

        height, width = hsv.shape[:2]
        x1 = max(0, int(math.floor(box[0])))
        y1 = max(0, int(math.floor(box[1])))
        x2 = min(width, int(math.ceil(box[2])))
        y2 = min(height, int(math.ceil(box[3])))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return None

        region = hsv[y1:y2, x1:x2]
        mask = cv2.inRange(region, np.array(marker.lower), np.array(marker.upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.config.min_area_marker:
            return None

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        cx = x1 + moments["m10"] / moments["m00"]
        cy = y1 + moments["m01"] / moments["m00"]
        return (cx, cy), area


def get_identity_resolver(config: VisionConfig) -> IdentityResolver:
    if config.yolo_identity == "marker":
        return MarkerIdentityResolver(config)
    if config.yolo_identity == "positional":
        return PositionalIdentityResolver()
    raise ValueError(
        f"Estrategia de identidade desconhecida: {config.yolo_identity!r} (use marker ou positional)"
    )
