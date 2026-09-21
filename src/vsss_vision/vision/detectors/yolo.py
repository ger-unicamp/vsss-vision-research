"""Detector por rede neural (YOLO, via Ultralytics).

Contrato idêntico ao `ColorDetector`: entra frame retificado (BGR), sai
`DetectionResult` em pixel. Tudo depois — suavização, conversão para
metros, métricas, latência — é compartilhado, então a comparação entre os
dois mede o detector, não a infraestrutura em volta dele.

Esquema de classes
------------------
O padrão é **por cor de time** (`ball`, `robot_yellow`, `robot_blue`), não
por robô. As regras do VSSS mandam que a cor de identificação alterne
entre partidas e que a etiqueta seja destacável; um modelo com classes
`robot0/robot1/robot2` fica preso às camisas de uma equipe e a uma cor.
É o que limita o modelo público do trabalho de referência, que detecta os
três robôs da própria equipe (camisas azuis) e a bola, e **nenhum robô
adversário**. Com classes por cor: trocar de lado é mudar `team_color` na
configuração, e o adversário é detectado pelo mesmo modelo.

O mapa `config.yolo_class_map` traduz nome de classe do modelo para papel
semântico, e aceita os dois esquemas — um esquema por robô é expressável
acrescentando `"robot_id"` à entrada. Isso mantém a reprodução do trabalho
de referência a uma mudança de configuração de distância.

Identidade e orientação
-----------------------
A classe diz o time, não qual robô. `vision/identity.py` resolve o
`robot_id` (marcador dentro da caixa, ou ordem posicional). A orientação
vem dos keypoints quando o modelo é `pose` (`theta = atan2(front - back)`,
mesma convenção do trabalho de referência); num modelo só de detecção,
cai para o vetor corpo -> marcador.

Latência
--------
`profile()` expõe pré-processamento, inferência e pós-processamento
separadamente, como o Ultralytics os reporta. A medição de `detect()` no
`LatencyRecorder` cobre os três — o pré-processamento (resize, normalização)
e a transferência para a GPU fazem parte do que o laço de controle espera,
e medir só o *forward* subestima a latência real.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np

from vsss_vision.vision.config import VisionConfig
from vsss_vision.vision.detectors.base import DetectedBall, DetectedRobot, Detector, DetectionResult
from vsss_vision.vision.identity import RobotCandidate, get_identity_resolver

logger = logging.getLogger("vsss_vision.yolo")

_INSTALL_HINT = (
    "Ultralytics nao esta instalado. Rode `uv sync --extra yolo` "
    "(o detector por rede neural e opcional para manter o pipeline classico leve)."
)


class YoloDetector(Detector):
    """Detecção por YOLO (Ultralytics), com identidade e orientação resolvidas."""

    def __init__(self, config: VisionConfig, model: Any | None = None) -> None:
        self.config = config
        self._resolver = get_identity_resolver(config)
        self._speed: dict[str, float] = {}
        self._unknown_classes: set[str] = set()

        # `model` injetável para teste sem pesos nem GPU; em uso normal é None.
        self._model = model if model is not None else self._load_model()
        self._names: dict[int, str] = dict(getattr(self._model, "names", {}) or {})
        self._device = self._resolve_device(config.yolo_device)
        self._validate_class_map()

    # --- carga e configuração ------------------------------------------

    def _load_model(self) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as error:  # pragma: no cover - depende do ambiente
            raise ImportError(_INSTALL_HINT) from error

        path = self.config.yolo_model_path
        logger.info("Carregando modelo YOLO de %s (task=%s)", path, self.config.yolo_task)
        # Carregado uma única vez: recarregar por frame contaminaria a
        # medição de latência com I/O de disco.
        return YOLO(path, task=self.config.yolo_task)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested
        try:  # pragma: no cover - depende do ambiente
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            return "cpu"

    def _validate_class_map(self) -> None:
        """Falha cedo e alto se o mapa de classes não bate com o modelo.

        Um mapa errado não causa exceção durante a inferência: o detector
        simplesmente não devolve nada, e o experimento reporta 100% de
        detecções perdidas como se fosse resultado. Melhor quebrar aqui.
        """

        if not self._names:
            logger.warning("Modelo sem `names`; nao foi possivel validar o mapa de classes.")
            return

        model_classes = set(self._names.values())
        mapped = set(self.config.yolo_class_map)
        overlap = model_classes & mapped
        if not overlap:
            raise ValueError(
                "Nenhuma classe do modelo aparece em yolo_class_map. "
                f"Classes do modelo: {sorted(model_classes)}. "
                f"Mapeadas na configuracao: {sorted(mapped)}. "
                "Ver docs/research/methodology.md, secao de esquema de classes."
            )
        for missing in sorted(model_classes - mapped):
            logger.warning("Classe %r do modelo nao esta em yolo_class_map; sera ignorada.", missing)

    def warmup(self, frame_shape: tuple[int, int] = (480, 640)) -> None:
        """Roda uma inferência descartável para pagar a carga de pesos e a alocação de GPU."""

        self.detect(np.zeros((*frame_shape, 3), dtype=np.uint8))

    def profile(self) -> dict[str, float]:
        return dict(self._speed)

    # --- inferência ------------------------------------------------------

    def detect(self, frame: np.ndarray) -> DetectionResult:
        results = self._model.predict(
            frame,
            imgsz=self.config.yolo_imgsz,
            conf=self.config.yolo_confidence,
            iou=self.config.yolo_iou,
            device=self._device,
            half=self.config.yolo_half,
            max_det=self.config.yolo_max_detections,
            verbose=False,
        )
        frame_shape = frame.shape[:2]
        if not results:
            self._speed = {}
            return DetectionResult(robots=[], ball=None, frame_shape=frame_shape)

        result = results[0]
        self._speed = {
            f"yolo_{name}": float(value)
            for name, value in (getattr(result, "speed", None) or {}).items()
            if value is not None
        }

        candidates, identified, ball = self._parse(result)
        robots = identified + self._resolver.resolve(frame, candidates)
        return DetectionResult(robots=robots, ball=ball, frame_shape=frame_shape)

    def _parse(
        self, result: Any
    ) -> tuple[list[RobotCandidate], list[DetectedRobot], Optional[DetectedBall]]:
        """Separa as caixas em: candidatos a identificar, já identificados, e bola."""

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return [], [], None

        class_ids = self._to_list(boxes.cls)
        confidences = self._to_list(boxes.conf)
        xyxy = self._to_list(boxes.xyxy)
        keypoints = self._keypoints(result)

        candidates: list[RobotCandidate] = []
        identified: list[DetectedRobot] = []
        best_ball: Optional[tuple[float, DetectedBall]] = None

        for index, class_id in enumerate(class_ids):
            role = self._role(int(class_id))
            if role is None:
                continue

            x1, y1, x2, y2 = (float(value) for value in xyxy[index])
            center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            confidence = float(confidences[index])

            if role["kind"] == "ball":
                # Só existe uma bola em campo: a de maior confiança vence.
                if best_ball is None or confidence > best_ball[0]:
                    best_ball = (confidence, DetectedBall(x=center_x, y=center_y))
                continue

            team = self._team(role.get("color"))
            theta = self._theta(keypoints, index)
            robot_id = role.get("robot_id")
            if robot_id is not None:
                # Esquema por robô: a própria classe carrega a identidade.
                identified.append(
                    DetectedRobot(robot_id=int(robot_id), team=team, x=center_x, y=center_y, theta=theta)
                )
            else:
                candidates.append(
                    RobotCandidate(
                        x=center_x,
                        y=center_y,
                        box=(x1, y1, x2, y2),
                        team=team,
                        confidence=confidence,
                        theta=theta,
                    )
                )

        return candidates, identified, (best_ball[1] if best_ball else None)

    def _role(self, class_id: int) -> Optional[dict[str, Any]]:
        name = self._names.get(class_id)
        if name is None:
            return None
        role = self.config.yolo_class_map.get(name)
        if role is None and name not in self._unknown_classes:
            self._unknown_classes.add(name)
            logger.warning("Classe %r fora de yolo_class_map; deteccoes dela serao ignoradas.", name)
        return role

    def _team(self, color: Optional[str]) -> str:
        """Cor do time -> `own`/`opponent`, segundo `config.team_color`.

        Trocar de lado entre partidas (o que as regras preveem) é mudar
        `team_color`, sem tocar no modelo.
        """

        if color is None:
            return "own"
        return "own" if color == self.config.team_color else "opponent"

    def _keypoints(self, result: Any) -> Optional[list[Any]]:
        if self.config.yolo_task != "pose":
            return None
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            return None
        coordinates = getattr(keypoints, "xy", None)
        if coordinates is None:
            return None
        return self._to_list(coordinates)

    def _theta(self, keypoints: Optional[list[Any]], index: int) -> Optional[float]:
        """Orientação a partir dos keypoints de frente e trás, quando existirem."""

        if keypoints is None or index >= len(keypoints):
            return None

        points = keypoints[index]
        front_index = self.config.yolo_keypoints.get("front", 0)
        back_index = self.config.yolo_keypoints.get("back", 1)
        if len(points) <= max(front_index, back_index):
            return None

        front_x, front_y = float(points[front_index][0]), float(points[front_index][1])
        back_x, back_y = float(points[back_index][0]), float(points[back_index][1])
        # Keypoint ausente vem como (0, 0) no Ultralytics; tratar como
        # "sem orientacao" em vez de devolver um angulo inventado.
        if (front_x, front_y) == (0.0, 0.0) or (back_x, back_y) == (0.0, 0.0):
            return None

        # Y da imagem cresce para baixo; invertido para que theta=0 aponte
        # para a direita no referencial do campo (ver geometry.py).
        return math.atan2(-(front_y - back_y), front_x - back_x)

    @staticmethod
    def _to_list(tensor: Any) -> list[Any]:
        """Aceita tensor do PyTorch, array do numpy ou lista — o teste injeta listas."""

        for attribute in ("cpu", "numpy", "tolist"):
            if hasattr(tensor, attribute):
                tensor = getattr(tensor, attribute)()
        return list(tensor)
