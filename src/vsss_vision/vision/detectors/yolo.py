"""Detector por rede neural — arquitetura pronta, implementação pendente.

É o objeto central das duas propostas de pesquisa: comparar detecção por
rede neural com a segmentação por cor sob as mesmas cenas, mesma escala
física e mesma instrumentação de latência.

Ponto de partida sugerido: `ayssag/BallnetPose` (Hugging Face) e o
`ballnet_dataset` (Roboflow Universe), publicados junto do TCC de
referência (Marques, 2025) — servem de linha de base pública e evitam
começar o treino do zero. Ver `docs/research/related-work.md`.

Implementação:
  1. `uv sync --extra yolo` (traz `ultralytics`; deixada opcional para
     que o pipeline clássico continue leve).
  2. Carregar o modelo de `config.yolo_model_path` **uma vez** em
     `__init__` — carregar por frame contaminaria a medição de latência.
  3. Em `detect()`, converter caixas/keypoints para
     `DetectedRobot`/`DetectedBall` em pixel do frame retificado, o mesmo
     contrato que `ColorDetector` cumpre.
  4. Orientação: um modelo *pose* com keypoints de frente e trás do robô
     dá `theta` direto (`atan2` entre os dois). Um modelo só de detecção
     não dá — nesse caso devolver `theta=None` e deixar claro no relatório
     que a métrica de orientação não se aplica, em vez de preenchê-la com
     um valor derivado de outra fonte.

A medição de latência deve incluir o pré-processamento (resize,
normalização) e a transferência para a GPU, não só o `forward` — é a
latência ponta a ponta que o laço de controle sente.
"""
from __future__ import annotations

import numpy as np

from vsss_vision.vision.config import VisionConfig
from vsss_vision.vision.detectors.base import Detector, DetectionResult


class YoloDetector(Detector):
    """Ainda não implementado — ver o roteiro no topo do arquivo."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config

    def detect(self, frame: np.ndarray) -> DetectionResult:
        raise NotImplementedError(
            "YoloDetector ainda nao foi implementado. "
            "Ver o roteiro em src/vsss_vision/vision/detectors/yolo.py e "
            "docs/research/related-work.md."
        )
